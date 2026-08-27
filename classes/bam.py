# coding: UTF-8
#!/usr/bin/env python
"""
bam.py  -  optimised BAM reader for AS detection
=================================================

Key changes vs. original
-------------------------
1. pysam.AlignmentFile.fetch()  replaces  pysam.view()
   pysam.view() decodes the entire BAM to a text string then re-splits and
   re-parses every field.  AlignmentFile.fetch() returns native AlignedSegment
   objects; all field access is done in C, giving ~10-30- speedup for large
   BAMs.

2. chrom parameter on parseBam()
   When the chromosome is known (e.g. from the parallel dispatch in
   detect_AS.py) only reads on that chromosome are streamed, cutting I/O
   to 1/N of the file for N chromosomes.

3. O(log n) gene lookup via GTF.find_gene()
   The original code iterated ALL genes on a chromosome for every read.
   GTF now exposes a bisect-based find_gene() that is O(log n_genes).

4. Native pysam flag / tag access
   - read.is_paired, read.is_duplicate, read.mate_is_unmapped  (C-level)
     replace the hand-rolled _checkFlag() which also rebuilt a list on every call.
   - read.get_tag('NH')  replaces _get_tag() which rebuilt a dict from raw
     tag strings on every call.

5. CIGAR via cigartuples  (list of (op_code, length) pairs, C-level)
   replaces string split + re-parse.

6. Bug fix: the original _get_coverage() was passed to map() without
   consuming the result.  In Python 3, map() is lazy, so _get_coverage()
   never executed - self.cov and self.inc_o were never populated for exonic
   reads, silently breaking RI detection.  Now replaced with an explicit
   for loop.

7. defaultdict(int) for all counters - no need to test key existence before
   incrementing.

8. All print / logging statements removed from inner loops.
"""

import sys
import os
import logging
from collections import defaultdict
import pysam

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s %(message)s',
    filename='log_read_bams.txt'
)

# CIGAR operation codes (pysam cigartuples)
_OP_MATCH  = 0   # M
_OP_INS    = 1   # I
_OP_DEL    = 2   # D
_OP_SKIP   = 3   # N  (intron in RNA-seq)
_OP_SOFT   = 4   # S
_OP_HARD   = 5   # H
_OP_PAD    = 6   # P
_OP_EQUAL  = 7   # =
_OP_DIFF   = 8   # X

# Operations that disqualify a read (same set as original CIGAR string check)
_SKIP_OPS = {_OP_INS, _OP_DEL, _OP_SOFT, _OP_HARD, _OP_PAD, _OP_EQUAL, _OP_DIFF}


class Bam():
    def __init__(self):
        # All counters use defaultdict(int) - no existence checks needed.
        # Structure: [chrom][gene][junction_key] -> int
        self.cov   = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.exc   = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.inc_o = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.inc_m = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parseBam(self, file: str, gtf, junc, chrom: str = None):
        """
        Parse a BAM file and populate coverage / junction count structures.

        Parameters
        ----------
        file  : path to BAM file (must be coordinate-sorted and indexed for
                chromosome-specific fetch; index created with samtools index)
        gtf   : GTF object (must have find_gene() and exons attributes)
        junc  : Junction object
        chrom : if provided, only reads on this chromosome are processed.
                Pass None to process all chromosomes (slower).
        """
        try:
            bam_fh = pysam.AlignmentFile(file, 'rb')
        except Exception as exc:
            logging.error('Cannot open BAM file %s: %s', file, exc)
            raise

        try:
            if chrom:
                # Resolve chromosome naming mismatch between the chrom list
                # (often derived from the GTF) and the BAM header.
                # e.g. GTF says "chr14" but BAM header says "14", or vice versa.
                bam_refs = set(bam_fh.references)
                resolved = None
                if chrom in bam_refs:
                    resolved = chrom
                elif chrom.startswith('chr') and chrom[3:] in bam_refs:
                    resolved = chrom[3:]            # chr14 -> 14
                elif ('chr' + chrom) in bam_refs:
                    resolved = 'chr' + chrom        # 14 -> chr14
                elif chrom == 'MT' and 'chrM' in bam_refs:
                    resolved = 'chrM'
                elif chrom == 'chrM' and 'MT' in bam_refs:
                    resolved = 'MT'

                if resolved is None:
                    # Contig genuinely absent from this BAM - skip quietly.
                    logging.warning(
                        'Contig "%s" not found in BAM %s (header has %d refs). '
                        'Skipping this chromosome for this file.',
                        chrom, file, len(bam_refs))
                    bam_fh.close()
                    return self

                read_iter = bam_fh.fetch(resolved)
            else:
                read_iter = bam_fh.fetch()

            for read in read_iter:
                self._process_read(read, gtf, junc)
        finally:
            bam_fh.close()

        logging.debug('parseBam done: exc=%d  cov=%d', len(self.exc), len(self.cov))
        return self

    # ------------------------------------------------------------------
    # Internal read processing
    # ------------------------------------------------------------------

    def _process_read(self, read, gtf, junc):
        """Process one AlignedSegment.  Returns immediately on any filter."""

        # --- 1. Basic quality filters (C-level attribute access) -----------
        if not read.is_paired:
            return
        if read.is_duplicate:
            return
        if read.mate_is_unmapped:
            return
        if read.is_unmapped:
            return

        # Unique-mapping filter: NH tag must equal 1
        try:
            if read.get_tag('NH') != 1:
                return
        except KeyError:
            return   # NH tag absent - skip

        # --- 2. CIGAR filter ----------------------------------------------
        cigar = read.cigartuples
        if cigar is None:
            return

        # Disqualify reads containing D/I/S/H/P/=/X operations
        ops = {op for op, _ in cigar}
        if ops & _SKIP_OPS:
            return

        # Classify read type by CIGAR composition
        m_segs = [(op, ln) for op, ln in cigar if op == _OP_MATCH]
        n_segs = [(op, ln) for op, ln in cigar if op == _OP_SKIP]

        if len(m_segs) == 1 and len(n_segs) == 0:
            exonic_read = True
        elif len(m_segs) == 2 and len(n_segs) == 1:
            exonic_read = False
        else:
            return   # complex CIGAR - skip

        # --- 3. Gene lookup (O(log n) via GTF interval index) -------------
        # reference_start is 0-based in pysam; the original used 1-based SAM
        # POS, so we add 1 to maintain the same coordinate for gene matching.
        pos  = read.reference_start + 1   # 1-based, matches original
        chrom_name = read.reference_name

        gene = gtf.find_gene(chrom_name, pos)
        if not gene:
            return

        # --- 4. Dispatch to exonic or junction processing -----------------
        if exonic_read:
            self._handle_exonic(chrom_name, gene, pos, m_segs[0][1], junc)
        else:
            self._handle_junction(chrom_name, gene, pos, m_segs, n_segs[0][1], junc)

    # ------------------------------------------------------------------
    def _handle_exonic(self, chrom, gene, pos, m_len, junc):
        """
        Accumulate intronic coverage for retained intron (RI) detection.

        l_edge / r_edge define the exonic read span (same coordinates as
        the original code used).
        """
        l_edge = pos
        r_edge = l_edge + m_len

        junc_keys = junc.junction_anno.get(chrom, {}).get(gene, {})
        if not junc_keys:
            return

        for j_key, _ in junc_keys.items():
            j_parts  = j_key.split('_')
            j_start  = int(j_parts[0])
            j_end    = int(j_parts[1])

            # Check whether the exonic read overlaps this junction region
            overlaps = (
                (l_edge <= j_start < r_edge) or
                (l_edge <= j_end   < r_edge) or
                (j_start <= l_edge and j_end >= r_edge)
            )
            if not overlaps:
                continue

            # Coverage count for RI
            self.cov[chrom][gene][j_key] += 1

            # Inclusion boundary counts
            if l_edge <= j_end < r_edge:
                self.inc_o[chrom][gene][j_end] += 1
            elif l_edge <= j_start < r_edge:
                self.inc_o[chrom][gene][j_start] += 1

    # ------------------------------------------------------------------
    def _handle_junction(self, chrom, gene, pos, m_segs, n_len, junc):
        """
        Classify a junction read and accumulate exc / inc_m / inc_o counts.

        Coordinate derivation (identical to original):
            l_start = pos - 1            (0-based)
            l_edge  = l_start + m_segs[0][1]
            r_edge  = l_edge  + n_len
            r_end   = r_edge  + m_segs[1][1]
        """
        l_start = pos - 1
        l_edge  = l_start + m_segs[0][1]
        r_edge  = l_edge  + n_len
        r_end   = r_edge  + m_segs[1][1]

        read_key = f'{l_edge}_{r_edge}'

        gene_junc_anno   = junc.junction_anno.get(chrom, {})
        gene_splice_anno = junc.splice_anno.get(chrom, {})

        if gene in gene_junc_anno and read_key in gene_junc_anno[gene]:
            # Known GTF junction - exclusion read
            self.exc[chrom][gene][read_key] += 1

        elif gene in gene_splice_anno and read_key in gene_splice_anno[gene]:
            # Novel splice - inclusion (matched) read
            self.inc_m[chrom][gene][read_key] += 1

        else:
            # Unmatched junction - compute inclusion boundary overlaps
            if gene not in gene_junc_anno:
                return   # single-exon gene, no junctions
            for j_key in gene_junc_anno[gene]:
                j_parts = j_key.split('_')
                j_start = int(j_parts[0])
                j_end   = int(j_parts[1])

                side = ''
                if (l_start <= j_start < l_edge) or (r_edge <= j_start < r_end):
                    side = 'LEFT'
                elif (l_start <= j_end < l_edge) or (r_edge <= j_end < r_end):
                    side = 'RIGHT'

                if side:
                    key_in = j_start if side == 'LEFT' else j_end
                    self.inc_o[chrom][gene][key_in] += 1
