# coding: UTF-8
#!/usr/bin/env python

import sys
import re
import os
import gzip
import bisect
import logging
from collections import defaultdict
from classes.file import File
import pandas as pd

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s %(message)s',
    filename='log_parse.GTF.txt'
)


class GTF(File):
    def __init__(self, inFile):
        super(GTF, self).__init__(inFile)
        self.genes = defaultdict(dict)
        self.exons = defaultdict(lambda: defaultdict(dict))
        self.pos   = defaultdict(dict)
        self.tx_lines = defaultdict(lambda: defaultdict(dict))
        self.ex_lines = defaultdict(lambda: defaultdict(dict))

        # --- Interval index for fast gene lookup (built after parseGTF) ---
        # Per chromosome, three parallel sorted lists (sorted by gene start):
        #   _gi_starts[chrom] : list of gene start positions (int)
        #   _gi_ends[chrom]   : list of gene end positions   (int)
        #   _gi_ids[chrom]    : list of gene_id strings
        # find_gene(chrom, pos) uses bisect for O(log n) lookup instead of
        # the original O(n_genes) linear scan.
        self._gi_starts = {}
        self._gi_ends   = {}
        self._gi_ids    = {}

        # Fallback per-transcript exon counter for GTFs lacking exon_number
        self._exon_counter = {}

    # ------------------------------------------------------------------
    def parseGTF(self):
        handle = self._openFile()
        for line in handle:
            if line.startswith('#'):
                continue

            fields = line.rstrip('\n').split('\t')
            if len(fields) < 9:
                continue   # malformed line - skip

            el_chr, _, el, el_start, el_end, _, strand, _, tags = fields[:9]

            # Robust attribute parsing.
            # GTF attribute format is:  key "value"; key "value"; ...
            # Splitting on a single space breaks when the value itself
            # contains spaces (e.g. gene_name "RP11 34 protein"), producing
            # >2 elements and a "dictionary update sequence" ValueError.
            # Fix: split each attribute on the FIRST space only (maxsplit=1)
            # and skip any malformed / empty attribute fragments.
            d = {}
            for kv in tags.strip().strip(';').split(';'):
                kv = kv.strip()
                if not kv:
                    continue
                parts = kv.split(' ', 1)   # split on first space only
                if len(parts) != 2:
                    continue               # malformed attribute - skip
                key, val = parts[0].strip(), parts[1].strip()
                d[key] = val

            # gene_id is required; skip line if absent
            if 'gene_id' not in d:
                continue
            gene_id       = d['gene_id'].replace('"', '')
            transcript_id = d.get('transcript_id', '').replace('"', '')

            if el == 'transcript':
                self.genes[el_chr][gene_id] = [
                    el_chr, int(el_start), int(el_end), strand, tags
                ]
                self.tx_lines[el_chr][gene_id][transcript_id] = line

            elif el == 'exon':
                # exon_number may be absent in some GTFs (e.g. certain ENCODE
                # or StringTie outputs). Fall back to an auto-incrementing
                # per-transcript counter so downstream code still works.
                if 'exon_number' in d:
                    exon_no = d['exon_number'].replace('"', '')
                else:
                    cnt_key = (gene_id, transcript_id)
                    self._exon_counter[cnt_key] = \
                        self._exon_counter.get(cnt_key, 0) + 1
                    exon_no = str(self._exon_counter[cnt_key])

                # Store coordinates as ints - avoids repeated int() calls
                # everywhere downstream.
                self.exons[gene_id][transcript_id][exon_no] = [
                    el_chr, int(el_start), int(el_end), strand
                ]
                self.ex_lines[gene_id][transcript_id][exon_no] = line
                key = '_'.join([el_start, el_end])
                if key not in self.pos[el_chr]:
                    self.pos[el_chr][key] = []
                self.pos[el_chr][key].extend(
                    [gene_id, transcript_id, exon_no])

        self._closeFile(handle)
        self._build_interval_index()

    # ------------------------------------------------------------------
    def _build_interval_index(self):
        """
        Build sorted parallel arrays per chromosome for O(log n) gene lookup.
        Only genes that have exon information are indexed (same filter as the
        original linear scan used).
        """
        for chrom, gene_dict in self.genes.items():
            items = []
            for gene_id, info in gene_dict.items():
                if gene_id not in self.exons:
                    continue                      # no exon info - skip
                items.append((info[1], info[2], gene_id))   # (start, end, id)

            if not items:
                continue

            items.sort()                          # sort by start position
            self._gi_starts[chrom] = [x[0] for x in items]
            self._gi_ends[chrom]   = [x[1] for x in items]
            self._gi_ids[chrom]    = [x[2] for x in items]

        logging.debug('Interval index built for %d chromosomes',
                      len(self._gi_starts))

    # ------------------------------------------------------------------
    def find_gene(self, chrom: str, pos: int) -> str:
        """
        Return the gene_id whose genomic interval contains `pos`, or '' if none.

        Replaces the original O(n_genes) linear scan in bam.py with an
        O(log n) bisect search.  Behaviour is identical: returns the first
        gene found (genes are expected to be non-overlapping in practice).

        Parameters
        ----------
        chrom : chromosome name (must match GTF seqname field)
        pos   : 1-based genomic position (same coordinate as SAM POS column)
        """
        starts = self._gi_starts.get(chrom)
        if not starts:
            return ''

        ends = self._gi_ends[chrom]
        ids  = self._gi_ids[chrom]

        # Find the rightmost gene whose start <= pos, then scan backwards
        # for the first one whose end also covers pos.
        # Because genes are sorted by start, idx is the last candidate.
        idx = bisect.bisect_right(starts, pos) - 1
        while idx >= 0:
            if ends[idx] >= pos:
                return ids[idx]
            idx -= 1

        return ''
