# coding: UTF-8
#!/usr/bin/env python
"""
build_ground_truth.py
=====================
Build a high-confidence splice junction ground truth set from long-read
RNA-seq BAM files (PacBio Iso-seq or Oxford Nanopore) for use as Tier 2
benchmark data when evaluating alternative splicing detection tools.

Strategy
--------
  1. Extract all splice junctions from each long-read BAM by walking
     CIGAR strings and collecting N (intron-skip) operations.
  2. Apply a per-replicate minimum read count filter.
  3. Keep only junctions supported in ALL replicates (reproducibility
     filter) — this is the strongest guard against long-read-specific
     artifacts such as RT template switching and mis-spliced reads.
  4. Optionally cross-reference against a GTF to add gene annotation.
  5. Write the final ground truth TSV with label = 1.

Long-read aligner notes
-----------------------
  PacBio  : align with minimap2 -ax splice:hq  OR  STARlong
  ONT     : align with minimap2 -ax splice      OR  minimap2 -ax splice -k14
  Both    : BAM must be coordinate-sorted and indexed (samtools sort + index)

  minimap2 alignment (if starting from FASTQ):
    minimap2 -ax splice:hq -uf --secondary=no \\
        genome.fa pacbio_reads.fastq.gz | \\
        samtools sort -o pacbio_rep1.bam && samtools index pacbio_rep1.bam

    minimap2 -ax splice -uf -k14 --secondary=no \\
        genome.fa ont_reads.fastq.gz | \\
        samtools sort -o ont_rep1.bam && samtools index ont_rep1.bam

Output columns
--------------
  chrom | intron_start | intron_end | strand |
  read_count_rep1 .. read_count_repN |
  total_read_count | n_reps_supported | label | source

  Coordinates are 0-based half-open to match detect_AS.py / pysam convention.

Usage
-----
  # Two replicates (minimum recommended)
  python build_ground_truth.py \\
      --bam rep1.bam rep2.bam \\
      --output longread_truth.tsv

  # Three replicates, stricter per-rep filter
  python build_ground_truth.py \\
      --bam rep1.bam rep2.bam rep3.bam \\
      --min-reads 3 \\
      --output longread_truth.tsv

  # With GTF annotation for gene names
  python build_ground_truth.py \\
      --bam rep1.bam rep2.bam \\
      --gtf annotation.gtf \\
      --output longread_truth.tsv

  # Require junction in at least 2 of 3 replicates (relaxed)
  python build_ground_truth.py \\
      --bam rep1.bam rep2.bam rep3.bam \\
      --min-reps 2 \\
      --output longread_truth.tsv

  # ONT data (noisier — use higher min-reads)
  python build_ground_truth.py \\
      --bam ont_rep1.bam ont_rep2.bam \\
      --min-reads 5 \\
      --platform ont \\
      --output ont_truth.tsv

Dependencies
------------
  pip install pysam pandas numpy
"""

import os
import sys
import argparse
import logging
from collections import defaultdict
from datetime import datetime as dt
import pandas as pd
import numpy as np
import pysam


# ===========================================================================
# Per-replicate junction extraction
# ===========================================================================

def extract_junctions_from_bam(bam_file: str,
                                 min_reads: int,
                                 skip_secondary: bool = True,
                                 skip_supplementary: bool = True,
                                 min_mapq: int = 0) -> pd.DataFrame:
    """
    Extract splice junctions from one long-read BAM file by walking
    CIGAR strings and collecting N (BAM_CREF_SKIP) operations.

    Parameters
    ----------
    bam_file           : path to coordinate-sorted, indexed BAM
    min_reads          : minimum reads supporting a junction in this replicate
    skip_secondary     : skip secondary alignments (recommended: True)
    skip_supplementary : skip supplementary/chimeric alignments (recommended: True)
    min_mapq           : minimum mapping quality (0 = keep all;
                         use 1 to exclude unmapped, 10 for stricter filter)

    Returns
    -------
    DataFrame with columns:
        chrom, intron_start, intron_end, strand, read_count
    One row per junction passing min_reads threshold.

    Coordinate convention (0-based half-open, same as pysam):
        intron_start = position of first intron base
        intron_end   = position of first base of downstream exon
        (equivalent to STAR SJ.out.tab col2-1 and col3)
    """
    if not os.path.isfile(bam_file):
        raise FileNotFoundError(f'BAM file not found: {bam_file}')

    bai = bam_file + '.bai'
    bai2 = bam_file.replace('.bam', '.bai')
    if not os.path.isfile(bai) and not os.path.isfile(bai2):
        raise FileNotFoundError(
            f'BAM index not found for {bam_file}\n'
            f'Run: samtools index {bam_file}'
        )

    logging.info('Extracting junctions from: %s', bam_file)

    junc_counts  = defaultdict(int)
    junc_strands = defaultdict(lambda: defaultdict(int))
    n_reads_total    = 0
    n_reads_junction = 0

    bam = pysam.AlignmentFile(bam_file, 'rb')

    for read in bam.fetch():
        # --- Basic filters ---------------------------------------------------
        if read.is_unmapped:
            continue
        if skip_secondary and read.is_secondary:
            continue
        if skip_supplementary and read.is_supplementary:
            continue
        if read.mapping_quality < min_mapq:
            continue
        if read.cigartuples is None:
            continue

        n_reads_total += 1
        chrom = read.reference_name
        pos   = read.reference_start   # 0-based current position

        has_junction = False

        for op, length in read.cigartuples:
            if op == 0:    # M  — match / mismatch
                pos += length
            elif op == 1:  # I  — insertion (does not consume reference)
                pass
            elif op == 2:  # D  — deletion
                pos += length
            elif op == 3:  # N  — intron skip  ← SPLICE JUNCTION
                intron_start = pos           # 0-based first intron base
                intron_end   = pos + length  # 0-based first base of next exon
                key = (chrom, intron_start, intron_end)

                junc_counts[key] += 1

                # Tally strand votes per junction
                strand = '-' if read.is_reverse else '+'
                junc_strands[key][strand] += 1

                pos += length
                has_junction = True

            elif op == 4:  # S  — soft clip (does not consume reference)
                pass
            elif op == 5:  # H  — hard clip
                pass
            elif op == 7:  # =  — sequence match
                pos += length
            elif op == 8:  # X  — sequence mismatch
                pos += length

        if has_junction:
            n_reads_junction += 1

    bam.close()

    logging.info('Reads total: %d  with junction: %d',
                 n_reads_total, n_reads_junction)

    # --- Assign strand by majority vote --------------------------------------
    rows = []
    for (chrom, intron_start, intron_end), count in junc_counts.items():
        if count < min_reads:
            continue

        strand_votes = junc_strands[(chrom, intron_start, intron_end)]
        strand = max(strand_votes, key=strand_votes.get)

        rows.append({
            'chrom':        chrom,
            'intron_start': intron_start,
            'intron_end':   intron_end,
            'strand':       strand,
            'read_count':   count,
        })

    df = pd.DataFrame(rows) if rows else pd.DataFrame(
        columns=['chrom', 'intron_start', 'intron_end', 'strand', 'read_count'])

    nm = os.path.basename(bam_file)
    logging.info('%s: %d junctions passed min_reads=%d',
                 nm, len(df), min_reads)
    print(f'  {nm}: {n_reads_total:,} reads  '
          f'{len(junc_counts):,} raw junctions  '
          f'{len(df):,} passed min_reads={min_reads}')

    return df


# ===========================================================================
# GTF gene annotation (optional)
# ===========================================================================

def build_gene_interval_index(gtf_file: str) -> dict:
    """
    Build a simple chrom -> sorted list of (start, end, gene_id) tuples
    for fast overlap lookup.
    """
    import gzip
    logging.info('Building gene index from GTF: %s', gtf_file)

    gene_index = defaultdict(list)
    opener = gzip.open if gtf_file.endswith('.gz') else open
    mode   = 'rt' if gtf_file.endswith('.gz') else 'r'

    with opener(gtf_file, mode) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] not in ('gene', 'transcript'):
                continue
            chrom  = fields[0]
            start  = int(fields[3])
            end    = int(fields[4])
            strand = fields[6]
            attrs  = fields[8]
            gene_id = ''
            for kv in attrs.strip(';').split(';'):
                kv = kv.strip()
                if kv.startswith('gene_id'):
                    gene_id = kv.split(' ', 1)[1].replace('"', '').strip()
                    break
            if gene_id:
                gene_index[chrom].append((start, end, strand, gene_id))

    # Sort by start position for binary search
    for chrom in gene_index:
        gene_index[chrom].sort(key=lambda x: x[0])

    logging.info('Gene index built: %d chromosomes', len(gene_index))
    return dict(gene_index)


def lookup_gene(chrom: str, intron_start: int,
                 intron_end: int, gene_index: dict) -> str:
    """Return gene_id overlapping the intron, or '' if none found."""
    intervals = gene_index.get(chrom, [])
    for g_start, g_end, g_strand, gene_id in intervals:
        if g_start > intron_end:
            break
        if g_end >= intron_start and g_start <= intron_end:
            return gene_id
    return ''


# ===========================================================================
# Core: build ground truth from multiple replicates
# ===========================================================================

def build_ground_truth(bam_files: list,
                        min_reads: int,
                        min_reps: int,
                        gtf_file: str,
                        platform: str,
                        output_file: str,
                        min_mapq: int) -> pd.DataFrame:
    """
    Build a high-confidence ground truth junction set from multiple
    long-read BAM replicates.

    Parameters
    ----------
    bam_files   : list of BAM file paths (at least 2 recommended)
    min_reads   : minimum reads per junction per replicate
    min_reps    : minimum number of replicates that must support a junction
                  (default = len(bam_files), i.e. ALL replicates)
    gtf_file    : optional GTF for gene annotation
    platform    : 'pacbio' or 'ont' (affects default settings and reporting)
    output_file : path for output TSV
    min_mapq    : minimum mapping quality filter

    Returns
    -------
    pd.DataFrame  ground truth junction set with label=1
    """
    start = dt.now()
    n_bams = len(bam_files)

    if min_reps > n_bams:
        print(f'WARNING: --min-reps ({min_reps}) > number of BAMs ({n_bams}). '
              f'Setting min_reps = {n_bams}')
        min_reps = n_bams

    # ── Step 1: Extract junctions from each replicate ───────────────────────
    print(f'\n--- Extracting junctions from {n_bams} BAM file(s) ---')
    rep_dfs   = []
    rep_names = []

    for i, bam in enumerate(bam_files, 1):
        rep_name = f'rep{i}_{os.path.basename(bam).replace(".bam", "")}'
        rep_names.append(rep_name)
        df = extract_junctions_from_bam(
            bam, min_reads=min_reads,
            skip_secondary=True, skip_supplementary=True,
            min_mapq=min_mapq)
        df = df.rename(columns={'read_count': f'read_count_{rep_name}'})
        rep_dfs.append(df)

    # ── Step 2: Merge all replicates on coordinates ──────────────────────────
    print(f'\n--- Merging replicates (min_reps={min_reps}) ---')

    merge_keys = ['chrom', 'intron_start', 'intron_end']

    if len(rep_dfs) == 1:
        # Single replicate — no reproducibility filter
        merged = rep_dfs[0].copy()
        merged['strand'] = merged['strand']
        merged['n_reps_supported'] = 1
        count_cols = [f'read_count_{rep_names[0]}']
    else:
        # Full outer join across all replicates
        merged = rep_dfs[0]
        for i, df in enumerate(rep_dfs[1:], start=1):
            # Use rep_names[i] directly -- df.columns[-1] already has
            # the read_count_ prefix so must NOT be prefixed again here
            rc_col = f'read_count_{rep_names[i]}'
            merged = pd.merge(merged,
                              df[merge_keys + [rc_col, 'strand']],
                              on=merge_keys + ['strand'],
                              how='outer')

        # Fill missing counts with 0 (junction absent in that replicate)
        count_cols = [c for c in merged.columns if c.startswith('read_count_')]
        for c in count_cols:
            merged[c] = merged[c].fillna(0).astype(int)

        # Count how many replicates support each junction
        merged['n_reps_supported'] = (
            merged[count_cols].gt(0).sum(axis=1))

        # ── Step 3: Reproducibility filter ───────────────────────────────────
        before = len(merged)
        merged = merged[merged['n_reps_supported'] >= min_reps]
        after  = len(merged)

        print(f'  Junctions in any replicate  : {before:,}')
        print(f'  Junctions in >= {min_reps} replicate(s): {after:,}  '
              f'({before - after:,} removed)')

    # ── Step 4: Compute total read count across replicates ───────────────────
    merged['total_read_count'] = merged[count_cols].sum(axis=1).astype(int)

    # ── Step 5: Filter very short / very long introns (likely artifacts) ─────
    merged['intron_length'] = merged['intron_end'] - merged['intron_start']

    min_intron = 50       # < 50 bp likely alignment artifact
    max_intron = 1_000_000  # > 1 Mb extremely unlikely, filter conservatively

    before = len(merged)
    merged = merged[
        (merged['intron_length'] >= min_intron) &
        (merged['intron_length'] <= max_intron)
    ]
    print(f'  After intron length filter  : {len(merged):,}  '
          f'({before - len(merged):,} removed, '
          f'length outside [{min_intron}, {max_intron:,}] bp)')

    # ── Step 6: GTF gene annotation (optional) ────────────────────────────────
    if gtf_file and os.path.isfile(gtf_file):
        print(f'\n--- Annotating with GTF: {gtf_file} ---')
        gene_index = build_gene_interval_index(gtf_file)
        merged['gene_id'] = merged.apply(
            lambda r: lookup_gene(r['chrom'], r['intron_start'],
                                  r['intron_end'], gene_index),
            axis=1)
        n_annotated = (merged['gene_id'] != '').sum()
        print(f'  Junctions with gene annotation: {n_annotated:,} / {len(merged):,}')
    else:
        merged['gene_id'] = ''

    # ── Step 7: Assign label and source ──────────────────────────────────────
    merged['label']  = 1
    merged['source'] = f'long_read_{platform}'

    # ── Step 8: Order columns cleanly ────────────────────────────────────────
    id_cols    = ['chrom', 'intron_start', 'intron_end',
                  'strand', 'intron_length', 'gene_id']
    count_cols_out = sorted(count_cols) + ['total_read_count', 'n_reps_supported']
    meta_cols  = ['label', 'source']
    all_cols   = id_cols + count_cols_out + meta_cols
    merged     = merged[[c for c in all_cols if c in merged.columns]]
    merged     = merged.sort_values(
        ['chrom', 'intron_start', 'intron_end']).reset_index(drop=True)

    # ── Step 9: Write output ─────────────────────────────────────────────────
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    merged.to_csv(output_file, sep='\t', index=False)

    elapsed = (dt.now() - start).seconds

    # ── Summary ──────────────────────────────────────────────────────────────
    print(f'\n=== Ground Truth Summary ===')
    print(f'  Platform              : {platform}')
    print(f'  BAM files             : {n_bams}')
    print(f'  Min reads per rep     : {min_reads}')
    print(f'  Min reps required     : {min_reps}')
    print(f'  Final junctions       : {len(merged):,}')
    print(f'  Chromosomes           : {merged["chrom"].nunique()}')
    if 'gene_id' in merged.columns:
        print(f'  Annotated genes       : {merged["gene_id"].nunique():,}')
    print(f'  Intron length (median): '
          f'{int(merged["intron_length"].median()):,} bp')
    print(f'  Intron length (min)   : {merged["intron_length"].min():,} bp')
    print(f'  Intron length (max)   : {merged["intron_length"].max():,} bp')
    if 'n_reps_supported' in merged.columns and n_bams > 1:
        print(f'  Replicate support:')
        for n, cnt in merged['n_reps_supported'].value_counts().sort_index().items():
            print(f'    {n} rep(s): {cnt:,} junctions')
    print(f'  Output                : {output_file}')
    print(f'  Elapsed               : {elapsed}s')
    print('============================\n')

    logging.info('Done. %s (%d junctions, %ds)',
                 output_file, len(merged), elapsed)
    return merged


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Build a high-confidence splice junction ground truth set\n'
            'from long-read RNA-seq BAM files (PacBio or ONT).\n\n'
            'Before running, ensure BAMs are sorted and indexed:\n'
            '  samtools sort -o rep1_sorted.bam rep1.bam\n'
            '  samtools index rep1_sorted.bam\n\n'
            'Align long reads with minimap2:\n'
            '  PacBio: minimap2 -ax splice:hq -uf --secondary=no \\\n'
            '          genome.fa reads.fastq.gz | samtools sort -o rep1.bam\n'
            '  ONT:    minimap2 -ax splice -uf -k14 --secondary=no \\\n'
            '          genome.fa reads.fastq.gz | samtools sort -o rep1.bam'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--bam', metavar='FILE', nargs='+', required=True,
        help='One or more long-read BAM files (replicates). '
             'At least 2 replicates strongly recommended.')
    parser.add_argument('-o', '--output', metavar='FILE',
        default='longread_truth.tsv',
        help='Output ground truth TSV (default: longread_truth.tsv)')
    parser.add_argument('--gtf', metavar='FILE', default=None,
        help='GTF annotation file for gene name annotation (optional)')
    parser.add_argument('--min-reads', metavar='INT', type=int, default=3,
        help='Min reads supporting a junction per replicate.\n'
             'Recommended: 3 for PacBio, 5 for ONT (noisier)\n'
             '(default: 3)')
    parser.add_argument('--min-reps', metavar='INT', type=int, default=None,
        help='Min number of replicates a junction must appear in.\n'
             'Default: ALL replicates (most stringent).\n'
             'Use e.g. --min-reps 2 with 3 BAMs for relaxed filter.')
    parser.add_argument('--platform', choices=['pacbio', 'ont'],
        default='pacbio',
        help='Sequencing platform — affects default parameter recommendations\n'
             '(default: pacbio)')
    parser.add_argument('--min-mapq', metavar='INT', type=int, default=0,
        help='Minimum mapping quality. Use 0 for minimap2 output\n'
             '(minimap2 sets MAPQ=0 for multi-mappers; primary alignments\n'
             'typically have MAPQ>=1). Use 10 for stricter filter.\n'
             '(default: 0)')
    parser.add_argument('--log', metavar='FILE',
        default='build_ground_truth.log')
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.DEBUG, filemode='w',
        format='[%(levelname)s] %(asctime)s %(message)s',
        filename=args.log
    )
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(console)

    # Validate BAM files exist
    for bam in args.bam:
        if not os.path.isfile(bam):
            print(f'ERROR: BAM file not found: {bam}')
            sys.exit(1)

    # Default min_reps = all replicates
    min_reps = args.min_reps if args.min_reps else len(args.bam)

    # Platform-specific warnings
    if args.platform == 'ont' and args.min_reads < 5:
        print(f'NOTE: ONT data is noisier than PacBio. '
              f'Consider --min-reads 5 (current: {args.min_reads})')
    if len(args.bam) == 1:
        print('WARNING: Only 1 BAM provided. Reproducibility filter '
              'cannot be applied.\nStrongly recommend using at least 2 replicates.')

    print(f'Start Time    = {dt.now().strftime("%H:%M:%S")}')
    print(f'BAM files     = {args.bam}')
    print(f'Platform      = {args.platform}')
    print(f'Min reads     = {args.min_reads}')
    print(f'Min reps      = {min_reps} / {len(args.bam)}')
    print(f'Min MAPQ      = {args.min_mapq}')
    print(f'GTF           = {args.gtf or "not provided"}')
    print(f'Output        = {args.output}')

    build_ground_truth(
        bam_files   = args.bam,
        min_reads   = args.min_reads,
        min_reps    = min_reps,
        gtf_file    = args.gtf,
        platform    = args.platform,
        output_file = args.output,
        min_mapq    = args.min_mapq,
    )

    print(f'End Time = {dt.now().strftime("%H:%M:%S")}')


if __name__ == '__main__':
    main()
