# coding: UTF-8
#!/usr/bin/env python
"""
label_gtex_junctions.py
=======================
Label GTEx junction read counts (GCT format) as true splicing events
(label = 1) using a STREAMING approach that never loads the full sample
matrix into memory.

Memory problem with the naive approach
---------------------------------------
  GTEx v11 junction file: 523,817 junctions x 19,788 samples
  Full matrix memory: ~523,817 x 19,788 x 8 bytes = ~83 GB  -> OOM kill

Streaming solution
------------------
  Read the file ONE LINE AT A TIME.
  For each junction row, compute mean / n_expressed / max directly from
  the raw values on that line and immediately discard the 19,788 numbers.
  Peak memory usage: one line at a time + output rows only (~tens of MB).

Usage
-----
    python label_gtex_junctions.py \\
        --gtex   GTEx_Analysis_2025-08-22_v11_STARv2.7.11b_junctions.gct.gz \\
        --output gtex_true_junctions.tsv

    # With FASTA to annotate splice sites
    python label_gtex_junctions.py \\
        --gtex   GTEx_Analysis_2025-08-22_v11_STARv2.7.11b_junctions.gct.gz \\
        --fasta  /path/to/genome.fa \\
        --output gtex_true_junctions.tsv

    # Merge with simulation true junctions
    python label_gtex_junctions.py \\
        --gtex   GTEx_Analysis_2025-08-22_v11_STARv2.7.11b_junctions.gct.gz \\
        --merge  true_junctions.tsv \\
        --output all_true_junctions.tsv

Output columns
--------------
  chrom | intron_start | intron_end | strand | gene_id |
  mean_read_count | n_samples_expressed | max_read_count |
  splice_site | source | label

  Coordinates are 0-based half-open to match detect_AS.py / pysam.

Dependencies
------------
    pip install pandas numpy pysam   (pysam optional — only for --fasta)
"""

import os
import sys
import gzip
import argparse
import logging
from datetime import datetime as dt
import pandas as pd
import numpy as np

try:
    import pysam
    _PYSAM_AVAILABLE = True
except ImportError:
    _PYSAM_AVAILABLE = False

COMPLEMENT = str.maketrans('ACGT', 'TGCA')

_DEFAULT_SKIP_PATTERNS = [
    'chrUn', '_random', '_alt', 'chrEBV', 'chrM', 'chrMT', 'Un_', 'random_'
]
_CANONICAL_SITES = {('GT', 'AG'), ('GC', 'AG'), ('AT', 'AC')}

# How often to print progress
_PROGRESS_INTERVAL = 50_000


# ===========================================================================
# Helpers
# ===========================================================================

def should_skip_chrom(chrom, skip_patterns):
    return any(pat in chrom for pat in skip_patterns)


def parse_junction_name(name: str) -> tuple:
    """
    Parse a GTEx junction Name into (chrom, intron_start_0based,
    intron_end_0based, strand).

    GTEx formats observed:
        chr1_14830_14969          most common
        chr1_14830_14969_+        with strand
        chr1_14830_14969_1        strand as 1/2/0

    STAR coords are 1-based -> convert to 0-based half-open:
        intron_start_0based = col2 - 1
        intron_end_0based   = col3      (last intron base = exclusive end)

    Returns (None, None, None, None) on failure.
    """
    strand = ''
    try:
        name_norm = name.replace(':', '_').replace('-', '_')
        parts = name_norm.split('_')

        chrom_parts = []
        coord_parts = []
        found_numeric = False
        for p in parts:
            if not found_numeric and not p.lstrip('-').isdigit():
                chrom_parts.append(p)
            else:
                found_numeric = True
                coord_parts.append(p)

        chrom = '_'.join(chrom_parts)
        if len(coord_parts) < 2:
            return None, None, None, None

        start_1based = int(coord_parts[0])
        end_1based   = int(coord_parts[1])

        if len(coord_parts) >= 3:
            raw = coord_parts[2]
            strand = {'+': '+', '-': '-', '1': '+', '2': '-'}.get(raw, '')

        return chrom, start_1based - 1, end_1based, strand

    except (ValueError, IndexError):
        return None, None, None, None


def get_dinucleotides(genome, chrom, intron_start, intron_end):
    try:
        donor    = genome.fetch(chrom, intron_start,   intron_start + 2).upper()
        acceptor = genome.fetch(chrom, intron_end - 2, intron_end      ).upper()
        return donor, acceptor
    except Exception:
        return 'NN', 'NN'


# ===========================================================================
# Core: streaming GCT parser — never loads full matrix
# ===========================================================================

def stream_gtex_gct(gtex_file: str,
                    min_reads: float,
                    min_samples: int,
                    skip_patterns: list) -> list:
    """
    Stream the GCT file line by line, computing per-junction statistics
    without ever holding the full matrix in memory.

    For each line we only keep:
        Name, Description, mean_read_count, n_samples_expressed, max_read_count

    Everything else is discarded immediately after computing the stats.

    Returns a list of dicts — one per junction that passes all filters.
    """
    opener = gzip.open if gtex_file.endswith('.gz') else open
    mode   = 'rt'

    results        = []
    n_total        = 0
    n_skipped_parse = 0
    n_skipped_chrom = 0
    n_skipped_reads = 0
    n_skipped_samp  = 0

    with opener(gtex_file, mode) as fh:

        # --- GCT header lines ------------------------------------------------
        first  = fh.readline().strip()
        second = fh.readline().strip()

        if not first.startswith('#1.2'):
            logging.warning('File does not start with #1.2 — may not be GCT')

        try:
            n_junctions, n_samples = map(int, second.split('\t')[:2])
        except ValueError:
            n_junctions, n_samples = None, None

        print(f'GCT header: {n_junctions:,} junctions x '
              f'{n_samples:,} samples' if n_junctions else 'GCT header parsed')
        print(f'Processing line by line (progress every '
              f'{_PROGRESS_INTERVAL:,} junctions) ...\n')

        # --- Column header line ----------------------------------------------
        header_line = fh.readline().rstrip('\n')
        headers     = header_line.split('\t')
        # Find Name and Description indices (should be 0 and 1)
        try:
            name_idx = headers.index('Name')
            desc_idx = headers.index('Description')
        except ValueError:
            name_idx, desc_idx = 0, 1
        # All other columns are sample count columns
        sample_indices = [i for i in range(len(headers))
                          if i not in (name_idx, desc_idx)]
        n_sample_cols  = len(sample_indices)
        logging.info('Sample columns detected: %d', n_sample_cols)

        # --- Stream data lines -----------------------------------------------
        for raw_line in fh:
            n_total += 1

            if n_total % _PROGRESS_INTERVAL == 0:
                print(f'  Processed: {n_total:,}  '
                      f'Kept so far: {len(results):,}', flush=True)

            line   = raw_line.rstrip('\n')
            fields = line.split('\t')

            if len(fields) < 3:
                n_skipped_parse += 1
                continue

            name = fields[name_idx]
            desc = fields[desc_idx] if desc_idx < len(fields) else ''

            # --- Parse junction coordinates ----------------------------------
            chrom, intron_start, intron_end, strand = parse_junction_name(name)
            if chrom is None:
                n_skipped_parse += 1
                continue

            # --- Skip non-standard chromosomes --------------------------------
            if should_skip_chrom(chrom, skip_patterns):
                n_skipped_chrom += 1
                continue

            # --- Compute stats from sample columns ---------------------------
            # Use only the sample columns (skip Name, Description)
            total        = 0.0
            n_expressed  = 0
            max_val      = 0.0
            valid_cols   = 0

            for idx in sample_indices:
                if idx >= len(fields):
                    continue
                try:
                    val = float(fields[idx])
                except ValueError:
                    continue
                total      += val
                valid_cols += 1
                if val > 0:
                    n_expressed += 1
                if val > max_val:
                    max_val = val

            if valid_cols == 0:
                n_skipped_reads += 1
                continue

            mean_val = total / valid_cols

            # --- Apply read count filter -------------------------------------
            if mean_val < min_reads:
                n_skipped_reads += 1
                continue

            # --- Apply sample count filter -----------------------------------
            if n_expressed < min_samples:
                n_skipped_samp += 1
                continue

            # --- Keep this junction ------------------------------------------
            results.append({
                'Name':               name,
                'chrom':              chrom,
                'intron_start':       intron_start,
                'intron_end':         intron_end,
                'strand':             strand,
                'gene_id':            desc if desc not in ('', '.') else '',
                'mean_read_count':    round(mean_val, 3),
                'n_samples_expressed': n_expressed,
                'max_read_count':     max_val,
            })

    # --- Streaming complete --------------------------------------------------
    print(f'\nStreaming complete:')
    print(f'  Total lines read      : {n_total:,}')
    print(f'  Parse failures        : {n_skipped_parse:,}')
    print(f'  Skipped (chrom)       : {n_skipped_chrom:,}')
    print(f'  Skipped (low reads)   : {n_skipped_reads:,}')
    print(f'  Skipped (few samples) : {n_skipped_samp:,}')
    print(f'  Passed all filters    : {len(results):,}')

    logging.info('Streaming done: %d kept / %d total', len(results), n_total)
    return results


# ===========================================================================
# Optional: annotate splice sites from FASTA
# ===========================================================================

def annotate_splice_sites(df: pd.DataFrame, fasta_file: str) -> pd.DataFrame:
    """
    Add a 'splice_site' column (e.g. 'GT-AG') using the reference FASTA.
    Processes in batches to give progress feedback.
    Skips silently if pysam is unavailable or FASTA is missing.
    """
    if not _PYSAM_AVAILABLE:
        logging.warning('pysam not available — splice site annotation skipped')
        df['splice_site'] = 'unknown'
        return df

    if not os.path.isfile(fasta_file):
        logging.warning('FASTA not found: %s — splice site annotation skipped',
                        fasta_file)
        df['splice_site'] = 'unknown'
        return df

    print(f'\nAnnotating splice sites for {len(df):,} junctions ...')
    genome = pysam.FastaFile(fasta_file)
    sites  = []

    for i, (_, row) in enumerate(df.iterrows()):
        if i % _PROGRESS_INTERVAL == 0 and i > 0:
            print(f'  Annotated: {i:,} / {len(df):,}', flush=True)
        donor, acceptor = get_dinucleotides(
            genome, row['chrom'], int(row['intron_start']), int(row['intron_end']))
        sites.append(f'{donor}-{acceptor}')

    genome.close()
    df['splice_site'] = sites
    print(f'  Splice site annotation done')
    return df


# ===========================================================================
# Main labeling function
# ===========================================================================

def label_gtex_junctions(gtex_file: str,
                          output_file: str,
                          fasta_file: str    = None,
                          min_reads: float   = 1.0,
                          min_samples: int   = 10,
                          canonical_only: bool = False,
                          skip_patterns: list  = None,
                          merge_file: str    = None) -> pd.DataFrame:

    start = dt.now()
    logging.info('=== label_gtex_junctions.py started: %s ===',
                 start.strftime('%H:%M:%S'))

    if skip_patterns is None:
        skip_patterns = _DEFAULT_SKIP_PATTERNS

    # --- Stream GCT and compute stats ----------------------------------------
    rows = stream_gtex_gct(gtex_file, min_reads, min_samples, skip_patterns)

    if not rows:
        print('ERROR: no junctions passed filters. '
              'Try lowering --min-reads or --min-samples.')
        sys.exit(1)

    df = pd.DataFrame(rows)
    df = df.drop(columns=['Name'])   # no longer needed

    # --- Deduplication -------------------------------------------------------
    before = len(df)
    df = df.drop_duplicates(
        subset=['chrom', 'intron_start', 'intron_end']
    ).reset_index(drop=True)
    print(f'After deduplication           : {len(df):,}  '
          f'({before - len(df):,} removed)')

    # --- Splice site annotation ----------------------------------------------
    if fasta_file:
        df = annotate_splice_sites(df, fasta_file)

        if canonical_only:
            before_c = len(df)
            df = df[df['splice_site'].apply(
                lambda s: tuple(s.split('-')) in _CANONICAL_SITES
                          if '-' in s else False
            )].reset_index(drop=True)
            print(f'After canonical filter        : {len(df):,}  '
                  f'({before_c - len(df):,} non-canonical removed)')
    else:
        df['splice_site'] = 'unknown'

    # --- Assign label and source ---------------------------------------------
    df['source'] = 'GTEx'
    df['label']  = 1

    # --- Merge with simulation junctions -------------------------------------
    if merge_file:
        if not os.path.isfile(merge_file):
            logging.warning('--merge file not found: %s — skipping', merge_file)
        else:
            sim_df = pd.read_csv(merge_file, sep='\t')
            for col in ['mean_read_count', 'n_samples_expressed',
                        'max_read_count', 'splice_site']:
                if col not in sim_df.columns:
                    sim_df[col] = np.nan if col != 'splice_site' else 'unknown'
            if 'source' not in sim_df.columns:
                sim_df['source'] = 'simulation'
            sim_df['label'] = 1

            merged = pd.concat([df, sim_df], ignore_index=True)
            before_m = len(merged)
            merged = merged.drop_duplicates(
                subset=['chrom', 'intron_start', 'intron_end']
            ).reset_index(drop=True)
            merged['label'] = 1

            print(f'\nMerge with {os.path.basename(merge_file)}:')
            print(f'  GTEx junctions        : {len(df):,}')
            print(f'  Simulation junctions  : {len(sim_df):,}')
            print(f'  Combined unique       : {len(merged):,}  '
                  f'({before_m - len(merged):,} overlaps removed)')
            df = merged

    # --- Write output --------------------------------------------------------
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    df.to_csv(output_file, sep='\t', index=False)

    elapsed = (dt.now() - start).seconds

    # --- Summary -------------------------------------------------------------
    print(f'\n=== Summary ===')
    print(f'  Total true junctions  : {len(df):,}')
    print(f'  Chromosomes covered   : {df["chrom"].nunique()}')
    print(f'  Label = 1 (all rows)  : {(df["label"]==1).sum():,}')
    if 'splice_site' in df.columns and df['splice_site'].ne('unknown').any():
        print(f'  Splice site breakdown :')
        for site, cnt in df['splice_site'].value_counts().head(5).items():
            print(f'    {site:<12}: {cnt:,}')
    print(f'  Output                : {output_file}')
    print(f'  Elapsed               : {elapsed}s')
    print('===============\n')

    logging.info('Done. %s (%d rows, %ds)', output_file, len(df), elapsed)
    return df


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Label GTEx junction read counts (GCT) as true splicing events.\n'
            'Uses streaming to handle the full 523k x 19k matrix without OOM.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--gtex', metavar='FILE', required=True,
        help='GTEx junction GCT file (.gct or .gct.gz)')
    parser.add_argument('-o', '--output', metavar='FILE',
        default='gtex_true_junctions.tsv',
        help='Output TSV (default: gtex_true_junctions.tsv)')
    parser.add_argument('--fasta', metavar='FILE', default=None,
        help='Indexed FASTA for splice site annotation (optional)')
    parser.add_argument('--merge', metavar='FILE', default=None,
        help='Merge with true_junctions.tsv from extract_true_junctions.py')
    parser.add_argument('--min-reads', metavar='FLOAT', type=float,
        default=1.0,
        help='Min mean read count across all samples (default: 1.0)')
    parser.add_argument('--min-samples', metavar='INT', type=int,
        default=10,
        help='Min number of samples with count > 0 (default: 10)')
    parser.add_argument('--canonical-only', action='store_true',
        help='Keep only GT-AG, GC-AG, AT-AC junctions. Requires --fasta.')
    parser.add_argument('--keep-nonstandard-chroms', action='store_true',
        help='Do not skip chrUn / random / alt / EBV / chrM contigs.')
    parser.add_argument('--log', metavar='FILE',
        default='label_gtex_junctions.log')
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

    if args.canonical_only and not args.fasta:
        print('ERROR: --canonical-only requires --fasta')
        sys.exit(1)

    skip_patterns = ([] if args.keep_nonstandard_chroms
                     else _DEFAULT_SKIP_PATTERNS)

    print(f'Start Time     = {dt.now().strftime("%H:%M:%S")}')
    print(f'GTEx file      = {args.gtex}')
    print(f'Min reads      = {args.min_reads}')
    print(f'Min samples    = {args.min_samples}')
    print(f'Canonical only = {args.canonical_only}')
    print(f'FASTA          = {args.fasta or "not provided"}')
    print(f'Merge with     = {args.merge or "none"}')
    print(f'Output         = {args.output}')

    label_gtex_junctions(
        gtex_file      = args.gtex,
        output_file    = args.output,
        fasta_file     = args.fasta,
        min_reads      = args.min_reads,
        min_samples    = args.min_samples,
        canonical_only = args.canonical_only,
        skip_patterns  = skip_patterns,
        merge_file     = args.merge,
    )


if __name__ == '__main__':
    main()
