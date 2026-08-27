# coding: UTF-8
#!/usr/bin/env python
"""
build_training_set.py
=====================
Merge simulation positives, GTEx positives, and negatives into a single
shuffled training set ready for Random Forest / ML artifact filtering.

Handles TWO input formats for positive files:

  Format A — extract_true_junctions.py output (true_junctions.tsv)
  -----------------------------------------------------------------
  Columns: chrom | intron_start | intron_end | strand | gene_id |
           transcript_id | label

  Format B — assign_labels.py output (labeled_junctions.tsv)
  -----------------------------------------------------------
  Columns: gene::tx | id | coverage | exclusion1 | exclusion2 |
           inclusion | over_boundaries | PSI | source_file |
           chrom_resolved | strand_resolved | label
  The script filters to label=1 rows and parses chrom/coordinates
  from chrom_resolved and the id column (format: intron_start_intron_end).

The format is detected automatically from the column names.

Input files
-----------
  Positives (label = 1):
    --sim-pos   true_junctions.tsv OR labeled_junctions.tsv
    --gtex-pos  gtex_true_junctions.tsv

  Negatives (label = 0):
    --neg       negatives.tsv

Usage
-----
    python build_training_set.py \\
        --sim-pos   labeled_junctions.tsv \\
        --gtex-pos  gtex_true_junctions.tsv \\
        --neg       negatives.tsv \\
        --max-ratio 3 \\
        --output    final_training_set.tsv

Dependencies
------------
    pip install pandas numpy
"""

import os
import sys
import argparse
import logging
from datetime import datetime as dt
import pandas as pd
import numpy as np


REQUIRED_COLS = ['chrom', 'intron_start', 'intron_end',
                 'strand', 'gene_id', 'source', 'label']

# Column aliases for Format A variations
_COL_ALIASES = {
    'intron_start': 'intron_start',
    'start':        'intron_start',
    'junc_start':   'intron_start',
    'chromStart':   'intron_start',
    'Start':        'intron_start',
    'intron_end':   'intron_end',
    'end':          'intron_end',
    'junc_end':     'intron_end',
    'chromEnd':     'intron_end',
    'End':          'intron_end',
    'chrom':        'chrom',
    'chr':          'chrom',
    'chromosome':   'chrom',
    'Chromosome':   'chrom',
    'seqname':      'chrom',
    'strand':       'strand',
    'Strand':       'strand',
    'gene_id':      'gene_id',
    'gene':         'gene_id',
    'gene_name':    'gene_id',
    'Description':  'gene_id',
    'geneID':       'gene_id',
}


# ===========================================================================
# Format detection
# ===========================================================================

def detect_format(df: pd.DataFrame, filepath: str) -> str:
    """
    Return 'A' for extract_true_junctions format,
           'B' for assign_labels / events format,
           'gtex' for label_gtex_junctions format.
    """
    cols = set(df.columns)

    # Format B: assign_labels output has these event-specific columns
    if 'chrom_resolved' in cols and 'id' in cols and 'gene::tx' in cols:
        logging.info('%s detected as Format B (assign_labels output)',
                     os.path.basename(filepath))
        return 'B'

    # GTEx format: has mean_read_count and n_samples_expressed
    if 'mean_read_count' in cols and 'n_samples_expressed' in cols:
        logging.info('%s detected as GTEx format',
                     os.path.basename(filepath))
        return 'gtex'

    # Format A: has intron_start / intron_end directly (or aliases)
    aliased = {_COL_ALIASES.get(c, c) for c in cols}
    if 'intron_start' in aliased and 'intron_end' in aliased:
        logging.info('%s detected as Format A (extract_true_junctions output)',
                     os.path.basename(filepath))
        return 'A'

    # Unknown — print columns and exit with helpful message
    print(f'\nERROR: Cannot determine format of: {filepath}')
    print(f'Columns found: {list(df.columns)}')
    print(
        '\nExpected one of:\n'
        '  Format A (extract_true_junctions): chrom, intron_start, intron_end, ...\n'
        '  Format B (assign_labels):          gene::tx, id, chrom_resolved, ...\n'
        '  GTEx format:                       chrom, intron_start, mean_read_count, ...\n'
    )
    sys.exit(1)


# ===========================================================================
# Format-specific parsers
# ===========================================================================

def _parse_format_b(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """
    Convert assign_labels.py output (labeled_junctions.tsv) to the
    common schema.

    Key steps:
      - Filter to label=1 rows only (true splicing events)
      - Parse intron_start and intron_end from the 'id' column
        (format: intron_start_intron_end, e.g. 74312349_74312515)
        For MXE (compound id: start1_end1Nstart2_end2) use first component
      - Use chrom_resolved as chrom
      - Use strand_resolved as strand
      - Use gene part of gene::tx as gene_id
    """
    logging.info('Parsing Format B: %s', os.path.basename(filepath))

    # Filter to true positives only
    n_before = len(df)
    df = df[df['label'] == 1].copy()
    n_after  = len(df)
    n_neg_dropped = n_before - n_after
    print(f'  Format B detected (assign_labels output)')
    print(f'  Total rows    : {n_before:,}')
    print(f'  label=1 rows  : {n_after:,}  (label=0 rows dropped: {n_neg_dropped:,})')

    if n_after == 0:
        print(f'  WARNING: no label=1 rows found in {filepath}')
        return pd.DataFrame(columns=REQUIRED_COLS)

    # Parse coordinates from id column
    def parse_id(id_val):
        """Parse intron_start, intron_end from id string."""
        try:
            # MXE: 'start1_end1Nstart2_end2' -> take first component
            first = str(id_val).split('N')[0]
            parts = first.strip().split('_')
            if len(parts) >= 2:
                return int(parts[0]), int(parts[1])
        except (ValueError, IndexError):
            pass
        return None, None

    parsed = df['id'].apply(lambda x: pd.Series(parse_id(x),
                                                  index=['intron_start',
                                                         'intron_end']))
    df['intron_start'] = parsed['intron_start']
    df['intron_end']   = parsed['intron_end']

    # Drop rows where id could not be parsed
    n_before_parse = len(df)
    df = df.dropna(subset=['intron_start', 'intron_end'])
    n_unparsed = n_before_parse - len(df)
    if n_unparsed:
        logging.warning('Format B: %d rows dropped — could not parse id column',
                        n_unparsed)
        print(f'  Unparseable id rows dropped: {n_unparsed:,}')

    df['intron_start'] = df['intron_start'].astype(int)
    df['intron_end']   = df['intron_end'].astype(int)

    # Map columns to canonical names
    df['chrom']  = df['chrom_resolved']
    df['strand'] = df['strand_resolved'].fillna('')

    # Extract gene_id from 'gene::tx' (format: gene_id::transcript_id)
    df['gene_id'] = df['gene::tx'].apply(
        lambda x: str(x).split('::')[0] if '::' in str(x) else str(x)
    )

    # Source: use source_file if present, else 'simulation'
    if 'source_file' in df.columns:
        df['source'] = df['source_file'].fillna('simulation')
    else:
        df['source'] = 'simulation'

    df['label'] = 1
    return df


def _parse_format_a(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """
    Parse extract_true_junctions.py or compatible format.
    Rename column aliases to canonical names.
    """
    logging.info('Parsing Format A: %s', os.path.basename(filepath))
    print(f'  Format A detected (extract_true_junctions output)')

    # Rename aliases
    rename_map = {}
    for col in df.columns:
        canonical = _COL_ALIASES.get(col)
        if canonical and canonical != col:
            rename_map[col] = canonical
    if rename_map:
        df = df.rename(columns=rename_map)
        logging.info('Renamed columns: %s', rename_map)

    # Validate required coordinate columns
    for col in ('intron_start', 'intron_end', 'chrom'):
        if col not in df.columns:
            raise KeyError(
                f'Column "{col}" not found in {filepath}.\n'
                f'Actual columns: {list(df.columns)}\n'
                f'If your file uses a different name, add it to _COL_ALIASES.'
            )

    df['intron_start'] = pd.to_numeric(df['intron_start'], errors='coerce')
    df['intron_end']   = pd.to_numeric(df['intron_end'],   errors='coerce')
    df = df.dropna(subset=['intron_start', 'intron_end'])
    df['intron_start'] = df['intron_start'].astype(int)
    df['intron_end']   = df['intron_end'].astype(int)

    if 'strand'  not in df.columns: df['strand']  = ''
    if 'gene_id' not in df.columns: df['gene_id'] = ''
    if 'source'  not in df.columns: df['source']  = 'simulation'

    df['label'] = 1
    return df


def _parse_gtex_format(df: pd.DataFrame, filepath: str) -> pd.DataFrame:
    """Parse label_gtex_junctions.py output."""
    logging.info('Parsing GTEx format: %s', os.path.basename(filepath))
    print(f'  GTEx format detected')

    # Rename aliases
    rename_map = {c: _COL_ALIASES[c] for c in df.columns if c in _COL_ALIASES}
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ('intron_start', 'intron_end', 'chrom'):
        if col not in df.columns:
            raise KeyError(f'Column "{col}" not found in GTEx file: {filepath}')

    df['intron_start'] = df['intron_start'].astype(int)
    df['intron_end']   = df['intron_end'].astype(int)
    if 'strand'  not in df.columns: df['strand']  = ''
    if 'gene_id' not in df.columns: df['gene_id'] = ''
    if 'source'  not in df.columns: df['source']  = 'GTEx'

    df['label'] = 1
    return df


# ===========================================================================
# Unified loader
# ===========================================================================

def load_positive_file(filepath: str, default_source: str) -> pd.DataFrame:
    """
    Load any positive-class file regardless of format.
    Auto-detects Format A, Format B, or GTEx format.
    """
    if not os.path.isfile(filepath):
        print(f'ERROR: file not found: {filepath}')
        sys.exit(1)

    with open(filepath) as fh:
        first_line = fh.readline()
    sep = '\t' if '\t' in first_line else ','

    df = pd.read_csv(filepath, sep=sep, low_memory=False)
    print(f'\nLoading: {filepath}')
    print(f'  Columns found: {list(df.columns)}')
    print(f'  Total rows   : {len(df):,}')

    fmt = detect_format(df, filepath)

    if fmt == 'B':
        df = _parse_format_b(df, filepath)
    elif fmt == 'gtex':
        df = _parse_gtex_format(df, filepath)
    else:
        df = _parse_format_a(df, filepath)

    df['source'] = df['source'].fillna(default_source)
    return df


def load_negative_file(filepath: str) -> pd.DataFrame:
    """Load negatives.tsv from generate_negatives.py."""
    if not os.path.isfile(filepath):
        print(f'ERROR: file not found: {filepath}')
        sys.exit(1)

    with open(filepath) as fh:
        first_line = fh.readline()
    sep = '\t' if '\t' in first_line else ','

    df = pd.read_csv(filepath, sep=sep, low_memory=False)
    print(f'\nLoading negatives: {filepath}')
    print(f'  Columns found: {list(df.columns)}')
    print(f'  Total rows   : {len(df):,}')

    # Rename aliases
    rename_map = {c: _COL_ALIASES[c] for c in df.columns if c in _COL_ALIASES}
    if rename_map:
        df = df.rename(columns=rename_map)

    for col in ('intron_start', 'intron_end', 'chrom'):
        if col not in df.columns:
            raise KeyError(
                f'Column "{col}" not found in negatives file: {filepath}.\n'
                f'Actual columns: {list(df.columns)}'
            )

    df['intron_start'] = df['intron_start'].astype(int)
    df['intron_end']   = df['intron_end'].astype(int)
    if 'strand'  not in df.columns: df['strand']  = ''
    if 'gene_id' not in df.columns: df['gene_id'] = ''
    if 'source'  not in df.columns: df['source']  = 'artifact'
    df['label'] = 0
    return df


# ===========================================================================
# Dedup, contamination, balance
# ===========================================================================

def deduplicate(df: pd.DataFrame, label_name: str) -> pd.DataFrame:
    source_priority = {'GTEx': 0, 'simulation': 1}
    df = df.copy()
    df['_p'] = df['source'].map(source_priority).fillna(99).astype(int)
    df = df.sort_values('_p')
    before = len(df)
    df = df.drop_duplicates(
        subset=['chrom', 'intron_start', 'intron_end'], keep='first')
    df = df.drop(columns=['_p'])
    removed = before - len(df)
    if removed:
        print(f'  {label_name} duplicates removed: {removed:,}  '
              f'({before:,} -> {len(df):,})')
    return df.reset_index(drop=True)


def remove_contamination(positives: pd.DataFrame,
                          negatives: pd.DataFrame) -> pd.DataFrame:
    pos_keys = set(zip(positives['chrom'],
                       positives['intron_start'],
                       positives['intron_end']))
    mask = negatives.apply(
        lambda r: (r['chrom'], r['intron_start'], r['intron_end'])
                  not in pos_keys, axis=1)
    n_removed = (~mask).sum()
    if n_removed:
        print(f'  Contaminated negatives removed: {n_removed:,}')
    return negatives[mask].reset_index(drop=True)


def balance_classes(positives, negatives, max_ratio, seed):
    n_pos, n_neg = len(positives), len(negatives)
    ratio = n_neg / max(n_pos, 1)
    print(f'\n  Current ratio (neg:pos): {ratio:.2f}  '
          f'({n_neg:,} neg / {n_pos:,} pos)')
    if ratio > max_ratio:
        target = int(n_pos * max_ratio)
        print(f'  Downsampling negatives: {n_neg:,} -> {target:,}')
        negatives = negatives.sample(n=target, random_state=seed)
    elif (n_pos / max(n_neg, 1)) > max_ratio:
        target = int(n_neg * max_ratio)
        print(f'  Downsampling positives: {n_pos:,} -> {target:,}')
        positives = positives.sample(n=target, random_state=seed)
    else:
        print(f'  Class balance within limit — no downsampling needed')
    return positives.reset_index(drop=True), negatives.reset_index(drop=True)


# ===========================================================================
# Align columns and stats
# ===========================================================================

def align_and_concat(frames):
    all_cols = list(REQUIRED_COLS)
    seen = set(all_cols)
    for df in frames:
        for c in df.columns:
            if c not in seen:
                all_cols.append(c)
                seen.add(c)
    aligned = []
    for df in frames:
        df = df.copy()
        for c in all_cols:
            if c not in df.columns:
                df[c] = np.nan
        aligned.append(df[all_cols])
    return pd.concat(aligned, ignore_index=True)


def write_stats_report(df: pd.DataFrame, output_path: str):
    lines = []
    sep = '=' * 55
    lines += [sep, 'TRAINING SET STATISTICS',
              f'Generated : {dt.now().strftime("%Y-%m-%d %H:%M:%S")}', sep, '']

    n_total = len(df)
    n_pos   = (df['label'] == 1).sum()
    n_neg   = (df['label'] == 0).sum()

    lines.append(f'Total junctions     : {n_total:,}')
    lines.append(f'Positives (label=1) : {n_pos:,}  ({100*n_pos/max(n_total,1):.1f}%)')
    lines.append(f'Negatives (label=0) : {n_neg:,}  ({100*n_neg/max(n_total,1):.1f}%)')
    lines.append(f'Ratio (neg:pos)     : {n_neg/max(n_pos,1):.2f}')
    lines.append('')

    lines.append('--- By source ---')
    for src, grp in df.groupby('source'):
        n1 = (grp['label'] == 1).sum()
        n0 = (grp['label'] == 0).sum()
        lines.append(f'  {src:<25}: {len(grp):>8,}  (+{n1:,} / -{n0:,})')
    lines.append('')

    lines.append('--- By chromosome (top 10) ---')
    for chrom, cnt in df['chrom'].value_counts().head(10).items():
        lines.append(f'  {chrom:<15}: {cnt:,}')
    lines.append('')

    lines.append('--- Intron length statistics ---')
    df = df.copy()
    df['_len'] = df['intron_end'] - df['intron_start']
    for lbl, name in [(1, 'Positives'), (0, 'Negatives')]:
        sub = df[df['label'] == lbl]['_len']
        if len(sub):
            lines.append(f'  {name}:  min={sub.min():,}  '
                         f'median={int(sub.median()):,}  '
                         f'max={sub.max():,} bp')
    lines.append('')
    lines.append(sep)

    report = '\n'.join(lines)
    with open(output_path, 'w') as fh:
        fh.write(report)
    print('\n' + report)
    logging.info('Stats report: %s', output_path)


# ===========================================================================
# Main
# ===========================================================================

def build_training_set(sim_pos_file, gtex_pos_file, neg_file,
                        output_file, max_ratio=None, seed=42):

    start = dt.now()
    logging.info('=== build_training_set.py started: %s ===',
                 start.strftime('%H:%M:%S'))

    pos_frames = []

    if sim_pos_file:
        sim_pos = load_positive_file(sim_pos_file, 'simulation')
        pos_frames.append(sim_pos)
        print(f'  Rows after parsing: {len(sim_pos):,}')

    if gtex_pos_file:
        if not os.path.isfile(gtex_pos_file):
            logging.warning('GTEx file not found: %s — skipping', gtex_pos_file)
        else:
            gtex_pos = load_positive_file(gtex_pos_file, 'GTEx')
            pos_frames.append(gtex_pos)
            print(f'  Rows after parsing: {len(gtex_pos):,}')

    if not pos_frames:
        print('ERROR: no positive rows loaded.')
        sys.exit(1)

    negatives = load_negative_file(neg_file)

    print('\n--- Deduplication ---')
    all_pos   = pd.concat(pos_frames, ignore_index=True)
    all_pos   = deduplicate(all_pos, 'positives')
    negatives = deduplicate(negatives, 'negatives')

    print('\n--- Contamination check ---')
    negatives = remove_contamination(all_pos, negatives)

    if max_ratio is not None:
        print(f'\n--- Class balancing (max ratio {max_ratio}:1) ---')
        all_pos, negatives = balance_classes(all_pos, negatives, max_ratio, seed)

    final = align_and_concat([all_pos, negatives])
    final = final.sample(frac=1, random_state=seed).reset_index(drop=True)

    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    final.to_csv(output_file, sep='\t', index=False)

    stats_file = output_file.replace('.tsv', '_stats.txt')
    if stats_file == output_file:
        stats_file = output_file + '_stats.txt'
    write_stats_report(final, stats_file)

    elapsed = (dt.now() - start).seconds
    print(f'\nFinal training set : {output_file}')
    print(f'Stats report       : {stats_file}')
    print(f'Total rows         : {len(final):,}')
    print(f'Elapsed            : {elapsed}s')
    logging.info('Done. %s (%d rows, %ds)', output_file, len(final), elapsed)
    return final


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Merge simulation positives, GTEx positives, and negatives\n'
            'into a final shuffled training set.\n\n'
            'Accepts both extract_true_junctions.py output\n'
            'AND assign_labels.py output as --sim-pos input.\n'
            'Format is detected automatically from column names.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--sim-pos', metavar='FILE', default=None,
        help='true_junctions.tsv OR labeled_junctions.tsv  (label=1)')
    parser.add_argument('--gtex-pos', metavar='FILE', default=None,
        help='gtex_true_junctions.tsv from label_gtex_junctions.py  (label=1)')
    parser.add_argument('--neg', metavar='FILE', required=True,
        help='negatives.tsv from generate_negatives.py  (label=0)')
    parser.add_argument('-o', '--output', metavar='FILE',
        default='final_training_set.tsv',
        help='Output TSV (default: final_training_set.tsv)')
    parser.add_argument('--max-ratio', metavar='FLOAT', type=float,
        default=None,
        help='Max neg:pos ratio — majority downsampled if exceeded.\n'
             'Recommended: 3.0  (default: no balancing)')
    parser.add_argument('--seed', metavar='INT', type=int, default=42,
        help='Random seed (default: 42)')
    parser.add_argument('--log', metavar='FILE',
        default='build_training_set.log')
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

    if not args.sim_pos and not args.gtex_pos:
        print('ERROR: provide at least one of --sim-pos or --gtex-pos')
        sys.exit(1)

    print(f'Start Time = {dt.now().strftime("%H:%M:%S")}')
    build_training_set(
        sim_pos_file  = args.sim_pos,
        gtex_pos_file = args.gtex_pos,
        neg_file      = args.neg,
        output_file   = args.output,
        max_ratio     = args.max_ratio,
        seed          = args.seed,
    )


if __name__ == '__main__':
    main()
