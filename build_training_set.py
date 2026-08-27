# coding: UTF-8
#!/usr/bin/env python
"""
build_training_set.py
=====================
Merge positive and negative junction sets into a single training set
ready for Random Forest / ML artifact filtering.

Input files
-----------
  Positives (label = 1):
    --sim-pos   true_junctions.tsv       from extract_true_junctions.py
    --gtex-pos  gtex_true_junctions.tsv  from label_gtex_junctions.py

  Negatives (label = 0):
    --neg       negatives.tsv            from generate_negatives.py

Output
------
  final_training_set.tsv  — shuffled, balanced training set with columns:
    chrom | intron_start | intron_end | strand | gene_id |
    source | label | [any extra columns present in input files]

  final_training_set_stats.txt — class balance and source breakdown report

Key steps
---------
  1. Load and standardise each input file to a common column schema
  2. Deduplicate coordinates within each class
  3. Remove any negatives that overlap with positives (contamination check)
  4. Balance classes (downsample majority class if ratio exceeds --max-ratio)
  5. Shuffle and write final training set
  6. Write a summary statistics report

Usage
-----
    python build_training_set.py \\
        --sim-pos  true_junctions.tsv \\
        --gtex-pos gtex_true_junctions.tsv \\
        --neg      negatives.tsv \\
        --output   final_training_set.tsv

    # With class balancing (cap negatives at 3x positives)
    python build_training_set.py \\
        --sim-pos   true_junctions.tsv \\
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


# ===========================================================================
# Column schema shared across all input files
# ===========================================================================

# These columns must exist in the output — all inputs are normalised to this
REQUIRED_COLS = ['chrom', 'intron_start', 'intron_end', 'strand',
                 'gene_id', 'source', 'label']


# ===========================================================================
# Step 1 — Load and normalise each input file
# ===========================================================================

def _normalise(df: pd.DataFrame, label: int, source_tag: str) -> pd.DataFrame:
    """
    Ensure a DataFrame has the required columns, fill missing ones,
    enforce label value, and add source tag where absent.

    Parameters
    ----------
    df         : raw loaded DataFrame
    label      : 1 for positives, 0 for negatives
    source_tag : fallback source string if 'source' column is missing
    """
    df = df.copy()

    # Enforce label
    df['label'] = label

    # Fill missing required columns with defaults
    if 'gene_id' not in df.columns:
        # Try 'Description' (GTEx GCT output) or 'gene_name'
        for alt in ('Description', 'gene_name', 'gene'):
            if alt in df.columns:
                df['gene_id'] = df[alt]
                break
        else:
            df['gene_id'] = ''

    if 'strand' not in df.columns:
        df['strand'] = ''

    if 'source' not in df.columns:
        df['source'] = source_tag
    else:
        # Keep existing source but tag with origin for traceability
        df['source'] = df['source'].fillna(source_tag)

    # Ensure coordinate columns are int
    df['intron_start'] = df['intron_start'].astype(int)
    df['intron_end']   = df['intron_end'].astype(int)

    # Keep only columns that exist
    keep = REQUIRED_COLS + [c for c in df.columns if c not in REQUIRED_COLS]
    df = df[[c for c in keep if c in df.columns]]

    return df


def load_simulation_positives(path: str) -> pd.DataFrame:
    """Load true_junctions.tsv from extract_true_junctions.py."""
    logging.info('Loading simulation positives: %s', path)
    df = pd.read_csv(path, sep='\t')
    return _normalise(df, label=1, source_tag='simulation')


def load_gtex_positives(path: str) -> pd.DataFrame:
    """Load gtex_true_junctions.tsv from label_gtex_junctions.py."""
    logging.info('Loading GTEx positives: %s', path)
    df = pd.read_csv(path, sep='\t')
    return _normalise(df, label=1, source_tag='GTEx')


def load_negatives(path: str) -> pd.DataFrame:
    """Load negatives.tsv from generate_negatives.py."""
    logging.info('Loading negatives: %s', path)
    df = pd.read_csv(path, sep='\t')
    return _normalise(df, label=0, source_tag='artifact')


# ===========================================================================
# Step 2 — Deduplicate within each class
# ===========================================================================

def deduplicate(df: pd.DataFrame, label_name: str) -> pd.DataFrame:
    """
    Remove duplicate (chrom, intron_start, intron_end) rows.
    When duplicates exist across sources (e.g. a junction in both
    simulation and GTEx), keep the row whose source is 'GTEx' first,
    then 'simulation', then others — so the most authoritative source wins.
    """
    source_priority = {'GTEx': 0, 'simulation': 1}
    df = df.copy()
    df['_priority'] = df['source'].map(source_priority).fillna(99).astype(int)
    df = df.sort_values('_priority')
    before = len(df)
    df = df.drop_duplicates(subset=['chrom', 'intron_start', 'intron_end'],
                             keep='first')
    df = df.drop(columns=['_priority'])
    after = len(df)
    if before != after:
        logging.info('%s: removed %d duplicate coordinates (%d -> %d)',
                     label_name, before - after, before, after)
    return df.reset_index(drop=True)


# ===========================================================================
# Step 3 — Contamination check
# ===========================================================================

def remove_contamination(positives: pd.DataFrame,
                          negatives: pd.DataFrame) -> pd.DataFrame:
    """
    Remove any negative rows whose coordinates match a positive.
    This prevents mislabeled training examples where a real junction
    was accidentally placed in the negative class.

    Returns the cleaned negatives DataFrame.
    """
    pos_keys = set(
        zip(positives['chrom'],
            positives['intron_start'],
            positives['intron_end'])
    )

    mask = negatives.apply(
        lambda r: (r['chrom'], r['intron_start'], r['intron_end'])
                  not in pos_keys,
        axis=1
    )

    n_contaminated = (~mask).sum()
    if n_contaminated > 0:
        logging.info(
            'Contamination check: removed %d negatives that matched '
            'a positive junction', n_contaminated)
        print(f'  Removed {n_contaminated:,} contaminated negatives '
              f'(coordinates found in positive set)')

    return negatives[mask].reset_index(drop=True)


# ===========================================================================
# Step 4 — Class balancing
# ===========================================================================

def balance_classes(positives: pd.DataFrame,
                    negatives: pd.DataFrame,
                    max_ratio: float,
                    seed: int) -> tuple:
    """
    If negatives > positives * max_ratio, downsample negatives.
    If positives > negatives * max_ratio, downsample positives.

    Returns (positives, negatives) after optional downsampling.
    """
    n_pos = len(positives)
    n_neg = len(negatives)

    if n_pos == 0 or n_neg == 0:
        return positives, negatives

    actual_ratio = n_neg / n_pos

    if actual_ratio > max_ratio:
        target_neg = int(n_pos * max_ratio)
        logging.info(
            'Downsampling negatives: %d -> %d (ratio %.1f -> %.1f)',
            n_neg, target_neg, actual_ratio, max_ratio)
        print(f'\n  Class imbalance: {n_neg:,} negatives vs {n_pos:,} positives '
              f'(ratio {actual_ratio:.1f}:1)')
        print(f'  Downsampling negatives to {target_neg:,} '
              f'(max ratio {max_ratio}:1)')
        negatives = negatives.sample(n=target_neg, random_state=seed)

    elif (n_pos / n_neg) > max_ratio:
        target_pos = int(n_neg * max_ratio)
        logging.info(
            'Downsampling positives: %d -> %d',
            n_pos, target_pos)
        print(f'\n  Class imbalance: {n_pos:,} positives vs {n_neg:,} negatives')
        print(f'  Downsampling positives to {target_pos:,} '
              f'(max ratio {max_ratio}:1)')
        positives = positives.sample(n=target_pos, random_state=seed)

    else:
        print(f'\n  Class balance OK: {n_pos:,} positives, '
              f'{n_neg:,} negatives (ratio {actual_ratio:.2f}:1)')

    return positives.reset_index(drop=True), negatives.reset_index(drop=True)


# ===========================================================================
# Step 5 — Align columns across DataFrames
# ===========================================================================

def align_columns(frames: list) -> list:
    """
    Ensure all DataFrames share the same column set before concatenation.
    Columns present in some but not all frames are filled with NaN.
    Required columns always appear first in a fixed order.
    """
    # Union of all columns
    all_cols = []
    seen = set()
    # Required columns first
    for c in REQUIRED_COLS:
        if c not in seen:
            all_cols.append(c)
            seen.add(c)
    # Then any extra columns from any frame
    for df in frames:
        for c in df.columns:
            if c not in seen:
                all_cols.append(c)
                seen.add(c)

    aligned = []
    for df in frames:
        missing = [c for c in all_cols if c not in df.columns]
        for m in missing:
            df = df.copy()
            df[m] = np.nan
        aligned.append(df[all_cols])

    return aligned


# ===========================================================================
# Step 6 — Write statistics report
# ===========================================================================

def write_stats_report(df: pd.DataFrame, output_path: str):
    """Write a human-readable summary of the training set composition."""
    lines = []
    lines.append('=' * 55)
    lines.append('TRAINING SET STATISTICS')
    lines.append(f'Generated: {dt.now().strftime("%Y-%m-%d %H:%M:%S")}')
    lines.append('=' * 55)
    lines.append('')

    n_total = len(df)
    n_pos   = (df['label'] == 1).sum()
    n_neg   = (df['label'] == 0).sum()

    lines.append(f'Total junctions   : {n_total:,}')
    lines.append(f'Positives (label=1): {n_pos:,}  '
                 f'({100*n_pos/max(n_total,1):.1f}%)')
    lines.append(f'Negatives (label=0): {n_neg:,}  '
                 f'({100*n_neg/max(n_total,1):.1f}%)')
    lines.append(f'Class ratio (neg:pos): '
                 f'{n_neg/max(n_pos,1):.2f}')
    lines.append('')

    lines.append('--- Breakdown by source ---')
    for src, grp in df.groupby('source'):
        n1 = (grp['label'] == 1).sum()
        n0 = (grp['label'] == 0).sum()
        lines.append(f'  {src:<30}: '
                     f'{len(grp):>7,} total  '
                     f'(+{n1:,} / -{n0:,})')
    lines.append('')

    lines.append('--- Breakdown by chromosome ---')
    chrom_counts = df['chrom'].value_counts()
    for chrom, count in chrom_counts.items():
        lines.append(f'  {chrom:<15}: {count:,}')
    lines.append('')

    lines.append('--- Intron length statistics ---')
    df = df.copy()
    df['intron_length'] = df['intron_end'] - df['intron_start']
    for label, name in [(1, 'Positives'), (0, 'Negatives')]:
        sub = df[df['label'] == label]['intron_length']
        if len(sub) > 0:
            lines.append(f'  {name}:')
            lines.append(f'    min    : {sub.min():,} bp')
            lines.append(f'    median : {int(sub.median()):,} bp')
            lines.append(f'    mean   : {int(sub.mean()):,} bp')
            lines.append(f'    max    : {sub.max():,} bp')
    lines.append('')
    lines.append('=' * 55)

    report = '\n'.join(lines)
    with open(output_path, 'w') as fh:
        fh.write(report)

    print('\n' + report)
    logging.info('Stats report written: %s', output_path)


# ===========================================================================
# Main function
# ===========================================================================

def build_training_set(sim_pos_file: str,
                        gtex_pos_file: str,
                        neg_file: str,
                        output_file: str,
                        max_ratio: float = None,
                        seed: int = 42) -> pd.DataFrame:
    """
    Full pipeline to merge all sources into a final training set.

    Parameters
    ----------
    sim_pos_file  : path to true_junctions.tsv (simulation positives)
    gtex_pos_file : path to gtex_true_junctions.tsv (GTEx positives)
                    pass None to skip
    neg_file      : path to negatives.tsv
    output_file   : path for final_training_set.tsv
    max_ratio     : maximum negative:positive ratio (None = no balancing)
    seed          : random seed for shuffling and downsampling
    """
    start = dt.now()
    logging.info('=== build_training_set.py started: %s ===',
                 start.strftime('%H:%M:%S'))

    # ── Load ─────────────────────────────────────────────────────────────────
    pos_frames = []

    if sim_pos_file:
        if not os.path.isfile(sim_pos_file):
            logging.error('Simulation positives file not found: %s', sim_pos_file)
            sys.exit(1)
        sim_pos = load_simulation_positives(sim_pos_file)
        pos_frames.append(sim_pos)
        print(f'Simulation positives loaded : {len(sim_pos):,}')

    if gtex_pos_file:
        if not os.path.isfile(gtex_pos_file):
            logging.warning('GTEx positives file not found: %s — skipping',
                            gtex_pos_file)
        else:
            gtex_pos = load_gtex_positives(gtex_pos_file)
            pos_frames.append(gtex_pos)
            print(f'GTEx positives loaded       : {len(gtex_pos):,}')

    if not pos_frames:
        print('ERROR: no positive files loaded. Provide --sim-pos and/or --gtex-pos')
        sys.exit(1)

    if not os.path.isfile(neg_file):
        logging.error('Negatives file not found: %s', neg_file)
        sys.exit(1)
    negatives = load_negatives(neg_file)
    print(f'Negatives loaded            : {len(negatives):,}')

    # ── Deduplicate within positives ─────────────────────────────────────────
    print('\n--- Deduplication ---')
    all_pos = pd.concat(pos_frames, ignore_index=True)
    print(f'  Positives before dedup : {len(all_pos):,}')
    all_pos = deduplicate(all_pos, 'positives')
    print(f'  Positives after dedup  : {len(all_pos):,}')

    print(f'  Negatives before dedup : {len(negatives):,}')
    negatives = deduplicate(negatives, 'negatives')
    print(f'  Negatives after dedup  : {len(negatives):,}')

    # ── Contamination check ──────────────────────────────────────────────────
    print('\n--- Contamination check ---')
    negatives = remove_contamination(all_pos, negatives)
    print(f'  Negatives after contamination check: {len(negatives):,}')

    # ── Class balancing ──────────────────────────────────────────────────────
    if max_ratio is not None:
        print(f'\n--- Class balancing (max ratio {max_ratio}:1) ---')
        all_pos, negatives = balance_classes(
            all_pos, negatives, max_ratio, seed)

    # ── Align columns and concatenate ────────────────────────────────────────
    aligned = align_columns([all_pos, negatives])
    final   = pd.concat(aligned, ignore_index=True)

    # ── Shuffle ──────────────────────────────────────────────────────────────
    final = final.sample(frac=1, random_state=seed).reset_index(drop=True)

    # ── Write output ─────────────────────────────────────────────────────────
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    final.to_csv(output_file, sep='\t', index=False)

    # ── Stats report ─────────────────────────────────────────────────────────
    stats_file = output_file.replace('.tsv', '_stats.txt')
    if stats_file == output_file:
        stats_file = output_file + '_stats.txt'
    write_stats_report(final, stats_file)

    elapsed = (dt.now() - start).seconds
    print(f'\nFinal training set : {output_file}')
    print(f'Stats report       : {stats_file}')
    print(f'Total rows         : {len(final):,}')
    print(f'Elapsed            : {elapsed}s')
    logging.info('Done. Output: %s (%d rows, %ds)',
                 output_file, len(final), elapsed)

    return final


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Merge simulation positives, GTEx positives, and negatives into '
            'a final shuffled training set for ML artifact filtering.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument('--sim-pos', metavar='FILE', default=None,
        help='true_junctions.tsv from extract_true_junctions.py  (label=1)')
    parser.add_argument('--gtex-pos', metavar='FILE', default=None,
        help='gtex_true_junctions.tsv from label_gtex_junctions.py  (label=1)')
    parser.add_argument('--neg', metavar='FILE', required=True,
        help='negatives.tsv from generate_negatives.py  (label=0)')
    parser.add_argument('-o', '--output', metavar='FILE',
        default='final_training_set.tsv',
        help='Output training set TSV (default: final_training_set.tsv)')
    parser.add_argument('--max-ratio', metavar='FLOAT', type=float,
        default=None,
        help='Maximum negative:positive ratio. '
             'Majority class is downsampled if exceeded. '
             'Recommended: 3.0  (default: no balancing)')
    parser.add_argument('--seed', metavar='INT', type=int, default=42,
        help='Random seed for shuffling and downsampling (default: 42)')
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

    print(f'Start Time  = {dt.now().strftime("%H:%M:%S")}')

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
