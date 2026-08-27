# coding: UTF-8
#!/usr/bin/env python
"""
assign_labels.py
================
Assign true/artifact labels to junctions detected by detect_AS.py
by comparing them against the ground-truth junction set produced by
extract_true_junctions.py.

Label convention
----------------
  1  = junction is present in the GTF used for simulation  (true splicing event)
  0  = junction is NOT in the GTF                          (alignment artifact / novel)

Usage
-----
    python assign_labels.py \\
        --events    output_dir/results_group1_ES_sample01_events.txt \\
        --true      true_junctions.tsv \\
        --gtf       annotation.gtf \\
        --output    labeled_junctions.tsv

    # Process all events files in a directory at once
    python assign_labels.py \\
        --events-dir  output_dir/ \\
        --true        true_junctions.tsv \\
        --gtf         annotation.gtf \\
        --output      labeled_junctions.tsv

Input — events files (from detect_AS.py _writeEvent)
------------------------------------------------------
  Tab-separated, one header row, columns:
    gene::tx | id | coverage | exclusion1 | exclusion2 |
    inclusion | over_boundaries | PSI  [+ optional splice site columns]

  The 'id' column holds the junction key:
    - Most AS types  : intron_start_intron_end   (e.g.  43094691_43095845)
    - MXE events     : start1_end1Nstart2_end2   (compound key)

Input — true_junctions.tsv (from extract_true_junctions.py)
-------------------------------------------------------------
  Tab-separated, columns:
    chrom | intron_start | intron_end | strand | gene_id | transcript_id | label

Input — GTF
-----------
  Used to build a gene_id -> (chrom, strand) lookup so that chromosome
  can be resolved from the gene::tx column in the events files.

Output — labeled_junctions.tsv
-------------------------------
  One row per detected junction with all original event columns preserved
  plus three new columns appended:
    chrom_resolved  : chromosome resolved from GTF gene lookup
    strand_resolved : strand resolved from GTF gene lookup
    label           : 1 = true event, 0 = artifact

Dependencies
------------
    pip install pandas
"""

import os
import sys
import glob
import argparse
import logging
import gzip
from collections import defaultdict
from datetime import datetime as dt
import pandas as pd


# ===========================================================================
# Step 1 — Build gene -> (chrom, strand) index from GTF
# ===========================================================================

def build_gene_index(gtf_file: str) -> dict:
    """
    Parse the GTF file and return a dict mapping gene_id -> (chrom, strand).

    Only 'transcript' or 'gene' feature lines are needed for this lookup —
    we stop collecting exon-level detail early to keep memory low.

    Parameters
    ----------
    gtf_file : str  path to GTF (plain text or .gz)

    Returns
    -------
    dict  { gene_id: (chrom, strand) }
    """
    logging.info('Building gene index from GTF: %s', gtf_file)

    gene_index = {}
    opener = gzip.open if gtf_file.endswith('.gz') else open
    mode   = 'rt' if gtf_file.endswith('.gz') else 'r'

    with opener(gtf_file, mode) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            # Only need gene or transcript lines for chrom/strand lookup
            if fields[2] not in ('gene', 'transcript', 'exon'):
                continue

            chrom  = fields[0]
            strand = fields[6]
            attr   = fields[8]

            # Parse gene_id from attribute string
            gene_id = ''
            for kv in attr.strip(';').split(';'):
                kv = kv.strip()
                if kv.startswith('gene_id'):
                    gene_id = kv.split(' ', 1)[1].replace('"', '').strip()
                    break

            if gene_id and gene_id not in gene_index:
                gene_index[gene_id] = (chrom, strand)

    logging.info('Gene index built: %d unique genes', len(gene_index))
    return gene_index


# ===========================================================================
# Step 2 — Load true junction set from extract_true_junctions output
# ===========================================================================

def load_true_junctions(true_file: str) -> tuple:
    """
    Load true_junctions.tsv and build an O(1) lookup set.

    Returns
    -------
    (true_set, true_df)
        true_set : set of (chrom, intron_start, intron_end) tuples
                   — strand is intentionally omitted because the events
                   output does not carry strand directly; chromosome +
                   coordinates are sufficient for unambiguous matching
        true_df  : full DataFrame (for reporting)
    """
    logging.info('Loading true junctions: %s', true_file)

    true_df = pd.read_csv(true_file, sep='\t')

    required = {'chrom', 'intron_start', 'intron_end'}
    missing  = required - set(true_df.columns)
    if missing:
        raise ValueError(
            f'true_junctions file is missing columns: {missing}\n'
            f'Expected output from extract_true_junctions.py'
        )

    true_set = set(
        zip(true_df['chrom'],
            true_df['intron_start'].astype(int),
            true_df['intron_end'].astype(int))
    )

    logging.info('True junction set loaded: %d unique junctions', len(true_set))
    return true_set, true_df


# ===========================================================================
# Step 3 — Parse junction coordinates from the events 'id' column
# ===========================================================================

def parse_junction_ids(id_value: str) -> list:
    """
    Extract one or more (intron_start, intron_end) pairs from an id value.

    Most AS types:  '43094691_43095845'           -> [(43094691, 43095845)]
    MXE events:     '43094691_43095845N43096000_43097000'
                                                  -> [(43094691, 43095845),
                                                      (43096000, 43097000)]

    Returns list of (int, int) tuples — one per component junction.
    Returns [] if parsing fails.
    """
    results = []
    # MXE uses 'N' to join two junction keys
    parts = id_value.split('N') if 'N' in id_value else [id_value]
    for part in parts:
        coords = part.strip().split('_')
        if len(coords) >= 2:
            try:
                results.append((int(coords[0]), int(coords[1])))
            except ValueError:
                logging.warning('Could not parse junction id component: %s', part)
    return results


# ===========================================================================
# Step 4 — Load and label one events file
# ===========================================================================

def label_events_file(events_file: str,
                      true_set: set,
                      gene_index: dict) -> pd.DataFrame:
    """
    Read one events file, resolve chrom from the GTF gene index, compare
    each junction against the true set, and return a labeled DataFrame.

    Parameters
    ----------
    events_file : str   path to one *_events.txt file from detect_AS.py
    true_set    : set   (chrom, intron_start, intron_end) true junction keys
    gene_index  : dict  gene_id -> (chrom, strand)

    Returns
    -------
    pd.DataFrame with all original columns plus:
        source_file, chrom_resolved, strand_resolved, label
    """
    df = pd.read_csv(events_file, sep='\t')

    if df.empty:
        logging.warning('Empty file skipped: %s', events_file)
        return pd.DataFrame()

    # Validate required columns
    for col in ('gene::tx', 'id'):
        if col not in df.columns:
            logging.warning('Missing column "%s" in %s — skipping', col, events_file)
            return pd.DataFrame()

    chrom_col  = []
    strand_col = []
    label_col  = []

    for _, row in df.iterrows():
        gene_tx  = str(row['gene::tx'])
        id_value = str(row['id'])

        # Resolve gene_id from 'gene::tx' column
        gene_id  = gene_tx.split('::')[0] if '::' in gene_tx else gene_tx
        chrom, strand = gene_index.get(gene_id, ('', ''))

        chrom_col.append(chrom)
        strand_col.append(strand)

        if not chrom:
            # Gene not found in GTF — cannot match, treat as unknown (label 0)
            logging.debug('Gene not found in GTF index: %s', gene_id)
            label_col.append(0)
            continue

        # Parse junction coordinate(s) from id column
        junc_pairs = parse_junction_ids(id_value)

        if not junc_pairs:
            label_col.append(0)
            continue

        # Label = 1 if ANY component junction of this event is in the true set
        # For MXE (two junctions), both should be true — use ALL for strictness
        is_true = all(
            (chrom, start, end) in true_set
            for start, end in junc_pairs
        )
        label_col.append(1 if is_true else 0)

    df['source_file']     = os.path.basename(events_file)
    df['chrom_resolved']  = chrom_col
    df['strand_resolved'] = strand_col
    df['label']           = label_col

    return df


# ===========================================================================
# Step 5 — Process all events files and write output
# ===========================================================================

def assign_labels(events_files: list,
                  true_file: str,
                  gtf_file: str,
                  output_file: str) -> pd.DataFrame:
    """
    Main function: assign labels to all junctions across all events files.

    Parameters
    ----------
    events_files : list  list of paths to *_events.txt files
    true_file    : str   path to true_junctions.tsv
    gtf_file     : str   path to GTF annotation file
    output_file  : str   path for labeled output TSV

    Returns
    -------
    pd.DataFrame  the full labeled junction table
    """
    start = dt.now()
    logging.info('=== assign_labels.py started: %s ===', start.strftime('%H:%M:%S'))

    if not events_files:
        raise ValueError('No events files provided or found.')

    # --- Load inputs ---------------------------------------------------------
    gene_index          = build_gene_index(gtf_file)
    true_set, true_df   = load_true_junctions(true_file)

    print(f'\nInputs loaded:')
    print(f'  Events files      : {len(events_files)}')
    print(f'  True junctions    : {len(true_set):,}')
    print(f'  Genes in GTF index: {len(gene_index):,}')

    # --- Process each events file --------------------------------------------
    all_frames = []
    for ef in sorted(events_files):
        logging.info('Processing: %s', ef)
        frame = label_events_file(ef, true_set, gene_index)
        if not frame.empty:
            all_frames.append(frame)
            n1 = (frame['label'] == 1).sum()
            n0 = (frame['label'] == 0).sum()
            logging.info('  %s -> %d true, %d artifact',
                         os.path.basename(ef), n1, n0)

    if not all_frames:
        raise RuntimeError('No labeled rows produced. Check input file formats.')

    result = pd.concat(all_frames, ignore_index=True)

    # --- Summary -------------------------------------------------------------
    n_total    = len(result)
    n_true     = (result['label'] == 1).sum()
    n_artifact = (result['label'] == 0).sum()
    n_unknown  = (result['chrom_resolved'] == '').sum()

    print(f'\n=== Labeling Summary ===')
    print(f'  Total junctions   : {n_total:,}')
    print(f'  Label = 1 (true)  : {n_true:,}  ({100*n_true/max(n_total,1):.1f}%)')
    print(f'  Label = 0 (artifact): {n_artifact:,}  ({100*n_artifact/max(n_total,1):.1f}%)')
    print(f'  Unresolved chrom  : {n_unknown:,}')
    if n_true > 0:
        print(f'  Class ratio (0:1) : {n_artifact/n_true:.2f}')
    print(f'========================\n')

    # --- Write output --------------------------------------------------------
    out_dir = os.path.dirname(output_file)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)

    result.to_csv(output_file, sep='\t', index=False)

    elapsed = (dt.now() - start).seconds
    logging.info('Output written: %s (%d rows, %ds)', output_file, n_total, elapsed)
    print(f'Output written : {output_file}')
    print(f'Rows           : {n_total:,}')
    print(f'Elapsed        : {elapsed}s')

    return result


# ===========================================================================
# CLI
# ===========================================================================

def collect_events_files(args) -> list:
    """Collect events file paths from --events and/or --events-dir."""
    files = []

    if args.events:
        for pattern in args.events:
            matched = glob.glob(pattern)
            if not matched:
                logging.warning('No files matched pattern: %s', pattern)
            files.extend(matched)

    if args.events_dir:
        for d in args.events_dir:
            if not os.path.isdir(d):
                logging.warning('Not a directory: %s', d)
                continue
            matched = glob.glob(os.path.join(d, '*_events.txt'))
            if not matched:
                logging.warning('No *_events.txt files found in: %s', d)
            files.extend(matched)

    # Deduplicate while preserving order
    seen = set()
    unique = []
    for f in files:
        if f not in seen:
            seen.add(f)
            unique.append(f)

    return unique


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Assign true/artifact labels to junctions detected by detect_AS.py\n'
            'by comparing against the ground-truth set from extract_true_junctions.py.\n\n'
            'Output: labeled TSV with label=1 (true event) or label=0 (artifact).'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    input_grp = parser.add_argument_group('Input — events files (use one or both)')
    input_grp.add_argument('--events', metavar='FILE', nargs='+', default=None,
        help='One or more events files (or glob patterns) from detect_AS.py.\n'
             'e.g. --events output_dir/*_events.txt')
    input_grp.add_argument('--events-dir', metavar='DIR', nargs='+', default=None,
        help='One or more directories to search for *_events.txt files.\n'
             'e.g. --events-dir output_dir/')

    parser.add_argument('--true', metavar='FILE', required=True,
        help='true_junctions.tsv from extract_true_junctions.py')
    parser.add_argument('--gtf', metavar='FILE', required=True,
        help='GTF annotation file (same one used for simulation and detect_AS.py)')
    parser.add_argument('-o', '--output', metavar='FILE',
        default='labeled_junctions.tsv',
        help='Output labeled TSV file (default: labeled_junctions.tsv)')
    parser.add_argument('--log', metavar='FILE',
        default='assign_labels.log',
        help='Log file path (default: assign_labels.log)')

    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level    = logging.DEBUG,
        filemode = 'w',
        format   = '[%(levelname)s] %(asctime)s %(message)s',
        filename = args.log
    )
    # Also print INFO to stdout
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(console)

    if not args.events and not args.events_dir:
        print('ERROR: provide at least one of --events or --events-dir')
        sys.exit(1)

    for f in [args.true, args.gtf]:
        if not os.path.isfile(f):
            print(f'ERROR: file not found: {f}')
            sys.exit(1)

    events_files = collect_events_files(args)
    if not events_files:
        print('ERROR: no events files found. Check --events / --events-dir paths.')
        sys.exit(1)

    print(f'Start Time  = {dt.now().strftime("%H:%M:%S")}')
    print(f'Events files: {len(events_files)}')
    for f in events_files:
        print(f'  {f}')

    assign_labels(
        events_files = events_files,
        true_file    = args.true,
        gtf_file     = args.gtf,
        output_file  = args.output,
    )


if __name__ == '__main__':
    main()
