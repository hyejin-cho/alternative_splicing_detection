# coding: UTF-8
#!/usr/bin/env python
"""
extract_true_junctions.py
=========================
Extract all annotated intron junctions from a GTF file used for simulation.
Every junction produced here is a TRUE SPLICING EVENT (label = 1) and forms
the positive class for ML-based artifact filtering.

Usage
-----
    python extract_true_junctions.py \\
        --gtf  Homo_sapiens.GRCh38.110.gtf \\
        --output true_junctions.tsv

Output columns
--------------
    chrom | intron_start | intron_end | strand | gene_id | transcript_id | label

Coordinate convention
---------------------
    intron_start : 0-based first base of the intron  (= GTF exon end)
    intron_end   : 0-based first base of the downstream exon (= GTF exon start - 1)
    This matches the BAM / pysam half-open coordinate system used by
    detect_AS.py and splice_site_features.py.

Dependencies
------------
    pip install pandas
"""

import os
import sys
import argparse
import logging
from collections import defaultdict
from datetime import datetime as dt
import pandas as pd


# ===========================================================================
# Core extraction function
# ===========================================================================

def extract_true_junctions_from_gtf(gtf_file: str) -> pd.DataFrame:
    """
    Parse a GTF file and return all unique intron junctions as a DataFrame.

    Each row represents one intron between two consecutive exons of the same
    transcript.  Duplicate junctions shared across transcripts are collapsed
    to one row (keeping the first occurrence of gene_id / transcript_id).

    Parameters
    ----------
    gtf_file : str  path to GTF annotation file (plain text or .gz)

    Returns
    -------
    pd.DataFrame with columns:
        chrom, intron_start, intron_end, strand,
        gene_id, transcript_id, label (always 1)
    """
    logging.info('Parsing GTF: %s', gtf_file)

    # exons[transcript_id] = list of dicts
    exons = defaultdict(list)

    opener = __import__('gzip').open if gtf_file.endswith('.gz') else open
    mode   = 'rt' if gtf_file.endswith('.gz') else 'r'

    n_exon_lines = 0
    with opener(gtf_file, mode) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9:
                continue
            if fields[2] != 'exon':
                continue

            chrom  = fields[0]
            start  = int(fields[3])   # 1-based inclusive (GTF)
            end    = int(fields[4])   # 1-based inclusive (GTF)
            strand = fields[6]

            # Parse attribute string into a dict
            attr_str = fields[8]
            attrs = {}
            for kv in attr_str.strip(';').split(';'):
                kv = kv.strip()
                if not kv:
                    continue
                parts = kv.split(' ', 1)
                if len(parts) == 2:
                    attrs[parts[0]] = parts[1].replace('"', '').strip()

            gene_id = attrs.get('gene_id', '')
            tx_id   = attrs.get('transcript_id', '')
            if not gene_id or not tx_id:
                continue

            exons[tx_id].append({
                'chrom':         chrom,
                'start':         start,
                'end':           end,
                'strand':        strand,
                'gene_id':       gene_id,
                'transcript_id': tx_id,
            })
            n_exon_lines += 1

    logging.info('Exon records parsed: %d across %d transcripts',
                 n_exon_lines, len(exons))

    # --- Build intron junctions from consecutive exon pairs --------------
    rows = []
    for tx_id, ex_list in exons.items():

        # Sort by genomic start position
        ex_list = sorted(ex_list, key=lambda x: x['start'])

        for i in range(len(ex_list) - 1):
            ex1 = ex_list[i]
            ex2 = ex_list[i + 1]

            # Sanity check: both exons must be on the same chrom and strand
            if ex1['chrom'] != ex2['chrom'] or ex1['strand'] != ex2['strand']:
                continue

            # Coordinate conversion — GTF is 1-based inclusive,
            # output is 0-based half-open (BAM convention):
            #
            #   GTF:  exon1_end (1-based)  →  intron_start (0-based) = exon1_end
            #   GTF:  exon2_start (1-based) →  intron_end   (0-based) = exon2_start - 1
            #
            intron_start = ex1['end']           # 0-based first base of intron
            intron_end   = ex2['start'] - 1     # 0-based first base of next exon

            if intron_end <= intron_start:      # malformed annotation — skip
                continue

            rows.append({
                'chrom':         ex1['chrom'],
                'intron_start':  intron_start,
                'intron_end':    intron_end,
                'strand':        ex1['strand'],
                'gene_id':       ex1['gene_id'],
                'transcript_id': tx_id,
                'label':         1,             # TRUE splicing event
            })

    df = pd.DataFrame(rows)

    if df.empty:
        logging.warning('No junctions extracted — check GTF format.')
        return df

    # Deduplicate: same intron coordinates + strand may appear in multiple
    # transcripts.  Keep first occurrence (retains one gene_id/tx_id).
    before = len(df)
    df = df.drop_duplicates(
        subset=['chrom', 'intron_start', 'intron_end', 'strand']
    ).reset_index(drop=True)
    after = len(df)

    logging.info('Junctions before dedup: %d', before)
    logging.info('Junctions after  dedup: %d  (%d duplicates removed)',
                 after, before - after)

    return df


# ===========================================================================
# Summary helper
# ===========================================================================

def print_summary(df: pd.DataFrame):
    """Print a concise summary of the extracted junction table."""
    if df.empty:
        print('No junctions extracted.')
        return

    print('\n=== Junction Summary ===')
    print(f'  Total unique junctions : {len(df):,}')
    print(f'  Chromosomes            : {df["chrom"].nunique()}')
    print(f'  Genes                  : {df["gene_id"].nunique():,}')
    print(f'  Transcripts            : {df["transcript_id"].nunique():,}')
    print(f'  Strand + / -           : '
          f'{(df["strand"]=="+").sum():,} / {(df["strand"]=="-").sum():,}')

    # Intron length stats
    df['intron_length'] = df['intron_end'] - df['intron_start']
    print(f'  Intron length (min)    : {df["intron_length"].min():,} bp')
    print(f'  Intron length (median) : {int(df["intron_length"].median()):,} bp')
    print(f'  Intron length (max)    : {df["intron_length"].max():,} bp')
    print(f'  Label = 1 (all rows)   : {(df["label"]==1).sum():,}')
    print('========================\n')


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Extract true splicing junctions from a GTF file.\n'
            'Output is a TSV with label=1 for every annotated intron junction.\n'
            'Use as the positive class (true events) for ML artifact filtering.'
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('-g', '--gtf', metavar='FILE', required=True,
        help='GTF annotation file (plain text or .gz)')
    parser.add_argument('-o', '--output', metavar='FILE',
        default='true_junctions.tsv',
        help='Output TSV file path')
    parser.add_argument('--log', metavar='FILE',
        default='extract_true_junctions.log',
        help='Log file path')
    parser.add_argument('--no-header', action='store_true',
        help='Write output without a header row')
    return parser.parse_args()


def main():
    args = parse_args()

    logging.basicConfig(
        level    = logging.DEBUG,
        filemode = 'w',
        format   = '[%(levelname)s] %(asctime)s %(message)s',
        filename = args.log
    )

    # Also log to stdout at INFO level
    console = logging.StreamHandler(sys.stdout)
    console.setLevel(logging.INFO)
    console.setFormatter(logging.Formatter('[%(levelname)s] %(message)s'))
    logging.getLogger().addHandler(console)

    if not os.path.isfile(args.gtf):
        logging.error('GTF file not found: %s', args.gtf)
        sys.exit(1)

    start = dt.now()
    print(f'Start Time = {start.strftime("%H:%M:%S")}')
    print(f'GTF        = {args.gtf}')
    print(f'Output     = {args.output}')

    # --- Extract ---------------------------------------------------------
    df = extract_true_junctions_from_gtf(args.gtf)

    if df.empty:
        logging.error('No junctions extracted. Exiting.')
        sys.exit(1)

    # --- Print summary ---------------------------------------------------
    print_summary(df)

    # --- Save output -----------------------------------------------------
    # Drop the temporary intron_length column if it was added by print_summary
    out_cols = ['chrom', 'intron_start', 'intron_end',
                'strand', 'gene_id', 'transcript_id', 'label']
    df[out_cols].to_csv(
        args.output,
        sep       = '\t',
        index     = False,
        header    = not args.no_header
    )

    elapsed = (dt.now() - start).seconds
    print(f'Output written : {args.output}')
    print(f'Rows           : {len(df):,}')
    print(f'Elapsed        : {elapsed}s')
    logging.info('Done. Output: %s  (%d rows, %ds)', args.output, len(df), elapsed)


if __name__ == '__main__':
    main()
