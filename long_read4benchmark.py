#coding: UTF-8
#!/usr/bin/env python

# extract_longread_junctions.py
# Use long-read alignments to build a high-confidence junction set
# that serves as ground truth for evaluating short-read tools
import os

# Get the current working directory as a string
cwd = os.getcwd()

os.chdir("/home/hycho/dt_home/SI/benchmark")
cwd = os.getcwd()
print(cwd)


import pysam
import pandas as pd
from collections import defaultdict

def extract_junctions_from_longread_bam(bam_file: str,
                                         min_reads: int = 3) -> pd.DataFrame:
    """
    Extract splice junctions from a long-read BAM file.
    Long reads span entire introns so junction coordinates are
    directly observed, not inferred from paired-end fragments.

    Parameters
    ----------
    bam_file  : aligned long-read BAM (minimap2 or STARlong)
    min_reads : minimum reads supporting a junction (default 3)
                use higher value (5-10) for noisier ONT data
                use lower value (2-3) for cleaner PacBio data
    """
    bam    = pysam.AlignmentFile(bam_file, 'rb')
    junc_counts  = defaultdict(int)
    junc_strands = {}

    for read in bam.fetch():
        if read.is_unmapped or read.is_secondary:
            continue
        if read.cigartuples is None:
            continue

        chrom = read.reference_name
        pos   = read.reference_start

        # Walk CIGAR to find N operations (introns)
        for op, length in read.cigartuples:
            if op == 0:   # M — match
                pos += length
            elif op == 3: # N — intron skip
                intron_start = pos
                intron_end   = pos + length
                key = (chrom, intron_start, intron_end)
                junc_counts[key]  += 1
                junc_strands[key]  = (
                    '-' if read.is_reverse else '+')
                pos += length
            elif op in (2, 7, 8):  # D, =, X
                pos += length

    bam.close()

    rows = []
    for (chrom, start, end), count in junc_counts.items():
        if count >= min_reads:
            rows.append({
                'chrom':        chrom,
                'intron_start': start,
                'intron_end':   end,
                'strand':       junc_strands[(chrom, start, end)],
                'read_count':   count,
                'label':        1,    # ground truth = true junction
                'source':       'long_read',
            })

    df = pd.DataFrame(rows)
    print(f'Long-read junctions: {len(df):,}  '
          f'(min_reads={min_reads})')
    return df


def build_ground_truth(longread_bam_rep1: str,
                        longread_bam_rep2: str,
                        min_reads_per_rep: int = 2,
                        output_file: str = 'longread_truth.tsv') -> pd.DataFrame:
    """
    Build a high-confidence ground truth by requiring a junction to be
    supported in BOTH long-read replicates.
    Reproducibility across replicates is the strongest filter
    for separating real junctions from long-read-specific artifacts.
    """
    rep1 = extract_junctions_from_longread_bam(
        longread_bam_rep1, min_reads=min_reads_per_rep)
    rep2 = extract_junctions_from_longread_bam(
        longread_bam_rep2, min_reads=min_reads_per_rep)

    keys_rep1 = set(zip(rep1['chrom'],
                        rep1['intron_start'],
                        rep1['intron_end']))
    keys_rep2 = set(zip(rep2['chrom'],
                        rep2['intron_start'],
                        rep2['intron_end']))

    # Keep only junctions seen in both replicates
    shared_keys = keys_rep1 & keys_rep2
    truth = rep1[rep1.apply(
        lambda r: (r['chrom'], r['intron_start'], r['intron_end'])
                  in shared_keys, axis=1
    )].copy()
    truth['label'] = 1

    print(f'Rep1 junctions : {len(rep1):,}')
    print(f'Rep2 junctions : {len(rep2):,}')
    print(f'Shared (truth) : {len(truth):,}')

    truth.to_csv(output_file, sep='\t', index=False)
    return truth
