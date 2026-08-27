# coding: UTF-8
#!/usr/bin/env python
"""
generate_negatives.py
=====================
Generate negative-label (label=0) junctions for ML artifact filter training.

Negative junctions come from four sources, each capturing a different
artifact type that the Random Forest needs to learn to recognize:

  Source 1 — Pipeline negatives (from assign_labels.py output)
             Junctions detected on simulated BAMs that are NOT in the
             simulation GTF.  These are alignment artifacts produced under
             realistic sequencing conditions.  This is the most important
             source and should always be included.

  Source 2 — Repetitive region artifacts
             Reads sampled from RepeatMasker-annotated repeats (SINEs, LINEs,
             pseudogenes) aligned to the genome.  Multi-mapping from repeat
             regions is the single largest source of false junctions in
             real RNA-seq data.

  Source 3 — Low-quality junction filter
             Junctions from real (non-simulated) RNA-seq BAMs that fail
             multiple quality thresholds simultaneously.  These are junctions
             that the aligner reported but that have hallmarks of artifacts:
             low unique read support, non-canonical splice sites, high
             soft-clip rate at boundaries.

  Source 4 — Random / chimeric junctions
             Coordinates sampled from annotated exon boundaries but paired
             across different genes or chromosomes.  These are biologically
             impossible junctions that anchor the negative class boundary.

Usage
-----
    # Source 1 only (minimum requirement — always run this first)
    python generate_negatives.py \\
        --labeled     labeled_junctions.tsv \\
        --output      negatives.tsv

    # Sources 1 + 2 + 3 + 4 (recommended for a balanced training set)
    python generate_negatives.py \\
        --labeled     labeled_junctions.tsv \\
        --gtf         annotation.gtf \\
        --bam         real_sample.bam \\
        --fasta       genome.fa \\
        --repeat-bed  repeatmasker.bed \\
        --output      negatives.tsv \\
        --max-per-source 2000

Output
------
  negatives.tsv — tab-separated with columns:
    chrom | intron_start | intron_end | strand |
    source | reason | label (always 0)

Dependencies
------------
    pip install pandas numpy pysam
"""

import os
import sys
import argparse
import logging
import random
import gzip
from collections import defaultdict
from datetime import datetime as dt
import pandas as pd
import numpy as np

try:
    import pysam
    _PYSAM_AVAILABLE = True
except ImportError:
    _PYSAM_AVAILABLE = False
    logging.warning('pysam not installed — Sources 2 and 3 will be skipped.')

COMPLEMENT = str.maketrans('ACGT', 'TGCA')


# ===========================================================================
# Shared helpers
# ===========================================================================

def reverse_complement(seq: str) -> str:
    return seq[::-1].translate(COMPLEMENT)


def get_dinucleotides(genome, chrom, intron_start, intron_end):
    """Return (donor, acceptor) dinucleotides. Returns ('NN','NN') on error."""
    try:
        donor    = genome.fetch(chrom, intron_start,   intron_start + 2).upper()
        acceptor = genome.fetch(chrom, intron_end - 2, intron_end      ).upper()
        return donor, acceptor
    except Exception:
        return 'NN', 'NN'


def is_canonical(donor, acceptor):
    return (donor, acceptor) in {('GT', 'AG'), ('GC', 'AG'), ('AT', 'AC')}


# ===========================================================================
# GTF helpers — exon boundary index
# ===========================================================================

def build_exon_boundary_index(gtf_file: str) -> dict:
    """
    Parse GTF and return a dict:
        exon_boundaries[chrom] = list of (exon_end, strand, gene_id)

    Used by Source 4 to sample real exon boundaries and pair them
    across different genes to create chimeric / impossible junctions.
    """
    logging.info('Building exon boundary index from: %s', gtf_file)
    boundaries = defaultdict(list)

    opener = gzip.open if gtf_file.endswith('.gz') else open
    mode   = 'rt' if gtf_file.endswith('.gz') else 'r'

    with opener(gtf_file, mode) as fh:
        for line in fh:
            if line.startswith('#'):
                continue
            fields = line.strip().split('\t')
            if len(fields) < 9 or fields[2] != 'exon':
                continue
            chrom  = fields[0]
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
                boundaries[chrom].append((end, strand, gene_id))

    logging.info('Exon boundaries loaded: %d chromosomes',
                 len(boundaries))
    return dict(boundaries)


# ===========================================================================
# Source 1 — Pipeline negatives from assign_labels.py output
# ===========================================================================

def source1_pipeline_negatives(labeled_file: str,
                                max_records: int = None) -> pd.DataFrame:
    """
    Extract junctions already labeled 0 by assign_labels.py.

    These are real alignment outputs on simulated BAMs where the junction
    was not in the simulation GTF — the ground-truth definition of an
    alignment artifact.

    Parameters
    ----------
    labeled_file : str  path to labeled_junctions.tsv from assign_labels.py
    max_records  : int  optional cap on number of records to return

    Returns
    -------
    DataFrame with columns: chrom, intron_start, intron_end, strand,
                             source, reason, label
    """
    logging.info('Source 1: loading pipeline negatives from %s', labeled_file)

    df = pd.read_csv(labeled_file, sep='\t')

    required = {'label', 'id', 'chrom_resolved'}
    missing  = required - set(df.columns)
    if missing:
        raise ValueError(
            f'labeled_junctions.tsv is missing columns: {missing}\n'
            'Run assign_labels.py first to generate this file.'
        )

    # Keep only label=0 rows with a resolved chromosome
    negatives = df[(df['label'] == 0) &
                   (df['chrom_resolved'] != '') &
                   (df['chrom_resolved'].notna())].copy()

    if negatives.empty:
        logging.warning('Source 1: no label=0 rows found in %s', labeled_file)
        return pd.DataFrame()

    # Parse intron_start and intron_end from the 'id' column (start_end format)
    # For MXE (compound ids like start1_end1Nstart2_end2) take the first component
    def parse_coords(id_val):
        first = str(id_val).split('N')[0]
        parts = first.split('_')
        if len(parts) >= 2:
            try:
                return int(parts[0]), int(parts[1])
            except ValueError:
                pass
        return None, None

    negatives[['intron_start', 'intron_end']] = negatives['id'].apply(
        lambda x: pd.Series(parse_coords(x))
    )
    negatives = negatives.dropna(subset=['intron_start', 'intron_end'])
    negatives['intron_start'] = negatives['intron_start'].astype(int)
    negatives['intron_end']   = negatives['intron_end'].astype(int)

    result = pd.DataFrame({
        'chrom':        negatives['chrom_resolved'].values,
        'intron_start': negatives['intron_start'].values,
        'intron_end':   negatives['intron_end'].values,
        'strand':       negatives.get('strand_resolved', '+').values,
        'source':       'pipeline',
        'reason':       'detected_not_in_gtf',
        'label':        0,
    })

    if max_records and len(result) > max_records:
        result = result.sample(n=max_records, random_state=42)

    logging.info('Source 1: %d pipeline negatives', len(result))
    return result.reset_index(drop=True)


# ===========================================================================
# Source 2 — Repetitive region artifacts
# ===========================================================================

def source2_repeat_artifacts(repeat_bed: str,
                              bam_file: str,
                              fasta_file: str,
                              max_records: int = 2000,
                              seed: int = 42) -> pd.DataFrame:
    """
    Find junctions from reads that originated in repetitive regions.

    Strategy: scan the BAM for junction reads whose alignment start or end
    falls inside a RepeatMasker-annotated region.  These reads are likely
    multi-mappers creating spurious junctions.

    Parameters
    ----------
    repeat_bed  : str  RepeatMasker BED (download from UCSC Table Browser
                       Genome Browser → Tools → Table Browser →
                       Variation & Repeats → RepeatMasker)
    bam_file    : str  path to indexed BAM file (real or simulated)
    fasta_file  : str  indexed reference FASTA
    max_records : int  maximum number of artifact junctions to return
    seed        : int  random seed for reproducible sampling

    Returns
    -------
    DataFrame with columns: chrom, intron_start, intron_end, strand,
                             source, reason, label
    """
    if not _PYSAM_AVAILABLE:
        logging.warning('Source 2 skipped: pysam not available')
        return pd.DataFrame()

    logging.info('Source 2: scanning BAM for repeat-region artifacts')
    logging.info('  repeat_bed : %s', repeat_bed)
    logging.info('  bam_file   : %s', bam_file)

    # --- Load repeat regions into a chrom->list of (start,end) dict ----------
    repeats = defaultdict(list)
    with open(repeat_bed) as fh:
        for line in fh:
            if line.startswith('#') or line.startswith('track'):
                continue
            parts = line.strip().split('\t')
            if len(parts) < 3:
                continue
            try:
                chrom = parts[0]
                start = int(parts[1])
                end   = int(parts[2])
                repeats[chrom].append((start, end))
            except ValueError:
                continue

    # Build fast lookup: sort intervals for each chrom
    repeat_sorted = {
        chrom: sorted(ivs) for chrom, ivs in repeats.items()
    }

    def in_repeat(chrom, pos):
        """Quick check: is pos inside any repeat interval on chrom?"""
        ivs = repeat_sorted.get(chrom, [])
        for start, end in ivs:
            if start > pos:
                break
            if start <= pos <= end:
                return True
        return False

    # --- Scan BAM for junction reads near repeat boundaries ------------------
    genome  = pysam.FastaFile(fasta_file)
    bam_fh  = pysam.AlignmentFile(bam_file, 'rb')

    artifacts = []
    random.seed(seed)

    for read in bam_fh.fetch():
        if read.is_unmapped or read.is_duplicate:
            continue
        if read.cigartuples is None:
            continue

        # Only process junction reads (contain N operation in CIGAR)
        ops = [op for op, _ in read.cigartuples]
        if 3 not in ops:   # 3 = BAM_CREF_SKIP = intron
            continue

        chrom = read.reference_name
        pos   = read.reference_start

        # Reconstruct junction coordinates from CIGAR
        m_segs = [(op, ln) for op, ln in read.cigartuples if op == 0]
        n_segs = [(op, ln) for op, ln in read.cigartuples if op == 3]

        if len(m_segs) < 2 or len(n_segs) < 1:
            continue

        l_edge = pos + m_segs[0][1]
        n_len  = n_segs[0][1]
        r_edge = l_edge + n_len

        # Flag as artifact if read start or junction boundary is in a repeat
        if not (in_repeat(chrom, pos) or
                in_repeat(chrom, l_edge) or
                in_repeat(chrom, r_edge)):
            continue

        donor, acceptor = get_dinucleotides(genome, chrom, l_edge, r_edge)

        artifacts.append({
            'chrom':        chrom,
            'intron_start': l_edge,
            'intron_end':   r_edge,
            'strand':       '-' if read.is_reverse else '+',
            'source':       'repeat_region',
            'reason':       f'repeat_overlap|{donor}-{acceptor}',
            'label':        0,
        })

        if len(artifacts) >= max_records * 3:   # collect extra, sample below
            break

    bam_fh.close()
    genome.close()

    if not artifacts:
        logging.warning('Source 2: no repeat artifacts found')
        return pd.DataFrame()

    result = pd.DataFrame(artifacts).drop_duplicates(
        subset=['chrom', 'intron_start', 'intron_end'])

    if len(result) > max_records:
        result = result.sample(n=max_records, random_state=seed)

    logging.info('Source 2: %d repeat-region artifacts', len(result))
    return result.reset_index(drop=True)


# ===========================================================================
# Source 3 — Low-quality junctions from real BAM
# ===========================================================================

def source3_low_quality_junctions(bam_file: str,
                                   fasta_file: str,
                                   true_set: set,
                                   max_records: int = 2000,
                                   min_mapq: int = 10,
                                   max_unique_reads: int = 3,
                                   seed: int = 42) -> pd.DataFrame:
    """
    Identify junctions in a real RNA-seq BAM that fail multiple quality
    criteria simultaneously.  A junction is a candidate artifact if it:
      - Has fewer than max_unique_reads supporting reads  AND
      - Has non-canonical splice site dinucleotides       AND
      - Is not present in the true (GTF) junction set

    Using multiple criteria avoids mislabeling rare but real junctions as
    artifacts based on any single filter alone.

    Parameters
    ----------
    bam_file        : str  path to indexed BAM file (real RNA-seq sample)
    fasta_file      : str  indexed reference FASTA
    true_set        : set  (chrom, intron_start, intron_end) from true junctions
    max_records     : int  maximum artifact junctions to return
    min_mapq        : int  minimum mapping quality for reads to be counted
    max_unique_reads: int  junctions with <= this many reads are candidates
    seed            : int  random seed

    Returns
    -------
    DataFrame with columns: chrom, intron_start, intron_end, strand,
                             source, reason, label
    """
    if not _PYSAM_AVAILABLE:
        logging.warning('Source 3 skipped: pysam not available')
        return pd.DataFrame()

    logging.info('Source 3: scanning for low-quality junctions in %s', bam_file)

    genome = pysam.FastaFile(fasta_file)
    bam_fh = pysam.AlignmentFile(bam_file, 'rb')

    # Count unique-mapping reads per junction
    junc_counts  = defaultdict(int)
    junc_strands = {}

    for read in bam_fh.fetch():
        if read.is_unmapped or read.is_duplicate:
            continue
        if read.mapping_quality < min_mapq:
            continue
        if read.cigartuples is None:
            continue
        try:
            if read.get_tag('NH') != 1:
                continue
        except KeyError:
            continue

        ops = [op for op, _ in read.cigartuples]
        if 3 not in ops:
            continue

        chrom  = read.reference_name
        pos    = read.reference_start
        m_segs = [(op, ln) for op, ln in read.cigartuples if op == 0]
        n_segs = [(op, ln) for op, ln in read.cigartuples if op == 3]

        if len(m_segs) < 2 or len(n_segs) < 1:
            continue

        l_edge = pos + m_segs[0][1]
        r_edge = l_edge + n_segs[0][1]
        key    = (chrom, l_edge, r_edge)

        junc_counts[key]  += 1
        junc_strands[key]  = '-' if read.is_reverse else '+'

    bam_fh.close()

    # --- Apply quality filters -------------------------------------------
    artifacts = []
    for (chrom, intron_start, intron_end), count in junc_counts.items():

        # Must not be a known true junction
        if (chrom, intron_start, intron_end) in true_set:
            continue

        # Condition 1: low read support
        if count > max_unique_reads:
            continue

        # Condition 2: non-canonical splice site
        donor, acceptor = get_dinucleotides(
            genome, chrom, intron_start, intron_end)
        if is_canonical(donor, acceptor):
            continue

        artifacts.append({
            'chrom':        chrom,
            'intron_start': intron_start,
            'intron_end':   intron_end,
            'strand':       junc_strands.get((chrom, intron_start, intron_end), '+'),
            'source':       'low_quality',
            'reason':       f'reads<={max_unique_reads}|non_canonical_{donor}-{acceptor}',
            'label':        0,
        })

    genome.close()

    if not artifacts:
        logging.warning('Source 3: no low-quality artifacts found')
        return pd.DataFrame()

    result = pd.DataFrame(artifacts).drop_duplicates(
        subset=['chrom', 'intron_start', 'intron_end'])

    if len(result) > max_records:
        result = result.sample(n=max_records, random_state=seed)

    logging.info('Source 3: %d low-quality artifacts', len(result))
    return result.reset_index(drop=True)


# ===========================================================================
# Source 4 — Chimeric / random junctions
# ===========================================================================

def source4_chimeric_junctions(gtf_file: str,
                                true_set: set,
                                n_junctions: int = 1000,
                                seed: int = 42) -> pd.DataFrame:
    """
    Generate biologically impossible junctions by pairing exon boundaries
    from different genes on the same chromosome.

    These anchor the negative class with junctions that have realistic
    coordinates (real exon boundaries) but impossible biology (inter-gene
    chimeras).  They are especially useful for teaching the model that
    annotation distance features are informative.

    Parameters
    ----------
    gtf_file    : str  GTF annotation file
    true_set    : set  (chrom, intron_start, intron_end) to avoid
    n_junctions : int  target number of chimeric junctions to generate
    seed        : int  random seed

    Returns
    -------
    DataFrame with columns: chrom, intron_start, intron_end, strand,
                             source, reason, label
    """
    logging.info('Source 4: generating %d chimeric junctions', n_junctions)

    boundaries = build_exon_boundary_index(gtf_file)
    rng        = np.random.default_rng(seed)

    artifacts = []
    attempts  = 0
    max_attempts = n_junctions * 20

    for chrom, exon_list in boundaries.items():
        if len(exon_list) < 4:
            continue

        exon_arr = np.array([e[0] for e in exon_list])   # exon end positions
        genes    = [e[2] for e in exon_list]

        while len(artifacts) < n_junctions and attempts < max_attempts:
            attempts += 1

            # Pick two exon boundaries from DIFFERENT genes
            i, j = rng.choice(len(exon_list), size=2, replace=False)
            if genes[i] == genes[j]:
                continue

            start = int(exon_arr[i])
            end   = int(exon_arr[j])

            if start >= end:
                start, end = end, start

            if end - start < 50:     # too short to be a real intron
                continue
            if end - start > 1e6:    # unrealistically long
                continue
            if (chrom, start, end) in true_set:
                continue

            artifacts.append({
                'chrom':        chrom,
                'intron_start': start,
                'intron_end':   end,
                'strand':       exon_list[i][1],
                'source':       'chimeric',
                'reason':       f'inter_gene_{genes[i]}_x_{genes[j]}',
                'label':        0,
            })

        if len(artifacts) >= n_junctions:
            break

    if not artifacts:
        logging.warning('Source 4: no chimeric junctions generated')
        return pd.DataFrame()

    result = pd.DataFrame(artifacts[:n_junctions]).drop_duplicates(
        subset=['chrom', 'intron_start', 'intron_end'])

    logging.info('Source 4: %d chimeric junctions', len(result))
    return result.reset_index(drop=True)


# ===========================================================================
# Combine all sources
# ===========================================================================

def combine_negatives(frames: list,
                      true_set: set,
                      output_file: str) -> pd.DataFrame:
    """
    Merge all negative sources, remove any accidental overlap with the
    true junction set, deduplicate, and write to output_file.
    """
    frames = [f for f in frames if f is not None and not f.empty]
    if not frames:
        raise RuntimeError('No negative junctions generated from any source.')

    combined = pd.concat(frames, ignore_index=True)

    # Remove any row that accidentally matches a true junction
    before = len(combined)
    combined = combined[
        ~combined.apply(
            lambda r: (r['chrom'], r['intron_start'], r['intron_end'])
                      in true_set,
            axis=1
        )
    ]
    after = len(combined)
    if before != after:
        logging.info('Removed %d rows that overlapped true junction set',
                     before - after)

    # Deduplicate on coordinates — keep first occurrence (highest-priority source)
    combined = combined.drop_duplicates(
        subset=['chrom', 'intron_start', 'intron_end']
    ).reset_index(drop=True)

    combined['label'] = 0   # enforce — all negatives

    os.makedirs(os.path.dirname(output_file) or '.', exist_ok=True)
    combined.to_csv(output_file, sep='\t', index=False)

    # --- Summary -------------------------------------------------------------
    print('\n=== Negative Junction Summary ===')
    print(f'  Total negatives   : {len(combined):,}')
    for src, grp in combined.groupby('source'):
        print(f'  Source {src:<20}: {len(grp):,}')
    print(f'  Output            : {output_file}')
    print('=================================\n')

    return combined


# ===========================================================================
# CLI
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            'Generate negative-label (label=0) junctions for ML training.\n'
            'Four sources are supported; Source 1 is the minimum requirement.'
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    # Required
    parser.add_argument('--labeled', metavar='FILE', required=True,
        help='labeled_junctions.tsv from assign_labels.py  [Source 1]')
    parser.add_argument('-o', '--output', metavar='FILE',
        default='negatives.tsv',
        help='Output negatives TSV (default: negatives.tsv)')

    # Source 2 + 3 + 4
    parser.add_argument('--gtf', metavar='FILE', default=None,
        help='GTF annotation file  [Source 4]')
    parser.add_argument('--bam', metavar='FILE', default=None,
        help='Indexed BAM file  [Sources 2 and 3]')
    parser.add_argument('--fasta', metavar='FILE', default=None,
        help='Indexed reference FASTA  [Sources 2 and 3]')
    parser.add_argument('--repeat-bed', metavar='FILE', default=None,
        help='RepeatMasker BED file  [Source 2]')
    parser.add_argument('--true', metavar='FILE', default=None,
        help='true_junctions.tsv to avoid false negatives in Sources 3 and 4')

    parser.add_argument('--max-per-source', metavar='INT', type=int,
        default=2000,
        help='Maximum negatives per source  (default: 2000)')
    parser.add_argument('--seed', metavar='INT', type=int, default=42,
        help='Random seed  (default: 42)')
    parser.add_argument('--log', metavar='FILE',
        default='generate_negatives.log')
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

    print(f'Start Time = {dt.now().strftime("%H:%M:%S")}')

    # Load true set for Sources 3 and 4 contamination check
    true_set = set()
    if args.true:
        if not os.path.isfile(args.true):
            print(f'ERROR: --true file not found: {args.true}')
            sys.exit(1)
        true_df  = pd.read_csv(args.true, sep='\t')
        true_set = set(zip(true_df['chrom'],
                           true_df['intron_start'].astype(int),
                           true_df['intron_end'].astype(int)))
        print(f'True junction set loaded: {len(true_set):,} junctions')

    frames = []

    # --- Source 1 (always run) -----------------------------------------------
    if not os.path.isfile(args.labeled):
        print(f'ERROR: --labeled file not found: {args.labeled}')
        sys.exit(1)
    frames.append(
        source1_pipeline_negatives(args.labeled,
                                   max_records=args.max_per_source)
    )

    # --- Source 2 (repeat artifacts) -----------------------------------------
    if args.repeat_bed and args.bam and args.fasta:
        for f in [args.repeat_bed, args.bam, args.fasta]:
            if not os.path.isfile(f):
                logging.warning('File not found, skipping Source 2: %s', f)
                break
        else:
            frames.append(
                source2_repeat_artifacts(
                    args.repeat_bed, args.bam, args.fasta,
                    max_records=args.max_per_source, seed=args.seed)
            )
    else:
        print('Source 2 skipped (provide --repeat-bed, --bam, --fasta to enable)')

    # --- Source 3 (low-quality junctions) ------------------------------------
    if args.bam and args.fasta:
        frames.append(
            source3_low_quality_junctions(
                args.bam, args.fasta, true_set,
                max_records=args.max_per_source, seed=args.seed)
        )
    else:
        print('Source 3 skipped (provide --bam and --fasta to enable)')

    # --- Source 4 (chimeric junctions) ---------------------------------------
    if args.gtf:
        frames.append(
            source4_chimeric_junctions(
                args.gtf, true_set,
                n_junctions=args.max_per_source, seed=args.seed)
        )
    else:
        print('Source 4 skipped (provide --gtf to enable)')

    # --- Combine and write ---------------------------------------------------
    result = combine_negatives(frames, true_set, args.output)

    print(f'End Time = {dt.now().strftime("%H:%M:%S")}')
    return result


if __name__ == '__main__':
    main()
