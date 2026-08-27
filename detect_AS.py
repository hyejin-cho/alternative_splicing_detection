# coding: UTF-8
#!/usr/bin/env python

import sys
import re
import os
import gzip
import logging
import argparse
import configparser
import multiprocessing
import itertools
from collections import defaultdict
import collections, functools, operator
from classes.file import File
from classes.gtf import GTF
from classes.bam import Bam
from classes.junctions import Junction
from classes.events import EVENT
import pandas as pd
import statistics
import subprocess as sp
from multiprocessing import Pool, freeze_support
from functools import partial
from numpy import *
from stats.calcDE import *
from datetime import datetime as dt


class HelpFormatter(argparse.RawDescriptionHelpFormatter,
                    argparse.ArgumentDefaultsHelpFormatter):
    pass


class Conf():
    def __init__(self):
        self.settings = dict()

    def getConfig(self, cfg_file):
        config = configparser.ConfigParser()
        config.read(cfg_file)
        self.settings['base_dir']      = config['default']['src_dir']
        self.settings['stat_file']     = config['stats']['r_file']
        self.settings['tmp_site_file'] = config['output']['tmp_site']
        self.settings['novel_gtf']     = config['output']['novel_gtf']
        self.settings['splice_file']   = config['output']['de_splice_file']


# ---------------------------------------------------------------------------
# Chromosome helpers
# ---------------------------------------------------------------------------

def get_chromosomes_from_gtf(gtf):
    """Extract chromosome list from the parsed GTF object."""
    for attr in ('chroms', 'chromosomes', 'seqnames', 'chrom_list'):
        if hasattr(gtf, attr):
            chroms = list(getattr(gtf, attr))
            if chroms:
                logging.debug('Chromosomes from GTF attr "%s": %s', attr, chroms)
                return chroms

    if hasattr(gtf, 'genes'):
        chroms = list(gtf.genes.keys())
        if chroms:
            logging.debug('Chromosomes from gtf.genes keys: %s', chroms)
            return chroms

    raise AttributeError(
        'Could not determine chromosome list from the GTF object. '
        'Please add a "chroms" attribute to your GTF class, or pass '
        '--chroms on the command line.'
    )


def get_chromosomes_from_bam(bam_file):
    """Fallback: read chromosome names from BAM header via pysam."""
    try:
        import pysam
        with pysam.AlignmentFile(bam_file, 'rb') as bam:
            chroms = list(bam.references)
        logging.debug('Chromosomes from BAM header (%s): %s', bam_file, chroms)
        return chroms
    except ImportError:
        raise ImportError(
            'pysam is not installed and chromosome list could not be inferred '
            'from the GTF object. Install pysam or add a "chroms" attribute to '
            'your GTF class.'
        )


# ---------------------------------------------------------------------------
# Worker functions
# ---------------------------------------------------------------------------

def _detect(file, gtf, junction, chrom, fasta_path, gtf_file,
            out_dir, out_prefix, group):
    """
    Detect alternative splicing events for a single (BAM file, chromosome) pair.

    Parameters
    ----------
    file       : str      path to BAM file
    gtf        : GTF      parsed GTF object (shared, read-only in workers)
    junction   : Junction junction object (shared, read-only in workers)
    chrom      : str      chromosome to restrict processing to
    fasta_path : str|None path to indexed FASTA for splice site features
    gtf_file   : str|None path to GTF for splice site annotation index
    out_dir    : str      directory to write output files
    out_prefix : str      prefix for output filenames
    group      : str      sample group label ('group1' or 'group2')
    """
    tmp    = file.split('/')
    nm     = tmp[-1]
    sample = nm[:nm.find('.bam')]

    logging.debug('[%s | %s] Start reading BAM ...', sample, chrom)
    bam = Bam().parseBam(file, gtf, junction, chrom=chrom)

    logging.debug('[%s | %s] bam.exc length: %d', sample, chrom, len(bam.exc))
    logging.debug('[%s | %s] bam.cov length: %d', sample, chrom, len(bam.cov))
    logging.debug('[%s | %s] Finish reading BAM: %s',
                  sample, chrom, dt.now().time())

    event = EVENT()
    event.form_splicings(
        gtf, bam, junction, sample,
        fasta_path = fasta_path,
        gtf_file   = gtf_file,
        chrom      = chrom,
        out_dir    = out_dir,
        out_prefix = out_prefix,
        group      = group,
    )

    logging.debug('[%s | %s] Finish AS detection: %s',
                  sample, chrom, dt.now().time())


def _detect_parallel(task):
    """
    Unpack a (bam_file, chrom, group) task tuple and call _detect.
    All shared objects come from pool globals set by init_detect().
    """
    bam_file, chrom, group = task
    _detect(
        bam_file, g_gtf, g_junction, chrom,
        g_fasta_path, g_gtf_file,
        g_out_dir, g_out_prefix, group,
    )


def init_detect(gtf, junction, fasta_path, gtf_file, out_dir, out_prefix):
    """Pool initializer: store shared objects as process-level globals."""
    global g_gtf, g_junction
    global g_fasta_path, g_gtf_file
    global g_out_dir, g_out_prefix
    g_gtf        = gtf
    g_junction   = junction
    g_fasta_path = fasta_path
    g_gtf_file   = gtf_file
    g_out_dir    = out_dir
    g_out_prefix = out_prefix


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    bam1_files = args.bam1.strip().split(',')
    bam2_files = args.bam2.strip().split(',')
    bam_files  = bam1_files + bam2_files

    gtf_file   = args.gtf
    fasta_path = args.fasta          # None when --fasta is not supplied
    cutoff     = args.cutoff
    wd         = args.dir
    out_prefix = args.output

    os.makedirs(wd, exist_ok=True)

    real_path = os.path.dirname(os.path.realpath(__file__))

    # --- Configuration -------------------------------------------------------
    conf = Conf()
    conf.getConfig(os.path.join(real_path, 'config.ini'))

    # --- Parse GTF -----------------------------------------------------------
    gtf = GTF(gtf_file)
    gtf.parseGTF()
    now = dt.now()
    logging.debug('Finish reading GTF: %s', now.strftime('%H:%M:%S'))
    print('Finish reading GTF:', now.strftime('%H:%M:%S'))

    # --- Determine chromosome list -------------------------------------------
    if args.chroms:
        chroms = [c.strip() for c in args.chroms.split(',')]
        print(f'Using user-specified chromosomes ({len(chroms)}): {chroms}')
    else:
        try:
            chroms = get_chromosomes_from_gtf(gtf)
        except AttributeError:
            print('Could not get chroms from GTF; falling back to BAM header ...')
            chroms = get_chromosomes_from_bam(bam_files[0])
        print(f'Detected {len(chroms)} chromosomes: {chroms}')
    logging.debug('Chromosome list: %s', chroms)

    # --- Build junction objects ----------------------------------------------
    junction = Junction()
    junction.findJunctionsFromGTF(gtf)
    junction.detectNovelSplices(gtf, args.splicing)
    now = dt.now()
    print('Finish reading junctions:', now.strftime('%H:%M:%S'))
    logging.debug('splice length: %d', len(junction.splices))

    # --- Splice site feature status ------------------------------------------
    if fasta_path:
        print(f'Splice site features ENABLED  (fasta={fasta_path})')
    else:
        print('Splice site features DISABLED  '
              '(pass --fasta /path/to/genome.fa to enable)')

    # --- Output location summary ---------------------------------------------
    print(f'Output directory : {wd}')
    print(f'Output prefix    : {out_prefix}')
    print(f'Output pattern   : {out_prefix}_{{group}}_{{TYPE}}_{{sample}}_events.txt')

    # --- Build task list: samples × chromosomes ------------------------------
    # Build (bam_file, chrom, group) tuples so each worker knows its group
    tasks_g1 = [(b, ch, 'group1') for b, ch in itertools.product(bam1_files, chroms)]
    tasks_g2 = [(b, ch, 'group2') for b, ch in itertools.product(bam2_files, chroms)]
    tasks    = tasks_g1 + tasks_g2
    n_tasks  = len(tasks)
    print(f'Total tasks: {len(bam_files)} samples x {len(chroms)} chroms '
          f'= {n_tasks}')
    for t in tasks:
        logging.debug('Task queued: BAM=%s  chrom=%s', t[0], t[1])

    # --- Determine pool size -------------------------------------------------
    cpu_cnt   = multiprocessing.cpu_count()
    n_workers = min(args.processes if args.processes else cpu_cnt, n_tasks)
    print(f'Pool size: {n_workers} workers (CPUs available: {cpu_cnt})')

    # --- Run parallel detection ----------------------------------------------
    with Pool(
        processes   = n_workers,
        initializer = init_detect,
        initargs    = [gtf, junction,
                       fasta_path, gtf_file,
                       wd, out_prefix],
    ) as pool:
        pool.map(_detect_parallel, tasks)

    now = dt.now()
    print('End Time:', now.strftime('%H:%M:%S'))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        formatter_class=HelpFormatter,
        description='''
            name:
            detect_AS.py -- detect Alternative Splicing with parallelisation
                            across samples AND chromosomes

            demo (without splice site features):
            detect_AS.py -b1 s1.bam,s2.bam -b2 s3.bam,s4.bam \\
                         -g annotation.gtf -s novel_splices.txt  \\
                         -d output_dir -o results

            demo (with splice site features):
            detect_AS.py -b1 s1.bam,s2.bam -b2 s3.bam,s4.bam \\
                         -g annotation.gtf -s novel_splices.txt  \\
                         -d output_dir -o results                 \\
                         --fasta /path/to/genome.fa
        '''
    )
    parser.add_argument('-b1', '--bam1', metavar='FILE1,FILE2,...',
        required=True,
        help='BAM files for sample group 1 (comma-separated)')
    parser.add_argument('-b2', '--bam2', metavar='FILE1,FILE2,...',
        required=True,
        help='BAM files for sample group 2 (comma-separated)')
    parser.add_argument('-g', '--gtf', metavar='FILE', required=True,
        help='GTF annotation file')
    parser.add_argument('-s', '--splicing', metavar='FILE', required=True,
        help='Combined novel splices file '
             '(output of combine_novel_splices_SJ.pl)')
    parser.add_argument('-c', '--cutoff', metavar='INT', type=int, default=2,
        help='Cutoff value for novel splice sites')
    parser.add_argument('-d', '--dir', metavar='DIR', type=str, required=True,
        help='Output directory (created automatically if it does not exist)')
    parser.add_argument('-o', '--output', metavar='PREFIX', type=str,
        required=True,
        help='Output filename prefix  '
             '(e.g. "results" produces results_ES_sample01_events.txt)')
    parser.add_argument('--fasta', metavar='FILE', type=str, default=None,
        help='Indexed reference genome FASTA '
             '(requires .fai index: samtools faidx genome.fa). '
             'Enables splice site feature columns in every output file. '
             'Omit to run without splice site features.')
    parser.add_argument('--chroms', metavar='chr1,chr2,...', type=str,
        default=None,
        help='Comma-separated chromosomes to process. '
             'Inferred from GTF (or BAM header) if omitted.')
    parser.add_argument('-p', '--processes', metavar='INT', type=int,
        default=None,
        help='Number of parallel worker processes. '
             'Defaults to the number of available CPU cores.')

    args = parser.parse_args()

    logging.basicConfig(
        level    = logging.DEBUG,
        filemode = 'w',
        format   = '[%(levelname)s] %(asctime)s %(message)s',
        filename = 'log_detect_as.txt'
    )

    now = dt.now()
    print('Start Time =', now.strftime('%H:%M:%S'))

    freeze_support()
    main(args)
