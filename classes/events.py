# coding: UTF-8
#!/usr/bin/env python
"""
events.py  -  optimised AS event detection and output
======================================================

Key changes vs. original
-------------------------
1. _define_MXE called ONCE after all chromosomes are processed
   The original called it inside the per-chromosome loop, causing it to
   re-scan all accumulated ES events on every chromosome iteration.

2. Integer comparisons replace range() membership testing in _define_MXE
   `int(c_pos[0]) in range(int(n_pos[0]), int(n_pos[1])+1)` is O(1) in
   Python 3 but still allocates a range object.  Replaced with plain
   arithmetic comparisons.

3. Coordinates already ints from gtf.py / junctions.py
   No repeated int() / str() conversions on the hot path.

4. Splice site features integrated via SpliceSiteAnnotator
   (unchanged from previous integration; annotator is initialised once per
   sample and closed when writing is complete).
"""

import sys
import re
import os
import os.path
import logging
from collections import defaultdict
import pysam
import pandas as pd
import csv

try:
    from .splice_site_features import (SpliceSiteAnnotator,
                                      SPLICE_FEATURE_COLUMNS,
                                      _DEFAULT_FEATURES)
    _SPLICE_FEATURES_AVAILABLE = True
except ImportError:
    _SPLICE_FEATURES_AVAILABLE = False
    SPLICE_FEATURE_COLUMNS = []
    _DEFAULT_FEATURES      = {}
    logging.warning('splice_site_features module not found - '
                    'splice site columns omitted from output.')

logging.basicConfig(
    level=logging.DEBUG,
    format='[%(levelname)s] %(asctime)s %(message)s',
    filename='log_read_bams.txt'
)


class EVENT():
    def __init__(self):
        self.events = defaultdict(lambda: defaultdict(dict))

        # Splice site feature storage (see events.py integration notes)
        self.junction_meta  = {}   # (chrom, junc_key) -> strand
        self.gene_key_chrom = {}   # gene_key -> chrom
        self.splice_features = {}  # (chrom, junc_key) -> feature dict

    # ==================================================================
    # Splice site feature helpers
    # ==================================================================

    def _store_junction_meta(self, chrom, junc_key, strand, gene_key):
        meta_key = (chrom, junc_key)
        if meta_key not in self.junction_meta:
            self.junction_meta[meta_key] = strand
        if gene_key not in self.gene_key_chrom:
            self.gene_key_chrom[gene_key] = chrom

    def _compute_splice_feature(self, chrom, junc_key, annotator):
        if annotator is None:
            return
        meta_key = (chrom, junc_key)
        if meta_key in self.splice_features:
            return
        strand = self.junction_meta.get(meta_key, '+')
        try:
            start, end = map(int, junc_key.split('_'))
            self.splice_features[meta_key] = annotator.get_features(
                chrom, start, end, strand)
        except Exception as exc:
            logging.warning('Splice feature failed %s %s: %s',
                            chrom, junc_key, exc)
            self.splice_features[meta_key] = dict(_DEFAULT_FEATURES)

    def _get_splice_row(self, gene_key, junc_key) -> list:
        if not _SPLICE_FEATURES_AVAILABLE or not SPLICE_FEATURE_COLUMNS:
            return []
        lookup_key = junc_key.split('N')[0] if 'N' in junc_key else junc_key
        chrom      = self.gene_key_chrom.get(gene_key)
        if chrom is None:
            return [str(_DEFAULT_FEATURES.get(c, ''))
                    for c in SPLICE_FEATURE_COLUMNS]
        feats = self.splice_features.get((chrom, lookup_key), _DEFAULT_FEATURES)
        return [str(feats.get(col, '')) for col in SPLICE_FEATURE_COLUMNS]

    # ==================================================================
    # MXE detection
    # ==================================================================

    def _define_MXE(self, gtf, junction):
        """
        Identify mutually exclusive exon pairs from the ES event set.
        Called ONCE after all chromosomes have been processed (not per-chrom).
        """
        for gene_key in self.events['ES']:
            gene, tx = gene_key.split('::')
            if len(gtf.exons[gene][tx].keys()) < 4:
                continue

            es_keys = list(self.events['ES'][gene_key])
            for i, c_key in enumerate(es_keys):
                if i == len(es_keys) - 1:
                    break
                n_key  = es_keys[i + 1]
                c_pos  = c_key.split('_')
                n_pos  = n_key.split('_')
                c0, c1 = int(c_pos[0]), int(c_pos[1])
                n0, n1 = int(n_pos[0]), int(n_pos[1])

                # Overlapping check - plain integer comparison
                if (n0 <= c0 <= n1) or (n0 <= c1 <= n1):
                    cov1, exc_s1, exc_l1, ijc_m1, ijc_o1, psi1 = \
                        self.events['ES'][gene_key][c_key]
                    cov2, exc_s2, exc_l2, ijc_m2, ijc_o2, psi2 = \
                        self.events['ES'][gene_key][n_key]
                    psi     = (float(psi1) + float(psi2)) / 2
                    mxe_key = c_key + 'N' + n_key
                    self.events['MXE'][gene_key][mxe_key] = [
                        str(int(cov1) + int(cov2)),
                        str(int(exc_s1) + int(exc_s2)),
                        str(int(exc_l1) + int(exc_l2)),
                        str(int(ijc_m1) + int(ijc_m2)),
                        str(int(ijc_o1) + int(ijc_o2)),
                        psi
                    ]

    # ==================================================================
    # Output
    # ==================================================================

    def _writeEvent(self, sample, out_dir='.', out_prefix='', group=''):
        """
        Write one TSV file per AS type into out_dir.

        Output filename pattern:
            {out_dir}/{out_prefix}_{group}_{as_type}_{sample}_events.txt
        (empty components are skipped)

        Columns: gene::tx | id | coverage | exclusion1 | exclusion2 |
                 inclusion | over_boundaries | PSI
                 [+ splice site feature columns when annotator is active]
        """
        os.makedirs(out_dir, exist_ok=True)

        base_header   = ['gene::tx', 'id', 'coverage', 'exclusion1',
                         'exclusion2', 'inclusion', 'over_boundaries', 'PSI']
        splice_header = SPLICE_FEATURE_COLUMNS if _SPLICE_FEATURES_AVAILABLE else []
        header        = base_header + splice_header

        for as_type in self.events:
            parts   = [p for p in [out_prefix, group, as_type, sample] if p]
            base_nm = '_'.join(parts) + '_events.txt'
            file_nm = os.path.join(out_dir, base_nm)
            with open(file_nm, 'w') as out:
                out.write('\t'.join(header) + '\n')
                for g_key in self.events[as_type]:
                    for e_id in self.events[as_type][g_key]:
                        line = [g_key, e_id]
                        line.extend(
                            map(str, self.events[as_type][g_key][e_id]))
                        line.extend(self._get_splice_row(g_key, e_id))
                        out.write('\t'.join(line) + '\n')

    # ==================================================================
    # Main detection
    # ==================================================================

    def form_splicings(self, gtf, bam, junction, sample,
                       fasta_path=None, gtf_file=None, chrom=None,
                       out_dir='.', out_prefix='', group=''):
        """
        Detect alternative splicing events and write per-sample output files.

        Parameters
        ----------
        gtf        : GTF      parsed GTF object
        bam        : Bam      parsed BAM object
        junction   : Junction junction index
        sample     : str      sample name (used in output filenames)
        fasta_path : str|None FASTA path for splice site features (optional)
        gtf_file   : str|None GTF path for annotation index (optional)
        chrom      : str|None restrict to this chromosome (optional)
        out_dir    : str      directory to write output files (default: '.')
        out_prefix : str      prefix for output filenames  (default: '')
        group      : str      sample group label ('group1' or 'group2')
        """
        # Initialise splice site annotator once per sample call
        annotator = None
        if fasta_path and gtf_file and _SPLICE_FEATURES_AVAILABLE:
            try:
                annotator = SpliceSiteAnnotator(fasta_path, gtf_file)
            except Exception as exc:
                logging.error('SpliceSiteAnnotator init failed: %s', exc)

        chroms_to_process = ([chrom] if chrom
                             else list(junction.splices.keys()))

        for n_chr in chroms_to_process:
            if n_chr not in junction.splices:
                continue

            for s_key, s_junc_dict in junction.splices[n_chr].items():
                s_info  = s_key.split('_')
                s_start = int(s_info[0])
                s_end   = int(s_info[1])

                for j_key, j_meta in s_junc_dict.items():
                    j_info  = j_key.split('_')
                    j_start = int(j_info[0])
                    j_end   = int(j_info[1])
                    as_type = j_meta[0]
                    s_gene  = j_meta[1]

                    # --- inclusion / exclusion counts ---------------------
                    bam_inc_m = bam.inc_m[n_chr][s_gene]
                    bam_inc_o = bam.inc_o[n_chr][s_gene]
                    bam_cov   = bam.cov[n_chr][s_gene]
                    bam_exc   = bam.exc[n_chr][s_gene]

                    ijc_m = bam_inc_m.get(s_key, 0)

                    ijc_o = 0
                    if j_start in bam_inc_o:
                        ijc_o = bam_inc_o[j_start]
                    elif j_end in bam_inc_o:
                        ijc_o = bam_inc_o[j_end]
                    if s_start in bam_inc_o and s_start != j_start:
                        ijc_o += bam_inc_o[s_start]
                    elif s_end in bam_inc_o and s_end != j_end:
                        ijc_o += bam_inc_o[j_end]

                    cov  = bam_cov.get(j_key, 0) + bam_cov.get(s_key, 0)
                    exc  = bam_exc.get(j_key, 0)

                    # --- ES events ----------------------------------------
                    if as_type == 'ES':
                        splice_anno_list = junction.splice_anno.get(
                            n_chr, {}).get(s_gene, {}).get(s_key, [])
                        for s_list in splice_anno_list:
                            jc_tx, jc_ex, jc_strand = (
                                str(s_list[0]), str(s_list[1]), str(s_list[2]))
                            ex_list = jc_ex.split('_')
                            exc_s   = ex_list[0]
                            exc_l   = ex_list[-1]
                            exc_n   = str(int(exc_s) + 1)

                            # Exon coordinates (already ints from gtf.py)
                            ex_s_end   = gtf.exons[s_gene][jc_tx][exc_s][2]
                            ex_p_start = gtf.exons[s_gene][jc_tx][exc_n][1]
                            j_key_s    = f'{ex_s_end}_{ex_p_start - 1}'

                            exc_p      = str(int(exc_l) - 1)
                            ex_l_start = gtf.exons[s_gene][jc_tx][exc_l][1]
                            ex_p_end   = gtf.exons[s_gene][jc_tx][exc_p][2]
                            j_key_l    = f'{ex_p_end}_{ex_l_start - 1}'

                            exc_s_cnt = bam_exc.get(j_key_s, 0)
                            exc_l_cnt = bam_exc.get(j_key_l, 0)
                            denom     = (exc_s_cnt + exc_l_cnt) + 2 * ijc_m
                            psi       = ((exc_s_cnt + exc_l_cnt) / denom
                                         if denom > 0 else 0)

                            gene_key = f'{s_gene}::{jc_tx}'
                            self.events[as_type][gene_key][s_key] = [
                                cov, exc_s_cnt, exc_l_cnt, ijc_m, ijc_o, psi]

                            self._store_junction_meta(
                                n_chr, s_key, jc_strand, gene_key)
                            self._compute_splice_feature(
                                n_chr, s_key, annotator)

                    # --- A3SS / A5SS / IR events --------------------------
                    else:
                        junc_anno_gene = junction.junction_anno.get(
                            n_chr, {}).get(s_gene, {})
                        if j_key not in junc_anno_gene:
                            continue

                        psi = ijc_m / (ijc_m + exc) if (ijc_m + exc) > 0 else 0

                        for j_list in junc_anno_gene[j_key]:
                            j_tx, j_ex, strand = (
                                str(j_list[0]), str(j_list[1]), str(j_list[2]))
                            gene_key = f'{s_gene}::{j_tx}'
                            self.events[as_type][gene_key][s_key] = [
                                cov, exc, '_', ijc_m, ijc_o, psi]

                            self._store_junction_meta(
                                n_chr, s_key, strand, gene_key)
                            self._compute_splice_feature(
                                n_chr, s_key, annotator)

                            # RI
                            if cov > 0:
                                denom_r = cov + 2 * exc
                                psi_r   = cov / denom_r if denom_r > 0 else 0
                                self.events['IR'][gene_key][j_key] = [
                                    cov, exc, '_', ijc_m, ijc_o, psi_r]
                                self._store_junction_meta(
                                    n_chr, j_key, strand, gene_key)
                                self._compute_splice_feature(
                                    n_chr, j_key, annotator)

        # MXE called ONCE after ALL chromosomes are processed
        self._define_MXE(gtf, junction)

        # Write output files for this sample
        self._writeEvent(sample, out_dir=out_dir, out_prefix=out_prefix, group=group)

        if annotator is not None:
            annotator.close()
