#coding: UTF-8
#!/usr/bin/env python

import sys
import re
import os
import logging
from collections import defaultdict
import pysam
import pandas as pd
import csv

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(asctime)s %(message)s', filename='log_read_bams.txt')
  
class EVENT():
    def __init__(self):
        self.events = defaultdict(lambda: defaultdict(dict))
        
    def _define_MXE(self, splicing):
        check_dict = defaultdict(lambda: defaultdict(dict))
        for gene_key in self.events["ES"]:
            gene, tx = gene_key.split("_")
            #get splice list in same gene and transcripts and next exon
            #skip if exons < 4 (exons should be >=4 for MXE
            if len(splicing.junction_list[gene][tx]) < 4:
                continue
                
            for ex_no in splicing.junction_list[gene][tx]:
                #current jc_key and count
                chr1, jc_key1, _ = splicing.junction_list[gene][tx][ex_no]
                ejc1, l_ijc, _, _ = self.events["ES"][gene_key][jc_key1]
                _, start, _ = jc_key1.split("_")
                next_ex = ex_no+1
                #check adjacent exon
                if next_ex not in splicing.junction_list[gene][tx]:
                    continue
                
                chr2, jc_key2, _ = splicing.junction_list[gene][tx][next_ex]
                for jc_list in splicing.splices[chr][jc_key2]:
                    type, jc_gene, jc_tx, jc_ex = map(str, jc_list)
                    if type == "ES":    #MXE
                        r_ijc, _, ejc2 = self.events["ES"][gene_key][jc_key1]
                        new_key = jc_key1.split("_")
                        new_key[1] = start
                        psi = (l_ijc+r_ijc)/(l_ijc+r_ijc + ejc1 + ejc2)
                        self.events["MXE"][gene_key]["_".join(new_key)] = [ejc1, ejc2, l_ijc, r_ijc]
                            

    def _writeEvent(self, sample):
        header = ['gene_tx', 'event_key', 'counts']
        for type in self.events:
            file_nm = type+"_"+sample+"_events.csv"
            with open(file_nm, 'w') as csvfile:
                out = csv.DictWriter(csvfile, fieldnames = header)
                out.writeheader()
                out.writerows(self.events[type])


    def form_splicings(self, gtf, bam, junction, sample):
    
        #find alternative splicing first
        for chr in junction.splice_info:
            for s_key in junction.splice_info[chr]:   #11/22/2021
                #get linked junction position
                j_start = if "LEFT" in junction.junction_link[n_chr][s_key] else 0
                j_end = if "RIGHT" in junction.junction_link[n_chr][s_key] else 0
                #check splicing has original linked junction
                if j_start == 0 and j_end == 0:
                    print("There is no linked junction for splicing %s-%s" %(chr, s_key))
                    sys.exit()
                #get junction
                s_start, s_end = s_key.split("_")
                j_key = "_".join(j_start, s_end) if j_start > 0 else "_".join(s_start, j_end)
                
                
                for jc_list in junc.splice_info[chr][s_key]: #multiple transcripts
                    jc_type, jc_gene, jc_tx, ex_no = map(int, jc_list)
                    #get count
                    cov = bam.cov[jc_gene][jc_tx][j_key]
                    ijc_m = bam.inc_m[jc_gene][jc_tx][s_key]
                    ijc_o = bam.inc_o[jc_gene][jc_tx][j_start] if j_start > 0 else bam.inc_o[jc_gene][jc_tx][j_end]
                    #get ex_info
                    
                    exc = bam.exc[jc_gene][jc_tx][ex_info]
                    #update inc_o (junction / splice diff keys)
                                            #get psi without normalization
                            psi = (l_ijc+r_ijc)/(l_ijc+r_ijc + 2*ejc)
                            self.events[as_type][gene_key][jc_key] = [ejc, l_ijc, r_ijc, psi] #exclude_jc read, include_jc left include_jc right
                            
        for gene in bam.j_counts:
            for tx in bam.j_counts[gene]:
                for ex_info in bam.j_counts[gene][tx]:
                    ex1, ex2 = ex_info.strip().split("_")
                    ####
                    #get junction counts
                    ###
                    #get splicing key first
                    for list in junction.junction_list[gene][tx][ex1]:
                        chr, jc_key1, j_type1 = map(str, list)
                        if (j_type1 eq "KNOWN"):
                            continue
                        #Find exclude junction counts
                        bam.jc_counts[j_gene][j_tx][jc_key1]
                    _, jc_key2, j_type2 = map(str, junction.junction_list[j_gene][j_tx][ex2])
                    
                    
        for chr in gtf.genes:
            for gene in gtf.genes[chr]:
                _, gx_start, gx_end = map(str, gtf.genes[chr][gene][0:3])
                gx_start = int(gx_start)
                gx_end = int(gx_end)
                if gene not in gtf.exons:  #no exon information from gtf file
                    continue
                
                for tx in gtf.exons[gene]:
                    gene_key = "_".join([gene, tx])
                    #Detect ES/A3SS/A5SS
                    last_ex = list(gtf.exons[gene][tx].keys())[-1]
                    for jc_key in bam.jc_counts[gene][tx]:
                        ejc = bam.jc_counts[gene][tx][jc_key]
                        for jc_list in junction.splices[chr][jc_key]: #multiple transcripts
                            as_type, _, jc_tx, ex_no = map(str, j_list)
                            if tx != jc_tx:
                                continue
                            
                            if as_type == "ES":
                                #get included junction reads
                                l_ijc = 0
                                if int(ex_no) > 1: #not first exon for ES
                                    ex_info = "_".join([ex_no-1, ex_no])
                                    l_ijc = bam.j_counts[gene][tx][ex_info]
                                r_ijc = 0
                                if ex_no < last_ex:
                                    ex_info = "_".join([ex_no, ex_no+1])
                                    r_ijc = bam.j_counts[gene][tx][ex_info]
                            elif as_type == "A3SS":
                                #get included junction reads
                                l_ijc = 0
                                if ex_no > 1: #not first exon for ES
                                    ex_info = "_".join([ex_no-1, ex_no])
                                    l_ijc = bam.j_counts[gene][tx][ex_info]
                                
                                r_ijc = bam.ijc_counts[chr][gene][jc_key+"_RIGHT"]
                            elif as_type == "A5SS":
                                #get included junction reads
                                l_ijc = bam.ijc_counts[chr][gene][jc_key+"_LEFT"]
                                r_ijc = 0
                                if ex_no < last_ex:
                                    ex_info = "_".join([ex_no, ex_no+1])
                                    r_ijc = bam.j_counts[gene][tx][ex_info]

                            #get psi without normalization
                            psi = (l_ijc+r_ijc)/(l_ijc+r_ijc + 2*ejc)
                            self.events[as_type][gene_key][jc_key] = [ejc, l_ijc, r_ijc, psi] #exclude_jc read, include_jc left include_jc right
                            
                        #For RI
                        for j_key in junction.junctions[chr]:
                            for j_list in junction.junctions[chr][j_key]: #multiple transcripts
                                _, j_tx, ex_info = map(str, j_list)
                                if tx != j_tx:
                                    continue
                                ejc = bam.j_counts[gene][tx][ex_info]
                                #how to get ijc for RI?
                                l_ijc = bam.ijc_counts[chr][gene][j_key+"_LEFT"]
                                r_ijc = bam.ijc_counts[chr][gene][j_key+"_RIGHT"]
                                cov = bam.cov[gene][tx][j_key]
                                #get psi without normalization
                                psi = (l_ijc+r_ijc)/(l_ijc+r_ijc + 2*ejc)
                                self.events["IR"][gene_key][j_key] = [ejc, l_ijc, r_ijc, psi, cov]
            #detect MXE
            self._define_MXE(junction)
             
            #write event for R
            self._writeEvent(sample)
                    
    
