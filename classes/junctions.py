                                                                          #coding: UTF-8
#!/usr/bin/env python

import sys
import re
import os
import gzip
import logging
from collections import defaultdict
from classes.file import File
import pandas as pd

logging.basicConfig(level=logging.DEBUG, format='[%(levelname)s] %(asctime)s %(message)s', filename='log_parse.GTF.txt')

#read junction file all together

class Junction():
    def __init__(self):
        self.junctions = defaultdict(dict)
        self.junction_info = defaultdict(dict)
        self.junction_link = defaultdict(dict)  #linke between junction and splicing
        self.splices = defaultdict(dict)
        self.splice_info = defaultdict(dict)
        self.junction_list = defaultdict(lambda: defaultdict(dict))
        self.position_list = defaultdict(dict)
        
    def findJunctionsFromGTF(self, gtf):
        for gene in gtf.exons:
            for tx_id in gtf.exons[gene]:
                last_ex = list(gtf.exons[gene][tx_id].keys())[-1]
                if last_ex == 1:    #can't have a junction
                    break
                #print('last exon: %s' %last_ex)
                def _find_junction(ex_no):
                    chr, ex_start, ex_end, strand = map(str, gtf.exons[gene][tx_id][ex_no])
                    n_ex = int(ex_no)+1
                    n_start, n_end = map(int, gtf.exons[gene][tx_id][str(n_ex)][1:3])
                    j_key = "_".join([str(ex_end), str(n_start-1)])
                    ex_info = "_".join([ex_no, str(n_ex)])
                    
                    self.position_list[chr][j_key] = ["_", "J"] #type is junction
                    if gene not in self.junctions[chr]:
                        self.junctions[chr][gene] = []
                    self.junctions[chr][gene].append(j_key)
                    if j_key not in self.junction_info[chr]:
                        self.junction_info[chr][j_key] = []
                    self.junction_info[chr][j_key].append([gene, tx_id, ex_info, strand])
                    if ex_no not in self.junction_list[gene][tx_id]:
                        self.junction_list[gene][tx_id][ex_no] = []
                    self.junction_list[gene][tx_id][ex_no].append([chr, j_key, "KNOWN"])
                    
                map(_find_junction, range(last_ex-1))   #call find_junction
    
    def detectNovelSplices(self, gtf, filelist):
        for f in filelist:
            self._readNovelSplices(gtf, f)
        
    def _readNovelSplices(self, gtf, file_name):
        file = open(file_name, 'r')
        lines = file.readlines()
        
        for line in lines:
            n_info = line.strip().split("\t")
            
            n_chr = n_info[0];
            n_info[1] = str(int(n_info[1])+1)   #fix position (should be match with gtf end)
            n_key = "_".join(n_info[1::3])
            print('nkey:%s' %n_key)
            if n_key in self.junction_info[n_chr]: #duplicates with gtf junction
                continue
            if n_key in self.position_list[n_chr]:   #duplicates with alternative splicing
                continue
            
            #find junction type (ES/A3SS/A5SS/RI)
            for j_key in self.junction_info[n_chr]:
                #print('Junction key %s' %j_key)
                j_start, j_end = j_key.split("_")
                j_start = int(j_start)
                j_end = int(j_end)
                if j_end < int(n_info[1]):   #pass to next (too small)
                    continue
                elif j_start == int(n_info[1]):    #match start position (A3SS, ES)
                    for j_list in self.junction_info[n_chr][j_key]: #multiple transcripts
                        j_gene, j_tx, ex_info, j_strand = map(str, j_list)
                        #check with last(second) exon
                        ex_no = ex_info.strip().split("_")[1]   #second exon
                        _, ex_start, ex_end, strand = map(str, gtf.exons[j_gene][j_tx][ex_no])
                        ex_start = int(ex_start)
                        ex_end = int(ex_end)
                        if int(n_info[2]) != ex_start and int(n_info[2]) <= ex_end: #A3SS (otherwise ES)
                            if j_gene not in self.splices[n_chr]:
                                self.splices[n_chr][j_gene] = []
                            self.splices[n_chr][j_gene].append(n_key)
                            type = "A3SS" if strand == "+" else "A5SS"
                            if n_key not in self.splice_info[n_chr]:    #can be mutiple for transcripts
                                self.splice_info[n_chr][n_key] = []
                            self.splice_info[n_chr][n_key].append([type, j_gene, j_tx, ex_info, strand])
                            
                            self.junction_link[n_chr][n_key]["RIGHT"] = j_end
                            
                            if ex_no not in self.junction_list[j_gene][j_tx]:
                                self.junction_list[j_gene][j_tx][ex_no] = []
                            self.junction_list[j_gene][j_tx][ex_no].append([n_chr, n_key, "NOVEL"])
                            self.position_list[chr][n_key] = [n_info[2], "S"]  #type is splicing
                            break
                        elif int(n_info[2]) > ex_end:
                            if j_gene not in self.splices[n_chr]:
                                self.splices[n_chr][j_gene] = []
                            self.splices[n_chr][j_gene].append(n_key)
                            
                            if n_key not in self.splice_info[n_chr]:    #can be mutiple for transcripts
                                self.splice_info[n_chr][n_key] = []
                            self.splice_info[n_chr][n_key].append(["ES", j_gene, j_tx, ex_no, strand])
                            self.junction_link[n_chr][n_key]["RIGHT"] = j_end
                            if ex_no not in self.junction_list[j_gene][j_tx]:
                                self.junction_list[j_gene][j_tx][ex_no] = []
                            self.junction_list[j_gene][j_tx][ex_no].append([n_chr, n_key, "NOVEL"])
                            self.position_list[chr][n_key] = ["_", "S"]
                            break
                elif j_end == int(n_info[2]):    #match end position (A5SS, ES)
                    for j_list in self.junctions[n_chr][j_key]: #multiple transcripts
                        j_gene, j_tx, ex_info, _ = map(str, j_list)
                        #check with first exon
                        ex_no = ex_info.strip().split("_")[0]
                        chr, ex_start, ex_end, strand = map(str, gtf.exons[j_gene][j_tx][ex_no])
                        ex_start = int(ex_start)
                        ex_end = int(ex_end)
                        if int(n_info[1]) != ex_end and int(n_info[1]) >= ex_start:
                            if j_gene not in self.splices[n_chr]:
                                self.splices[n_chr][j_gene] = []
                            self.splices[n_chr][j_gene].append(n_key)
                            type = "A5SS" if strand == "+" else "A3SS"
                            if n_key not in self.splice_info[n_chr]:    #can be mutiple for transcripts
                                self.splice_info[n_chr][n_key] = []
                            self.splice_info[n_chr][n_key].append([type, j_gene, j_tx, ex_info, strand])
                            self.junction_link[n_chr][n_key]["LEFT"] = j_start
                            if ex_no not in self.junction_list[j_gene][j_tx]:
                                self.junction_list[j_gene][j_tx][ex_no] = []
                            self.junction_list[j_gene][j_tx][ex_no].append([n_chr, n_key, "NOVEL"])
                            self.position_list[chr][n_key] = [n_info[1], "S"]
                            break
                        elif int(n_info[1]) < ex_start:
                            if j_gene not in self.splices[n_chr]:
                                self.splices[n_chr][j_gene] = []
                            self.splices[n_chr][j_gene].append(n_key)
                            
                            if n_key not in self.splice_info[n_chr]:    #can be mutiple for transcripts
                                self.splice_info[n_chr][n_key] = []
                            self.splice_info[n_chr][n_key].append(["ES", j_gene, j_tx, ex_no, strand])
                            self.junction_link[n_chr][n_key]["LEFT"] = j_start
                            if ex_no not in self.junction_list[j_gene][j_tx]:
                                self.junction_list[j_gene][j_tx][ex_no] = []
                            self.junction_list[j_gene][j_tx][ex_no].append([n_chr, n_key, "NOVEL"])
                            self.position_list[chr][n_key] = ["_", "S"]
                            break
                else:   #not match two point check ES with exons
                    for j_list in self.junctions[n_chr][j_key]:
                        j_gene, j_tx, _, _ = map(str, j_list)
                        last_ex = len(gtf.exons[j_gene][j_tx])
                        for ex_no in gtf.exons[j_gene][j_tx]:
                            _, ex_start, ex_end, strand = map(str, gtf.exons[j_gene][j_tx][ex_no])
                            ex_start = int(ex_start)
                            ex_end = int(ex_end)
                            if ex_no == 1 and int(n_info[2]) <= ex_start: #check in first exon range
                                break
                            elif ex_no == last_ex and int(n_info[1]) > ex_end: #check in last exon range
                                break
                            #check bp points skip exon
                            if int(n_info[1]) < ex_start and int(n_info[2]) > ex_end:
                                if j_gene not in self.splices[n_chr]:
                                    self.splices[n_chr][j_gene] = []
                                self.splices[n_chr][j_gene].append(n_key)
                            
                                if n_key not in self.splice_info[n_chr]:    #can be mutiple for transcripts
                                    self.splice_info[n_chr][n_key] = []
                                self.splice_info[n_chr][n_key].append(["ES", j_gene, j_tx, ex_no, strand])
                                #self.junction_link[n_chr][n_key]["LEFT"] = j_start
                                #self.junction_link[n_chr][n_key]["RIGHT"] = j_end
                                if ex_no not in self.junction_list[j_gene][j_tx]:
                                    self.junction_list[j_gene][j_tx][ex_no] = []
                                self.junction_list[j_gene][j_tx][ex_no].append([n_chr, n_key, "NOVEL"])
                                self.position_list[chr][n_key] = ["_", "S"]
                                break
                    
        file.close()

    
