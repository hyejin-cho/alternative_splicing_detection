#coding: UTF-8
#!/usr/bin/env python

import sys
import re
import os
import gzip
import logging
import argparse
from collections import defaultdict
import collections, functools, operator
from classes.file import File
from classes.gtf import GTF
from classes.bam import Bam
import pandas as pd
import statistics
import subprocess as sp
from numpy import *

def runR(file1, file2):
    p = sp.Popen("R --vanilla --slave", shell=True, stdin=sp.PIPE)
    p.communicate(input=r_script.encode())
    p.stdin.close()
    p.wait()
    return

def detectDiffAS(gtf, tmp_file, sample_count, start_column, output):
    df = pd.read_csv(tmp_file, sep="\t")
    count_columns = 3   #cov, IJC, EJC
    #fc_column = si_column + 2
    s2_start = count_columns * sample_count + start_column
    si_pos = df.columns.get_loc("SI1")
    
    df = df.apply(_calcCov, axis=1, args=[gtf, sample_count, count_columns, start_column, s2_start, si_pos])
    #df.to_csv('tmp_test.txt', index = None, sep='\t', header=True)
    
    #change columns order
    cols = df.columns.tolist()
    c_cols = cols[:si_pos+2]
    c_cols.append(cols[-1])
    c_cols.extend(cols[si_pos+2:len(cols)-1])

    df=df[c_cols]
    df.to_csv(output, index = None, sep='\t', header=True)

def _calcCov(x, gtf, sample_count, count_columns, start_column, s2_start, si_pos):
    #type = x.iloc[2]
    type = x['TYPE']
    cov_rlt1 = list()
    cov_rlt2 = list()
    
    if type == 'RI':
        #change name from index to position
        chr = x['Chr']
        gene = x['Gene']
        idx = int(x['Name'])
        #print("ri idx:" + idx)
        exons = gtf.exons[chr][gene]
        #ri_name = exons[idx][2]+"_"+exons[idx+1][1]
        ri_pos = [int(exons[idx][2]),int(exons[idx+1][1])]
        ri_pos[0] = ri_pos[0]+1
        ri_pos[1] = ri_pos[1]-1
        ri_name = "_".join(map(str, ri_pos))
        x['Name'] = ri_name #change index to name(positions)
        
        ri_length = int(exons[idx+1][1]) - int(exons[idx][2]) + 1
        for i in range(sample_count):   #calc coverage
            cov1 = 0.0
            pos1 = start_column + (count_columns * i)
            if int(x.iloc[pos1]) > 0:
                cov1 = float(int(x.iloc[pos1])/ri_length)
            
            cov2 = 0.0
            pos2 = s2_start + (count_columns * i)
            
            if int(x.iloc[pos1]) > 0:
                cov2 = float(int(x.iloc[pos2])/ri_length)
            
            x.iloc[pos1] = cov1    #change counts to coverages
            x.iloc[pos2] = cov2

    else:   #other AS type, just change name start+1, end -1
        pos =  x['Name'].split("_")
        pos[0] = int(pos[0])+1
        pos[1] = int(pos[1])-1
        new_nm = "_".join(map(str, pos))
        x['Name'] = new_nm

    #get delta SI
    
    d_si_pos = si_pos + 2
    d_si = abs(float(x.iloc[si_pos])-float(x.iloc[si_pos+1]))
    #x.insert(d_si_pos, )
    x['Delta_SI'] = d_si
    
    return x
