#coding: UTF-8
#!/usr/bin/env python

import sys
import re
import os
import pybam

def _checkFlag(flag):
    flag_list = [1,2,4,8,16,32,64,128,512,1024,2048]
    rlt = 1
    if (flag % 2) ==0:  #no paired
        return 0
    if (flag >= 1024) and (flag < 2048): #dup
        return 0
    elif (flag >= 2048) and ((flag-2048) >=1024):    #dup
        return 0
    #check mate_is unmapped
    mate_val = flag - 9 #(1+8)
    if mate_val in flag_list:
        return 0
    return rlt

def _get_tag(tags, key):
    t_dict = {tags[i][0]: tags[i][2] for i in range(len(tags))}
    return t_dict.get(key)

for line in pybam.read('../data/hisat2/gs_s1.bam'):
    chr = line.sam_rname
    flag = line.sam_flag
    cigar = line.sam_cigar_string
    tags = line.sam_tags_list
    #print('tag type:%s' %type(tags))
    #print('tag size:%s' %len(tags))
    #print(tags)
    #print(tags[0])
    #print(tags[0][0])
    
    #print('tag type:%s row-size:%s col-size:%s' %(type(tags), len(tags), len(tags[1])))
    start = line.sam_pos0
    
    #flag 1-pair 8-mate is unmapped 1024 - duplicate (even number then pass (not paired)
    if _checkFlag(flag) == 0:
        continue
    #cigar = line.cigarstring
    if 'D' in cigar or 'I' in cigar or 'S' in cigar or 'H' in cigar or 'P' in cigar or 'X' in cigar or '=' in cigar: ## skip
        continue;

    tag_val = _get_tag(tags, 'NH')
    if tag_val != 1: #unique contig only
        continue

    print("chr:%s, tag:%s, cigar:%s" % (chr, tag_val, cigar))
    
bamfile.close()


