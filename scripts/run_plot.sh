#!/bin/sh

#  splicing.sh
#  Created by Hyejin Cho on 8/20/19.
#  

#module load Python/2.7.14
#module load samtools
#module load R/3.6.0-Python-3.6.6-foss-2018b

gtf=$1
program_dir=/home/hycho/share_home/SI/src/scripts
input=$2
region=$3
out=$4

python $program_dir/sashimi-plot.py -b $input -c $region -g $gtf -M 3 -C 3 -O 3 --alpha 0.25 --base-size=20 --ann-height=2 -o $out --width=12 -P /home/hycho/share_home/SI/src/scripts/palette.txt

#python $program_dir/sashimi-plot.py -b $input -c $region -g $gtf -M 3 -C 3 -O 3 --shrink --alpha 0.25 --base-size=20 --ann-height=2 -o $out --width=12 -P palette.txt
# --labels $label
#-F png




