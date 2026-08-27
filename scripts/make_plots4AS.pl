#!/opt/easybuild/software/Perl/5.28.0-GCCcore-7.3.0/bin/perl

use strict;
use warnings;
use Getopt::Long;

use Scalar::Util qw(looks_like_number);

my $usage = <<"USAGE";
usage: $0 gtf_file tsv_file as_file out_dir;
USAGE

die $usage unless @ARGV==4;

my $gtf = $ARGV[0];
my $out_dir = $ARGV[3];
my $tsv = $ARGV[1];

#get gtf information
open GTF, $gtf;
my %transcripts;
while(<GTF>){
    chomp;
    my @arr = split/\t/;
    my $gene;
    
    next unless $arr[2] eq "transcript";
    my @tag = split/; /, $arr[8];
    for (@tag){
        if (/^gene_id/) {
            $_ =~ /gene_id \"(.+)\"/;
            $gene = $1;
        }
    }
    
    my @pos;
    if (exists $transcripts{$gene}) { #compare
        @pos = @{$transcripts{$gene}};
        $pos[1] = $arr[3] if $arr[3] < $pos[1];
        $pos[2] = $arr[4] if $arr[4] > $pos[2];
    }else{
        @pos = @arr[0,3,4];
    }
    
    $transcripts{$gene} = \@pos if $arr[2] eq "transcript";
    
}
close GTF;

my $cmd = "./run_plot.sh";

unless (-d $out_dir) {  #create directory
    mkdir $out_dir;
}

#make head
my $head = "#!/bin/bash\n";
$head .= "#SBATCH --job-name=".$out_dir."_job\n";
$head .= "#SBATCH --mail-type=END,FAIL\n";
$head .= "#SBATCH --mail-user=email\@coh.org\n";
$head .= "#SBATCH -n 8\n";
$head .= "#SBATCH -N 2\n";
$head .= "#SBATCH --mem=10G\n";
$head .= "#SBATCH --output=".$out_dir."_run_%j.log\n";
$head .= "#SBATCH --time=96:00:00\n\n";
$head .= "module load Python/2.7.14\n";
$head .= "module load samtools/1.6\n";
$head .= "module load R/3.5.1\n";

print $head,"\n";

open FILE, $ARGV[2];
my %plotted;

while(<FILE>) {
    chomp;
    next if /^Chr/;
    my @arr = split/\t/;
    next if exists $plotted{$arr[1]};   #no duplication
    
    #print STDERR $arr[0],"\n";
    my @info = @{$transcripts{$arr[1]}};
    my $gene = $arr[1];
    print STDERR "gene:$gene|" if $gene =~ /ARHGAP1/;
    $gene =~ s/\(//g;
    $gene =~ s/\)//g;
    my $line = "$cmd $gtf $tsv ".$info[0].":".$info[1]."-".$info[2]." ".$out_dir."/".$gene;
    print $line,"\n";
    $plotted{$arr[1]} = 1;
}
close FILE;


print STDERR "Done.\n";



