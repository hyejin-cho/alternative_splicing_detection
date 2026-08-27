#!/opt/easybuild/software/Perl/5.28.0-GCCcore-7.3.0/bin/perl

use strict;
use warnings;
use Getopt::Long;

use Scalar::Util qw(looks_like_number);

my $usage = <<"USAGE";
usage: $0 splice_file chr;
USAGE

die $usage unless @ARGV==2;

my $chr = $ARGV[1];

open FILE, $ARGV[0];
while(<FILE>){
    chomp;
    my @arr = split/\t/;
    
    next unless $arr[0] eq $chr;
    
    print join("\t", @arr), "\n";
}
close FILE;

