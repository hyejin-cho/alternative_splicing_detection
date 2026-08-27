#!/usr/bin/env perl
# coding: UTF-8
#
# combine_novel_splices_SJ.pl
# ===========================
# Combine novel de-novo splice junctions from multiple STAR SJ.out.tab files
# for use as the novel splice input to detect_AS.py.
#
# A junction is NOVEL when column 6 == 0 (not annotated in the GTF supplied
# to STAR).
#
# Usage
# -----
#   perl combine_novel_splices_SJ.pl -o combined_novel_splices.txt \
#        sample_01_SJ.out.tab sample_02_SJ.out.tab ...
#
#   Optional flags:
#     -o FILE    output file name (default: combined_novel_splices.txt)
#     -m INT     minimum uniquely-mapping reads crossing the junction
#                across ALL input files combined (default: 1)
#     -h         print this help and exit
#
# Input format  — STAR SJ.out.tab (9 columns, tab-separated, no header)
# -----------------------------------------------------------------------
#   col1  chromosome
#   col2  intron start (1-based, first base of intron)
#   col3  intron end   (1-based, last base of intron)
#   col4  strand       (0=undefined, 1=+, 2=-)
#   col5  intron motif (0=non-canonical, 1=GT/AG, 2=CT/AC,
#                       3=GC/AG, 4=CT/GC, 5=AT/AC, 6=GT/AT)
#   col6  annotated    (0=novel/de-novo, 1=annotated)
#   col7  unique reads crossing the junction
#   col8  multi-mapping reads crossing the junction
#   col9  maximum spliced alignment overhang
#
# Output format — tab-separated, sorted by chrom then start
# ----------------------------------------------------------
#   col1  chromosome
#   col2  adjusted start  = STAR_col2 - 2
#          ** This coordinate is required by junctions.py which adds +1
#             internally so that the value equals the GTF exon1_end (1-based),
#             matching the junction key format {exon1_end}_{exon2_start-1}. **
#   col3  intron end (1-based, last base of intron) = STAR_col3 (unchanged)
#   col4  strand (0=undefined, 1=+, 2=-) from STAR — as-is
#   col5  intron motif — as-is
#   col6  0 (always novel — annotated entries are excluded)
#   col7  sum of unique reads across all input files
#   col8  sum of multi-mapping reads across all input files
#   col9  maximum overhang seen across all input files
#
# Coordinate conversion detail
# ----------------------------
#   STAR SJ.out.tab uses 1-based intron coordinates:
#     intron_start (col2) = exon1_end + 1
#     intron_end   (col3) = exon2_start - 1
#
#   junctions.py buildsjunction keys as:
#     j_key = "{exon1_end}_{exon2_start-1}"
#   and reads the novel splice file then does:
#     n_info[1] = n_info[1] + 1   (to match exon1_end)
#
#   Therefore: output col2 must equal exon1_end - 1 = STAR_col2 - 2
#   so that after the +1 in junctions.py:
#     n_info[1] = (STAR_col2 - 2) + 1 = STAR_col2 - 1 = exon1_end  [correct]
#
# Dependencies
# ------------
#   Core Perl only — no non-standard modules required.
#

use strict;
use warnings;
use Getopt::Long;

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

my $output   = 'combined_novel_splices.txt';
my $min_reads = 1;
my $help     = 0;

GetOptions(
    'o=s' => \$output,
    'm=i' => \$min_reads,
    'h'   => \$help,
) or die usage();

if ($help || @ARGV == 0) {
    print usage();
    exit 0;
}

my @sj_files = @ARGV;
print "Input files     : " . scalar(@sj_files) . "\n";
print "Output file     : $output\n";
print "Min unique reads: $min_reads\n\n";

# ---------------------------------------------------------------------------
# Pass 1 — read all SJ.out.tab files, keep novel junctions (col6 == 0)
# ---------------------------------------------------------------------------
# Storage structure:
#   $junctions{chrom}{adj_start}{end} = {
#       strand   => INT,
#       motif    => INT,
#       unique   => INT,   # summed across files
#       multi    => INT,   # summed across files
#       overhang => INT,   # max across files
#   }
# ---------------------------------------------------------------------------

my %junctions;
my $total_lines   = 0;
my $novel_kept    = 0;
my $annotated_skipped = 0;

for my $file (@sj_files) {
    unless (-f $file) {
        warn "WARNING: File not found, skipping: $file\n";
        next;
    }

    open(my $fh, '<', $file)
        or die "Cannot open $file: $!\n";

    my $file_novel = 0;
    my $file_lines = 0;

    while (my $line = <$fh>) {
        chomp $line;
        next if $line =~ /^\s*$/;   # skip blank lines

        my @f = split(/\t/, $line);
        unless (@f == 9) {
            warn "WARNING: Expected 9 columns, got " . scalar(@f)
               . " in $file line $. — skipping\n";
            next;
        }

        $file_lines++;
        $total_lines++;

        my ($chrom, $col2, $col3, $strand,
            $motif, $annotated, $unique, $multi, $overhang) = @f;

        # --- Novel filter: col6 must be 0 --------------------------------
        if ($annotated != 0) {
            $annotated_skipped++;
            next;
        }

        # --- Coordinate conversion ---------------------------------------
        # Output start = STAR_col2 - 2
        # junctions.py adds +1 to this, giving STAR_col2 - 1 = exon1_end
        my $adj_start = $col2 - 2;
        my $end       = $col3;   # unchanged — = exon2_start - 1 = j_end

        # Guard against negative coordinates (malformed input)
        if ($adj_start < 0) {
            warn "WARNING: Adjusted start < 0 for $chrom:$col2-$col3 "
               . "in $file — skipping\n";
            next;
        }

        # --- Accumulate counts -------------------------------------------
        if (exists $junctions{$chrom}{$adj_start}{$end}) {
            # Junction seen before — sum reads, take max overhang
            $junctions{$chrom}{$adj_start}{$end}{unique}   += $unique;
            $junctions{$chrom}{$adj_start}{$end}{multi}    += $multi;
            $junctions{$chrom}{$adj_start}{$end}{overhang}  =
                $overhang > $junctions{$chrom}{$adj_start}{$end}{overhang}
                ? $overhang
                : $junctions{$chrom}{$adj_start}{$end}{overhang};
        } else {
            # First time seeing this junction
            $junctions{$chrom}{$adj_start}{$end} = {
                strand   => $strand,
                motif    => $motif,
                unique   => $unique,
                multi    => $multi,
                overhang => $overhang,
            };
            $novel_kept++;
        }

        $file_novel++;
    }

    close($fh);
    printf "  %-40s  %6d total  %6d novel\n",
           $file, $file_lines, $file_novel;
}

print "\n";
print "Total lines read      : $total_lines\n";
print "Annotated (skipped)   : $annotated_skipped\n";
print "Unique novel junctions: $novel_kept\n";

# ---------------------------------------------------------------------------
# Pass 2 — apply minimum read filter and write output
# ---------------------------------------------------------------------------

open(my $out_fh, '>', $output)
    or die "Cannot write to $output: $!\n";

my $written  = 0;
my $filtered = 0;

# Sort chromosomes: numeric chroms first (1..22), then X, Y, M, then rest
my @chroms = sort {
    my ($a_num) = $a =~ /^(?:chr)?(\d+)$/;
    my ($b_num) = $b =~ /^(?:chr)?(\d+)$/;
    if (defined $a_num && defined $b_num) {
        $a_num <=> $b_num;
    } elsif (defined $a_num) {
        -1;
    } elsif (defined $b_num) {
        1;
    } else {
        $a cmp $b;
    }
} keys %junctions;

for my $chrom (@chroms) {
    # Sort by adjusted start position, then end position
    for my $adj_start (sort { $a <=> $b } keys %{ $junctions{$chrom} }) {
        for my $end (sort { $a <=> $b } keys %{ $junctions{$chrom}{$adj_start} }) {

            my $j = $junctions{$chrom}{$adj_start}{$end};

            # Apply minimum unique-read filter across all samples combined
            if ($j->{unique} < $min_reads) {
                $filtered++;
                next;
            }

            print $out_fh join("\t",
                $chrom,
                $adj_start,        # STAR_col2 - 2 (junctions.py adds +1)
                $end,              # STAR_col3 unchanged
                $j->{strand},
                $j->{motif},
                0,                 # always novel
                $j->{unique},     # summed across files
                $j->{multi},      # summed across files
                $j->{overhang},   # max across files
            ) . "\n";

            $written++;
        }
    }
}

close($out_fh);

print "Filtered (< $min_reads unique reads): $filtered\n";
print "Written to output     : $written\n";
print "Output file           : $output\n";
print "\nDone.\n";

# ---------------------------------------------------------------------------
# Usage string
# ---------------------------------------------------------------------------

sub usage {
    return <<'USAGE';

Usage:
  perl combine_novel_splices_SJ.pl [options] SJ1.out.tab SJ2.out.tab ...

Options:
  -o FILE    Output file name (default: combined_novel_splices.txt)
  -m INT     Minimum unique reads summed across all files (default: 1)
  -h         Print this help and exit

Example:
  perl combine_novel_splices_SJ.pl -o novel_splices.txt -m 2 \
       sample_01_SJ.out.tab sample_02_SJ.out.tab sample_03_SJ.out.tab

USAGE
}
