#!/usr/bin/perl

use warnings;
use strict;
use utf8::all;

my $filename = shift or die "Error: missing file argument!";
open(FILE, "$filename") or die;

my %xml;
while (<FILE>) {
    my($n,$xml) = split(/\t/);
    $xml{$n} .= $xml;
}

my $N=0;
while (<>) {
    print $xml{$N} if exists $xml{$N};
    print;
    $N++;
}
