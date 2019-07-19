#!/usr/bin/perl

use warnings;
use strict;
use utf8::all;

my $filename = shift or die "Error: missing file argument!";
open(FILE, ">$filename") or die;

my $N=0;
while (<>) {
    if (/^<.*>$/) {
	print FILE "$N\t$_";
    }
    else {
	print;
	$N++;
    }
}
