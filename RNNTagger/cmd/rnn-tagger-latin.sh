#!/bin/sh

# Set these paths appropriately

BIN=./bin
SCRIPTS=./scripts
LIB=./lib
PyRNN=./PyRNN
PyNMT=./PyNMT
TMP=/tmp/rnn-tagger$$
LANGUAGE=latin

TOKENIZER=${SCRIPTS}/tokenize.pl
ABBR_LIST=${LIB}/Tokenizer/${LANGUAGE}-abbreviations
MWL=${SCRIPTS}/mwl-lookup.perl
MWLFILE=${LIB}/MWL/${LANGUAGE}-mwls
TAGGER=$PyRNN/rnn-annotate.py
RNNPAR=${LIB}/PyRNN/${LANGUAGE}
REFORMAT=${SCRIPTS}/reformat.pl
LEMMATIZER=$PyNMT/nmt-translate.py
NMTPAR=${LIB}/PyNMT/${LANGUAGE}

$TOKENIZER -a $ABBR_LIST $1 |
    $MWL -f $MWLFILE > $TMP.tok

tr '[:upper:]' '[:lower:]' < $TMP.tok > $TMP.lc
$TAGGER $RNNPAR $TMP.lc > $TMP.tagged

$REFORMAT $TMP.tagged > $TMP.reformatted

$LEMMATIZER --print_source $NMTPAR $TMP.reformatted > $TMP.lemmas

$SCRIPTS/lemma-lookup.pl $TMP.lemmas $TMP.tagged | tr '~' ' ' > $TMP.lemmatized

paste $TMP.tok $TMP.lemmatized | cut -f1,3,4

rm $TMP.tok $TMP.lc  $TMP.tagged  $TMP.reformatted $TMP.lemmas $TMP.lemmatized
