#!/bin/sh

# Set these paths appropriately

BIN=./bin
SCRIPTS=./scripts
LIB=./lib
PyRNN=./PyRNN
PyNMT=./PyNMT
TMP=/tmp/rnn-tagger$$
LANGUAGE=russian

TOKENIZER=${SCRIPTS}/tokenize.pl
TAGGER=$PyRNN/rnn-annotate.py
RNNPAR=${LIB}/PyRNN/${LANGUAGE}
REFORMAT=${SCRIPTS}/reformat.pl
LEMMATIZER=$PyNMT/nmt-translate.py
NMTPAR=${LIB}/PyNMT/${LANGUAGE}

$TOKENIZER $1 > $TMP.tok

$TAGGER $RNNPAR $TMP.tok > $TMP.tagged

$REFORMAT $TMP.tagged > $TMP.reformatted

$LEMMATIZER --print_source $NMTPAR $TMP.reformatted > $TMP.lemmas

$SCRIPTS/lemma-lookup.pl $TMP.lemmas $TMP.tagged 

rm $TMP.tok  $TMP.tagged  $TMP.reformatted $TMP.lemmas
