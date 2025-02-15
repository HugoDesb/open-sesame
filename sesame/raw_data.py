# -*- coding: utf-8 -*-
import nltk
lemmatizer = nltk.stem.WordNetLemmatizer()

from conll09 import CoNLL09Element, CoNLL09Example
from sentence import Sentence

#from rnntagger import RnnTagger

def make_data_instance(text, index):
    """
    Takes a line of text and creates a CoNLL09Example instance from it.
    """
    """
    tagger = RnnTagger()
    tagger.tag(text.lstrip().rstrip())
    
    tokenized = tagger.tokens
    pos_tagged = tagger.pos_tag

    lemmatized = tagger.lemmas
    """

    tokenized = nltk.tokenize.word_tokenize(text.lstrip().rstrip())
    pos_tagged = [p[1] for p in nltk.pos_tag(tokenized)]
    lemmatized = [lemmatizer.lemmatize(tokenized[i])
                    if not pos_tagged[i].startswith("V") else lemmatizer.lemmatize(tokenized[i], pos='v')
                    for i in range(len(tokenized))]

    conll_lines = ["{}\t{}\t_\t{}\t_\t{}\t{}\t_\t_\t_\t_\t_\t_\t_\tO\n".format(
        i+1, tokenized[i], lemmatized[i], pos_tagged[i], index) for i in range(len(tokenized))]
    elements = [CoNLL09Element(conll_line) for conll_line in conll_lines]

    sentence = Sentence(syn_type=None, elements=elements)

    instance = CoNLL09Example(sentence, elements)

    return instance
