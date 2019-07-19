#!/usr/bin/python3

import sys
import argparse
import torch
import warnings
warnings.simplefilter("ignore")

sys.path.insert(0,'.')
from PyRNN.Data import Data
from PyRNN.RNNTagger import RNNTagger
from PyRNN.CRFTagger import CRFTagger


def annotate_sentence(model, data, words):

   # map words to numbers and create Torch variables
   fwd_charIDs, bwd_charIDs = data.words2charIDvec(words)
   fwd_charIDs = model.long_tensor(fwd_charIDs)
   bwd_charIDs = model.long_tensor(bwd_charIDs)

   # optional word embeddings
   word_embs = None if data.word_emb_size<=0 else model.float_tensor(data.words2vecs(words))

   # run the model
   if type(model) is RNNTagger:
      tagscores = model(fwd_charIDs, bwd_charIDs, word_embs)
      _, tagIDs = tagscores.max(dim=-1)
   elif type(model) is CRFTagger:
      tagIDs = model(fwd_charIDs, bwd_charIDs, word_embs)
   else:
      sys.exit("Error in function annotate_sentence")

   tags = data.IDs2tags(tagIDs)
   return tags



###########################################################################
# main function
###########################################################################

if __name__ == "__main__":

   parser = argparse.ArgumentParser(description='Annotation program of the RNN-Tagger.')

   parser.add_argument('path_param', type=str,
                       help='name of parameter file')
   parser.add_argument('path_data', type=str,
                       help='name of the file with input data')
   parser.add_argument('--crf_beam_size', type=int, default=10,
                       help='size of the CRF beam (if the system contains a CRF layer)')
   parser.add_argument('--gpu', type=int, default=0,
                       help='selection of the GPU (default is GPU 0)')


   if len(sys.argv) < 3:
      sys.stderr.write("\nUsage: "+sys.argv[0]+" parameter-file data-file\n\n")
      sys.exit(1)

   args = parser.parse_args()

   # load parameters
   data  = Data(args.path_param+".io")   # read the symbol mapping tables
   model = torch.load(args.path_param+".rnn", map_location="cpu")   # read the model
   
   if args.gpu >= 0 and torch.cuda.is_available():
      if args.gpu >= torch.cuda.device_count():
         args.gpu = 0
      torch.cuda.set_device(args.gpu)
      model = model.cuda()

   model.eval()
   for i, words in enumerate(data.sentences(args.path_data)):
      print(i, end="\r", file=sys.stderr, flush=True)

      tags = annotate_sentence(model, data, words)

      for word, tag in zip(words, tags):
         print(word, tag, sep="\t")
      print("")
