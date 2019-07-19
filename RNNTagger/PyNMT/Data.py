
import sys
from collections import Counter
import pickle
import numpy

pad_string = '<pad>'
unk_string = '<unk>'

         
### helper functions ###############################################
         
def build_dict(file_name, max_vocab_size):
   """ 
   reads a list of sentences from a file and returns
   - a dictionary which maps the most frequent words to indices and
   - a table which maps indices to the most frequent words
   """
   
   word_freq = Counter()
   with open(file_name) as file:
      for line in file:
         word_freq.update(line.split())
         
   if max_vocab_size <= 0:
      max_vocab_size = len(word_freq)
            
   words, _ = zip(*word_freq.most_common(max_vocab_size))
   # ID of pad_string must be 0
   words = [pad_string, unk_string] + list(words)
   word2ID = {w:i for i,w in enumerate(words)}
   
   return word2ID, words


def pad_batch(batch):
   """ pad sequences in batch with 0s to obtain sequences of identical length """
   
   seq_len = list(map(len, batch))
   max_len = max(seq_len)
   padded_batch = [seq + [0]*(max_len-len(seq)) for seq in batch]
   return padded_batch, seq_len


def rstrip_zeros( wordIDs ):
   wordIDs = list(wordIDs)
   if 0 not in wordIDs:
      return wordIDs
   return wordIDs[:wordIDs.index(0)]

     
def words2IDs(words, word2ID):
   """ maps a list of words to a list of IDs """
   unkID = word2ID[unk_string]
   return [word2ID.get(w, unkID) for w in words]

   
### class Data ####################################################

class Data(object):
   """
   class for reading a tagged training and development corpus or a test corpus
   """

   def __init__(self, *args):
      if len(args) == 2:
         self.init_test( *args )  # Initialisation for translation
      else:
         self.init_train( *args ) # Initialisation for training
   
         
   ### functions needed during training ##########################

   def init_train(self, path_train_src, path_train_tgt, path_dev_src, path_dev_tgt,
                  max_src_vocab_size, max_tgt_vocab_size, max_len, batch_size):

      self.max_len    = max_len
      self.batch_size = batch_size
      
      self.path_train_src = path_train_src
      self.path_train_tgt = path_train_tgt
      self.path_dev_src   = path_dev_src
      self.path_dev_tgt   = path_dev_tgt

      self.src2ID, self.ID2src = build_dict(self.path_train_src, max_src_vocab_size)
      self.tgt2ID, self.ID2tgt = build_dict(self.path_train_tgt, max_tgt_vocab_size)
      self.src_vocab_size = len(self.ID2src)
      self.tgt_vocab_size = len(self.ID2tgt)


   def train_batches(self):
      return self.batches()
      
   def dev_batches(self):
      return self.batches(train=False)
      
   def batches(self, train=True):
      """ yields the next batch of training examples """

      if train:
         src_file = open(self.path_train_src)
         tgt_file = open(self.path_train_tgt)
      else:
         src_file = open(self.path_dev_src)
         tgt_file = open(self.path_dev_tgt)

      num_batches_in_big_batch = 5
      big_batch_size = self.batch_size * num_batches_in_big_batch
      big_batch = []
      while True:
         for src_line, tgt_line in zip(src_file, tgt_file):
            srcIDs = words2IDs(src_line.split(), self.src2ID)
            tgtIDs = words2IDs(tgt_line.split(), self.tgt2ID)

            # filter out very long sentences
            if self.max_len > 0 \
                 and (len(srcIDs) > self.max_len or len(tgtIDs) > self.max_len):
               continue
            
            big_batch.append((srcIDs, tgtIDs))

            if len(big_batch) == big_batch_size:
               # sort by source sentence length
               big_batch.sort(key=lambda x: len(x[0]), reverse=True)

               # extract the mini-batches
               for i in range(num_batches_in_big_batch):
                  batch = big_batch[i*self.batch_size:(i+1)*self.batch_size]
                  src_vecs, tgt_vecs = zip(*batch)
                  yield pad_batch(src_vecs), pad_batch(tgt_vecs)
                  
               big_batch = []

         if train:
            # reread the two files
            src_file.seek(0)
            tgt_file.seek(0)
         else:
            # sort by source sentence length
            big_batch.sort(key=lambda x: len(x[0]), reverse=True)
            
            # extract the last mini-batches
            for i in range(num_batches_in_big_batch):
               batch = big_batch[i*self.batch_size:(i+1)*self.batch_size]
               if len(batch) > 0:
                   src_vecs, tgt_vecs = zip(*batch)
                   yield pad_batch(src_vecs), pad_batch(tgt_vecs)
            break
            

   def store_parameters( self, filename ):
      """ store parameters to a file """
      all_params = (self.ID2src, self.ID2tgt)
      with open(filename, "wb") as file:
         pickle.dump( all_params, file )

      
   ### functions needed for translation ############################

   def init_test(self, filename, batch_size):
      """ load parameters from a file """

      self.batch_size = batch_size
      with open(filename, "rb") as file:
         data = pickle.load( file )
         self.ID2src, self.ID2tgt = data
         self.src2ID = {w:i for i,w in enumerate(self.ID2src)}
         self.tgt2ID = {w:i for i,w in enumerate(self.ID2tgt)}

         
   def build_batch(self, batch):
      batch_IDs = [words2IDs(srcWords, self.src2ID) for srcWords in batch]
      result = sorted(enumerate(batch_IDs), key=lambda x: len(x[1]), reverse=True)
      sent_idx, sorted_batch_IDs = zip(*result)

      rev_sent_idx = [None] * len(sent_idx)
      for i,k in enumerate(sent_idx):
         rev_sent_idx[k] = i

      return batch, rev_sent_idx, pad_batch(sorted_batch_IDs)

   
   def test_batches(self, file):
      """ yields the next batch of test sentences """

      batch = []
      for line in file:
         srcWords = line.split()
         batch.append(srcWords)
         if len(batch) == self.batch_size:
            yield self.build_batch(batch)
            batch = []

      if len(batch) > 0:
         yield self.build_batch(batch)

   def source_words(self, wordIDs):
      return [self.ID2src[id] for id in wordIDs]

   def target_words(self, wordIDs):
      return [self.ID2tgt[id] for id in wordIDs if id > 0]

