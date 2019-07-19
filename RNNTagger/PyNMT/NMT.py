
import sys
import math
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.autograd import Variable


class NNModule(nn.Module):

   def __init__(self):
      super(NNModule, self).__init__()

   def on_gpu(self):
      return next(self.parameters()).is_cuda
   
   def variable(self, x):
      return Variable(x).cuda() if self.on_gpu() else Variable(x)
   
   def long_tensor(self, array):
      return self.variable(torch.LongTensor(array))
   
   def zero_tensor(self, *size):
      return self.variable(torch.zeros(*size))
   
   def zero_long_tensor(self, *size):
      return self.zero_tensor(*size).long()
   

### Encoder ###################################################################

class Encoder(NNModule):

   def __init__(self, vocab_size, word_emb_size, rnn_size, rnn_depth,
                dropout_rate, emb_dropout_rate, use_lstm=None, embeddings=None):
      
      super(Encoder, self).__init__()

      if embeddings is None:
         self.embeddings = nn.Embedding(vocab_size, word_emb_size, padding_idx=0)
      else:
         self.embeddings = nn.Embedding.from_pretrained(embeddings)
         
      rnn = nn.LSTM if use_lstm else nn.GRU
      self.rnn = rnn(word_emb_size, rnn_size, batch_first=True, bidirectional=True)
      self.deep_rnns = nn.ModuleList(
         [rnn(2*rnn_size, rnn_size, batch_first=True, bidirectional=True)
          for _ in range(rnn_depth-1)])

      self.dropout = nn.Dropout(dropout_rate)
      self.emb_dropout = nn.Dropout(emb_dropout_rate)

      
   def forward(self, wordIDs, seq_len):

      # look up the source word embeddings
      word_embs = self.embeddings(wordIDs)
      word_embs = self.emb_dropout(word_embs)
      
      # run the encoder BiRNN
      packed_input = pack_padded_sequence(word_embs, seq_len, batch_first=True)
      output, _ = self.rnn(packed_input)
      states, _ = pad_packed_sequence(output, batch_first=True)
         
      # run additional deep RNN layers with residual connections (if present)
      for rnn in self.deep_rnns:
         packed_input = pack_padded_sequence(self.dropout(states), seq_len, batch_first=True)
         output, _ = rnn(packed_input)
         output, _ = pad_packed_sequence(output, batch_first=True)
         states = states + output  # residual connections

      return self.dropout(states)


### Attention #################################################################

class Attention(NNModule):

   def __init__(self, enc_rnn_size, dec_rnn_size):
      
      super(Attention, self).__init__()

      self.projection = nn.Linear(enc_rnn_size*2+dec_rnn_size, dec_rnn_size)
      self.final_weights = nn.Parameter(torch.randn(dec_rnn_size))


   def forward(self, enc_states, dec_state, src_len=None):

      # replicate dec_state along a new (sentence length) dimension
      exp_dec_states = dec_state.unsqueeze(1).expand(-1,enc_states.size(1),-1)

      # replicate enc_state along the first (batch) dimension if it is 1
      # needed during beam search
      exp_enc_states = enc_states.expand(dec_state.size(0),-1,-1)

      # append the decoder state to each encoder state
      input = torch.cat((exp_enc_states, exp_dec_states), dim=-1)
      
      # apply an MLP layer
      proj_input = torch.tanh(self.projection(input))
      
      # single-neuron output layer
      scores = torch.matmul(proj_input, self.final_weights) / \
               math.sqrt(self.final_weights.size(0))

      if src_len:  # not beam search
         # mask all padding positions
         mask = [[0]*l + [-float('inf')]*(enc_states.size(1)-l) for l in src_len]
         mask = Variable(torch.Tensor(mask))
         if self.on_gpu():
            mask = mask.cuda()
         scores = scores + mask
      
      # softmax across all encoder positions
      attn_probs = F.softmax(scores, dim=-1)
      
      # weighted average of encoder representations
      enc_context = torch.sum(enc_states*attn_probs.unsqueeze(2), dim=1)
      
      return enc_context

         
### Decoder ###################################################################

class NMTDecoder(NNModule):

   def __init__(self, src_vocab_size, tgt_vocab_size, word_emb_size,
                enc_rnn_size, dec_rnn_size, enc_depth, dec_depth,
                dropout_rate, emb_dropout_rate, use_lstm=False,
                tie_embeddings=True, src_embeddings=None,
                tgt_embeddings=None):
      ''' intialize the model before training starts '''
      
      super(NMTDecoder, self).__init__()

      self.use_lstm = use_lstm
      self.tie_embeddings = tie_embeddings
      self.dec_rnn_size = dec_rnn_size
      self.dec_depth = dec_depth
      
      self.encoder = Encoder(src_vocab_size, word_emb_size, enc_rnn_size, enc_depth,
                             dropout_rate, emb_dropout_rate, self.use_lstm, src_embeddings)
      self.attention = Attention(enc_rnn_size, dec_rnn_size)

      if tgt_embeddings is None:
         self.tgt_embeddings = nn.Embedding(tgt_vocab_size, word_emb_size)
      else:
         self.tgt_embeddings = nn.Embedding.from_pretrained(tgt_embeddings)
         self.tie_embeddings = True

      rnnCell = nn.LSTMCell if self.use_lstm else nn.GRUCell
      self.dec_rnn = rnnCell(word_emb_size+enc_rnn_size*2, dec_rnn_size)
      self.deep_dec_rnns = nn.ModuleList(
         [rnnCell(dec_rnn_size, dec_rnn_size) for _ in range(1, dec_depth)]
      )

      self.dropout = nn.Dropout(dropout_rate)
      self.emb_dropout = nn.Dropout(emb_dropout_rate)

      if self.tie_embeddings:
         self.output_proj = nn.Linear(dec_rnn_size, word_emb_size)
         self.output_layer = nn.Linear(word_emb_size, tgt_vocab_size)
         self.output_layer.weight = self.tgt_embeddings.weight
      else:
         self.output_layer = nn.Linear(dec_rnn_size, tgt_vocab_size)
      

   def finetune_embeddings(self, flag=True):
      self.tgt_embeddings.weight.requires_grad = flag
      self.encoder.embeddings.weight.requires_grad = flag

   
   def init_decoder(self, src_wordIDs, src_len):
      src_wordIDs = self.long_tensor(src_wordIDs)
      enc_states  = self.encoder(src_wordIDs, src_len)

      # initialize the decoder state
      batch_size = enc_states.size(0)
      init_state =  self.zero_tensor(batch_size, self.dec_rnn_size)
      if self.use_lstm:
         init_state = (init_state, init_state)
      dec_rnn_states = [init_state for _ in range(self.dec_depth)]

      return enc_states, dec_rnn_states
   
   
   def decoder_step(self, prev_word_embs, enc_states, dec_rnn_states, src_len=None):
      ''' run a single decoder step '''

      # compute the source context vector
      hidden_state = dec_rnn_states[-1][0] if self.use_lstm else dec_rnn_states[-1]
      enc_context = self.attention(enc_states, hidden_state, src_len)
      enc_context = self.dropout(enc_context)
      
      # run the decoder RNN
      dec_input = torch.cat((prev_word_embs, enc_context), dim=-1)
      dec_rnn_states[0] = self.dec_rnn(dec_input, dec_rnn_states[0])
      dec_input = dec_rnn_states[0][0] if self.use_lstm else dec_rnn_states[0]

      # run additional deep decoder RNN layers with residual connections
      for i, rnn in enumerate(self.deep_dec_rnns):
         dec_rnn_states[i] = rnn(self.dropout(dec_input), dec_rnn_states[i])
         output = dec_rnn_states[i][0] if self.use_lstm else dec_rnn_states[i]
         dec_input = dec_input + output  # residual connections

      return dec_input, dec_rnn_states

   
   def compute_scores(self, hidden_states):
      hidden_states = self.dropout(hidden_states)
      if True:  #self.tie_embeddings:
         hidden_states = self.output_proj(hidden_states)
      return self.output_layer(hidden_states)

   
   def forward(self, src_wordIDs, src_len, tgt_wordIDs):
      ''' forward pass of the network during training and evaluation on dev data '''

      self.train(True)

      enc_states, dec_rnn_states = self.init_decoder(src_wordIDs, src_len)
      
      # look up the target word embeddings
      tgt_word_embs = self.tgt_embeddings(tgt_wordIDs)
      tgt_word_embs = self.emb_dropout(tgt_word_embs)

      hidden_states = []
      for i in range(tgt_word_embs.size(1)):
         hidden_state, dec_rnn_states = self.decoder_step(
            tgt_word_embs[:,i,:], enc_states,dec_rnn_states, src_len)
         hidden_states.append(hidden_state)
         
      hidden_states = torch.stack(hidden_states, dim=1)
      scores = self.compute_scores(hidden_states)

      return scores


   ### Translation ########################

   def translate(self, src_wordIDs, src_len, beam_size=0):
      ''' forward pass of the network during translation '''

      self.train(False)

      if beam_size > 0:
         return self.beam_translate(src_wordIDs, src_len, beam_size)

      enc_states, dec_rnn_states = self.init_decoder(src_wordIDs, src_len)

      tgt_wordIDs = []
      prev_wordIDs = self.zero_long_tensor(len(src_wordIDs))
      
      # target sentences may have twice the size of the source sentences plus 5
      for i in range(src_len[0]*2+5):

         # compute scores for the next target word candidates
         tgt_word_embs = self.tgt_embeddings(prev_wordIDs)
         hidden_state, dec_rnn_states = self.decoder_step(tgt_word_embs, enc_states, dec_rnn_states, src_len)
         scores = self.compute_scores(hidden_state)

         # extract the most likely target word for each sentence
         _, best_wordIDs = scores.topk(1, dim=-1)
         best_wordIDs = best_wordIDs.squeeze(1)
         
         tgt_wordIDs.append(best_wordIDs)
         prev_wordIDs = best_wordIDs
         lp = F.log_softmax(scores, dim=-1)
            
         # stop if all output symbols are boundary/padding symbols
         if (best_wordIDs == 0).all():
            break

      return torch.stack(tgt_wordIDs).t().cpu().data.numpy()


   ### Translation with beam decoding ########################

   def build_beam(self, logprobs, beam_size, hidden_states, cell_states):

      # get the threshold which needs to be exceeded by the hypotheses
      top_logprobs, _ = logprobs.view(-1).topk(beam_size+1)
      threshold = top_logprobs[-1]

      # extract the most likely extensions for each hypothesis
      top_logprobs, top_wordIDs = logprobs.topk(beam_size, dim=-1)

      # extract the most likely extended hypotheses overall
      new_wordIDs = []
      new_logprobs = []
      new_hidden_states = []
      if cell_states is not None:
         new_cell_states = []
      prev_pos = []
      for i in range(top_logprobs.size(0)):
         for k in range(top_logprobs.size(1)):
            if (top_logprobs[i,k] > threshold).all(): # without all() it doesn't work
               new_wordIDs.append(top_wordIDs[i,k])
               prev_pos.append(i)
               new_hidden_states.append(hidden_states[i])
               if cell_states is not None:
                  new_cell_states.append(cell_states[i])
               new_logprobs.append(top_logprobs[i,k])
      new_wordIDs = torch.stack(new_wordIDs).squeeze(1)
      new_logprobs = torch.stack(new_logprobs).squeeze(1)
      hidden_states = torch.stack(new_hidden_states)
      if cell_states is not None:
         cell_states = torch.stack(new_cell_states)

      return new_wordIDs, new_logprobs, hidden_states, cell_states, prev_pos

   
   def beam_translate(self, src_wordIDs, src_len, beam_size):
      ''' processes a single sentence with beam search '''
      
      enc_states, dec_rnn_states = self.init_decoder(src_wordIDs, src_len)
      
      tgt_wordIDs = []
      prev_pos = []
      prev_wordIDs = self.zero_long_tensor(1)
      prev_logprobs = self.zero_tensor(1)
      
      # target sentences have at most twice the size of the source sentences plus 5
      for i in range(src_wordIDs.size(1)*2+5):

         # compute scores for the next target word candidates
         tgt_word_embs = self.tgt_embeddings(prev_wordIDs)
         hidden_state, dec_rnn_states = self.decoder_step(tgt_word_embs, enc_states, dec_rnn_states, src_len)
         scores = self.compute_scores(hidden_state)

         # add the current logprob to the logprob of the previous hypothesis
         logprobs = prev_logprobs.unsqueeze(1) + F.log_softmax(scores, dim=-1)

         # extract the best hypotheses
         best_wordIDs, prev_logprobs, dec_rnn_states, cell_state, prev \
            = self.build_beam(logprobs, beam_size, dec_rnn_states, cell_state)

         # store information for computing the best translation at the end
         tgt_wordIDs.append(best_wordIDs.cpu().data.numpy().tolist())
         prev_pos.append(prev)
         prev_wordIDs = best_wordIDs
         
         # stop if all output symbols are boundary/padding symbols
         if (best_wordIDs == 0).all():
            break

      # extract the best translation
      # get the position of the most probable hypothesis
      _, pos = prev_logprobs.topk(1)
      pos = int(pos)

      # extract the best translation backward using prev_pos
      wordIDs = []
      for i in range(len(prev_pos)-1,0,-1):
         pos = prev_pos[i][pos]
         wordIDs.append(tgt_wordIDs[i-1][pos])
      wordIDs.reverse()

      return [wordIDs]
