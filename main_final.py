# %%
import os
from re import TEMPLATE
import torch
import tensorflow as tf
import glob
import struct
import numpy as np
import torch.nn as nn
from numpy import random
import logging
import pyrouge
from queue import Queue
from queue import Empty
from torch.optim import Adagrad
from tensorflow.core.example import example_pb2
import torch.nn.functional as Functional
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.nn.utils import clip_grad_norm_
import io
# %%
vocab_path = "finished_files/vocab"
vocab_size = 50000
batch_queue_max = 40000
train_data_path = "finished_files/chunked/test_000*"
val_data_path = "finished_files/chunked/val_*"
test_data_path = "finished_files/chunked/test_*"
decode_path = "decode/"
batch_size_max = 8
UNKNOWN_TOKEN = '[UNK]'
START_DECODING = '[START]' 
STOP_DECODING = '[STOP]' 
PAD_DECODING = '[PAD]'
SENTENCE_START = '<s>'
SENTENCE_END = '</s>'
max_enc_steps=400
min_dec_steps=35
max_dec_steps=100
data_read_finish = 0
embedding_dim = 128
hidden_dim = 256
trunc_norm_init_std=1e-4
lr_init = 0.15
adagrad_init_acc=0.1
beam_size = 4

random.seed(123)
torch.manual_seed(123)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(123)
    print "use CUDA"
device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
# device = 'cpu'
# %%
# vocabulary
word_to_id_dict = {'[UNK]':0, '[PAD]':1, '[START]':2, '[STOP]':3}
id_to_word_dict = {0:'[UNK]', 1:'[PAD]', 2:'[START]', 3:'[STOP]'}
# vocab_f = io.open(vocab_path, 'r', encoding='utf-8')
cnt = 4
with open(vocab_path, 'r') as vocab_f:
  for line in vocab_f:
      l = line.split()
      if len(l)!=2: continue
      w = l[0]
      word_to_id_dict[w] = cnt
      id_to_word_dict[cnt] = w
      cnt +=1
      if cnt >= vocab_size: break
print "finish", cnt, "vocabs"

def word_to_id(word):
    if word not in word_to_id_dict:
        return word_to_id_dict[UNKNOWN_TOKEN]
    else:
        return word_to_id_dict[word]

def id_to_word(id):
    if id < vocab_size:
        return id_to_word_dict[id]
    else:
        # print("id not in vocab")
        return '[UNK]'


# %%
# article: words to ids
def article_to_ids(words):
  ids, oovs = [], []
  for w in words:
    if word_to_id(w) == word_to_id(UNKNOWN_TOKEN): # give article oovs ids
      if w not in oovs:
        oovs.append(w)
      ids.append(len(word_to_id_dict) + oovs.index(w))
    else:
      ids.append(word_to_id(w))
  return ids, oovs
  
# abstract: words to ids
def abstract_to_ids(words, article_oovs):
  ids = []
  for w in words:
    if word_to_id(w) == word_to_id(UNKNOWN_TOKEN):
      if w in article_oovs:                       # give article oovs ids
        ids.append(len(word_to_id_dict) + article_oovs.index(w))
      else:                                       # give other oovs UNK
        ids.append(word_to_id(UNKNOWN_TOKEN))
    else:
      ids.append(word_to_id(w))
  return ids

# abstract to decoder input / target
def seq_to_dec_input_target(seq, seq_with_oov, max_len):
  start_id = word_to_id(START_DECODING)
  stop_id = word_to_id(STOP_DECODING)

  input = [start_id] + seq[:]         # add START to decoder_input
  target = seq_with_oov[:]
  if len(input) > max_len:            # truncate
    input = input[:max_len]
    target = target[:max_len]
  else:
    target.append(stop_id)          # add STOP to target
  return input, target


class Example(object):
  def __init__(self, article, abstract_sentences):
  # article part
    article_words = article.split()
    if len(article_words) > max_enc_steps:  # truncate
      article_words = article_words[:max_enc_steps]
    self.enc_input = [word_to_id(w) for w in article_words]
    self.enc_input_with_oovs, self.article_oovs = article_to_ids(article_words)
    self.enc_len = len(article_words) 
    

  # abstract part
    abstract = ' '.join(abstract_sentences)
    abstract_words = abstract.split()
    abstract_ids = [word_to_id(w) for w in abstract_words]
    abstract_ids_with_oovs = abstract_to_ids(abstract_words, self.article_oovs)
     
    self.dec_input, self.target = seq_to_dec_input_target(abstract_ids, abstract_ids_with_oovs, max_dec_steps)
    self.dec_len = len(self.dec_input)

  # Store the original strings
    self.original_article = article
    self.original_abstract = abstract
    self.original_abstract_sents = abstract_sentences


class Batch(object):
  def __init__(self, examples, batch_size):   
  # encoder input
    max_enc_len = max(example.enc_len for example in examples)
    # padding
    for example in examples:
      while len(example.enc_input) < max_enc_len:
        example.enc_input.append(word_to_id(PAD_DECODING))
      while len(example.enc_input_with_oovs) < max_enc_len:
        example.enc_input_with_oovs.append(word_to_id(PAD_DECODING))
      
    # batch array
    self.enc_batch = np.zeros((batch_size, max_enc_len), int)
    self.enc_lens = np.zeros(batch_size, int)
    self.enc_padding_mask = np.zeros((batch_size, max_enc_len), int)
    
    for i, example in enumerate(examples):
      self.enc_batch[i, :] = example.enc_input[:]
      self.enc_lens[i] = example.enc_len
      for j in range(example.enc_len):
        self.enc_padding_mask[i, j] = 1
      
    # article oovs
    self.max_article_oocs = max([len(example.article_oovs) for example in examples])
    self.article_oovs = [example.article_oovs for example in examples]
    self.enc_batch_with_oovs = np.zeros((batch_size, max_enc_len), int)
    for i, example in enumerate(examples):
      self.enc_batch_with_oovs[i, :] = example.enc_input_with_oovs[:]
    

  # decoder input
    # padding
    for example in examples:
      while len(example.dec_input) < max_dec_steps:
        example.dec_input.append(word_to_id(PAD_DECODING))
      while len(example.target) < max_dec_steps:
        example.target.append(word_to_id(PAD_DECODING))
    # batch array
    self.dec_batch = np.zeros((batch_size, max_dec_steps), int)
    self.target_batch = np.zeros((batch_size, max_dec_steps), int)
    self.dec_lens = np.zeros(batch_size, int)
    self.dec_padding_mask = np.zeros((batch_size, max_dec_steps), int)
    
    for i, example in enumerate(examples):
      self.dec_batch[i, :] = example.dec_input[:]
      self.target_batch[i, :] = example.target[:]
      self.dec_lens[i] = example.dec_len
      for j in range(example.dec_len):
        self.dec_padding_mask[i, j] = 1
    
    self.original_abstract = [example.original_abstract for example in examples]


# %%
data_read_finish = 0
# read_article_num = 800

def bytes_to_sentence(abstract):
  cur = 0
  sents = []
  while True:
    try:
      start = abstract.index(SENTENCE_START.encode(), cur)
      end = abstract.index(SENTENCE_END.encode(), start + 1)
      cur = end + len(SENTENCE_END)
      sents.append(abstract[start+len(SENTENCE_START) : end].decode())
    except ValueError as e:
      return sents

def text_generator(example_generator):
    global data_read_finish
    # cnt = 0
    while True:
      e = example_generator.next()
      try:
        article_text = e.features.feature['article'].bytes_list.value[0] 
        abstract_text = e.features.feature['abstract'].bytes_list.value[0]
        
      except StopIteration:
        if data_read_finish:
          return
        print("Failed to get article or abstract from example")   
        continue
      if len(article_text)==0:
        continue
      else:
        try:
          yield (article_text, abstract_text)
        except StopIteration:
          return
      if data_read_finish: return 

def example_generator(data_path):

  global data_read_finish
  filelist = glob.glob(data_path)
  filelist = sorted(filelist)
  # print("file_list:", filelist)
  for file in filelist:
    f = open(file, 'rb')
    print "file:", file
    flag = 1
    while flag:
      len_bytes = f.read(8)
      if not len_bytes: 
        break
      else:
        str_len = struct.unpack('q', len_bytes)[0]
        example_str = struct.unpack('%ds' % str_len, f.read(str_len))[0]
        yield example_pb2.Example.FromString(example_str)
  data_read_finish = 1


# turn data to example, and put example to example_queue
example_queue = Queue(Queue())
example_queue.maxsize = batch_queue_max*batch_size_max
input_gen = text_generator(example_generator(train_data_path))
while True:
  try:
    (article, abstract) = input_gen.next() 
    
  except StopIteration:
    break

  article = article.decode('utf-8')
  abstract_sentences = [sent.strip() for sent in bytes_to_sentence(abstract)]
  example = Example(article, abstract_sentences) 
  example_queue.put(example)

# turn examples to batch, and put batch to batch_queue
batch_queue = Queue(Queue())
test_batch_queue = Queue(Queue())
batch_queue.maxsize = batch_queue_max
test_batch_queue.maxsize = batch_queue_max
test_batches = []
batch_num = 0
batch_empty_flag = 0
while True:
  batch = []
  for i in range(batch_size_max):
    if example_queue.empty():
      batch_empty_flag = 1
      break
    batch.append(example_queue.get())
  if not batch: # empty batch
    break
  if not batch_empty_flag:
    batch = sorted(batch, key=lambda example: example.enc_len, reverse=True) # sort by encoder seq length(long to short)
    batch_queue.put(Batch(batch, len(batch)))
    batch_num+=1

print "batch_num:", batch_num



# %%
def init_lstm_wt(lstm):
    for names in lstm._all_weights:
        for name in names:
            if name.startswith('weight_'):
                wt = getattr(lstm, name)
                wt.data.uniform_(-0.02, 0.02)
            elif name.startswith('bias_'):
                # set forget bias to 1
                bias = getattr(lstm, name)
                n = bias.size(0)
                start, end = n // 4, n // 2
                bias.data.fill_(0.)
                bias.data[start:end].fill_(1.)
def init_linear_wt(linear):
    linear.weight.data.normal_(std=1e-4)
    if linear.bias is not None:
        linear.bias.data.normal_(std=1e-4)

#------ Model ------
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.normal_(std=1e-4)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=True)
        init_lstm_wt(self.lstm)
        self.w_h = nn.Linear(hidden_dim*2, hidden_dim*2, bias=False)
    
    def forward(self, input, seq_lens):
        emb = self.embedding(input)

        emb_packed = pack_padded_sequence(emb, seq_lens, batch_first=True)
        output, encoder_hidden = self.lstm(emb_packed)
        encoder_output, _ = pad_packed_sequence(output, batch_first=True)
        
        encoder_f = encoder_output.contiguous().view(-1, hidden_dim*2)
        encoder_feature = self.w_h(encoder_f)
        
        return encoder_output, encoder_feature, encoder_hidden

class Attention(nn.Module):
    def __init__(self):
        super(Attention, self).__init__()

        self.w_c = nn.Linear(1, hidden_dim*2, bias=False)
        self.w_s = nn.Linear(hidden_dim*2, hidden_dim*2)
        self.v = nn.Linear(hidden_dim*2, 1, bias=False)
    
    def forward(self, decoder_state, encoder_outputs, encoder_feature, enc_padding_mask, coverage):
        b, t_k, n = list(encoder_outputs.size())
        decoder_f = self.w_s(decoder_state)
        decoder_feature = decoder_f.unsqueeze(1).expand(b, t_k, n).contiguous() 
        decoder_feature = decoder_feature.view(-1, n)
        coverage_feature = self.w_c(coverage.view(-1, 1))  # B*t_k x 1 => B*t_k x n
        attn_features = encoder_feature + decoder_feature + coverage_feature    # B*t_k x n
        
        e = torch.tanh(attn_features)
        scores = self.v(e).view(-1, t_k)   # B x t_k

        attn = Functional.softmax(scores, dim=1)
        normalization_factor = attn.sum(1, keepdim=True)
        attn_dist = attn / normalization_factor

        
        attn = attn.view(-1, t_k)  # B x t_k
        coverage = coverage.view(-1, t_k) # B x t_k
        coverage = coverage + attn

        return attn, coverage


# %%
class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.embedding.weight.data.normal_(std=1e-4)
        self.input_linear = nn.Linear(2*hidden_dim + embedding_dim, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=1, batch_first=True, bidirectional=False)
        init_lstm_wt(self.lstm)
        self.attention = Attention()
        self.p_gen_linear = nn.Linear(4*hidden_dim + embedding_dim, 1)
        self.vocab_mapping = nn.Linear(3*hidden_dim, vocab_size)
        init_linear_wt(self.vocab_mapping)

    def forward(self, enc_outputs, enc_feature, enc_padding_mask, input, context_vec, coverage, enc_batch_with_oovs, dec_hidden_0, steps, training):
        if not training and steps == 0:
            h_dec, c_dec = dec_hidden_0
            dec_state = torch.cat((h_dec.view(-1, hidden_dim), c_dec.view(-1, hidden_dim)), 1)
            attn, coverage = self.attention(dec_state, enc_outputs, enc_feature, enc_padding_mask, coverage)
        
        
        # lstm input: context vector and decoder input embed
        emb = self.embedding(input)
        dec_input = self.input_linear(torch.cat((context_vec, emb), 1))
        # lstm
        lstm_output, dec_hidden = self.lstm(dec_input.unsqueeze(1), dec_hidden_0) # t_k x B x hidden_dim
        # attention
        h_dec, c_dec = dec_hidden
        dec_state = torch.cat((h_dec.view(-1, hidden_dim), c_dec.view(-1, hidden_dim)), 1)
        attn, coverage_tmp = self.attention(dec_state, enc_outputs, enc_feature, enc_padding_mask, coverage)
        if training or steps != 0:
            coverage = coverage_tmp
        
        # context vector: h*
        context_vec = torch.bmm(attn.unsqueeze(1), enc_outputs)  # B x 1 x 2*hidden_dim
        context_vec = context_vec.view(-1, 2*hidden_dim)   # B x 2*hidden_dim
        p_gen_input = torch.cat((context_vec, dec_state, dec_input), 1)  # B x (2*2*hidden_dim + emb_dim)
        p_gen = self.p_gen_linear(p_gen_input)
        p_gen = torch.sigmoid(p_gen)

        # P_vocab: lstm_output(decoder state), context_vec (totol dim: 3*hid)
        output = torch.cat((lstm_output.view(-1, hidden_dim), context_vec), 1) # B*t_k x 3*hidden_dim
        output = self.vocab_mapping(output)
        p_vocab = Functional.softmax(output, dim=1)
        vocab_distribution = p_gen.to(device) * p_vocab
        attn_distribution = ((1-p_gen)* attn)
        max_oov_id = torch.max(enc_batch_with_oovs)
        

        vocab_distribution_extend = Functional.pad(input=vocab_distribution, pad=(0, max_oov_id-len(word_to_id_dict)+1, 0, 0), mode='constant', value=1e-7).to(device)
        final_distribution = (vocab_distribution_extend.scatter_add(1, enc_batch_with_oovs, attn_distribution))
        
        return final_distribution, dec_hidden, context_vec, p_gen, coverage, attn_distribution

class ReduceState(nn.Module):
    def __init__(self):
        super(ReduceState, self).__init__()

        self.reduce_h = nn.Linear(2*hidden_dim, hidden_dim)
        init_linear_wt(self.reduce_h)
        self.reduce_c = nn.Linear(2*hidden_dim, hidden_dim)
        init_linear_wt(self.reduce_c)

    def forward(self, encoder_hidden):
        h, c = encoder_hidden # h: 2 x B x hidden_dim
        h_input = h.transpose(0, 1).contiguous().view(-1, 2*hidden_dim)
        c_input = c.transpose(0, 1).contiguous().view(-1, 2*hidden_dim)
        h_reduced = Functional.relu(self.reduce_h(h_input))
        c_reduced = Functional.relu(self.reduce_c(c_input))
        return (h_reduced.unsqueeze(0), c_reduced.unsqueeze(0)) # h: 1 x B x hidden_dim

class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.encoder = Encoder().to(device)
        self.decoder = Decoder().to(device)
        self.reduce_state = ReduceState().to(device)
        self.decoder.embedding.weight = self.encoder.embedding.weight

# %%
# ------ training setup ------
model_path = '/models'
model = Model()

params = list(model.encoder.parameters()) + list(model.decoder.parameters()) + list(model.reduce_state.parameters())
optimizer = Adagrad(params, lr=lr_init, initial_accumulator_value=adagrad_init_acc)

optimizer.load_state_dict(optimizer.state_dict())
for state in optimizer.state.values():
    for k, v in state.items():
        if torch.is_tensor(v):
            state[k] = v.to(device)

# %%
#------ train ------
# training iteration
training_iteration = 19
i = 19

state_dict = torch.load(str(i-1)+"_c.pth", map_location='cpu')
model.load_state_dict(state_dict)
model.to(device) 
# model.load_state_dict(torch.load(str(i-1)+"_c.pth"))
while i < training_iteration:
    batch_i = 0
    while(batch_i<batch_num):
        if batch_queue.qsize()==0:
            print("batch queue is empty")
            break
        
        batch = batch_queue.get()
        
        optimizer.zero_grad()
        # info from batch
        enc_batch = torch.from_numpy(batch.enc_batch)
        enc_padding_mask = torch.from_numpy(batch.enc_padding_mask)
        enc_lens = torch.from_numpy(batch.enc_lens)
        enc_batch_with_oovs = torch.from_numpy(batch.enc_batch_with_oovs)
        coverage = torch.zeros(enc_batch.size())
        dec_batch = torch.from_numpy(batch.dec_batch)
        target_batch = torch.from_numpy(batch.target_batch)
        context_vec = torch.zeros((batch_size_max, 2 * hidden_dim))
        dec_padding_mask = torch.from_numpy(batch.dec_padding_mask)
        dec_lens = torch.from_numpy(batch.dec_lens)
        max_dec_lens = np.max(batch.dec_lens)
        
        # input seq_len
        encoder_output, encoder_feature, encoder_hidden = model.encoder(enc_batch.to(device), enc_lens.to(device))
        dec_hidden_0 = model.reduce_state(encoder_hidden)
        step_loss_list = []
        for steps in range(min(max_dec_lens, max_dec_steps)):
            dec_input = dec_batch[:, steps]
            target = target_batch[:, steps]
            final_distribution, dec_hidden_0, context_vec, p_gen, coverage, attn_distribution = model.decoder(encoder_output.to(device), encoder_feature.to(device), enc_padding_mask.to(device), dec_input.to(device), context_vec.to(device), coverage.to(device), enc_batch_with_oovs.to(device), dec_hidden_0, steps, training=0)
            final_distribution = final_distribution.to(device)
            target = target.to(device)
            gold_probs = torch.gather(final_distribution, 1, target.unsqueeze(1).type(torch.int64)).squeeze().to(device)
            step_loss = -torch.log(gold_probs + 1e-12).to(device)
            step_coverage_loss = torch.sum(torch.min(attn_distribution, coverage.to(device)), 1)
            step_loss += step_coverage_loss
            step_mask = dec_padding_mask[:, steps].to(device)
            step_loss *= step_mask
            step_loss_list.append(step_loss)
        sum_losses = torch.sum(torch.stack(step_loss_list, 1), 1)
        batch_avg_loss = sum_losses/dec_lens.to(device)
        loss = torch.mean(batch_avg_loss).to(device)
        loss.backward()

        clip_grad_norm_(model.encoder.parameters(), 2.0)
        clip_grad_norm_(model.decoder.parameters(), 2.0)
        clip_grad_norm_(model.reduce_state.parameters(), 2.0)
        
        optimizer.step()
        
        batch_queue.put(batch)
        if batch_i%50==0:
          print("loss",batch_i,":",  loss.item())
        if batch_i%1000==0:
          torch.save(model, str(i)+"_c.pth")
          print("saved:"+str(i)+"-"+str(batch_i))
        batch_i+=1
        
    torch.save(model.state_dict(), str(i)+"_c.pth")
    print("saved:"+str(i))
    i+=1  


# %%
#------ beam search ------
class Beam(object):
    def __init__(self, tokens, log_probs, state, context, coverage):
        self.tokens = tokens
        self.log_probs = log_probs
        self.state = state
        self.context = context
        self.coverage = coverage

    def add(self, add_token, context, coverage, add_log_prob=0, state=state):
        return Beam(tokens = self.tokens + [add_token], log_probs = self.log_probs + [add_log_prob], state = state, context = context, coverage = coverage)



def beam_search(batch): # 1 example in a batch
    model.encoder = model.encoder.eval()
    model.decoder = model.decoder.eval()
    model.reduce_state = model.reduce_state.eval()

    batch_size = 1
    # init
    enc_batch = torch.from_numpy(batch.enc_batch)
    enc_padding_mask = torch.from_numpy(batch.enc_padding_mask)
    enc_lens = batch.enc_lens
    enc_batch_with_oovs = torch.from_numpy(batch.enc_batch_with_oovs)
    coverage = torch.zeros(enc_batch.size())
    dec_batch = torch.from_numpy(batch.dec_batch)
    target_batch = torch.from_numpy(batch.target_batch)
    context_vec = torch.zeros((batch_size, 2 * hidden_dim))
    dec_padding_mask = torch.from_numpy(batch.dec_padding_mask)
    dec_lens = torch.from_numpy(batch.dec_lens)


    # encoder
    
    enc_output, enc_feature, enc_hidden = model.encoder(enc_batch.to(device), enc_lens)
    dec_hidden_0 = model.reduce_state(enc_hidden)
    beams = [Beam(tokens=[word_to_id(START_DECODING)], log_probs=[0.0], state=dec_hidden_0, context = context_vec, coverage=(coverage))]
    # results = [word_to_id(START_DECODING)]
    results = []
    steps = 0
    flag=1
    while steps < max_dec_steps and len(results)<beam_size:
        
        new_beams = []
        for beam in beams:
          dec_input = torch.as_tensor([beam.tokens[-1] if beam.tokens[-1]<vocab_size else word_to_id('[UNK]')])
          
        # decoder
        # final_distribution, dec_hidden, context_vec, p_gen, coverage, attn_distribution = model.decoder(encoder_output.to(device), encoder_feature.to(device), enc_padding_mask.to(device), dec_input.to(device), context_vec.to(device), coverage.to(device), enc_batch_with_oovs.to(device), dec_hidden_0, steps, training=0)
          final_distribution, dec_hidden, context_vec, p_gen, coverage, attn_distribution = model.decoder(enc_output.to(device), enc_feature, enc_padding_mask, dec_input.to(device), beam.context.to(device), beam.coverage.to(device), enc_batch_with_oovs.to(device), dec_hidden_0, steps, training=1)
          log_probs = torch.log(final_distribution)   # beam_size x vocab_size
          dec_hidden_0 = dec_hidden
          flag=0
          tmp_prob, tmp2 = torch.topk(log_probs, beam_size*2)
          dec_h, dec_c = dec_hidden
          dec_h = dec_h.squeeze()
          dec_c = dec_c.squeeze()
          for i in range(beam_size*2):
            new_beam = beam.add(add_token=tmp2[0][i].item(), add_log_prob=tmp_prob[0][i].item(), context = context_vec, coverage=coverage, state=dec_hidden)
          # print("new_beam_token:", new_beam.tokens)
          
            new_beams.append(new_beam)
        steps += 1
   
        new_beams = sorted(new_beams, key=lambda b: sum(b.log_probs)/len(b.tokens), reverse=True)
 
        beams = []
        for beam in new_beams:
            if beam.tokens[-1] == word_to_id(STOP_DECODING):
                if steps >= min_dec_steps:
                    results.append(beam)
            else:
                beams.append(beam)
            if len(beams) == beam_size or len(results) == beam_size:
                break   
    if len(results)==0: results = beams
    beams = sorted(results, key=lambda b: sum(b.log_probs)/len(b.tokens), reverse=True)
    return beams[0].tokens



#------ test data ------
# turn data to example, and put example to example_queue
data_read_finish = 0
test_example_queue = Queue(Queue())
test_example_queue.maxsize = batch_queue_max*batch_size_max
test_input_gen = text_generator(example_generator(test_data_path))
while True:
  try:
    (article, abstract) = test_input_gen.next() 
    
  except StopIteration:
    break
  article = article.decode('utf-8')
  abstract_sentences = [sent.strip() for sent in bytes_to_sentence(abstract)]
  example = Example(article, abstract_sentences) 
  test_example_queue.put(example)

# turn examples to batch, and put batch to batch_queue
test_batch_queue = Queue(Queue())
test_batch_queue.maxsize = batch_queue_max
test_batches = []
test_batch_num = 0
batch_empty_flag = 0

while True:
  if test_example_queue.empty():
      batch_empty_flag = 1
      break
  else:
    batch = test_example_queue.get()
    b_list = []
    for i in range(1):
      b_list.append(batch)
    test_batch_queue.put(Batch(b_list, 1))
    test_batch_num+=1
print("test_batch_num:", test_batch_num)


#------ save result ------
f = io.open("result.txt", "w")
f.write(str(test_batch_num).decode('utf-8')+"\n")
for i in range(test_batch_num):
    if i%100==0: print(i)
    test_batch = test_batch_queue.get()

    result_id = beam_search(test_batch)
    result_wd = [id_to_word(id) for id in result_id]
    decoder_output = u' '.join(result_wd[1:-1])
    
    f.write(decoder_output+'\n')
    f.write(test_batch.original_abstract[0].decode('utf-8')+'\n')
