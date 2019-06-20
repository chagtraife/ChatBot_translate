import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import random
import os
import io
import string
import re
from sklearn.model_selection import train_test_split

# dùng eager mode trong tensor    
tf.enable_eager_execution()

def preprocess(sentence):
    sent = sentence.lower()
    sent = sent.strip()
    sent = re.sub("'", " ", sent)
    sent = re.sub("\s+", " ", sent)
    sent = ''.join([char for char in sent if char not in exclude])
    sent = "<start> " + sent + " <end>"
    return sent
    
# load data
en_filename = "./dataset/train.en.txt"
vi_filename = "./dataset/train.vi.txt"
raw_en_lines = open(en_filename, encoding='utf-8').read().strip().split("\n")
raw_vi_lines = open(vi_filename, encoding='utf-8').read().strip().split("\n")
exclude = list(string.punctuation) + list(string.digits)

en_lines = []
vi_lines = []
min_len, max_len = 10, 14

# với mục đích demo, mình sẽ không xử lí những câu quá dài,
# loại bỏ những câu có độ dài không nằm trong khoảng (min_len, max_len)
for eline, vline in zip(raw_en_lines, raw_vi_lines):
    eline = preprocess(eline)
    vline = preprocess(vline)
    if min_len < len(eline.split(" ")) < max_len
        and min_len < len(vline.split(" ")) < max_len:
        en_lines.append(eline)
        vi_lines.append(vline)


class Language():
    def __init__(self, lines):
        self.lines = lines
        self.word2id = {}
        self.id2word = {}
        self.vocab = set()
        self.max_len = 0
        self.min_len = 0
        self.vocab_size = 0
        self.init_language_params()

    def init_language_params(self):
        for line in self.lines:
            self.vocab.update(line.split(" "))
        self.word2id['<pad>'] = 0
        for id, word in enumerate(self.vocab):
            self.word2id[word] = id + 1
        for word, id in self.word2id.items():
            self.id2word[id] = word
        self.max_len = max([len(line.split(" ")) for line in self.lines])
        self.min_len = min([len(line.split(" ")) for line in self.lines])
        self.vocab_size = len(self.vocab) + 1
            
    def sentence_to_vector(self, sent):
        return np.array([self.word2id[word] for word in sent.split(" ")])
            
    def vector_to_sentence(self, vector):
        return " ".join([self.id2word[id] for id in vector])



inp_lang = Language(en_lines)
tar_lang = Language(vi_lines)
inp_vector = [inp_lang.sentence_to_vector(line) for line in inp_lang.lines]
tar_vector = [tar_lang.sentence_to_vector(line) for line in tar_lang.lines]

# add padding vào câu để tất cả các câu có độ dài bằng nhau
inp_tensor = tf.keras.preprocessing.sequence.pad_sequences(inp_vector, inp_lang.max_len, padding='post')
tar_tensor = tf.keras.preprocessing.sequence.pad_sequences(tar_vector, tar_lang.max_len, padding='post')
x_train, x_val, y_train, y_val = train_test_split(inp_tensor, tar_tensor, test_size=0.1)

BATCH_SIZE = 32
BUFFER_SIZE = x_train.shape[0]
N_BATCH = BUFFER_SIZE//BATCH_SIZE
hidden_unit = 1024
embedding_size = 256

dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
dataset = dataset.batch(BATCH_SIZE)



class Encode(tf.keras.Model):
    def __init__(self, embedding_size, vocab_size, hidden_units):
        super(Encode, self).__init__()
        self.Embedding = tf.keras.layers.Embedding(vocab_size,embedding_size)
        self.GRU = tf.keras.layers.GRU(
            hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform')
        self.hidden_units = hidden_units
        
    def call(self, x, hidden_state):
        x = self.Embedding(x)
        outputs, last_state = self.GRU(x, hidden_state)
        return outputs, last_state
    
    def init_hidden_state(self, batch_size):
        return tf.zeros([batch_size, self.hidden_units])


class Attention(tf.keras.Model):
    def __init__(self, hidden_units):
        super(Attention, self).__init__()
        self.W_out_encode = tf.keras.layers.Dense(hidden_unit)
        self.W_state = tf.keras.layers.Dense(hidden_unit)
        self.V = tf.keras.layers.Dense(1)
        
    def call(self, encode_outs, pre_state):
        pre_state = tf.expand_dims(pre_state, axis=1)
        pre_state = self.W_state(pre_state)
        encode_outs = self.W_out_encode(encode_outs)
        score = self.V(
            tf.nn.tanh(
                pre_state + encode_outs)
        )
        score = tf.nn.softmax(score, axis=1)
        context_vector = score*encode_outs
        context_vector = tf.reduce_sum(context_vector, axis=1)
        return context_vector, score


class Decode(tf.keras.Model):
    def __init__(self, vocab_size, embedding_size, hidden_units):
        super(Decode, self).__init__()
        self.hidden_units = hidden_units
        self.Embedding = tf.keras.layers.Embedding(vocab_size,embedding_size)
        self.Attention = Attention(hidden_units)
        self.GRU = tf.keras.layers.GRU(
            hidden_units,
            return_sequences=True,
            return_state=True,
            recurrent_initializer='glorot_uniform'
        )
        self.Fc = tf.keras.layers.Dense(vocab_size)
            
    def call(self, x, encode_outs, pre_state):
        x = tf.expand_dims(x, axis=1)
        x = self.Embedding(x)
        context_vector, attention_weight = self.Attention(encode_outs, pre_state)
        context_vector = tf.expand_dims(context_vector, axis=1)
        gru_inp = tf.concat([x, context_vector], axis=-1)
        out_gru, state = self.GRU(gru_inp)
        out_gru = tf.reshape(out_gru, (-1, out_gru.shape[2]))
        return self.Fc(out_gru), state


def loss_function(real, pred):
    mask = 1 - np.equal(real, 0)
    loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
    return tf.reduce_mean(loss_)