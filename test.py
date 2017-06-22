# -*- coding: utf-8 -*- 

import argparse
import math
import sys
import time
import random
import pickle

import numpy as np

from rnnlm import RNNLM
from chainer import Function, Variable, optimizers, serializers, utils


# list of numpy.ndarray
def load_data(filename):
    vocab = {}
    vocab["<eos>"] = 0
    v_count = 1
    data = []
    
    with open(filename, "r") as f:
        sents = []
        for line in f:
            words = line.split()
            words[-1] = "<eos>"
            sent = np.ndarray((len(words)), dtype=np.int32)
            for i, word in enumerate(words):
                if word not in vocab:
                    vocab[word] = v_count
                    v_count += 1
                sent[i] = vocab[word]
            sents.append(sent)
    return sents, vocab

def create_minibatch(data, batch_size=100, shuffle=True):
    assert type(data) == list
    assert type(data[0]) == np.ndarray
    
    if shuffle:
        random.shuffle(data)
    
    batches = []
    splitted = [data[i:i + batch_size]for i in range(0, len(data), batch_size)]
    
    for batch_l in splitted:
        max_len = max(map(len, batch_l))
        # -1で初期化したバッチサイズx最も長い文の長さの行列を作る
        batch = np.ndarray((len(batch_l), max_len), dtype=np.int32)
        batch.fill(-1)
        # batch_lの値をbatchにコピー
        for i, row in enumerate(batch):
            row[:len(batch_l[i])] = batch_l[i]
        batches.append(batch)
    return batches

train, vocab = load_data("./RNNLM_Chainer/ptb.test.txt")
batches = create_minibatch(train, batch_size=100, shuffle=True)


