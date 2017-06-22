# -*- coding: utf-8 -*- 

import numpy as np
import argparse
import pickle

from rnnlm import RNNLM
from chainer import serializers


def sent2ids(sent, vocab):
    ids = []
    for w in sent.split():
        id = vocab.get(w, None)
        if id:
            ids.append(id)
        else:
            return None
    return ids   
    

def test_rnnlm(sent, model, vocab, hidden_size):
    vocab_r = {v:k for k, v in vocab.items()}
    ids = sent2ids(sent, vocab)
    if ids:
        y = model.predict(ids).data
        next_w_id = np.argmax(y)
        next_w = vocab_r[next_w_id]
        print next_w
    else:
        print "error"
    pass

    
    
parser = argparse.ArgumentParser(description="A program for testing RNNLM.")
parser.add_argument("sentence",         action="store", type=str, help="a sentence that you want to test.")
parser.add_argument("model_file_path",  action="store", type=str, help="a model file path that you want to test.")
parser.add_argument("vocab_file_path",  action="store", type=str, help="a vocab file used to train the model.")
parser.add_argument("hidden_size",      action="store", type=int, help="a hidden size of RNN.")
args = parser.parse_args()

# train時にpickleしたvocabファイル(word->id)をロード
with open(args.vocab_file_path, "rb") as f:
    vocab = pickle.load(f)
# modelを読み込み
model = RNNLM(len(vocab), args.hidden_size)
serializers.load_npz(args.model_file_path, model)

test_rnnlm(args.sentence, model, vocab, args.hidden_size)
