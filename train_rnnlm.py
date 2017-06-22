# -*- coding: utf-8 -*- 

import sys, time, random, argparse, math, pickle
import numpy as np
from rnnlm import RNNLM
from chainer import optimizers, serializers


def load_data(filename):
    vocab = {}
    with open(filename, "r") as f:
        words = f.read().replace("\n", "<eos>").strip().split()
    dataset = np.ndarray((len(words),), dtype=np.int32)
    for i, word in enumerate(words):
        if word not in vocab:
            vocab[word] = len(vocab)
        dataset[i] = vocab[word]
    return dataset, vocab
    

def train_rnnlm(train, vocab, hidden_size, epoch_num, batch_size):
    # モデル初期化
    train_data, vocab = load_data("./RNNLM_Chainer/ptb.test.txt")
    eos_id = vocab['<eos>']

    model = RNNLM(len(vocab), hidden_size)
    optimizer = optimizers.Adam()
    optimizer.setup(model)

    # TODO:minibatchでできるようにする
    
    # train_dataを文のリストに変換する
    sents = []
    sent = []
    for word_id in train_data:
        sent.append(word_id)
        if word_id == eos_id:
            sents.append(sent)
            sent = []

    # 学習・保存
    for epoch_i in range(epoch_num):
        loss_sum = 0.0
        random.shuffle(sents)
        for i, s in enumerate(sents):
            loss = model(s,train=True)
            loss_sum += loss
            model.zerograds()
            loss.backward()
            optimizer.update()
            if (i % 100 == 0):
                print i, "/", len(sents)," finished"
        print "epoch " + str(epoch_i) + " finished"
        print "average loss is " + str(loss_sum/len(sents))
        outfile = "rnnlm-" + str(epoch_i) + ".model"
        serializers.save_npz(outfile, model)
        loss_sum = 0.0
    

parser = argparse.ArgumentParser(description="A program for training RNNLM. A sentences in corpus must be separated by \\n.")
parser.add_argument("train_file_path",          action="store",                         type=str, help="a train file path.")
parser.add_argument("-e", "--epoch-num",        action="store", default="10",           type=int, help="a epoch size for NN training.")
parser.add_argument("-s", "--hidden-size",      action="store", default=1000,           type=int, help="a hidden size of RNN.")
parser.add_argument("-b", "--batch-size",       action="store", default=100,            type=int, help="a batch size for NN training.")
parser.add_argument("-v", "--vocab-file-name",  action="store", default="vocab.pickle", type=str, help="the name of vocab file.")
args = parser.parse_args()

train, vocab = load_data(args.train_file_path)

with open(args.vocab_file_name, mode='wb') as f:
    pickle.dump(vocab, f)
    
train_rnnlm(train, vocab, args.hidden_size, args.epoch_num, args.batch_size)
