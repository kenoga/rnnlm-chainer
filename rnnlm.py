# -*- coding: utf-8 -*- 

import numpy as np
import chainer
from chainer import Function, Variable, optimizers, serializers, utils
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L


class RNNLM(chainer.Chain):
    def __init__(self, vocab_size, hidden_size):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        super(RNNLM, self).__init__(
            w_xh = L.EmbedID(vocab_size, hidden_size),
            w_hh = L.Linear(hidden_size, hidden_size),
            w_hy = L.Linear(hidden_size, vocab_size),
        )

    def __call__(self, s, train=True):
        # assert s[-1] == eos_id
        accum_loss = None
        # 中間層の初期値
        h = Variable(np.zeros((1,self.hidden_size), dtype=np.float32))
        for i in range(len(s) - 1):
            # next_w_id = eos_id if (i == len(s) - 1) else s[i+1]
            next_w_id = s[i+1]
            t = Variable(np.array([next_w_id], dtype=np.int32))
            x_h = self.w_xh(Variable(np.array([s[i]], dtype=np.int32)))
            h_h = self.w_hh(h)
            h = F.tanh(x_h + h_h)
            loss = F.softmax_cross_entropy(self.w_hy(h), t)
            accum_loss = loss if accum_loss is None else accum_loss + loss
        return accum_loss
    
    # 文が与えられたときに次の単語の確率分布を返す
    def predict(self, s):
        h = Variable(np.zeros((1, self.hidden_size), dtype=np.float32))
        for i in range(len(s)):
            x_h = self.w_xh(Variable(np.array([s[i]], dtype=np.int32)))
            h_h = self.w_hh(h)
            h = F.tanh(x_h + h_h)
            y = self.w_hy(h)
        return y
        