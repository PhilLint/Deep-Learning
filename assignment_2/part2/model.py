# MIT License
#
# Copyright (c) 2019 Tom Runia
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to conditions.
#
# Author: Deep Learning Course | Fall 2019
# Date Created: 2019-09-06
################################################################################

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch.nn as nn


class TextGenerationModel(nn.Module):

    def __init__(self, batch_size, seq_length, vocabulary_size,
                 lstm_num_hidden=256, lstm_num_layers=2, device='cuda:0'):

        super(TextGenerationModel, self).__init__()
        # Initialization here...

        self.batch_size = batch_size
        self.seq_length = seq_length
        self.vocabulary_size = vocabulary_size
        self.lstm_num_hidden = lstm_num_hidden
        self.lstm_num_layers = lstm_num_layers
        self.device = device

        # dimension (vocabulary_size, vocabulary_size) to fit the model input
        self.char_embedding = nn.Embedding(vocabulary_size, vocabulary_size)

        self.lstm = nn.LSTM(vocabulary_size, lstm_num_hidden, lstm_num_layers, dropout=1 - dropout_keep_prob,
                            batch_first=True)
        self.linear = nn.Linear(lstm_num_hidden, vocabulary_size)

    def forward(self, x, hidden=None):
        # Implementation here...
        net_out, hidden = self.lstm(self.char_embedding(x), hidden)
        # Transpose because the cross entropy loss wants(minibatch, classes, features)
        return self.linear(net_out).transpose(2, 1), hidden