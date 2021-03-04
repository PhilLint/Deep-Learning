################################################################################
# MIT License
#
# Copyright (c) 2019
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

import torch
import torch.nn as nn

################################################################################

class LSTM(nn.Module):

    def __init__(self, seq_length, input_dim, batch_size, num_hidden, num_classes, device='cpu'):
        super(LSTM, self).__init__()
        # initialize inputs
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = device

        # Initialize weight matrices
        # input modulation gate
        self.W_gx = nn.Parameter(torch.randn((input_dim, num_hidden)), requires_grad=True )
        self.W_gh = nn.Parameter(torch.randn((num_hidden, num_hidden)), requires_grad=True )
        # input gate
        self.W_ix = nn.Parameter(torch.randn((input_dim, num_hidden)),  requires_grad=True )
        self.W_ih = nn.Parameter(torch.randn((num_hidden, num_hidden)), requires_grad=True )
        # forget gate
        self.W_fx = nn.Parameter(torch.randn((input_dim, num_hidden)), requires_grad=True )
        self.W_fh = nn.Parameter(torch.randn((num_hidden, num_hidden)), requires_grad=True )
        # output gate
        self.W_ox = nn.Parameter(torch.randn((input_dim, num_hidden)), requires_grad=True )
        self.W_oh = nn.Parameter(torch.randn((num_hidden, num_hidden)), requires_grad=True )
        # linear output layer
        self.W_ph = nn.Parameter(torch.randn((num_hidden, num_classes)), requires_grad=True )

        # Initialize biases
        self.b_g = nn.Parameter(torch.randn(num_hidden), requires_grad=True )
        self.b_i = nn.Parameter(torch.randn(num_hidden), requires_grad=True )
        self.b_f = nn.Parameter(torch.randn(num_hidden), requires_grad=True )
        self.b_o = nn.Parameter(torch.randn(num_hidden), requires_grad=True )
        self.b_p = nn.Parameter(torch.randn(num_classes), requires_grad=True )


        # initialize hidden and cell states
        self.c = torch.empty(batch_size, num_hidden).to(device)
        self.h = torch.empty(batch_size, num_hidden).to(device)


    def forward(self, x):
        # similar to vanilla RNN: reset hidden and cell state with zeros before forward pass
        self.c = torch.zeros((self.batch_size, self.num_hidden)).to(self.device)
        self.h = torch.zeros((self.batch_size, self.num_hidden)).to(self.device)

        for seq in range(x.shape[1]):
            x_t = x[:, seq].unsqueeze(1)

            # compute input modulation gate (equation 4 of assignment sheet)
            g = torch.tanh(x_t @ self.W_gx + self.h @ self.W_gh + self.b_g)
            # compute input gate (equation 5 of assignment sheet)
            i = torch.sigmoid(x_t @ self.W_ix + self.h @ self.W_ih + self.b_i)
            # compute forget gate (equation 6 of assignment sheet)
            f = torch.sigmoid(x_t @ self.W_fx + self.h @ self.W_fh + self.b_f)
            # compute output gate (equation 7 of assignment sheet)
            o = torch.sigmoid(x_t @ self.W_ox + self.h @ self.W_oh + self.b_o)
            # compute cell state (equation 8 of assignment sheet)
            self.c = g * i + self.c * f
            # compute hidden state (equation 9 of assignment sheet)
            self.h = torch.tanh(self.c) * o
        # compute equation 10 of assignment sheet
        out = self.h @ self.W_ph + self.b_p
        return out