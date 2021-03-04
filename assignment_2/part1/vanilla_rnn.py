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

class VanillaRNN(nn.Module):

    def __init__(self, seq_length, input_dim, num_hidden, batch_size, num_classes, device):
        super(VanillaRNN, self).__init__()

        # Initialization of inputs
        self.seq_length = seq_length
        self.input_dim = input_dim
        self.num_hidden = num_hidden
        self.batch_size= batch_size
        self.num_classes = num_classes
        self.device = device

        # Initialize biases
        self.b_h = nn.Parameter(torch.randn(num_hidden), requires_grad = True)
        self.b_p = nn.Parameter(torch.randn(num_classes), requires_grad = True)

        # Initialize the weight matrices with
        self.W_hx = nn.Parameter(torch.randn((input_dim, num_hidden)), requires_grad = True)
        self.W_hh = nn.Parameter(torch.randn((num_hidden, num_hidden)), requires_grad = True)
        self.W_hp = nn.Parameter(torch.randn((num_hidden, num_classes)), requires_grad = True)

        # Initialize hidden state
        self.h = torch.empty(batch_size, num_hidden).to(device)

    def forward(self, x):
        # reset hidden states before forward pass
        self.h = torch.zeros((self.batch_size, self.num_hidden)).to(self.device)
        # loop through steps of sequence
        for seq in range(x.shape[1]):
            # reshape input to get all seq steps for all batch-elements: 128x1 input
            x_t = x[:, seq].unsqueeze(1)
            h_x = x_t @ self.W_hx
            h_h = self.h @ self.W_hh
            self.h = torch.tanh(h_x + h_h + self.b_h)

        out = self.h @ self.W_hp + self.b_p

        return out



