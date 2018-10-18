#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""
Char LSTM
"""

import torch
import torch.nn
import torch.nn.functional as F


class CharLSTM(nn.Module):
    def __init__(self,
                 vocab_size,
                 category_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 device=None):
        super(CharLSTM, self).__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.hidden_size = hidden_size + category_size
        self.num_layers = num_layers
        self.device = device

        self.embedding = nn.Embedding(vocab_size, embedding_size)
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(self.hidden_size, hidden_size,
                            num_layers, dropout=dropout)
        self.linear = nn.Linear(hidden_size, vocab_size)

        self.relu = nn.ReLU()
        self.softmax = nn.LogSoftmax(dim=2)

    def forward(self, categories, inputs, hidden_state):
        """
            category_size: [batch_size, category_size]
            inputs: [1, batch_size]
            hidden_state
        """
        inputs = inputs.view(1, -1)
        embedded = self.embedding(inputs)
        embedded = self.dropout(embedded)

        # embedded: [1, batch_size, embedding_size]
        lstm_input = torch.cat((embedded, categories.unsqueeze(0)), dim=2)
        output, hidden_state = self.lstm(lstm_input, hidden_state)

        output = self.relu(self.linear(output))

        output = self.softmax(output)
        return output, hidden_state

    def init_hidden(self, batch_size):
        initial_state_scale = math.sqrt(3.0 / self.hidden_size)
        initial_state1 = torch.empty(self.num_layers, batch_size, self.hidden_size), device = self.device)
        initial_state2 = torch.empty((self.num_layers, batch_size, self.hidden_size), device = self.device)
        torch.nn.init.uniform_(initial_state1, a = -initial_state_scale, b = initial_state_scale)
        torch.nn.init.uniform_(initial_state2, a = -initial_state_scale, b = initial_state_scale)
        return (initial_state1, initial_state2)
