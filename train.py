#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""
import time
import math

import torch
import torch.nn as nn
import torch.optim as optim

from model import CharLSTM
import data_set


lr = 0.001
vocab_size = data_set.n_letters
category_size = data_set.n_categories
embedding_size = 128
hidden_size = 128
num_layers = 1
dropout = 0.0
max_len = 20
batch_size = 32

device = torch.devcie('cuda')

criterion = nn.NLLLoss(ignore_index=0,
                       reduction='elementwise_mean')
model = CharLSTM(vocab_size,
                 category_size,
                 embedding_size,
                 hidden_size,
                 num_layers,
                 dropout,
                 device)

model = model.to(device=device)
print(model)
optimizer = optim.Adam(model.parameters(), lr)


def trainIters():
    all_losses = []
    total_loss = 0  # Reset every plot_every iters
    start = time.time()
    n_iters = 100000
    print_every = 5000
    plot_every = 500

    for iter in range(1, n_iters + 1):
        output, loss = train(*data_set.next_batch(max_len, batch_size, device))
        total_loss += loss

        if iter % print_every == 0:
            print('%s (%d %d%%) %.4f' %
                  (timeSince(start), iter, iter / n_iters * 100, loss))

        if iter % plot_every == 0:
            all_losses.append(total_loss / plot_every)
            total_loss = 0


def train(categories, inputs, inputs_length, targets):
    """
        categories: [batch_size, category_size]
        inputs: [max_len, batch_size]
        inputs_length: [batch_size]
        targets: [batch_size]
    """

    hidden_state = model.init_hidden()

    optimizer.zero_grad()

    loss = 0

    for i in range(max_len):
        output, hidden_state = model(categories, inputs[i], hidden_state)
        # output: [1, batch_size, vocab_size], targets[i]:[batch_size]
        output = output.view(-1, vocab_size)
        loss += criterion(output, targets[i])

    loss.backward()

    optimizer.step()

    return output, loss.item() / torch.mean(inputs_length.float())


def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


if __name__ == '__main__':
    trainIters()
