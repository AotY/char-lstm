#! /usr/bin/env python
# -*- coding: utf-8 -*-
# Copyright © 2018 LeonTao
#
# Distributed under terms of the MIT license.

"""

"""


from __future__ import unicode_literals, print_function, division
from io import open
import glob
import os
import unicodedata
import string
import random

import torch
import torch.nn as nn

# sos: - ,  eos: <
all_letters = '-' + string.ascii_letters + " .,;'<"
n_letters = len(all_letters) + 1 # Plus EOS marker

def findFiles(path): 
    return glob.glob(path)

# Turn a Unicode string to plain ASCII, thanks to http://stackoverflow.com/a/518232/2809427
def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
        and c in all_letters
    )

# Read a file and split into lines
def readLines(filename):
    lines = open(filename, encoding='utf-8').read().strip().split('\n')
    return [unicodeToAscii(line) for line in lines]

# Build the category_lines dictionary, a list of lines per category
category_lines = {}
all_categories = []
for filename in findFiles('data/names/*.txt'):
    category = os.path.splitext(os.path.basename(filename))[0]
    all_categories.append(category)
    lines = readLines(filename)
    category_lines[category] = lines

n_categories = len(all_categories)

if n_categories == 0:
    raise RuntimeError('Data not found. Make sure that you downloaded data '
        'from https://download.pytorch.org/tutorial/data.zip and extract it to '
        'the current directory.')


# Random item from a list
def randomChoice(l):
    return l[random.randint(0, len(l) - 1)]

# Get a random category and random line from that category
def randomTrainingPair():
    category = randomChoice(all_categories)
    line = randomChoice(category_lines[category])
    return category, line


# One-hot vector for category
def categoryTensor(category, device):
    li = all_categories.index(category)
    tensor = torch.zeros(1, n_categories, dtype=torch.long, device=device)
    tensor[0][li] = 1
    return tensor

# One-hot matrix of first to last letters (not including EOS) for input
def inputTensor(line, device):
    tensor = torch.zeros(len(line), dtype=torch.long, device=device)
    for li in range(len(line)):
        letter = line[li]
        tensor[li] = all_letters.find(letter)
    return tensor

# LongTensor of second letter to end (EOS) for target
def targetTensor(line, device):
    letter_indexes = [all_letters.find(line[li]) for li in range(1, len(line))]
    letter_indexes.append(n_letters - 1) # EOS
    return torch.LongTensor(letter_indexes, device=device)

# Make category, input, and target tensors from a random category, line pair
def randomTrainingExample(device):
    category, line = randomTrainingPair()
    category_tensor = categoryTensor(category, device)
    input = inputIndex(line)
    target = targetIndex(line)
    return category_tensor, input, target

def next_batch(max_len, batch_size, device=None):
    categories = torch.empty((batch_size, n_categories), dtype=torch.long, device=device)
    inputs = torch.empty((max_len, batch_size), dtype=torch.long, device=device)
    targets = torch.empty((max_len, batch_size), dtype=torch.long, device=device)
    inputs_length = []

    for i in range(batch_size):
        category_tensor, input, target = randomTrainingPair(device) 
        inputs_length.append(len(input))

        categories[i] = category_tensor
        for j, (i_index, t_index) in enumerate(zip(input, target)):
            inputs[j, i] = i_index
            targets[j, i] = t_index

    inputs_length = torch.LongTensor(inputs_length, device=device)
    return categories, inputs, inputs_length, targets


