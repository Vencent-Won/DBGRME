#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch

import pandas as pd

import torch.nn.functional as F


def read_data(datadir, dataname):
    if dataname == 'ml-1m':
        num_users, num_items = 6022, 3043
    elif dataname == 'ml-100k':
        num_users, num_items = 943, 1682
    elif dataname == 'yelp':
        num_users, num_items = 31668, 38048
    else:
        num_users, num_items = 29858, 40981
    path = './{}/{}'.format(datadir, dataname)
    df_train = pd.read_csv('{}/train_sparse.csv'.format(path))
    df_test = pd.read_csv('{}/test_sparse.csv'.format(path))
    return num_users, num_items, df_train, df_test

def uniform_sample(batch_data, train_dict, num_users, num_items, device):
    batch_data = batch_data.T
    if train_dict is None:
        neg_index = torch.randint(num_items, size=(batch_data.shape[1],), device=device)
    else:
        if train_dict.shape[0] != 2:
            train_dict = train_dict.t()
        weight = torch.ones(num_users, num_items).to(device)
        weight[train_dict[0], train_dict[1]] = 0
        weight[batch_data[0], batch_data[1]] = 0
        weight_prob = F.softmax(weight, dim=-1)[batch_data[0]]
        neg_index = torch.distributions.Categorical(weight_prob).sample()
    batch_data = torch.vstack([batch_data, neg_index])
    labels = torch.ones(batch_data.shape[1]*2, dtype=torch.float, device=device)
    labels[batch_data.shape[1]:] = 0
    return batch_data.T, labels

def minibatch(*tensors, batch_size):
    if len(tensors) == 1:
        tensor = tensors[0]
        for i in range(0, len(tensor), batch_size):
            yield tensor[i:i + batch_size]
    else:
        for i in range(0, len(tensors[0]), batch_size):
            yield tuple(x[i:i + batch_size] for x in tensors)

