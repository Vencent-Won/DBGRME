#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import sys

from torch import nn, optim
from config import RS_args_parser
from utils.metric import recall_compute
from utils.explight import set_seed, get_dump_path, initialize_exp, describe_model
from utils.data_process import read_data, uniform_sample
from model_lib.base_model import *
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm




def main(args):
    # device setting
    if (torch.cuda.is_available() and args.cuda):
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')

    # dataset prepare
    num_users, num_items, df_train, df_test = read_data(args.datadir, args.dataset)
    unique_users = df_train['user'].unique().tolist()
    unique_items = df_train['item'].unique().tolist()
    train_dict = df_train.groupby('user')['item'].apply(list).to_dict()
    train_edge_index = torch.vstack([torch.FloatTensor(df_train.values.T[0]),
                                     torch.FloatTensor(df_train.values.T[1]) + len(unique_users)]).to(device)
    test_edge_index = torch.vstack([torch.FloatTensor(df_test.values.T[0]),
                                    torch.FloatTensor(df_test.values.T[1]) + len(unique_users)]).to(device)
    test_item_counts = degree(test_edge_index[0].long())
    edge_index = torch.cat([train_edge_index, train_edge_index[[1, 0]]], dim=-1).long()
    edge_index, edge_weight = gcn_norm(edge_index, add_self_loops=False)
    Graph = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight)
    train_index0 = torch.vstack([torch.FloatTensor(df_train.values.T[0]),
                                 torch.FloatTensor(df_train.values.T[1])]).long().to(device)
    train_data_loader = DataLoader(train_index0.T, batch_size=args.batch_size, shuffle=True)
    test_data_loader = DataLoader(torch.LongTensor(np.sort(df_test['user'].unique())).to(device),
                                  batch_size=args.test_batch_size)

    # model initialization
    model = RS_Model(args, num_users, num_items, Graph, device=device)
    describe_model(model, args.logger_path, name='recommender system model')
    optimizer = optim.Adam(model.parameters(), lr=args.lr_benign)
    os.makedirs(args.model_path, exist_ok=True)
    # Training
    best_recall = 0.

    logger.info('-------------------------Training----------------------------')
    for epoch in range(args.epoch):
        model.train()
        epoch_train_loss = 0.
        for i, batch_data in enumerate(train_data_loader):
            optimizer.zero_grad()
            # sample positive and negative triple data
            batch_data, labels = uniform_sample(batch_data, None, num_users, num_items, device)
            loss = model.loss(torch.Tensor(batch_data[:, 0]).long(), torch.Tensor(batch_data[:, 1]).long(),
                              torch.Tensor(batch_data[:, 2]).long(), labels)
            loss.backward()
            optimizer.step()
            epoch_train_loss += loss.item()
        logger.info("epoch {}, training loss: {}".format(epoch, epoch_train_loss))

        if (epoch+1) % args.eval_gap == 0:
            logger.info('-------------------------Testing----------------------------')
            model.eval()
            recall = 0.
            for i, batch_users in enumerate(test_data_loader):
                if args.model_name in ['lgn', 'gcmc', 'ngcf']:
                    pred = model.getUsersRating(batch_users, Graph)
                else:
                    pred = model.getUsersRating(batch_users)
                recall += recall_compute(batch_users, num_users, num_items, pred, train_edge_index, test_edge_index,
                                         args.top_k, test_item_counts, device)
            if recall > best_recall:
                best_recall = recall
                logger.info("epoch {}, recall: {}".format(epoch, recall / len(df_test['user'].unique())))
                path = args.model_path + '/{}-{}-{}-{}.pt'.format(args.dataset, args.model_name, args.seed,
                                                                  args.hidden_dim_benign)
                torch.save(model.state_dict(), path)


if __name__ == '__main__':
    args = RS_args_parser()
    set_seed(args.seed)
    logger = initialize_exp(args)
    args.logger_path = get_dump_path(args)
    main(args)
