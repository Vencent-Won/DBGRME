#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import re
import sys
import torch

import numpy as np
import pandas as pd

from torch import nn, optim
from config import RSMSA_args_parser
from utils.metric import recall_compute

from utils.explight import set_seed, get_dump_path, initialize_exp
from model_lib.victim import LightGCN_T, NGCF_T, GCMC_T, NCF_T, BPR_T
from utils.data_process import read_data, uniform_sample
from model_lib.surrogate import Surrogate
from model_lib.generator import Generator, gumble_discrete
from torch_geometric.data import DataLoader
from torch_geometric.utils import degree
from torch_geometric.nn.conv.gcn_conv import gcn_norm


def Discretize(x, k, device):
    # transform the continuous variable to discrete variable
    topk_values, topk_indices = torch.topk(x, k, dim=-1)
    edge_index0 = torch.arange(x.shape[0]).repeat_interleave(k).to(device)
    edge_index = torch.vstack([edge_index0, topk_indices.view(x.shape[0] * k)]).long()
    return topk_indices, edge_index


def cal_agreement(scores_rank, labels_rank):
    temp = torch.hstack([scores_rank, labels_rank])
    temp, _ = temp.sort(-1)
    agreement = (temp[:, 1:] == temp[:, :-1]).sum()
    return agreement


def build_compute_graph(edge_index, size):
    query_edge_index = torch.hstack([edge_index, edge_index[[1, 0]]])
    query_edge_index, query_edge_weight = gcn_norm(query_edge_index, add_self_loops=False)
    query_graph = torch.sparse_coo_tensor(indices=query_edge_index, values=query_edge_weight, size=[size, size])
    return query_graph


# @torch.no_grad()
def get_synthesis_data(num_fakers, num_users, num_items, train_edge_index, top_k, device, shadow_data=None):
    if shadow_data is None:
        z = torch.randn((num_fakers, num_items), device=device)
        z = z.detach()
        indices, fake_edge_index = Discretize(z, top_k, device)
    else:
        indices, fake_edge_index = shadow_data
    fake_edge_index = fake_edge_index[:, torch.randperm(fake_edge_index.size(1))]
    temp_indices = pd.DataFrame(fake_edge_index.T.cpu().tolist(), columns=['user', 'item'])
    bpr_dict = temp_indices.groupby('user')['item'].apply(list).to_dict()
    # then prepare the target graph and clone graph
    target_fake_edge_index = torch.vstack([fake_edge_index[0] + num_users, fake_edge_index[1] + num_users + num_fakers])
    target_train_edge_index = torch.vstack([train_edge_index[0], train_edge_index[1] + num_fakers])
    target_query_edge_index = torch.hstack([target_train_edge_index, target_fake_edge_index])
    target_query_graph = build_compute_graph(target_query_edge_index, size=(num_users+num_fakers+num_items))

    clone_edge_index = torch.vstack([fake_edge_index[0], fake_edge_index[1] + num_fakers])
    clone_query_graph = build_compute_graph(clone_edge_index, size=(num_fakers+num_items))
    return bpr_dict, fake_edge_index, target_query_graph, clone_query_graph


def main(args):
    # device setting
    if (torch.cuda.is_available() and args.cuda):
        device = torch.device('cuda:{}'.format(args.gpu))
        torch.cuda.empty_cache()
    else:
        device = torch.device('cpu')
    T_MODELS = {'lgn': LightGCN_T, 'ngcf': NGCF_T, 'gcmc': GCMC_T, 'ncf': NCF_T, 'bpr': BPR_T}

    """ Dataset load """
    num_users, num_items, df_train, df_test = read_data(args.datadir, args.dataset)
    unique_users = df_train['user'].unique().tolist()
    unique_items = df_train['item'].unique().tolist()
    num_fakers = args.num_fakers
    train_edge_index = torch.vstack([torch.LongTensor(df_train.values.T[0]),
                                     torch.LongTensor(df_train.values.T[1]) + len(unique_users)]).to(device)
    test_edge_index = torch.vstack([torch.LongTensor(df_test.values.T[0]),
                                    torch.LongTensor(df_test.values.T[1]) + len(unique_users)]).to(device)
    test_item_counts = degree(test_edge_index[0].long())
    edge_index = torch.cat([train_edge_index, train_edge_index[[1, 0]]], dim=-1).long()
    edge_index, edge_weight = gcn_norm(edge_index, add_self_loops=False)
    Train_Graph = torch.sparse_coo_tensor(indices=edge_index, values=edge_weight)
    test_data_loader = DataLoader(torch.LongTensor(np.sort(df_test['user'].unique())).to(device),
                                  batch_size=args.test_batch_size)

    # build victim model and load pretrained weights
    V = T_MODELS[args.victim_model](num_users, num_items, num_fakers, args.hidden_dim_benign, args.num_layers_benign,
                                    args.regs_decay).to(device)
    path = args.model_path + '/{}-{}-{}-{}.pt'.format(args.dataset, args.victim_model, args.seed,
                                                      args.hidden_dim_benign)
    V.load_state_dict(torch.load(path, map_location=device), strict=False)
    for name, param in V.named_parameters():
        param.requires_grad = False
    V.eval()

    # build generator
    G = Generator(num_fakers, num_items, args.in_dim_generator, args.hidden_dim_generator,
                  args.out_dim_generator, args.num_layers_generator, args.test_batch_size).to(device)
    optG = torch.optim.Adam(G.parameters(), lr=args.lr_generator, eps=1e-9)
    # surrogate model initialization
    S = Surrogate(num_users, num_items, num_fakers, args.hidden_dim_clone, args.num_layers_clone, args.dropout,
                  args.regs_decay).to(device)
    optS = optim.Adam(S.parameters(), lr=args.lr_clone, eps=1e-9)
    for name, param in S.named_parameters():
        if "fakers" in name:
            param.requires_grad = False
    q = 0
    best_fidelity = 0.
    best_recall = 0.
    while q < args.query_budget:
        fake_user = torch.arange(num_fakers, dtype=torch.long).to(device)
        if (q == 0 & (not args.clone_first)) or (q > 0):
            G.train()
            S.eval()
            for n_iter in range(args.iter_generator):
                all_loss_G = 0.
                loss_generator = 0.
                z = torch.randn((num_fakers, args.in_dim_generator), device=device)
                shadow_scores = G(z)
                shadow_data = gumble_discrete(shadow_scores, args.gen_top_k, device)
                bpr_dict, fake_edge_index, g_target_query, g_clone_query = get_synthesis_data(num_fakers, num_users,
                                                                                              num_items,
                                                                                              train_edge_index,
                                                                                              args.gen_top_k, device,
                                                                                              shadow_data)
                q += num_fakers
                S.init_fakers()
                V.init_fakers()
                # query victim model
                with torch.no_grad():
                    target_item_out_all = V.getUsersRating(fake_user + num_users, g_target_query, mode='target_fake')
                    target_item_prob_all = nn.Sigmoid()(target_item_out_all)
                    target_item_prob_all[fake_edge_index[0], fake_edge_index[1]] = float('-inf')
                    _, target_item_list_all = target_item_prob_all.topk(args.fit_top_k, dim=-1)
                # query surrogate model
                edge_index0 = torch.arange(num_fakers).repeat_interleave(args.fit_top_k).to(device)
                fake_bpr_edge_index = torch.vstack(
                    [edge_index0, target_item_list_all.view(num_fakers * args.fit_top_k)]).long()
                train_data_loader = DataLoader(fake_bpr_edge_index.T, batch_size=args.train_bpr_batch_size,
                                               shuffle=True)
                for i, batch_data in enumerate(train_data_loader):
                    optG.zero_grad()
                    # sample positive and negative triple data
                    batch_data, labels = uniform_sample(batch_data, None, num_fakers, num_items, device)
                    loss_generator = -S.loss(batch_data[:, 0], batch_data[:, 1], batch_data[:, 2], g_clone_query,
                                             mode='fake_bpr')
                    loss_generator.backward()
                    optG.step()
                    all_loss_G += loss_generator.item()
                logger.info('query budget {}/{}, generate iter {}, loss {}'.format(q, args.query_budget, n_iter,
                                                                                   all_loss_G))

        if (q == 0 & args.clone_first) or (q > 0):
            # surrogate model training ...
            # synthesis data generate
            old_embeddings = S.embedding_item.weight.clone()
            for n_iter in range(args.iter_clone):
                S.train()
                G.eval()
                S.init_fakers()
                V.init_fakers()
                q += num_fakers
                with torch.no_grad():
                    z = torch.randn((num_fakers, args.in_dim_generator), device=device)
                    shadow_score = G(z)
                    shadow_data = gumble_discrete(shadow_score, args.gen_top_k, device)
                    bpr_dict, fake_edge_index, g_target_query, g_clone_query = get_synthesis_data(num_fakers, num_users,
                                                                                                  num_items,
                                                                                                  train_edge_index,
                                                                                                  args.gen_top_k,
                                                                                                  device, shadow_data)
                    target_item_out_all = V.getUsersRating(fake_user + num_users, g_target_query, mode='target_fake')
                    target_item_prob_all = nn.Sigmoid()(target_item_out_all)
                    target_item_prob_all[fake_edge_index[0], fake_edge_index[1]] = float('-inf')
                    _, target_item_list_all = target_item_prob_all.topk(args.fit_top_k, dim=-1)
                all_loss = 0.
                # fitting training
                ## query victim model for one step
                edge_index0 = torch.arange(num_fakers).repeat_interleave(args.fit_top_k).to(device)
                fake_bpr_edge_index = torch.vstack(
                    [edge_index0, target_item_list_all.view(num_fakers * args.fit_top_k)]).long()
                train_data_loader = DataLoader(fake_bpr_edge_index.T, batch_size=args.train_bpr_batch_size,
                                               shuffle=True)
                for i, batch_data in enumerate(train_data_loader):
                    optS.zero_grad()
                    # sample positive and negative triple data
                    batch_data, labels = uniform_sample(batch_data, None, num_fakers, num_items, device)
                    loss = S.loss(batch_data[:, 0], batch_data[:, 1], batch_data[:, 2], g_clone_query, mode='fake_bpr')
                    items = torch.cat((batch_data[:, 1], batch_data[:, 2]), dim=-1)
                    if (n_iter > 0 or q > 0) & (args.alpha_norm > 0):
                        new_embeddings = S.embedding_item.weight[items].clone()
                        diff = new_embeddings - old_embeddings[items]
                        loss += 0.5 * torch.mean(diff ** 2) * args.alpha_norm
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(S.parameters(), 0.5)
                    optS.step()
                    all_loss += loss.item()

                logger.info('query budget {}/{}, surrogate iter {}, loss {}'.format(q, args.query_budget, n_iter,
                                                                                    all_loss))
                if q % args.eval_gap == 0:
                    S.eval()
                    fidelity = 0.
                    recall = 0.
                    with torch.no_grad():
                        for batch_users in test_data_loader:
                            # query the victim model with train graph of dataset
                            target_item_prob = nn.Sigmoid()(V.getUsersRating(batch_users, Train_Graph, mode='test'))
                            mask = ((train_edge_index[0] >= min(batch_users)) &
                                    (train_edge_index[0] < max(batch_users)))
                            target_item_prob[train_edge_index[0, mask].long() - min(batch_users),
                                             train_edge_index[1, mask].long() - num_users] = float('-inf')
                            _, target_item_list = target_item_prob.topk(args.eval_top_k, dim=-1)
                            # query the surrogate model with train graph of dataset
                            clone_item_prob = nn.Sigmoid()(S.getUsersRating(batch_users, Train_Graph, mode='test'))
                            clone_item_prob[train_edge_index[0, mask].long() - min(batch_users),
                                            train_edge_index[1, mask].long() - num_users] = float('-inf')
                            _, clone_item_list = clone_item_prob.topk(args.eval_top_k, dim=-1)
                            # compute the fidelity and recall
                            fidelity += cal_agreement(clone_item_list, target_item_list) / args.eval_top_k
                            recall += recall_compute(batch_users, num_users, num_items, clone_item_prob, train_edge_index,
                                                     test_edge_index, args.eval_top_k, test_item_counts, device)
                        fidelity /= num_users
                        recall /= num_users
                        if fidelity > best_fidelity:
                            best_fidelity = fidelity
                            path = args.logger_path + '/{}-{}-{}-{}.pt'.format(args.dataset, args.victim_model,
                                                                               args.seed,
                                                                               args.hidden_dim_benign)
                            torch.save(S.state_dict(), path)
                            logger.info(
                                'query budget {}/{} best fidelity {} fidelity {}, recall {}'.format(q,
                                                                                                    args.query_budget,
                                                                                                    best_fidelity,
                                                                                                    fidelity,
                                                                                                    recall))
    logger.info('budget {}, best fidelity {}, best recall {}'.format(args.query_budget, best_fidelity,
                                                                     best_recall))



if __name__ == '__main__':
    arg = RSMSA_args_parser()
    set_seed(arg.seed)
    logger = initialize_exp(arg)
    arg.logger_path = get_dump_path(arg)
    main(arg)
