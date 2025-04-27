#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import torch

import numpy as np
import pandas as pd
import torch.nn.functional as F

from torch import nn, optim
from config import RSMSA_args_parser
from utils.metric import recall_compute

from utils.explight import set_seed, get_dump_path, initialize_exp
from model_lib.victim import LightGCN_T, NGCF_T, GCMC_T, NCF_T, BPR_T
from utils.data_process import read_data, uniform_sample
from model_lib.surrogate import Surrogate
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


@torch.no_grad()
def generate_auto_regressive(num_fakers, num_users, num_items, train_edge_index, top_k, max_len, batch_size, target_model, device):
    target_model.eval()
    # gen_loader = DataLoader(torch.range(0, num_fakers), shuffle=False, batch_size=batch_size)
    batch = torch.arange(num_fakers, device=device).long()
    # for i, batch in enumerate(gen_loader):
    seqs = torch.randint(1, num_items+1, (batch.shape[0], 1)).to(device)
    candidates = None
    for t in range(max_len):
        edge_index0 = torch.arange(batch.shape[0]).repeat_interleave(t+1).to(device)
        edge_index = torch.vstack([edge_index0, seqs.view(-1)]).long()
        shadow_data = (seqs, edge_index)
        bpr_dict, fake_edge_index, g_target_query, g_clone_query = get_synthesis_data(num_fakers, num_users,
                                                                                      num_items, train_edge_index,
                                                                                      None, device, shadow_data)
        target_item_out_all = target_model.getUsersRating(batch + num_users, g_target_query, mode='target_fake')
        target_item_prob_all = nn.Sigmoid()(target_item_out_all)
        _, target_item_list_all = target_item_prob_all.topk(top_k, dim=-1)
        selected_indices = torch.distributions.Categorical(F.softmax(torch.ones_like(target_item_list_all.float()),
                                                                     -1).to(device)).sample()
        row_indices = torch.arange(target_item_list_all.size(0))
        if t < (max_len-1):
            seqs = torch.cat((seqs, target_item_list_all[row_indices, selected_indices].unsqueeze(1)), -1)
        try:
            candidates = torch.cat((candidates, target_item_list_all.unsqueeze(1)), 1)
        except:
            candidates = target_item_list_all.unsqueeze(1)
    return seqs, candidates




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

    # build target model and load pretrained weights
    V = T_MODELS[args.victim_model](num_users, num_items, num_fakers, args.hidden_dim_benign, args.num_layers_benign,
                                    args.regs_decay).to(device)
    path = args.model_path + '/{}-{}-{}-{}.pt'.format(args.dataset, args.victim_model, args.seed,
                                                      args.hidden_dim_benign)
    V.load_state_dict(torch.load(path, map_location=device), strict=False)
    V.eval()
    for name, param in V.named_parameters():
        param.requires_grad = False

    # surrogate model initialization
    S = LightGCN_T(num_users, num_items, num_fakers, args.hidden_dim_clone, args.num_layers_clone, args.regs_decay).to(device)
    # S = Surrogate(num_users, num_items, num_fakers, args.hidden_dim_clone, args.num_layers_clone, args.dropout,
    #               args.regs_decay).to(device)
    optS = optim.Adam(S.parameters(), lr=args.lr_clone, eps=1e-9)
    # for name, param in S.named_parameters():
    #     if "embedding_user" in name:
    #         param.requires_grad = False
    train_dict = df_train.groupby('user')['item'].apply(list).to_dict()
    fake_user_index = np.random.choice(list(range(num_users)), int(num_users*0.1))
    fake_edge_index0 = [torch.ones(len(train_dict[user]), dtype=torch.long, device=device) * user for user in
                        fake_user_index]
    fake_edge_index1 = [torch.LongTensor(train_dict[user]).to(device) for user in fake_user_index]
    fake_bpr_edge_index = torch.vstack((torch.cat(fake_edge_index0), torch.cat(fake_edge_index1)))
    train_data_loader = DataLoader(fake_bpr_edge_index.T, batch_size=args.train_bpr_batch_size, shuffle=True)
    fake_edge_index = torch.vstack((fake_bpr_edge_index[0], fake_bpr_edge_index[1] + num_users))
    fake_edge_index = torch.cat([fake_edge_index, fake_edge_index[[1, 0]]], dim=-1).long()
    fake_edge_index, fake_edge_weight = gcn_norm(fake_edge_index, add_self_loops=False)
    fake_graph = torch.sparse_coo_tensor(indices=fake_edge_index, values=fake_edge_weight, size=[num_users+num_items, num_users+num_items])
    best_fidelity = 0.
    best_recall = 0.
    q = 0
    V.init_fakers()
    loss_func_1 = nn.MarginRankingLoss(margin=0.5)
    loss_func_2 = nn.MarginRankingLoss(margin=0.5)
    tokens, candidates = generate_auto_regressive(num_fakers, num_users, num_items, train_edge_index, args.fit_top_k,
                                                  args.gen_top_k, args.test_batch_size, V, device)
    # while q < args.query_budget:
    # for epoch in range(args.query_budget):
    #     S.train()
    #     if (q == 0 & args.clone_first) or (q > 0):
    #         # surrogate model training ...
    #         # synthesis data generate
    #
    #         old_embeddings = S.embedding_item.weight.clone()
    S.init_fakers()
    for n_iter in range(args.iter_clone):
        S.train()
        train_index = torch.randint(1, args.gen_top_k, (num_fakers, ), device=device)
        topk_indices = candidates[torch.arange(num_fakers, device=device), train_index, :]
        edge_index0 = torch.cat([torch.ones(temp, dtype=torch.long, device=device)*i for i, temp in enumerate(train_index)])
        edge_index1 = torch.cat([tokens[i, 0:temp] for i, temp in enumerate(train_index)])
        edge_index = torch.vstack([edge_index0, edge_index1]).long()
        shadow_data = (topk_indices, edge_index)
        bpr_dict, fake_edge_index, g_target_query, g_clone_query = get_synthesis_data(num_fakers, num_users, num_items,
                                                                                      train_edge_index, args.gen_top_k,
                                                                                      device, shadow_data)
        q += num_fakers
        if args.loss_type == 'bpr':
            edge_index0 = torch.arange(num_fakers).repeat_interleave(args.fit_top_k).to(device)
            fake_bpr_edge_index = torch.vstack(
                [edge_index0, topk_indices.view(num_fakers * args.fit_top_k)]).long()
            train_data_loader = DataLoader(fake_bpr_edge_index.T, batch_size=args.train_bpr_batch_size,
                                           shuffle=True)
            all_loss = 0.
            # fitting training
            ## query target model for one step
            for i, batch_data in enumerate(train_data_loader):
                optS.zero_grad()
                # sample positive and negative triple data
                batch_data, labels = uniform_sample(batch_data, None, num_fakers, num_items, device)
                loss = S.loss(batch_data[:, 0], batch_data[:, 1], batch_data[:, 2], g_clone_query, mode='fake_bpr')
                # items = torch.cat((batch_data[:, 1], batch_data[:, 2]), dim=-1)
                # if (n_iter > 0 or q > 0) & (args.alpha_norm > 0):
                #     new_embeddings = S.embedding_item.weight[items].clone()
                #     diff = new_embeddings - old_embeddings[items]
                #     loss += 0.5 * torch.mean(diff ** 2) * args.alpha_norm
                loss.backward()
                torch.nn.utils.clip_grad_norm_(S.parameters(), 0.5)
                optS.step()
                all_loss += loss.item()
            else:
                train_data_loader = DataLoader(torch.arange(num_fakers), batch_size=args.test_batch_size, shuffle=True)
                for i, batch_data in enumerate(train_data_loader):
                    optS.zero_grad()
                    clone_item_prob = nn.Sigmoid()(S.getUsersRating(batch_data, g_clone_query, mode='fake_bpr'))
                    weight = torch.ones_like(clone_item_prob).to(device)
                    weight[torch.arange(weight.size(0)).unsqueeze(1), topk_indices[batch_data]] = 0
                    neg_samples = torch.distributions.Categorical(F.softmax(weight, -1)).sample_n(
                        candidates.size(-1)).permute(1, 0)
                    neg_logits = torch.gather(clone_item_prob, -1, neg_samples)
                    pos_logits = torch.gather(clone_item_prob, -1, candidates)
                    logits_1 = pos_logits[:, :-1].reshape(-1)
                    logits_2 = pos_logits[:, 1:].reshape(-1)
                    loss = loss_func_1(logits_1, logits_2, torch.ones(logits_1.shape).to(device))
                    loss += loss_func_2(pos_logits, neg_logits, torch.ones(pos_logits.shape).to(device))
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(S.parameters(), 0.5)
                    optS.step()
                    all_loss += loss.item()

        logger.info('query budget {}/{}, surrogate loss {}'.format(q, args.query_budget, all_loss))
        if q % args.eval_gap == 0:
            S.eval()
            fidelity = 0.
            recall = 0.
            with torch.no_grad():
                for batch_users in test_data_loader:
                    # query the target model with train graph of dataset
                    target_item_prob = nn.Sigmoid()(V.getUsersRating(batch_users, Train_Graph, mode='test'))
                    mask = ((train_edge_index[0] >= min(batch_users)) & (train_edge_index[0] < max(batch_users)))
                    target_item_prob[train_edge_index[0, mask].long() - min(batch_users),
                                     train_edge_index[1, mask].long() - num_users] = float('-inf')
                    _, target_item_list = target_item_prob.topk(args.eval_top_k, dim=-1)
                    # query the surrogate model with train graph of dataset
                    clone_item_prob = nn.Sigmoid()(S.getUsersRating(batch_users, Train_Graph, mode='test'))
                    clone_item_prob[train_edge_index[0, mask].long() - min(batch_users),
                                    train_edge_index[1, mask].long() - num_users] = float('-inf')
                    _, clone_item_list = clone_item_prob.topk(args.eval_top_k, dim=-1)
                    # compute the fidelity
                    fidelity += cal_agreement(clone_item_list, target_item_list) / args.eval_top_k
                    # compute the recall
                    # recall += recall_compute(batch_users, num_users, num_items, clone_item_prob, train_edge_index,
                    #                          test_edge_index, args.eval_top_k, test_item_counts, device)
                fidelity /= num_users
                recall /= num_users
            if fidelity > best_fidelity:
                best_fidelity = fidelity
                best_recall = recall
                path = args.logger_path + '/{}-{}-{}-{}.pt'.format(args.dataset, args.victim_model, args.seed,
                                                                   args.hidden_dim_benign)
                torch.save(S.state_dict(), path)
                logger.info('query budget {}/{} best fidelity {} fidelity {}, recall {}'.format(q, args.query_budget,
                                                                                                best_fidelity, fidelity,
                                                                                                recall))
    logger.info('budget {}, best fidelity {}, best recall {}'.format(args.query_budget, best_fidelity, best_recall))



if __name__ == '__main__':
    arg = RSMSA_args_parser()
    set_seed(arg.seed)
    logger = initialize_exp(arg)
    arg.logger_path = get_dump_path(arg)
    main(arg)
