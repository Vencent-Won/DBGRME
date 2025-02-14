#!/usr/bin/env python
# -*- coding: utf-8 -*-
import torch



def recall_compute(batch_users, num_users, num_items, pred, train_edge_index, test_edge_index, top_k, test_item_counts,
                   device):
    mask = ((train_edge_index[0] >= min(batch_users)) &
            (train_edge_index[0] < max(batch_users)))
    pred[train_edge_index[0, mask].long() - min(batch_users),
         train_edge_index[1, mask].long() - num_users] = float('-inf')
    mask = ((test_edge_index[0] >= min(batch_users)) & (test_edge_index[0] < max(batch_users)))
    ground_truth = torch.zeros([len(batch_users), num_items], dtype=torch.bool).to(device)
    ground_truth[(test_edge_index[0, mask] - min(batch_users)).long(),
                 (test_edge_index[1, mask] - num_users).long()] = True
    top_k_index = pred.topk(top_k, dim=-1).indices
    isin_mat = ground_truth.gather(1, top_k_index)
    return float((isin_mat.sum(dim=-1) / test_item_counts[min(batch_users):max(batch_users) + 1].clamp(1e-6)).sum())

