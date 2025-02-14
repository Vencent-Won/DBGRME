#!/usr/bin/env python
# -*- coding: utf-8 -*-
import time
import torch
import random

import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F


class BPR(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(BPR, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.latent_dim = args.hidden_dim_benign
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.args = args
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def loss(self, user, pos, neg, labels):  # 统一输入为tensor
        user_emb = self.embedding_user.weight[user]
        pos_emb = self.embedding_item.weight[pos]
        neg_emb = self.embedding_item.weight[neg]

        pos_score = (user_emb * pos_emb).sum(-1)
        neg_score = (user_emb * neg_emb).sum(-1)

        loss = -(pos_score - neg_score).sigmoid().log().mean()
        return loss

    def loss_rank(self, user, pos, neg, rank, labels):
        user_emb = self.embedding_user.weight[user]
        pos_emb = self.embedding_item.weight[pos]
        neg_emb = self.embedding_item.weight[neg]

        pos_score = (user_emb * pos_emb).sum(-1)
        neg_score = (user_emb * neg_emb).sum(-1)

        loss_bpr = -(pos_score - neg_score).sigmoid().log().mean()
        loss_rank = 0.

        users_list = list(set(user.tolist()))
        s = 0
        for u in users_list:
            ratings = self.getUsersRating(torch.Tensor([u]).long().to(self.args.device))
            for i in range(rank[u].shape[0] - 1):
                f_i = rank[u][i]
                r_f = ratings[f_i.long()]

                for s_i in rank[u][i + 1:]:
                    r_s = ratings[s_i.long()]
                    loss_rank += torch.nn.functional.softplus(r_s - r_f)
                    s += 1
            # posEmb, negEmb = self.getEmbedding(pos, neg)
            # loss_reg += (1 / 2) * (posEmb.norm(2).pow(2) + negEmb.norm(2).pow(2))
        Loss = loss_bpr + loss_rank / s

        return Loss

    def getUsersRating(self, user):  # 统一输入为单个用户tensor
        user_emb = self.embedding_user.weight[user]
        item_emb = self.embedding_item.weight

        score = torch.matmul(user_emb, item_emb.T)
        return score


class NCF(nn.Module):
    def __init__(self, args, num_users, num_items):
        super(NCF, self).__init__()

        self.num_users = num_users
        self.num_items = num_items

        self.n_layers = args.num_layers_benign
        self.num_factors = args.hidden_dim_benign
        self.dropout = args.dropout_rate

        self.embedding_user = torch.nn.Embedding(num_users, self.num_factors * (2 ** (self.n_layers - 1)))
        self.embedding_item = torch.nn.Embedding(num_items, self.num_factors * (2 ** (self.n_layers - 1)))

        MLP_modules = []
        for i in range(self.n_layers):
            input_size = self.num_factors * (2 ** (self.n_layers - i))
            MLP_modules.append(nn.Dropout(p=self.dropout))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        self.predict_layer = nn.Linear(self.num_factors, 1)
        self.loss_function = nn.BCEWithLogitsLoss()

        self._init_weight_()

    def _init_weight_(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        # for m in self.MLP_layers:
        #     if isinstance(m, nn.Linear):
        #         nn.init.xavier_uniform_(m.weight)

        nn.init.kaiming_uniform_(self.predict_layer.weight,
                                 a=1, nonlinearity='sigmoid')

    def loss(self, user, pos, neg, labels):
        item = torch.cat((pos, neg), dim=0)
        user = torch.cat((user, user), dim=0)
        embed_user_MLP = self.embedding_user.weight[user]
        embed_item_MLP = self.embedding_item.weight[item]
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        prediction = self.predict_layer(output_MLP).view(-1)  # check
        loss = self.loss_function(prediction, labels)
        return loss

    def loss_rank(self, user, pos, neg, rank, labels):
        item = torch.cat((pos, neg), dim=0)
        user = torch.cat((user, user), dim=0)
        embed_user_MLP = self.embedding_user.weight[user]
        embed_item_MLP = self.embedding_item.weight[item]
        interaction = torch.cat((embed_user_MLP, embed_item_MLP), -1)
        output_MLP = self.MLP_layers(interaction)

        prediction = self.predict_layer(output_MLP).view(-1)  # check
        loss_bpr = self.loss_function(prediction, labels)

        loss_rank = 0.

        users_list = list(set(user.tolist()))
        s = 0
        for u in users_list:
            ratings = self.getUsersRating(torch.Tensor([u]).long().to(self.args.device))
            for i in range(rank[u].shape[0] - 1):
                f_i = rank[u][i]
                r_f = ratings[f_i.long()]
                for s_i in rank[u][i + 1:]:
                    r_s = ratings[s_i.long()]
                    loss_rank += torch.nn.functional.softplus(r_s - r_f)
                    s += 1
            # posEmb, negEmb = self.getEmbedding(pos, neg)
            # loss_reg += (1 / 2) * (posEmb.norm(2).pow(2) + negEmb.norm(2).pow(2))
        Loss = loss_bpr + loss_rank / s

        return Loss

    def getUsersRating(self, users):  # 统一输入为tensor
        embed_user_MLP = self.embedding_user.weight[users]
        embed_item_MLP = self.embedding_item.weight

        embed_user_MLP = embed_user_MLP.unsqueeze(1).expand(-1, self.num_items, -1)
        embed_item_MLP = embed_item_MLP.unsqueeze(0).expand(embed_user_MLP.shape[0], -1, -1)

        interaction = torch.cat((embed_user_MLP, embed_item_MLP), dim=2)

        output_MLP = self.MLP_layers(interaction)
        predictions = self.predict_layer(output_MLP).squeeze()
        return predictions



class GCMC(nn.Module):
    def __init__(self,
                 args,
                 num_users: int,
                 num_items: int,
                 Graph):
        super(GCMC, self).__init__()
        self.config = args
        self.num_users = num_users
        self.num_items = num_items
        self.__init_weight()
        self.Graph = Graph

    def __init_weight(self):
        self.latent_dim = self.config.hidden_dim_benign
        self.n_layers = self.config.num_layers_benign
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.gc_mlp = nn.ModuleList()
        self.bi_mlp = nn.ModuleList()
        for k in range(self.n_layers):
            self.gc_mlp.append(nn.Linear(self.latent_dim, self.latent_dim))
            self.bi_mlp.append(nn.Linear(self.latent_dim, self.latent_dim))

        self.f = nn.Sigmoid()

    def computer(self, Graph):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        ego_emb = torch.cat([users_emb, items_emb], 0)
        all_emb = [ego_emb]

        g_droped = Graph  # not using dropout
        for layer in range(self.n_layers):  # 3 layers
            side_emb = torch.sparse.mm(g_droped, ego_emb)
            ego_emb = F.leaky_relu(self.gc_mlp[layer](side_emb))
            mlp_emb = self.bi_mlp[layer](ego_emb)
            mlp_emb = nn.Dropout(0.1)(mlp_emb)  # !!!!!!!!!!!
            all_emb += [mlp_emb]
        all_emb = torch.cat(all_emb, 1)
        users = all_emb[:self.num_users, :]
        items = all_emb[self.num_users:, :]
        # print(users.shape, items.shape)
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer(self.Graph)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user.weight[users]
        pos_emb_ego = self.embedding_item.weight[pos_items]
        neg_emb_ego = self.embedding_item.weight[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def loss(self, users, pos, neg, labels):

        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(),
                                                                                      neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        Loss = reg_loss * self.config.weight_decay + loss
        return Loss

    def getUsersRating(self, users, Graph):
        all_users, all_items = self.computer(Graph)
        users_emb = all_users[users]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating


class NGCF(nn.Module):
    def __init__(self,
                 args,
                 num_users: int,
                 num_items: int,
                 Graph):
        super(NGCF, self).__init__()
        self.config = args
        self.num_users = num_users
        self.num_items = num_items
        self.__init_weight()
        self.Graph = Graph

    def __init_weight(self):
        self.latent_dim = self.config.hidden_dim_benign
        self.n_layers = self.config.num_layers_benign
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.gc_mlp = nn.ModuleList()
        self.bi_mlp = nn.ModuleList()
        for k in range(self.n_layers):
            self.gc_mlp.append(nn.Linear(self.latent_dim, self.latent_dim))
            self.bi_mlp.append(nn.Linear(self.latent_dim, self.latent_dim))

        self.f = nn.Sigmoid()

    def sparse_dropout(self, x, rate):
        noise_shape = x._nnz()
        random_tensor = 1 - rate
        random_tensor += torch.rand(noise_shape).to(x.device)
        dropout_mask = torch.floor(random_tensor).type(torch.bool)
        i = x._indices()
        v = x._values()

        i = i[:, dropout_mask]
        v = v[dropout_mask]

        out = torch.sparse_coo_tensor(i, v, x.shape).to(x.device)

        return out * (1. / (1 - rate))

    def computer(self, Graph):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        ego_emb = torch.cat([users_emb, items_emb], 0)
        all_emb = [ego_emb]

        g_droped = Graph  # not using dropout
        # g_droped = self.sparse_dropout(Graph, 0.1)

        for layer in range(self.n_layers):  # 3 layers
            side_emb = torch.sparse.mm(g_droped, ego_emb)
            sum_emb = F.leaky_relu(self.gc_mlp[layer](side_emb))
            bi_emb = torch.mul(ego_emb, side_emb)
            bi_emb = F.leaky_relu(self.bi_mlp[layer](bi_emb))
            ego_emb = sum_emb + bi_emb
            ego_emb = nn.Dropout(0.1)(ego_emb)  # !!!!!!!!!!!
            norm_emb = F.normalize(ego_emb, p=2, dim=1)
            all_emb += [norm_emb]
        all_emb = torch.cat(all_emb, 1)
        users = all_emb[:self.num_users, :]
        items = all_emb[self.num_users:, :]
        # print(users.shape, items.shape)
        return users, items

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer(self.Graph)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user.weight[users]
        pos_emb_ego = self.embedding_item.weight[pos_items]
        neg_emb_ego = self.embedding_item.weight[neg_items]
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def loss(self, users, pos, neg, labels):

        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(),
                                                                                      neg.long())
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)
        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        Loss = reg_loss * self.config.weight_decay + loss
        return Loss


    def getUsersRating(self, users, Graph):
        all_users, all_items = self.computer(Graph)
        users_emb = all_users[users]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating


class LightGCN(nn.Module):
    def __init__(self,
                 args,
                 num_users: int,
                 num_items: int,
                 Graph):
        super(LightGCN, self).__init__()
        self.config = args
        self.num_users = num_users
        self.num_items = num_items
        self.__init_weight()
        self.Graph = Graph

    def __init_weight(self):
        self.latent_dim = self.config.hidden_dim_benign
        self.n_layers = self.config.num_layers_benign
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        # world.cprint('use NORMAL distribution initilizer')
        self.f = nn.Sigmoid()

    def computer(self, G):
        users_emb = self.embedding_user.weight
        items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])  # [70839, 64]
        embs = [all_emb]  # list; len 1; initial u-i embeddings

        g_droped = G  # not using dropout

        for layer in range(self.n_layers):  # 3 layers
            all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=0)
        light_out = torch.mean(embs, dim=0)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        # print(users.shape, items.shape)
        return users, items

    def getEmbedding(self, users, pos_items, neg_items, G):
        all_users, all_items = self.computer(G)
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def getAllEmb(self):
        all_users, all_items = self.computer(self.Graph)
        return all_users, all_items

    def loss(self, users, pos, neg, labels):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(users.long(), pos.long(),
                                                                                      neg.long(), self.Graph)
        reg_loss = (1 / 2) * (userEmb0.norm(2).pow(2) +
                              posEmb0.norm(2).pow(2) +
                              negEmb0.norm(2).pow(2)) / float(len(users))
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))
        Loss = reg_loss * self.config.weight_decay + loss
        return Loss

    def getUsersRating(self, users, G):
        all_users, all_items = self.computer(G)

        users_emb = all_users[users]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating


MODELS = {'bpr': BPR, 'ncf': NCF, 'gcmc': GCMC, 'ngcf': NGCF, 'lgn': LightGCN}

def RS_Model(args, num_users, num_items, Graph=None, device='cpu'):
    if args.model_name in ['bpr', 'ncf']:
        model = MODELS[args.model_name](args, num_users, num_items).to(device)
    else:
        model = MODELS[args.model_name](args, num_users, num_items, Graph).to(device)
    return model
