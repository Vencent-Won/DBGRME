import torch

import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn import MLP



def bpr_loss_cal(batch_data0, batch_data1, weight_decay):
    useremb0, posemb0, negemb0 = batch_data0
    users_emb, pos_emb, neg_emb = batch_data1
    reg_loss = (1 / 2) * (useremb0.norm(2).pow(2) +
                          posemb0.norm(2).pow(2) +
                          negemb0.norm(2).pow(2)) / float(useremb0.shape[0]) * weight_decay
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=-1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=-1)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    loss += reg_loss * weight_decay
    return loss



class Surrogate(nn.Module):
    def __init__(self, num_users, num_items, num_fakers, hidden_dim, num_layers, dropout, weight_decay):
        super(Surrogate, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_fakers = num_fakers
        self.n_layers = num_layers
        self.latent_dim = hidden_dim
        self.dropout = dropout
        self.weight_decay = weight_decay
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_fakers = torch.nn.Embedding(num_embeddings=self.num_fakers, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.item_projector = MLP([self.latent_dim, self.latent_dim, self.latent_dim], dropout=self.dropout)
        self.user_projector = MLP([self.latent_dim, self.latent_dim, self.latent_dim], dropout=self.dropout)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)
        self.batch_norm0 = nn.BatchNorm1d(self.latent_dim)
        self.layer_norm0 = nn.LayerNorm(self.latent_dim)

        self.batch_norms = torch.nn.ModuleList()
        for layer in range(self.n_layers):
            self.batch_norms.append(torch.nn.BatchNorm1d(self.latent_dim))
        self.layer_norms = torch.nn.ModuleList()
        for layer in range(self.n_layers):
            self.layer_norms.append(nn.LayerNorm(self.latent_dim))

    def init_fakers(self):
        # nn.init.xavier_uniform_(self.embedding_fakers.weight)
        nn.init.normal_(self.embedding_fakers.weight, std=0.1)

    def get_user_embedding(self, mode='target_fake'):
        user_emb = self.embedding_user.weight
        if 'fake' in mode:
            fake_users_emb = self.embedding_fakers.weight
            if mode == 'target_fake':
                user_emb = torch.cat([user_emb, fake_users_emb], dim=0)
            else:
                user_emb = fake_users_emb
        return user_emb

    def computer(self, g_normed, mode='target_fake'):
        items_emb = self.embedding_item.weight
        items_emb = self.item_projector(items_emb)
        user_emb = self.get_user_embedding(mode)
        user_emb = self.user_projector(user_emb)
        all_emb = torch.cat([user_emb, items_emb], dim=0)
        all_emb = self.layer_norm0(all_emb)
        all_emb = self.batch_norm0(all_emb)
        embs = [all_emb]
        embs_ori = all_emb.clone()
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_normed, all_emb)
            all_emb = embs_ori + all_emb
            all_emb = self.layer_norms[layer](all_emb)
            all_emb = self.batch_norms[layer](all_emb)
            if layer == self.n_layers - 1:
                # remove relu for the last layer
                all_emb = F.dropout(all_emb, self.dropout, training=self.training)
            else:
                all_emb = F.dropout(F.relu(all_emb), self.dropout, training=self.training)
            embs.append(all_emb)
        # embs = torch.stack(embs, dim=1)
        # all_emb = torch.mean(embs, dim=1)
        return all_emb[:user_emb.shape[0]], all_emb[user_emb.shape[0]:]

    def loss(self, users, pos, neg, graph, mode='target_fake'):
        all_users, all_items = self.computer(graph, mode)
        batch_data1 = (all_users[users], all_items[pos], all_items[neg])
        batch_data0 = (self.get_user_embedding(mode)[users], self.embedding_item.weight[pos],
                       self.embedding_item.weight[neg])
        loss = bpr_loss_cal(batch_data0, batch_data1, self.weight_decay)
        return loss

    def getUsersRating(self, users, g_normed, mode='target_fake'):
        all_users, all_items = self.computer(g_normed, mode)
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating
