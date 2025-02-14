import torch

import torch.nn as nn

from torch_geometric.nn import MLP
from torch_geometric.data import DataLoader




def gumble_discrete(scores, k, device):
    (topk_values, topk_indices) = torch.topk(scores, k, dim=-1)
    # print(topk_values)
    # print(topk_indices)
    topk_indices = topk_indices - topk_values.detach() + topk_values
    edge_index0 = torch.arange(scores.shape[0]).repeat_interleave(k).to(device)
    edge_index = torch.vstack([edge_index0, topk_indices.view(scores.shape[0] * k)]).long()
    return topk_indices, edge_index


class Generator(nn.Module):
    def __init__(self, num_users, num_items, in_dim_generator, hidden_dim_generator, out_dim_generator, num_layers,
                 batch_size):
        super(Generator, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.in_dim = in_dim_generator
        self.latent_dim = hidden_dim_generator
        self.out_dim = out_dim_generator
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.in_dim)
        self.item_projector = MLP(in_channels=self.in_dim, hidden_channels=self.latent_dim, out_channels=self.out_dim,
                                  num_layers=self.num_layers, act='tanh')
        self.user_projector = MLP(in_channels=self.in_dim, hidden_channels=self.latent_dim, out_channels=self.out_dim,
                                  num_layers=self.num_layers, act='tanh')
        self.__init_weight()

    def __init_weight(self):
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def init_fakers(self):
        nn.init.normal_(self.embedding_user.weight, std=0.1)


    def forward(self, z):
        scores = None
        emb_items = self.embedding_item.weight
        emb_items = self.item_projector(emb_items)
        user_loader = DataLoader(z, batch_size=self.batch_size, shuffle=False)
        for i, batch_user in enumerate(user_loader):
            emb_usr = self.user_projector(batch_user)
            if i < 1:
                scores = torch.matmul(emb_usr, emb_items.t())
            else:
                scores = torch.vstack([scores, torch.matmul(self.user_projector(emb_usr), emb_items.t())])
        return scores



