import torch

import torch.nn as nn
import torch.nn.functional as F


def bpr_loss_cal(batch_data0, batch_data1, weight_decay):
    useremb0, posemb0, negemb0 = batch_data0
    users_emb, pos_emb, neg_emb = batch_data1
    reg_loss = (1 / 2) * (useremb0.norm(2).pow(2) +
                          posemb0.norm(2).pow(2) +
                          negemb0.norm(2).pow(2)) / float(users_emb.shape[0])
    pos_scores = torch.mul(users_emb, pos_emb)
    pos_scores = torch.sum(pos_scores, dim=1)
    neg_scores = torch.mul(users_emb, neg_emb)
    neg_scores = torch.sum(neg_scores, dim=1)
    loss = -F.logsigmoid(pos_scores - neg_scores).mean()
    loss += reg_loss * weight_decay
    return loss


class BPR_T(nn.Module):
    def __init__(self, num_users, num_items, num_fakers, hidden_dim, num_layers, weight_decay):
        super(BPR_T, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_fakers = num_fakers
        self.latent_dim = hidden_dim
        self.weight_decay = weight_decay
        self.__init_weight()

    def __init_weight(self):
        self.initializer = nn.init.normal_

        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        self.embedding_fakers = torch.nn.Embedding(num_embeddings=self.num_fakers, embedding_dim=self.latent_dim)

        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def init_fakers(self):
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

    def computer(self, mode='target_fake'):
        items_emb = self.embedding_item.weight
        user_emb = self.get_user_embedding(mode)
        bpr_out = torch.cat([user_emb, items_emb], dim=0)

        return bpr_out[:user_emb.shape[0]], bpr_out[user_emb.shape[0]:]

    def getUsersRating(self, users, graph, mode='target_fake'):
        all_users, all_items = self.computer(mode)
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating


class NCF_T(nn.Module):
    def __init__(self, num_users, num_items, num_fakers, hidden_dim, num_layers, weight_decay):
        super(NCF_T, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_fakers = num_fakers
        self.n_layers = num_layers
        self.latent_dim = hidden_dim
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim * (2 ** (self.n_layers - 1)))
        self.embedding_item = torch.nn.Embedding(self.num_items, self.latent_dim * (2 ** (self.n_layers - 1)))
        self.embedding_fakers = torch.nn.Embedding(self.num_fakers, self.latent_dim * (2 ** (self.n_layers - 1)))
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        MLP_modules = []
        for i in range(self.n_layers):
            input_size = self.latent_dim * (2 ** (self.n_layers - i))
            MLP_modules.append(nn.Dropout(p=0.1))
            MLP_modules.append(nn.Linear(input_size, input_size // 2))
            MLP_modules.append(nn.ReLU())
        self.MLP_layers = nn.Sequential(*MLP_modules)
        self.predict_layer = nn.Linear(self.latent_dim, 1)
        self.loss_function = nn.BCEWithLogitsLoss()


        nn.init.kaiming_uniform_(self.predict_layer.weight, a=1, nonlinearity='sigmoid')

    def init_fakers(self):
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

    def computer(self, mode='target_fake'):
        items_emb = self.embedding_item.weight
        user_emb = self.get_user_embedding(mode)
        ncf_out = torch.cat([user_emb, items_emb], dim=0)

        return ncf_out[:user_emb.shape[0]], ncf_out[user_emb.shape[0]:]


    def getUsersRating(self, users, _, mode='target_fake'):
        all_users, all_items = self.computer(mode)
        embed_user_MLP = all_users[users]
        embed_item_MLP = all_items

        embed_user_MLP = embed_user_MLP.unsqueeze(1).expand(-1, self.num_items, -1)
        embed_item_MLP = embed_item_MLP.unsqueeze(0).expand(embed_user_MLP.shape[0], -1, -1)

        interaction = torch.cat((embed_user_MLP, embed_item_MLP), dim=2)

        output_MLP = self.MLP_layers(interaction)
        predictions = self.predict_layer(output_MLP).squeeze()
        return predictions



class GCMC_T(nn.Module):
    def __init__(self, num_users, num_items, num_fakers, hidden_dim, num_layers, weight_decay):
        super(GCMC_T, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_fakers = num_fakers
        self.n_layers = num_layers
        self.latent_dim = hidden_dim
        self.weight_decay = weight_decay
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_fakers = torch.nn.Embedding(num_embeddings=self.num_fakers, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.gc_mlp = nn.ModuleList()
        self.bi_mlp = nn.ModuleList()
        for k in range(self.n_layers):
            self.gc_mlp.append(nn.Linear(self.latent_dim, self.latent_dim))
            self.bi_mlp.append(nn.Linear(self.latent_dim, self.latent_dim))

        # self.f = nn.Sigmoid()

    def init_fakers(self):
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
        user_emb = self.get_user_embedding(mode)

        emb = torch.cat([user_emb, items_emb], dim=0)
        all_emb = [emb]
        for layer in range(self.n_layers):  # 3 layers
            side_emb = torch.sparse.mm(g_normed, emb)
            emb = F.leaky_relu(self.gc_mlp[layer](side_emb))
            mlp_emb = self.bi_mlp[layer](emb)
            mlp_emb = nn.Dropout(0.1)(mlp_emb)  # !!!!!!!!!!!
            all_emb += [mlp_emb]
        all_emb = torch.cat(all_emb, 1)
        return all_emb[:user_emb.shape[0]], all_emb[user_emb.shape[0]:]


    def getUsersRating(self, users, g_normed, mode='target_fake'):
        all_users, all_items = self.computer(g_normed, mode)
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating


class NGCF_T(nn.Module):
    def __init__(self, num_users, num_items, num_fakers, hidden_dim, num_layers, weight_decay):
        super(NGCF_T, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_fakers = num_fakers
        self.n_layers = num_layers
        self.latent_dim = hidden_dim
        self.weight_decay = weight_decay
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(self.num_users, self.latent_dim)
        self.embedding_fakers = torch.nn.Embedding(self.num_fakers, self.latent_dim)
        self.embedding_item = torch.nn.Embedding(self.num_items, self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

        self.gc_mlp = nn.ModuleList()
        self.bi_mlp = nn.ModuleList()
        for k in range(self.n_layers):
            self.gc_mlp.append(nn.Linear(self.latent_dim, self.latent_dim))
            self.bi_mlp.append(nn.Linear(self.latent_dim, self.latent_dim))

        self.f = nn.Sigmoid()

    def init_fakers(self):
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
        user_emb = self.get_user_embedding(mode)
        ego_emb = torch.cat([user_emb, items_emb], dim=0)
        all_emb = [ego_emb]

        for layer in range(self.n_layers):  # 3 layers
            side_emb = torch.sparse.mm(g_normed, ego_emb)
            sum_emb = F.leaky_relu(self.gc_mlp[layer](side_emb))
            bi_emb = torch.mul(ego_emb, side_emb)
            bi_emb = F.leaky_relu(self.bi_mlp[layer](bi_emb))
            ego_emb = sum_emb + bi_emb
            ego_emb = nn.Dropout(0.1)(ego_emb)  # !!!!!!!!!!!
            norm_emb = F.normalize(ego_emb, p=2, dim=1)
            all_emb += [norm_emb]
        all_emb = torch.cat(all_emb, 1)
        users = all_emb[:user_emb.shape[0]]
        items = all_emb[user_emb.shape[0]:]
        # print(users.shape, items.shape)
        return users, items

    def getUsersRating(self, users, g_normed, mode='target_fake'):
        all_users, all_items = self.computer(g_normed, mode)
        users_emb = all_users[users]
        items_emb = all_items
        rating = torch.matmul(users_emb, items_emb.t())
        return rating


class LightGCN_T(nn.Module):
    def __init__(self, num_users, num_items, num_fakers, hidden_dim, num_layers, weight_decay):
        super(LightGCN_T, self).__init__()
        self.num_users = num_users
        self.num_items = num_items
        self.num_fakers = num_fakers
        self.n_layers = num_layers
        self.latent_dim = hidden_dim
        self.weight_decay = weight_decay
        self.__init_weight()

    def __init_weight(self):
        self.embedding_user = torch.nn.Embedding(num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_fakers = torch.nn.Embedding(num_embeddings=self.num_fakers, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        nn.init.normal_(self.embedding_user.weight, std=0.1)
        nn.init.normal_(self.embedding_item.weight, std=0.1)

    def init_fakers(self):
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
        user_emb = self.get_user_embedding(mode)
        all_emb = torch.cat([user_emb, items_emb], dim=0)
        embs = [all_emb]
        for layer in range(self.n_layers):
            all_emb = torch.sparse.mm(g_normed, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        light_out = torch.mean(embs, dim=1)
        return light_out[:user_emb.shape[0]], light_out[user_emb.shape[0]:]
        
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
