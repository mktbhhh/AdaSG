# __all__ = ["LightGCN", "lightgcn"]

"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import os
import world
import torch
from dataloader import BasicDataset
import dataloader
from torch import nn
import numpy as np


class BasicModel(nn.Module):
    def __init__(self):
        super(BasicModel, self).__init__()

    def getUsersRating(self, users):
        raise NotImplementedError


class LightGCN(BasicModel):
    def __init__(self, config, dataset: BasicDataset):
        super(LightGCN, self).__init__()

        self.config = config
        self.dataset = dataset
        self.__init_weight()

    def __init_weight(self):
        self.num_users = self.dataset.n_users
        self.num_items = self.dataset.m_items
        self.latent_dim = self.config.latent_dim_rec
        self.n_layers = self.config.lightGCN_n_layers
        self.keep_prob = self.config.keep_prob
        self.A_split = self.config.A_split
        self.full_precision_flag = True
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim
        )
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim
        )
        if self.config.pretrain == 0:
            #             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
            #             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
            #             print('use xavier initilizer')
            # random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint("use NORMAL distribution initilizer")
        else:
            self.embedding_user.weight.data.copy_(
                torch.from_numpy(self.config["user_emb"])
            )
            self.embedding_item.weight.data.copy_(
                torch.from_numpy(self.config["item_emb"])
            )
            print("use pretarined data")
        self.f = nn.Sigmoid()
        self.Graph = self.dataset.getSparseGraph()
        print(f"lgn is already to go(dropout:{self.config.dropout})")

        # print("save_txt")

    def __dropout_x(self, x, keep_prob):
        size = x.size()
        index = x.indices().t()
        values = x.values()
        random_index = torch.rand(len(values)) + keep_prob
        random_index = random_index.int().bool()
        index = index[random_index]
        values = values[random_index] / keep_prob
        g = torch.sparse.FloatTensor(index.t(), values, size)
        return g

    def __dropout(self, keep_prob):
        if self.A_split:
            graph = []
            for g in self.Graph:
                graph.append(self.__dropout_x(g, keep_prob))
        else:
            graph = self.__dropout_x(self.Graph, keep_prob)
        return graph

    def computer(self):
        """
        propagate methods for lightGCN
        """
        if not self.full_precision_flag:
            users_emb = self.embedding_user.get_quantization_weight()
            items_emb = self.embedding_item.get_quantization_weight()
        else:
            users_emb = self.embedding_user.weight
            items_emb = self.embedding_item.weight
        all_emb = torch.cat([users_emb, items_emb])
        #   torch.split(all_emb , [self.num_users, self.num_items])
        embs = [all_emb]
        if self.config.dropout:
            if self.training:
                print("droping")
                g_droped = self.__dropout(self.keep_prob)
            else:
                g_droped = self.Graph
        else:
            g_droped = self.Graph

        for layer in range(self.n_layers):
            if self.A_split:
                temp_emb = []
                for f in range(len(g_droped)):
                    temp_emb.append(torch.sparse.mm(g_droped[f], all_emb))
                side_emb = torch.cat(temp_emb, dim=0)
                all_emb = side_emb
            else:
                all_emb = torch.sparse.mm(g_droped, all_emb)
            embs.append(all_emb)
        embs = torch.stack(embs, dim=1)
        # print(embs.size())
        light_out = torch.mean(embs, dim=1)
        users, items = torch.split(light_out, [self.num_users, self.num_items])
        if not self.full_precision_flag:
            users = self.embedding_user.quantization_last_emb(users)
            items = self.embedding_item.quantization_last_emb(items)
        return users, items

    def getUsersRating(self, users):
        all_users, all_items = self.computer()
        users_emb = all_users[users.long()]
        items_emb = all_items
        rating = self.f(torch.matmul(users_emb, items_emb.t()))
        return rating

    # def getUserItemScore(self, userid, itemid):
    #     all_users, all_items = self.computer()
    #     users_emb = all_users[userid.long()]
    #     items_emb = all_items[itemid.long()]
    #     return self.f(torch.mul(users_emb, items_emb))

    def getEmbedding(self, users, pos_items, neg_items):
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        pos_emb = all_items[pos_items]
        neg_emb = all_items[neg_items]
        users_emb_ego = self.embedding_user(users)
        pos_emb_ego = self.embedding_item(pos_items)
        neg_emb_ego = self.embedding_item(neg_items)
        return users_emb, pos_emb, neg_emb, users_emb_ego, pos_emb_ego, neg_emb_ego

    def bpr_loss(self, users, pos, neg):
        (users_emb, pos_emb, neg_emb, userEmb0, posEmb0, negEmb0) = self.getEmbedding(
            users.long(), pos.long(), neg.long()
        )
        reg_loss = (
            (1 / 2)
            * (
                userEmb0.norm(2).pow(2)
                + posEmb0.norm(2).pow(2)
                + negEmb0.norm(2).pow(2)
            )
            / float(len(users))
        )
        pos_scores = torch.mul(users_emb, pos_emb)
        pos_scores = torch.sum(pos_scores, dim=1)
        neg_scores = torch.mul(users_emb, neg_emb)
        neg_scores = torch.sum(neg_scores, dim=1)

        loss = torch.mean(torch.nn.functional.softplus(neg_scores - pos_scores))

        return loss, reg_loss

    def forward(self, users, items):
        # compute embedding
        users_emb0 = self.embedding_user(users)
        items_emb0 = self.embedding_item(items)
        # all_users, all_items = self.computer()
        # print('forward')
        all_users, all_items = self.computer()
        users_emb = all_users[users]
        items_emb = all_items[items]
        inner_pro = torch.mul(users_emb, items_emb)
        # gamma = torch.sum(inner_pro, dim=1)
        return inner_pro


MODELS = {"lgn": LightGCN}


def getFileName(option, root):
    if option.model_name == "mf":
        file = f"mf-{option.dataset}-{option.latent_dim_rec}.pth.tar"
    elif option.model_name == "lgn":
        file = f"lgn-{option.dataset}-{option.lightGCN_n_layers}-{option.latent_dim_rec}.pth.tar"
    return os.path.join(root, file)


def get_lightgcn(
    model_name=None, pretrained=True, root=os.path.join("pretrained"), **kwargs
):
    dataset = kwargs["dataset"]
    settings = kwargs["settings"]

    if pretrained:
        if (model_name is None) or (not model_name):
            raise ValueError(
                "Parameter `model_name` should be properly initialized for loading pretrained model."
            )

        # dataset = dataloader.Loader(settings, path="dataset_path/"+world.dataset)
        Recmodel = MODELS[world.model_name](settings, dataset)
        Recmodel = Recmodel.to(world.device)

        weight_file = getFileName(settings, root)
        Recmodel.load_state_dict(
            torch.load(weight_file, map_location=torch.device("cpu"))
        )

    return Recmodel


def lightgcn(**kwargs):
    return get_lightgcn(model_name="lightgcn", **kwargs)
