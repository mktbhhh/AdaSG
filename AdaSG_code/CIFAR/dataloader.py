"""
data loder for loading data
"""
import os
import math
import torch
import torch.utils.data as data
import numpy as np
from PIL import Image
import torchvision
import torchvision.datasets as dsets
import torchvision.transforms as transforms
import struct

from torch.utils.data import Dataset
from options import Option
import world
from world import cprint
from scipy.sparse import csr_matrix
import scipy.sparse as sp
from time import time

__all__ = ["DataLoader", "PartDataLoader"]


class ImageLoader(data.Dataset):
    def __init__(self, dataset_dir, transform=None, target_transform=None):
        class_list = os.listdir(dataset_dir)
        datasets = []
        for cla in class_list:
            cla_path = os.path.join(dataset_dir, cla)
            files = os.listdir(cla_path)
            for file_name in files:
                file_path = os.path.join(cla_path, file_name)
                if os.path.isfile(file_path):
                    # datasets.append((file_path, tuple([float(v) for v in int(cla)])))
                    datasets.append((file_path, [float(cla)]))
                    # print(datasets)
                    # assert False

        self.dataset_dir = dataset_dir
        self.datasets = datasets
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        frames = []

        file_path, label = self.datasets[index]
        noise = torch.load(file_path, map_location=torch.device("cpu"))
        return noise, torch.Tensor(label)

    def __len__(self):
        return len(self.datasets)


class DataLoader(object):
    """
    data loader for CV data sets
    """

    def __init__(
        self,
        dataset,
        batch_size,
        n_threads=4,
        ten_crop=False,
        data_path="/home/dataset/",
        logger=None,
    ):
        """
        create data loader for specific data set
        :params n_treads: number of threads to load data, default: 4
        :params ten_crop: use ten crop for testing, default: False
        :params data_path: path to data set, default: /home/dataset/
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.n_threads = n_threads
        self.ten_crop = ten_crop
        self.data_path = data_path
        self.logger = logger
        self.dataset_root = data_path

        self.logger.info("|===>Creating data loader for " + self.dataset)

        if self.dataset in ["cifar100"]:
            self.train_loader, self.test_loader = self.cifar(dataset=self.dataset)

        elif self.dataset in ["imagenet"]:
            self.train_loader, self.test_loader = self.imagenet(dataset=self.dataset)

        # elif self.dataset in ["gowalla"]:
        #     self.gnn_loader = self.gowalla()

        else:
            assert False, "invalid data set"

    def getloader(self):
        """
        get train_loader and test_loader
        """
        return self.train_loader, self.test_loader

    def get_gnn_loader(self):
        return self.gnn_loader

    def imagenet(self, dataset="imagenet"):
        traindir = os.path.join(self.data_path, "train")
        testdir = os.path.join(self.data_path, "val")

        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

        train_loader = torch.utils.data.DataLoader(
            dsets.ImageFolder(
                traindir,
                transforms.Compose(
                    [
                        transforms.RandomResizedCrop(224),
                        transforms.RandomHorizontalFlip(),
                        transforms.ToTensor(),
                        normalize,
                    ]
                ),
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.n_threads,
            pin_memory=True,
        )

        test_transform = transforms.Compose(
            [
                transforms.Resize(256),
                # transforms.Scale(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]
        )

        test_loader = torch.utils.data.DataLoader(
            dsets.ImageFolder(testdir, test_transform),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.n_threads,
            pin_memory=False,
        )
        return train_loader, test_loader

    def cifar(self, dataset="cifar100"):
        """
        dataset: cifar
        """
        if dataset == "cifar10":
            norm_mean = [0.49139968, 0.48215827, 0.44653124]
            norm_std = [0.24703233, 0.24348505, 0.26158768]
        elif dataset == "cifar100":
            norm_mean = [0.50705882, 0.48666667, 0.44078431]
            norm_std = [0.26745098, 0.25568627, 0.27607843]

        else:
            assert False, "Invalid cifar dataset"

        test_data_root = self.dataset_root

        test_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize(norm_mean, norm_std)]
        )

        if self.dataset == "cifar10":
            test_dataset = dsets.CIFAR10(
                root=test_data_root, train=False, transform=test_transform
            )
        elif self.dataset == "cifar100":
            test_dataset = dsets.CIFAR100(
                root=test_data_root,
                train=False,
                transform=test_transform,
                download=True,
            )
        else:
            assert False, "invalid data set"

        test_loader = torch.utils.data.DataLoader(
            dataset=test_dataset,
            batch_size=200,
            shuffle=False,
            pin_memory=True,
            num_workers=self.n_threads,
        )
        return None, test_loader

    # def gowalla(self, dataset="gowalla"):
    #     dataset = Loader(option, path="../dataset_path/" + dataset)
    #     return dataset


class BasicDataset(Dataset):
    def __init__(self):
        print("init dataset")

    @property
    def n_users(self):
        raise NotImplementedError

    @property
    def m_items(self):
        raise NotImplementedError

    @property
    def trainDataSize(self):
        raise NotImplementedError

    @property
    def testDict(self):
        raise NotImplementedError

    @property
    def allPos(self):
        raise NotImplementedError

    def getUserItemFeedback(self, users, items):
        raise NotImplementedError

    def getUserPosItems(self, users):
        raise NotImplementedError

    def getUserNegItems(self, users):
        """
        not necessary for large dataset
        it's stupid to return all neg items in super large dataset
        """
        raise NotImplementedError

    def getSparseGraph(self):
        """
        build a graph in torch.sparse.IntTensor.
        Details in NGCF's matrix form
        A =
            |I,   R|
            |R^T, I|
        """
        raise NotImplementedError


class Loader(BasicDataset):
    """
    Dataset type for pytorch \n
    Incldue graph information
    gowalla dataset
    """

    def __init__(self, settings, path="./dataset_path/gowalla"):
        # train or test
        cprint(f"loading [{path}]")
        self.settings = settings
        self.split = self.settings.A_split
        self.folds = self.settings.A_n_fold
        self.mode_dict = {"train": 0, "test": 1}
        self.mode = self.mode_dict["train"]
        self.n_user = 0
        self.m_item = 0
        train_file = path + "/train.txt"
        test_file = path + "/test.txt"
        self.path = path
        trainUniqueUsers, trainItem, trainUser = [], [], []
        testUniqueUsers, testItem, testUser = [], [], []
        self.traindataSize = 0
        self.testDataSize = 0

        with open(train_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n").split(" ")
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    trainUniqueUsers.append(uid)
                    trainUser.extend([uid] * len(items))
                    trainItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.traindataSize += len(items)
        self.trainUniqueUsers = np.array(trainUniqueUsers)
        self.trainUser = np.array(trainUser)
        self.trainItem = np.array(trainItem)

        with open(test_file) as f:
            for l in f.readlines():
                if len(l) > 0:
                    l = l.strip("\n").split(" ")
                    items = [int(i) for i in l[1:]]
                    uid = int(l[0])
                    testUniqueUsers.append(uid)
                    testUser.extend([uid] * len(items))
                    testItem.extend(items)
                    self.m_item = max(self.m_item, max(items))
                    self.n_user = max(self.n_user, uid)
                    self.testDataSize += len(items)
        self.m_item += 1
        self.n_user += 1
        self.testUniqueUsers = np.array(testUniqueUsers)
        self.testUser = np.array(testUser)
        self.testItem = np.array(testItem)

        self.Graph = None
        print(f"{self.trainDataSize} interactions for training")
        print(f"{self.testDataSize} interactions for testing")
        print(
            f"{self.settings.dataset} Sparsity : {(self.trainDataSize + self.testDataSize) / self.n_users / self.m_items}"
        )

        # (users,items), bipartite graph
        self.UserItemNet = csr_matrix(
            (np.ones(len(self.trainUser)), (self.trainUser, self.trainItem)),
            shape=(self.n_user, self.m_item),
        )
        self.users_D = np.array(self.UserItemNet.sum(axis=1)).squeeze()
        self.users_D[self.users_D == 0.0] = 1
        self.items_D = np.array(self.UserItemNet.sum(axis=0)).squeeze()
        self.items_D[self.items_D == 0.0] = 1.0
        # pre-calculate
        self._allPos = self.getUserPosItems(list(range(self.n_user)))
        self.__testDict = self.__build_test()
        print(f"{self.settings.dataset} is ready to go")

    @property
    def n_users(self):
        return self.n_user

    @property
    def m_items(self):
        return self.m_item

    @property
    def trainDataSize(self):
        return self.traindataSize

    @property
    def testDict(self):
        return self.__testDict

    @property
    def allPos(self):
        return self._allPos

    def _split_A_hat(self, A):
        A_fold = []
        fold_len = (self.n_users + self.m_items) // self.folds
        for i_fold in range(self.folds):
            start = i_fold * fold_len
            if i_fold == self.folds - 1:
                end = self.n_users + self.m_items
            else:
                end = (i_fold + 1) * fold_len
            A_fold.append(
                self._convert_sp_mat_to_sp_tensor(A[start:end])
                .coalesce()
                .to(world.device)
            )
        return A_fold

    def _convert_sp_mat_to_sp_tensor(self, X):
        coo = X.tocoo().astype(np.float32)
        row = torch.Tensor(coo.row).long()
        col = torch.Tensor(coo.col).long()
        index = torch.stack([row, col])
        data = torch.FloatTensor(coo.data)
        return torch.sparse.FloatTensor(index, data, torch.Size(coo.shape))

    def getSparseGraph(self):
        print("loading adjacency matrix")
        if self.Graph is None:
            try:
                pre_adj_mat = sp.load_npz(self.path + "/s_pre_adj_mat.npz")
                print("successfully loaded...")
                norm_adj = pre_adj_mat
            except:
                print("generating adjacency matrix")
                s = time()
                adj_mat = sp.dok_matrix(
                    (self.n_users + self.m_items, self.n_users + self.m_items),
                    dtype=np.float32,
                )
                adj_mat = adj_mat.tolil()
                R = self.UserItemNet.tolil()
                adj_mat[: self.n_users, self.n_users :] = R
                adj_mat[self.n_users :, : self.n_users] = R.T
                adj_mat = adj_mat.todok()
                # adj_mat = adj_mat + sp.eye(adj_mat.shape[0])

                rowsum = np.array(adj_mat.sum(axis=1))
                d_inv = np.power(rowsum, -0.5).flatten()
                d_inv[np.isinf(d_inv)] = 0.0
                d_mat = sp.diags(d_inv)

                norm_adj = d_mat.dot(adj_mat)
                norm_adj = norm_adj.dot(d_mat)
                norm_adj = norm_adj.tocsr()
                end = time()
                print(f"costing {end-s}s, saved norm_mat...")
                sp.save_npz(self.path + "/s_pre_adj_mat.npz", norm_adj)

            if self.split == True:
                self.Graph = self._split_A_hat(norm_adj)
                print("done split matrix")
            else:
                self.Graph = self._convert_sp_mat_to_sp_tensor(norm_adj)
                self.Graph = self.Graph.coalesce().to(world.device)
                print("don't split the matrix")
        return self.Graph

    def __build_test(self):
        """
        return:
            dict: {user: [items]}
        """
        test_data = {}
        for i, item in enumerate(self.testItem):
            user = self.testUser[i]
            if test_data.get(user):
                test_data[user].append(item)
            else:
                test_data[user] = [item]
        return test_data

    def getUserItemFeedback(self, users, items):
        """
        users:
            shape [-1]
        items:
            shape [-1]
        return:
            feedback [-1]
        """
        # print(self.UserItemNet[users, items])
        return np.array(self.UserItemNet[users, items]).astype("uint8").reshape((-1,))

    def getUserPosItems(self, users):
        posItems = []
        for user in users:
            posItems.append(self.UserItemNet[user].nonzero()[1])
        return posItems

    # def getUserNegItems(self, users):
    #     negItems = []
    #     for user in users:
    #         negItems.append(self.allNeg[user])
    #     return negItems
