"""
basic trainer
"""
import time

import torch.autograd
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import utils as utils
import numpy as np
import torch
from torch import optim

import math
import world
import multiprocessing

__all__ = ["Trainer", "GnnTrainer"]

from utils import gnn_utils


def compute_entropy(p_logit, T):
    p = F.softmax(p_logit / T, dim=-1)
    entropy = torch.sum(p * (-F.log_softmax(p_logit / T, dim=-1)), 1)

    return entropy


def marginal_loss(teacher_out, student_out, num_class):
    max_entropy = -(1 / num_class) * math.log(1 / num_class) * num_class
    lambda_upper = 0.8
    lambda_low = 0.3
    h_info = compute_entropy(teacher_out - student_out, 1)
    h_info_prime = (h_info - h_info.min()) / (max_entropy - h_info.min())
    zero_vector = torch.zeros(16).cuda()  # batch size 16
    h_loss = torch.mean(torch.max(lambda_low - h_info_prime, zero_vector)) + torch.mean(
        torch.max(h_info_prime - lambda_upper, zero_vector)
    )

    return h_loss


class GnnTrainer(object):
    """
    trainer for training gnn network
    """

    def __init__(
        self,
        model,
        model_teacher,
        generator,
        dataset,
        settings,
        loss_class,
        lr_master_S,
        lr_master_G,
        logger,
        run_count=0,
        tensorboard_logger=None,
    ):
        """
        init trainer
        Args:
            model: 量化模型Q
            model_teacher: 全精度模型P
            generator: 数据生成器
            dataset: 数据集加载
            settings: 全局配置
            loss_class: 损失函数
            logger: 日志记录
            tensorboard_logger: tensorboard日志
        """
        self.settings = settings

        # self.model = utils.data_parallel(model, self.settings.nGPU, self.settings.GPU)
        self.model = model.cuda()
        # self.model_teacher = utils.data_parallel(
        #     model_teacher, self.settings.nGPU, self.settings.GPU
        # )
        self.model_teacher = model_teacher.cuda()

        # self.generator = utils.data_parallel(
        #     generator, self.settings.nGPU, self.settings.GPU
        # )
        self.generator = generator.cuda()

        self.tensorboard_logger = tensorboard_logger
        self.bpr = loss_class
        self.lr_master_S = lr_master_S
        self.lr_master_G = lr_master_G
        self.MSE_loss = nn.MSELoss().cuda()

        self.optimizer_S = optim.Adam(model.parameters(), lr=self.lr_master_S.lr)
        self.log_soft = nn.LogSoftmax(dim=1)

        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.settings.lr_G,
            betas=(self.settings.b1, self.settings.b2),
        )

        self.logger = logger
        self.run_count = run_count
        self.scalar_info = {}
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.save_BN_mean = []
        self.save_BN_var = []

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        lr_S = self.lr_master_S.get_lr(epoch)
        lr_G = self.lr_master_G.get_lr(epoch)
        # update learning rate of model optimizer
        for param_group in self.optimizer_S.param_groups:
            param_group["lr"] = lr_S

        for param_group in self.optimizer_G.param_groups:
            param_group["lr"] = lr_G

    def loss_fn_kd(self, output, labels, teacher_outputs, linear=None):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        """
        q_loss = 20 * 20 * compute_entropy(teacher_outputs - output, 20)
        q_loss = -torch.mean(q_loss)

        return q_loss

    def forward(self, gen_user, gen_item, teacher_outputs, labels=None, linear=None):
        """
        forward propagation
        """
        # forward and backward and optimize
        output = self.model.getUserItemScore(gen_user, gen_item)
        if labels is not None:
            loss = self.loss_fn_kd(output, labels, teacher_outputs, linear)
            return output, loss
        else:
            return output, None

    def backward_G(self, loss_G):
        """
        backward propagation
        """
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

    def backward_S(self, loss_S):
        """
        backward propagation
        """
        self.optimizer_S.zero_grad()
        loss_S.backward()
        self.optimizer_S.step()

    def backward(self, loss):
        """
        backward propagation
        """
        self.optimizer_G.zero_grad()
        self.optimizer_S.zero_grad()
        loss.backward()
        self.optimizer_G.step()
        self.optimizer_S.step()

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def train(self, epoch):
        """
        training
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        fp_acc = utils.AverageMeter()

        iters = 200
        self.update_lr(epoch)

        self.model.eval()
        self.model_teacher.eval()
        self.generator.train()

        start_time = time.time()
        end_time = start_time

        if epoch == 0:
            for m in self.model_teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.register_forward_hook(self.hook_fn_forward)

        for i in range(iters):
            start_time = time.time()
            data_time = start_time - end_time

            z = Variable(
                torch.randn(self.settings.batchSize, self.settings.latent_dim_rec)
            ).cuda()

            # Get labels ranging from 0 to n_classes for n rows
            labels = Variable(
                torch.randint(0, self.settings.latent_dim_rec, (self.settings.batchSize,))
            ).cuda()
            z = z.contiguous()
            labels = labels.contiguous()

            gen_user, gen_item = self.generator(z, labels)

            labels_loss = Variable(
                torch.zeros(self.settings.batchSize, self.settings.latent_dim_rec)
            ).cuda()
            labels_loss.scatter_(1, labels.unsqueeze(1), 1.0)

            self.mean_list.clear()
            self.var_list.clear()

            output_teacher_batch = self.model_teacher(
                gen_user, gen_item
            )
            output = self.model(gen_user, gen_item)

            # generation loss
            z_ds = output_teacher_batch - output
            z_as = output_teacher_batch + output
            loss_ds = ((-(labels_loss * self.log_soft(z_ds)).sum(dim=1))).mean()
            loss_as = ((-(labels_loss * self.log_soft(z_as)).sum(dim=1))).mean()
            loss_onehot = 0.1 * loss_ds + 0.1 * loss_as
            h_loss = marginal_loss(output_teacher_batch, output, self.settings.nClasses)

            # BN statistic loss
            BNS_loss = torch.zeros(1).cuda()
            for num in range(len(self.mean_list)):
                BNS_loss += self.MSE_loss(
                    self.mean_list[num], self.teacher_running_mean[num]
                ) + self.MSE_loss(self.var_list[num], self.teacher_running_var[num])
            BNS_loss = BNS_loss / len(self.mean_list)

            # loss of Generator
            loss_G = h_loss + loss_onehot + BNS_loss
            self.backward_G(loss_G)

            output, loss_S = self.forward(
                gen_user.detach(),
                gen_item.detach(),
                output_teacher_batch.detach(),
                labels,
                linear=labels_loss,
            )
            if epoch >= self.settings.warmup_epochs:
                self.backward_S(loss_S)

            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output,
                labels=labels,
                loss=loss_S,
                top5_flag=True,
                mean_flag=True,
            )

            top1_error.update(single_error, 1)
            top1_loss.update(single_loss, 1)

            end_time = time.time()

            gt = labels.data.cpu().numpy()
            d_acc = np.mean(
                np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt
            )

            fp_acc.update(d_acc)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [Balance loss: %f] [BNS_loss:%f] [S loss: %f] "
            % (
                epoch + 1,
                self.settings.nEpochs,
                i + 1,
                iters,
                100 * fp_acc.avg,
                loss_G.item(),
                loss_onehot.item(),
                BNS_loss.item(),
                loss_S.item(),
            )
        )

        self.scalar_info["accuracy every epoch"] = 100 * d_acc
        self.scalar_info["G loss every epoch"] = loss_G
        self.scalar_info["One-hot loss every epoch"] = loss_onehot
        self.scalar_info["S loss every epoch"] = loss_S

        self.scalar_info["training_top1error"] = top1_error.avg
        self.scalar_info["training_loss"] = top1_loss.avg

        if self.tensorboard_logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        return top1_error.avg, top1_loss.avg

    def test_one_batch(self, X):
        sorted_items = X[0].numpy()
        groundTrue = X[1]
        r = gnn_utils.getLabel(groundTrue, sorted_items)
        pre, recall, ndcg = [], [], []
        for k in world.topks:
            ret = gnn_utils.RecallPrecision_ATk(groundTrue, r, k)
            pre.append(ret['precision'])
            recall.append(ret['recall'])
            ndcg.append(gnn_utils.NDCGatK_r(groundTrue, r, k))
        return {'recall': np.array(recall),
                'precision': np.array(pre),
                'ndcg': np.array(ndcg)}

    def test(self, dataset, epoch, w=None, multicore=0):
        """
        testing
        """
        CORES = multiprocessing.cpu_count() // 2
        u_batch_size = world.test_u_batch_size
        testDict: dict = dataset.testDict
        # eval mode with no dropout
        self.model.eval()
        max_K = max(world.topks)
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.topks)),
                   'recall': np.zeros(len(world.topks)),
                   'ndcg': np.zeros(len(world.topks))}
        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in gnn_utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = self.model.getUsersRating(batch_users_gpu)
                # rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1 << 10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [
                #         utils.AUC(rating[i],
                #                   dataset,
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.test_one_batch(x))
            scale = float(u_batch_size / len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if world.tensorboard:
                w.add_scalars(f'Test/Recall@{world.topks}',
                              {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/Precision@{world.topks}',
                              {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/NDCG@{world.topks}',
                              {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            if multicore == 1:
                pool.close()
            print(results)

        # print(
        #     "[Epoch %d/%d] [precision: %.4f%%]"
        #     % (
        #         epoch + 1,
        #         self.settings.nEpochs,
        #         results['precision'][0],
        #     )
        # )
        self.run_count += 1

        return results

    def test_teacher(self, dataset, epoch, w=None, multicore=0):
        """
        testing
        """
        CORES = multiprocessing.cpu_count() // 2
        u_batch_size = world.test_u_batch_size
        testDict: dict = dataset.testDict
        # eval mode with no dropout
        self.model_teacher.eval()
        max_K = max(world.topks)
        if multicore == 1:
            pool = multiprocessing.Pool(CORES)
        results = {'precision': np.zeros(len(world.topks)),
                   'recall': np.zeros(len(world.topks)),
                   'ndcg': np.zeros(len(world.topks))}
        with torch.no_grad():
            users = list(testDict.keys())
            try:
                assert u_batch_size <= len(users) / 10
            except AssertionError:
                print(f"test_u_batch_size is too big for this dataset, try a small one {len(users) // 10}")
            users_list = []
            rating_list = []
            groundTrue_list = []
            # auc_record = []
            # ratings = []
            total_batch = len(users) // u_batch_size + 1
            for batch_users in gnn_utils.minibatch(users, batch_size=u_batch_size):
                allPos = dataset.getUserPosItems(batch_users)
                groundTrue = [testDict[u] for u in batch_users]
                batch_users_gpu = torch.Tensor(batch_users).long()
                batch_users_gpu = batch_users_gpu.to(world.device)

                rating = self.model_teacher.getUsersRating(batch_users_gpu)
                # rating = rating.cpu()
                exclude_index = []
                exclude_items = []
                for range_i, items in enumerate(allPos):
                    exclude_index.extend([range_i] * len(items))
                    exclude_items.extend(items)
                rating[exclude_index, exclude_items] = -(1 << 10)
                _, rating_K = torch.topk(rating, k=max_K)
                rating = rating.cpu().numpy()
                # aucs = [
                #         utils.AUC(rating[i],
                #                   dataset,
                #                   test_data) for i, test_data in enumerate(groundTrue)
                #     ]
                # auc_record.extend(aucs)
                del rating
                users_list.append(batch_users)
                rating_list.append(rating_K.cpu())
                groundTrue_list.append(groundTrue)
            assert total_batch == len(users_list)
            X = zip(rating_list, groundTrue_list)
            if multicore == 1:
                pre_results = pool.map(self.test_one_batch, X)
            else:
                pre_results = []
                for x in X:
                    pre_results.append(self.test_one_batch(x))
            scale = float(u_batch_size / len(users))
            for result in pre_results:
                results['recall'] += result['recall']
                results['precision'] += result['precision']
                results['ndcg'] += result['ndcg']
            results['recall'] /= float(len(users))
            results['precision'] /= float(len(users))
            results['ndcg'] /= float(len(users))
            # results['auc'] = np.mean(auc_record)
            if world.tensorboard:
                w.add_scalars(f'Test/Recall@{world.topks}',
                              {str(world.topks[i]): results['recall'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/Precision@{world.topks}',
                              {str(world.topks[i]): results['precision'][i] for i in range(len(world.topks))}, epoch)
                w.add_scalars(f'Test/NDCG@{world.topks}',
                              {str(world.topks[i]): results['ndcg'][i] for i in range(len(world.topks))}, epoch)
            if multicore == 1:
                pool.close()
            print("test_teacher==>")
            print(results)

        # print(
        #     "[Epoch %d/%d] [precision: %.4f%%]"
        #     % (
        #         epoch + 1,
        #         self.settings.nEpochs,
        #         results['precision'][0],
        #     )
        # )
        self.run_count += 1

        return results



class Trainer(object):
    """
    trainer for training network, use SGD
    """

    def __init__(
        self,
        model,
        model_teacher,
        generator,
        lr_master_S,
        lr_master_G,
        train_loader,
        test_loader,
        settings,
        logger,
        tensorboard_logger=None,
        opt_type="SGD",
        optimizer_state=None,
        run_count=0,
    ):
        """
        init trainer
        """

        self.settings = settings

        self.model = utils.data_parallel(model, self.settings.nGPU, self.settings.GPU)
        self.model_teacher = utils.data_parallel(
            model_teacher, self.settings.nGPU, self.settings.GPU
        )

        self.generator = utils.data_parallel(
            generator, self.settings.nGPU, self.settings.GPU
        )

        self.train_loader = train_loader
        self.test_loader = test_loader
        self.tensorboard_logger = tensorboard_logger
        self.log_soft = nn.LogSoftmax(dim=1)
        self.criterion = nn.CrossEntropyLoss().cuda()
        self.bce_logits = nn.BCEWithLogitsLoss().cuda()
        self.MSE_loss = nn.MSELoss().cuda()
        self.lr_master_S = lr_master_S
        self.lr_master_G = lr_master_G
        self.opt_type = opt_type
        if opt_type == "SGD":
            self.optimizer_S = torch.optim.SGD(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                momentum=self.settings.momentum,
                weight_decay=self.settings.weightDecay,
                nesterov=True,
            )
        elif opt_type == "RMSProp":
            self.optimizer_S = torch.optim.RMSprop(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1.0,
                weight_decay=self.settings.weightDecay,
                momentum=self.settings.momentum,
                alpha=self.settings.momentum,
            )
        elif opt_type == "Adam":
            self.optimizer_S = torch.optim.Adam(
                params=self.model.parameters(),
                lr=self.lr_master_S.lr,
                eps=1e-5,
                weight_decay=self.settings.weightDecay,
            )
        else:
            assert False, "invalid type: %d" % opt_type
        if optimizer_state is not None:
            self.optimizer_S.load_state_dict(optimizer_state)
        self.optimizer_G = torch.optim.Adam(
            self.generator.parameters(),
            lr=self.settings.lr_G,
            betas=(self.settings.b1, self.settings.b2),
        )

        self.logger = logger
        self.run_count = run_count
        self.scalar_info = {}
        self.mean_list = []
        self.var_list = []
        self.teacher_running_mean = []
        self.teacher_running_var = []
        self.save_BN_mean = []
        self.save_BN_var = []

        self.fix_G = False

    def update_lr(self, epoch):
        """
        update learning rate of optimizers
        :param epoch: current training epoch
        """
        lr_S = self.lr_master_S.get_lr(epoch)
        lr_G = self.lr_master_G.get_lr(epoch)
        # update learning rate of model optimizer
        for param_group in self.optimizer_S.param_groups:
            param_group["lr"] = lr_S

        for param_group in self.optimizer_G.param_groups:
            param_group["lr"] = lr_G

    def loss_fn_kd(self, output, labels, teacher_outputs, linear=None):
        """
        Compute the knowledge-distillation (KD) loss given outputs, labels.
        "Hyperparameters": temperature and alpha
        """
        q_loss = 20 * 20 * compute_entropy(teacher_outputs - output, 20)
        q_loss = -torch.mean(q_loss)

        return q_loss

    def forward(self, images, teacher_outputs, labels=None, linear=None):
        """
        forward propagation
        """
        # forward and backward and optimize
        output, output_1 = self.model(images, True)
        if labels is not None:
            loss = self.loss_fn_kd(output, labels, teacher_outputs, linear)
            return output, loss
        else:
            return output, None

    def backward_G(self, loss_G):
        """
        backward propagation
        """
        self.optimizer_G.zero_grad()
        loss_G.backward()
        self.optimizer_G.step()

    def backward_S(self, loss_S):
        """
        backward propagation
        """
        self.optimizer_S.zero_grad()
        loss_S.backward()
        self.optimizer_S.step()

    def backward(self, loss):
        """
        backward propagation
        """
        self.optimizer_G.zero_grad()
        self.optimizer_S.zero_grad()
        loss.backward()
        self.optimizer_G.step()
        self.optimizer_S.step()

    def hook_fn_forward(self, module, input, output):
        input = input[0]
        mean = input.mean([0, 2, 3])
        # use biased var in train
        var = input.var([0, 2, 3], unbiased=False)

        self.mean_list.append(mean)
        self.var_list.append(var)
        self.teacher_running_mean.append(module.running_mean)
        self.teacher_running_var.append(module.running_var)

    def hook_fn_forward_saveBN(self, module, input, output):
        self.save_BN_mean.append(module.running_mean.cpu())
        self.save_BN_var.append(module.running_var.cpu())

    def train(self, epoch):
        """
        training
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()
        fp_acc = utils.AverageMeter()

        iters = 200
        self.update_lr(epoch)

        self.model.eval()
        self.model_teacher.eval()
        self.generator.train()

        start_time = time.time()
        end_time = start_time

        if epoch == 0:
            for m in self.model_teacher.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.register_forward_hook(self.hook_fn_forward)

        for i in range(iters):
            start_time = time.time()
            data_time = start_time - end_time

            z = Variable(
                torch.randn(self.settings.batchSize, self.settings.latent_dim)
            ).cuda()

            # Get labels ranging from 0 to n_classes for n rows
            labels = Variable(
                torch.randint(0, self.settings.nClasses, (self.settings.batchSize,))
            ).cuda()
            z = z.contiguous()
            labels = labels.contiguous()
            images = self.generator(z, labels)

            labels_loss = Variable(
                torch.zeros(self.settings.batchSize, self.settings.nClasses)
            ).cuda()
            labels_loss.scatter_(1, labels.unsqueeze(1), 1.0)

            self.mean_list.clear()
            self.var_list.clear()
            output_teacher_batch, output_teacher_1 = self.model_teacher(
                images, out_feature=True
            )

            output, output_1 = self.model(images, True)

            # generation loss
            z_ds = output_teacher_batch - output
            z_as = output_teacher_batch + output
            loss_ds = ((-(labels_loss * self.log_soft(z_ds)).sum(dim=1))).mean()
            loss_as = ((-(labels_loss * self.log_soft(z_as)).sum(dim=1))).mean()
            loss_onehot = 0.1 * loss_ds + 0.1 * loss_as
            h_loss = marginal_loss(output_teacher_batch, output, self.settings.nClasses)

            # BN statistic loss
            BNS_loss = torch.zeros(1).cuda()
            for num in range(len(self.mean_list)):
                BNS_loss += self.MSE_loss(
                    self.mean_list[num], self.teacher_running_mean[num]
                ) + self.MSE_loss(self.var_list[num], self.teacher_running_var[num])
            BNS_loss = BNS_loss / len(self.mean_list)

            # loss of Generator
            loss_G = h_loss + loss_onehot + BNS_loss
            self.backward_G(loss_G)

            output, loss_S = self.forward(
                images.detach(),
                output_teacher_batch.detach(),
                labels,
                linear=labels_loss,
            )
            if epoch >= self.settings.warmup_epochs:
                self.backward_S(loss_S)

            single_error, single_loss, single5_error = utils.compute_singlecrop(
                outputs=output,
                labels=labels,
                loss=loss_S,
                top5_flag=True,
                mean_flag=True,
            )

            top1_error.update(single_error, images.size(0))
            top1_loss.update(single_loss, images.size(0))
            top5_error.update(single5_error, images.size(0))

            end_time = time.time()

            gt = labels.data.cpu().numpy()
            d_acc = np.mean(
                np.argmax(output_teacher_batch.data.cpu().numpy(), axis=1) == gt
            )

            fp_acc.update(d_acc)

        print(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%] [G loss: %f] [Balance loss: %f] [BNS_loss:%f] [S loss: %f] "
            % (
                epoch + 1,
                self.settings.nEpochs,
                i + 1,
                iters,
                100 * fp_acc.avg,
                loss_G.item(),
                loss_onehot.item(),
                BNS_loss.item(),
                loss_S.item(),
            )
        )

        self.scalar_info["accuracy every epoch"] = 100 * d_acc
        self.scalar_info["G loss every epoch"] = loss_G
        self.scalar_info["One-hot loss every epoch"] = loss_onehot
        self.scalar_info["S loss every epoch"] = loss_S

        self.scalar_info["training_top1error"] = top1_error.avg
        self.scalar_info["training_top5error"] = top5_error.avg
        self.scalar_info["training_loss"] = top1_loss.avg

        if self.tensorboard_logger is not None:
            for tag, value in list(self.scalar_info.items()):
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def test(self, epoch):
        """
        testing
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model.eval()
        self.model_teacher.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()

                labels = labels.cuda()
                images = images.cuda()
                output = self.model(images)

                loss = torch.ones(1)
                self.mean_list.clear()
                self.var_list.clear()

                single_error, single_loss, single5_error = utils.compute_singlecrop(
                    outputs=output,
                    loss=loss,
                    labels=labels,
                    top5_flag=True,
                    mean_flag=True,
                )

                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()

        print(
            "[Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
            % (
                epoch + 1,
                self.settings.nEpochs,
                i + 1,
                iters,
                (100.00 - top1_error.avg),
            )
        )

        self.scalar_info["testing_top1error"] = top1_error.avg
        self.scalar_info["testing_top5error"] = top5_error.avg
        self.scalar_info["testing_loss"] = top1_loss.avg
        if self.tensorboard_logger is not None:
            for tag, value in self.scalar_info.items():
                self.tensorboard_logger.scalar_summary(tag, value, self.run_count)
            self.scalar_info = {}
        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg

    def test_teacher(self, epoch):
        """
        testing
        """
        top1_error = utils.AverageMeter()
        top1_loss = utils.AverageMeter()
        top5_error = utils.AverageMeter()

        self.model_teacher.eval()

        iters = len(self.test_loader)
        start_time = time.time()
        end_time = start_time

        with torch.no_grad():
            for i, (images, labels) in enumerate(self.test_loader):
                start_time = time.time()
                data_time = start_time - end_time

                labels = labels.cuda()
                if self.settings.tenCrop:
                    image_size = images.size()
                    images = images.view(
                        image_size[0] * 10,
                        image_size[1] / 10,
                        image_size[2],
                        image_size[3],
                    )
                    images_tuple = images.split(image_size[0])
                    output = None
                    for img in images_tuple:
                        if self.settings.nGPU == 1:
                            img = img.cuda()
                        img_var = Variable(img, volatile=True)
                        temp_output, _ = self.forward(img_var)
                        if output is None:
                            output = temp_output.data
                        else:
                            output = torch.cat((output, temp_output.data))
                    single_error, single_loss, single5_error = utils.compute_tencrop(
                        outputs=output, labels=labels
                    )
                else:
                    if self.settings.nGPU == 1:
                        images = images.cuda()

                    output = self.model_teacher(images)

                    loss = torch.ones(1)
                    self.mean_list.clear()
                    self.var_list.clear()

                    single_error, single_loss, single5_error = utils.compute_singlecrop(
                        outputs=output,
                        loss=loss,
                        labels=labels,
                        top5_flag=True,
                        mean_flag=True,
                    )
                #
                top1_error.update(single_error, images.size(0))
                top1_loss.update(single_loss, images.size(0))
                top5_error.update(single5_error, images.size(0))

                end_time = time.time()
                iter_time = end_time - start_time

        print(
            "Teacher network: [Epoch %d/%d] [Batch %d/%d] [acc: %.4f%%]"
            % (
                epoch + 1,
                self.settings.nEpochs,
                i + 1,
                iters,
                (100.00 - top1_error.avg),
            )
        )

        self.run_count += 1

        return top1_error.avg, top1_loss.avg, top5_error.avg
