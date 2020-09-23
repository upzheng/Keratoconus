import os
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import utils.loss as loss
import network as network
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, cohen_kappa_score
from options.base_train_option import arg2str
from datetime import datetime
from utils.summary import write_scalars
from torch.autograd import Variable
from tensorboardX import SummaryWriter


class ImageSolver(object):

    def __init__(self, args):
        """
        Solver Initialization
        """

        self.args = args
        self.batch_size = args.batch_size
        self.label_weight = torch.from_numpy(np.array(args.label_weights)).float()

        self.converge = False
        self.lr = args.lr
        self.start_iter = args.start_iter
        self.hist_min_loss = 100000.0
        self.hist_max_recall = 0.0
        self.hist_max_tnnr = 0.0
        self.best = 'best_recall'

        self.model = getattr(network, args.model_name.lower())(args)
        if args.cuda:
            self.model = self.model.cuda()
            self.label_weight = self.label_weight.cuda()

        # self.optim = torch.optim.Adam(params=self.model.parameters(), lr=self.lr)
        self.optim = getattr(torch.optim, args.optim) \
            (filter(lambda p: p.requires_grad, self.model.parameters()), lr=self.lr, weight_decay=args.weight_decay)
        self.init_writer()

        self.loss = getattr(loss, args.loss)(weight=self.label_weight)
        self.model_path = os.path.join(args.save_folder, self.model.model_name())

        if args.resume:
            if os.path.isfile(self.args.resume):
                iter, index = self.model.load_model_best(args.resume)
                self.start_iter = iter
                if self.best == 'best_recall':
                    self.hist_max_recall = index
                else:
                    self.hist_max_tnnr = index
                print('load model at iter', iter)
            else:
                print("=> no checkpoint found at '{}'".format(args.resume))

    def init_writer(self):
        """ Tensorboard writer initialization
            """
        log_path = os.path.join(self.args.save_log, datetime.now().strftime('%b%d_%H-%M-%S') + '_' + self.args.exp_name + '_train')
        log_option_path = os.path.join(log_path, 'options.log')
        save_option_path = os.path.join(self.args.save_folder, self.args.exp_name, 'options.log')

        self.writer = SummaryWriter(log_path)
        with open(log_option_path, 'w') as f:
            f.write(arg2str(self.args))
        with open(save_option_path, 'w') as f:
            f.write(arg2str(self.args))

    def _adjust_learning_rate_iter(self, step):
        """Sets the learning rate to the initial LR decayed by 10 at every specified step
        # Adapted from PyTorch Imagenet example:
        # https://github.com/pytorch/examples/blob/master/imagenet/main.py
        """
        if self.args.lr_adjust == 'fix':
            if step in self.args.stepvalues:
                self.lr_current = self.lr_current * self.args.gamma
        elif self.args.lr_adjust == 'poly':
            self.lr_current = self.args.lr * (1 - step / self.args.max_iters) ** 0.9

        for param_group in self.optim.param_groups:
            param_group['lr'] = self.lr_current

    @staticmethod
    def accuracy(score, target):
        _, pred = score.max(1)
        equal = (pred == target).float()
        acc = accuracy_score(target.cpu().data.numpy(), pred.cpu().data.numpy())  # the same as equal.mean().data[0]
        return equal.mean().data[0]

    @staticmethod
    def metrics_2c(score, target):
        # for two classification
        _, predict = score.max(1)

        correct = (predict.data == target.data).byte()
        positive = (target.data == 1).byte()
        negative = (target.data == 0).byte()
        TP = (correct & positive).float().sum()

        recall = TP / target.float().sum()
        precision = TP / predict.float().sum()
        f1score = (2 * precision * recall) / (precision + recall)

        TN = (correct & negative).float().sum()
        specificity = TN / (1 - target).float().sum()
        if (predict.data == 0).float().sum() > 0:
            TNnR = TN/((predict.data == 0).float().sum())
        else:
            TNnR = TN/((predict.data == 0).float().sum()+0.1)
        return precision.data[0], recall.data[0], f1score.data[0], specificity.data[0], TNnR

    @staticmethod
    def metrics_mc(score, target):
        # for multi classes
        _, predict = score.max(1)
        r = recall_score(target.cpu().data.numpy(), predict.cpu().data.numpy(), average='macro')

        ap = precision_score(target.cpu().data.numpy(), predict.cpu().data.numpy(), average='macro')

        f1 = f1_score(target.cpu().data.numpy(), predict.cpu().data.numpy(), average='macro')

        kappa = cohen_kappa_score(target.cpu().data.numpy(), predict.cpu().data.numpy())
        return r, ap, f1, kappa

    def train_iter(self, step, dataloader):

        imgs, target = dataloader.next()

        # Train mode
        self.model.train()

        imgs = imgs.float()
        imgs, target = Variable(imgs).cuda(), Variable(target).cuda()
        score = self.model(imgs)
        loss = self.loss(score, target)

        # Backward
        loss.backward()
        self.optim.step()
        self.model.zero_grad()

        if step % self.args.display_freq == 0:

            # compute metrics
            # TODO more general metrics
            acc = self.accuracy(score, target)
            recall, prec, f1, kap = self.metrics_mc(score, target)

            print('Training - Loss: {:.4f} - Acc: {:.4f} - Precision: {:.4f} - Recall: {:.4f} - f1score:{:.4f} - kappa:{:.4f}' \
                  .format(loss.data[0], acc, prec, recall, f1, kap))

            # Record to tensorboard
            # TODO more general metrics
            scalars = [loss.data[0], acc, prec, recall, f1, kap]
            names = ['loss', 'acc', 'precision', 'recall', 'f1score', 'kappa']
            write_scalars(self.writer, scalars, names, step, 'train')

            # debug info
            if self.args.debug:
                print('lebel: {}'.format(target.cpu().data.tolist()))
                print('pred : {}'.format(score.max(1)[1].cpu().data.tolist()))

        del imgs, score, target, loss

    def train(self, train_dataloader, valid_dataloader=None):
        """
        Training Process
        @param train_dataloader: Training Data Loader
        @param valid_dataloader: Validation Data Loader
        """
        train_epoch_size = len(train_dataloader)
        train_iter = iter(train_dataloader)
        val_epoch_size = len(valid_dataloader)
        val_iter = iter(valid_dataloader)

        for step in range(self.start_iter, self.args.max_iters):

            self._adjust_learning_rate_iter(step)
            # A new training epoch begin
            if step % train_epoch_size == 0:
                print('Epoch: {} ----- step:{} - train_epoch size:{}'.format(step // train_epoch_size, step, train_epoch_size))
                train_iter = iter(train_dataloader)

            self.train_iter(step, train_iter)

            if (valid_dataloader is not None) and (step % self.args.val_freq == 0 or step == self.args.max_iters-1) and (step != 0):

                print('============Begin Validation============:step:{}'.format(step))

                # A new validation epoch begin
                val_iter = iter(valid_dataloader)


                total_loss, total_acc, total_prec, total_recall, total_f1, total_kap = self.validation(val_iter, val_epoch_size)

                if (total_recall > self.hist_max_recall) or ((total_recall == self.hist_max_recall) and total_loss.data[0] < self.hist_min_loss):
                    self.hist_max_recall = total_recall
                    self.model.save_model_best(step, best=self.best, index=self.hist_max_recall)

                if total_loss.data[0] < self.hist_min_loss:
                    self.hist_min_loss = total_loss.data[0]
                    self.model.save_model_best(step, best='min_loss')

                print('Validation - Loss: {:.4f} - Acc: {:.4f} - Presision: {:.4f} - Recall: {:.4f} - f1: {:.4f} - kappa: {:.4f}' \
                      .format(total_loss.data[0], total_acc, total_prec, total_recall, total_f1, total_kap))

                # Record to tensorboard
                # TODO more general metrics
                scalars = [total_loss.data[0], total_acc, total_prec, total_recall, total_f1, total_kap]
                names = ['loss', 'acc', 'precision', 'recall', 'f1score', 'kappa']
                write_scalars(self.writer, scalars, names, step, 'val')

                del total_loss, total_acc, total_prec, total_recall, total_f1, total_kap

                print('============End Validation============')

            # if step % self.args.save_freq == 0 and step != 0:
            #     self.model.save_model(step)

    def validation(self, val_iter, val_epoch_size):

        self.model.eval()

        total_score = []
        total_target = []
        with torch.no_grad():
            for i in range(val_epoch_size):
                imgs, target = next(val_iter)
                imgs = imgs.float()
                imgs, target = Variable(imgs).cuda(), \
                               Variable(target).cuda()

                val_score = self.model(imgs)
                if i == 0:
                    total_score = val_score
                    total_target = target
                else:
                    total_target = torch.cat((total_target, target), 0)
                    total_score = torch.cat((total_score, val_score), 0)

        del imgs, target

        # Compute Metrics
        val_loss = self.loss(total_score, total_target)
        val_acc = self.accuracy(total_score, total_target)
        recall, precision, f1, kap = self.metrics_mc(total_score, total_target)

        # debug info
        if self.args.debug:
            print('some val lebel: {}'.format(total_target[:16].cpu().data.tolist()))
            print('some val pred : {}'.format(total_score[:16].max(1)[1].cpu().data.tolist()))

        return val_loss, val_acc, precision, recall, f1, kap

    def restore(self):
        self.model.load_state_dict(torch.load(self.model_path))
        print('Model Restored')



