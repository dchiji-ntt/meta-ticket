
# Part of this code is based on https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import DataParallel
import numpy as np

from models.sinusoid_base import SinusoidBase
from models.maml_model import MAMLModel
from models.maml import fast_adapt


class SinusoidMAML(SinusoidBase):
    def __init__(self, outman, cfg, device, data_parallel):
        super().__init__(outman, cfg, device, data_parallel)

    def train_one_iter(self, it, batch):
        self.optimizer.zero_grad()
        self.meta_model.meta_zero_grad()
        self.meta_model.inner_zero_grad()

        # for the case: self.scheduler == CustomCosineLR
        step_before_train = hasattr(self.scheduler, "step_before_train") and self.scheduler.step_before_train
        if step_before_train:
            try:
                self.scheduler.step(epoch=it)
            except:
                self.scheduler.step()

        batch_size = batch[0].size(0)
        train_samples = batch[0][:, ::2, :].float()
        train_targets = batch[1][:, ::2, :].float()
        train_batch = list(zip(train_samples, train_targets))
        test_samples = batch[0][:, 1::2, :].float()
        test_targets = batch[1][:, 1::2, :].float()
        test_batch = list(zip(test_samples, test_targets))

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        mean_grad_norms = dict()
        accum_error = 0.0
        for train_sample, test_sample in zip(train_batch, test_batch):
            # Compute meta-training loss
            learner = self.meta_model.clone()
            eval_error, _, grads_dict = fast_adapt(train_sample,
                                                               test_sample,
                                                               learner,
                                                               self.criterion,
                                                               self.train_steps,
                                                               self.device)
            eval_error.backward()
            meta_train_error += eval_error.item()
            for name in grads_dict:
                grad = grads_dict[name]
                if name not in mean_grad_norms:
                    mean_grad_norms[name] = grad.abs().mean().item()
                else:
                    mean_grad_norms[name] += grad.abs().mean().item()

        # Average the accumulated gradients and optimize
        batch_size = len(train_batch)
        for p in self.meta_model.meta_parameters():
            p.grad.data.mul_(1.0 / batch_size)

        self.optimizer.step()

        self.optimizer.zero_grad()
        self.meta_model.meta_zero_grad()
        self.meta_model.inner_zero_grad()

        # for the case: self.scheduler == MultiStepLR
        if not step_before_train:
            try:
                self.scheduler.step(epoch=it)
            except:
                self.scheduler.step()

        for name in mean_grad_norms:
            mean_grad_norms[name] = mean_grad_norms[name] / batch_size

        return {
                'loss': meta_train_error / batch_size,
                'accuracy': meta_train_error / batch_size,
                'grad_norms': mean_grad_norms,
                }

    def evaluate(self, dataset_type='val'):
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        mean_grad_norms = dict()
        valid_batch_size = self.cfg['valid_batch_size']

        if dataset_type == 'val':
            loader = self.val_loader
        elif dataset_type == 'test':
            loader = self.test_loader
        else:
            raise NotImplementedError()

        for it, batch in enumerate(loader):
            if it >= valid_batch_size:
                break

            assert batch[0].size(0) == 1 and batch[1].size(0) == 1
            train_input = batch[0][0, ::2, :].float()
            train_target = batch[1][0, ::2, :].float()
            train_sample = (train_input, train_target)
            test_input = batch[0][0, 1::2, :].float()
            test_target = batch[1][0, 1::2, :].float()
            test_sample = (test_input, test_target)

            learner = self.meta_model.clone()

            eval_error, _, grads_dict = fast_adapt(train_sample,
                                                   test_sample,
                                                   learner,
                                                   self.criterion,
                                                   self.test_steps,
                                                   self.device)
            meta_valid_error += eval_error.item()
            for name in grads_dict:
                grad = grads_dict[name]
                if name not in mean_grad_norms:
                    mean_grad_norms[name] = grad.abs().mean().item()
                else:
                    mean_grad_norms[name] += grad.abs().mean().item()

        sparsity_dict = self.get_sparsity()
        for name in mean_grad_norms:
            mean_grad_norms[name] = mean_grad_norms[name] / valid_batch_size
            if (sparsity_dict is not None) and (name in sparsity_dict):
                mean_grad_norms[name] /= (1. - sparsity_dict[name])

        return {
                'loss': meta_valid_error / valid_batch_size,
                'accuracy': meta_valid_error / valid_batch_size,
                'grad_norms': mean_grad_norms,
                }

    def _get_meta_model(self):
        return MAMLModel(self.model, self.cfg['inner_lr'], self.cfg['no_update_keywords'], first_order=self.cfg['first_order'])


