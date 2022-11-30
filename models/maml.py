
# Part of this code is based on https://github.com/learnables/learn2learn/blob/master/examples/vision/maml_omniglot.py

import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import DataParallel
import numpy as np

from models.meta_learning import MetaLearning
from models.maml_model import MAMLModel


def accuracy(predictions, targets):
    predictions = predictions.argmax(dim=1).view(targets.shape)
    return (predictions == targets).sum().float() / targets.size(0)

def fast_adapt(train_sample, test_sample,
               learner, loss, adaptation_steps, device):
    train_input, train_target  = train_sample
    test_input, test_target = test_sample

    train_input, train_target = train_input.to(device), train_target.to(device)
    test_input, test_target = test_input.to(device), test_target.to(device)

    # Adapt the model
    grads_dict = dict()
    for step in range(adaptation_steps):
        train_error = loss(learner(train_input), train_target)
        named_grads = learner.adapt(train_error)
        for name, grad in named_grads:
            if name not in grads_dict:
                grads_dict[name] = grad
            else:
                grads_dict[name] += grad

    # Evaluate the adapted model
    predictions = learner(test_input)
    valid_error = loss(predictions, test_target)
    valid_accuracy = accuracy(predictions, test_target)
    return valid_error, valid_accuracy, grads_dict


class MAML(MetaLearning):
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

        train_batch = list(zip(*batch['train']))
        test_batch = list(zip(*batch['test']))

        meta_train_error = 0.0
        meta_train_accuracy = 0.0
        mean_grad_norms = dict()
        accum_error = 0.0
        for train_sample, test_sample in zip(train_batch, test_batch):
            # Compute meta-training loss
            learner = self.meta_model.clone()
            eval_error, eval_accuracy, grads_dict = fast_adapt(train_sample,
                                                               test_sample,
                                                               learner,
                                                               self.criterion,
                                                               self.train_steps,
                                                               self.device)
            eval_error.backward()
            meta_train_error += eval_error.item()
            meta_train_accuracy += eval_accuracy.item()
            for name in grads_dict:
                grad = grads_dict[name]
                if name not in mean_grad_norms:
                    mean_grad_norms[name] = grad.abs().mean().item()
                else:
                    mean_grad_norms[name] += grad.abs().mean().item()

        # Print some metrics
        #print('Iteration', it)
        #print('Meta Train Error', meta_train_error / batch_size)
        #print('Meta Train Accuracy', meta_train_accuracy / batch_size)
        #print('Meta Valid Error', )
        #print('Meta Valid Accuracy', meta_valid_accuracy / batch_size)

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
                'accuracy': meta_train_accuracy / batch_size,
                'grad_norms': mean_grad_norms,
                }

    def evaluate(self, dataset_type='val'):
        meta_valid_error = 0.0
        meta_valid_accuracy = 0.0
        mean_grad_norms = dict()

        if dataset_type == 'val':
            loader = self.val_loader
            batch_size = self.cfg['valid_batch_size']
        elif dataset_type == 'test':
            loader = self.test_loader
            batch_size = self.cfg['test_batch_size']
        else:
            raise NotImplementedError()

        for it, batch in enumerate(loader):
            if it >= batch_size:
                break
            train_batch = list(zip(*batch['train']))
            test_batch = list(zip(*batch['test']))

            train_sample = train_batch[0]
            test_sample = test_batch[0]
            assert (len(train_batch) == 1 and len(test_batch) == 1)

            learner = self.meta_model.clone()

            if self.cfg['const_params_test']:
                learner.reset(const=self.cfg['const_reset_scale'], kaiming=True, rate=1.0)

            eval_error, eval_accuracy, grads_dict = fast_adapt(train_sample,
                                                               test_sample,
                                                               learner,
                                                               self.criterion,
                                                               self.test_steps,
                                                               self.device)
            meta_valid_error += eval_error.item()
            meta_valid_accuracy += eval_accuracy.item()
            for name in grads_dict:
                grad = grads_dict[name]
                if name not in mean_grad_norms:
                    mean_grad_norms[name] = grad.abs().mean().item()
                else:
                    mean_grad_norms[name] += grad.abs().mean().item()

        sparsity_dict = self.get_sparsity()
        for name in mean_grad_norms:
            mean_grad_norms[name] = mean_grad_norms[name] / batch_size
            if (sparsity_dict is not None) and (name in sparsity_dict):
                mean_grad_norms[name] /= (1. - sparsity_dict[name])

        return {
                'loss': meta_valid_error / batch_size,
                'accuracy': meta_valid_accuracy / batch_size,
                'grad_norms': mean_grad_norms,
                }

    def _get_meta_model(self):
        return MAMLModel(self.model, self.cfg['inner_lr'], self.cfg['no_update_keywords'], first_order=self.cfg['first_order'])


