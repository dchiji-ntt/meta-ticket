
# Part of the following code is based on https://github.com/learnables/learn2learn/blob/master/learn2learn/algorithms/maml.py

import torch
from torch import nn

from models.maml import MAML, fast_adapt
from models.meta_ticket_model import MetaTicketModel
from utils.combined_optimizer import CombinedOptimizer

class MetaTicket(MAML):
    def __init__(self, outman, cfg, device, data_parallel):
        super().__init__(outman, cfg, device, data_parallel)

        lrs = dict()
        for name, p in self.meta_model.named_inner_parameters():
            lrs[name] = self.cfg['inner_lr']
            # to freeze specified params in inner loop
            for keyword in self.cfg['no_update_keywords']:
                if keyword in name:
                    lrs[name] = 0.0
                    break
        self.meta_model.lrs = lrs

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
        for train_sample, test_sample in zip(train_batch, test_batch):
            if self.cfg['const_params_train']:
                self.meta_model.reset(const=self.cfg['const_reset_scale'], kaiming=True)

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

        # Average the accumulated gradients and optimize
        batch_size = len(train_batch)
        for name, p in self.meta_model.named_meta_parameters():
            p.grad.data.mul_(1.0 / batch_size)

        self.optimizer.step()

        self.optimizer.zero_grad()
        self.meta_model.meta_zero_grad()
        self.meta_model.inner_zero_grad()

        assert hasattr(self.meta_model.module, 'rerandomize')
        self.meta_model.module.rerandomize()

        # for the case: self.scheduler == MultiStepLR
        if not step_before_train:
            try:
                self.scheduler.step(epoch=it)
            except:
                self.scheduler.step()

        sparsity_dict = self.get_sparsity()
        for name in mean_grad_norms:
            mean_grad_norms[name] = mean_grad_norms[name] / batch_size
            if (sparsity_dict is not None) and (name in sparsity_dict):
                mean_grad_norms[name] /= (1. - sparsity_dict[name])

        return {
                'loss': meta_train_error / batch_size,
                'accuracy': meta_train_accuracy / batch_size,
                'grad_norms': mean_grad_norms,
                }

    def _get_meta_model(self):
        return MetaTicketModel(self.model,
                           None,    # lrs will be set in MetaTicket.__init__()
                           init_mode=self.cfg['init_mode'],
                           ignore_params=self.cfg['ignore_params'],
                           allow_unused=True,
                           first_order=self.cfg['first_order'],
                           init_sparsity=self.cfg['init_sparsity'],
                           rerand_freq=self.cfg['rerand_freq'],
                           rerand_rate=self.cfg['rerand_rate'],
                           scale_delta_coeff=self.cfg['scale_delta_coeff'],
                           learnable_scale=self.cfg['learnable_scale'],
                           )

    def _get_optimizer(self):
        optims_dict = dict()

        optim_name = self.cfg['optimizer']
        lr = self.cfg['lr']
        weight_decay = self.cfg['weight_decay']
        score_params = []
        for p_name, p in self.meta_model.named_meta_parameters():
            if p_name.endswith('_score'):
                score_params.append(p)
        optims_dict['score'] = self._new_optimizer(optim_name,
                                                   score_params,
                                                   lr,
                                                   weight_decay)

        scale_deltas = []
        for p_name, p in self.meta_model.named_meta_parameters():
            if p_name.endswith('_scale_delta'):
                scale_deltas.append(p)
        self.outman.print('[DEBUG] scale_deltas:')
        self.outman.pprint(scale_deltas)
        if len(scale_deltas) > 0:
            optims_dict['scale_delta'] = self._new_optimizer(optim_name,
                                                             scale_deltas,
                                                             self.cfg['scale_lr'],
                                                             weight_decay)

        assert len(score_params) + len(scale_deltas) == len(self.meta_model.named_meta_parameters())

        aux_params = []
        aux_param_names = []
        if len(self.cfg['aux_params']) >= 1:
            for name, param in self.meta_model.named_inner_parameters():
                if name in [n for n,_ in self.meta_model.named_meta_parameters()]:
                    continue
                for kwd in self.cfg['aux_params']:
                    if kwd in name:
                        aux_params.append(param)
                        aux_param_names.append(name)
                        break
            self.outman.print('Aux Params (additionally optimized):')
            self.outman.pprint(aux_param_names)
            if len(aux_params) >= 1:
                aux_optim = self._new_optimizer(self.cfg['aux_optimizer'],
                                                aux_params,
                                                self.cfg['aux_lr'],
                                                self.cfg['aux_weight_decay'])
                optims_dict['aux'] = aux_optim

        optim = CombinedOptimizer(optims_dict)
        return optim


