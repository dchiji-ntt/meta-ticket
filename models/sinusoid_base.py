
import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import DataParallel

from torch.optim.lr_scheduler import MultiStepLR
from utils.schedulers import CustomCosineLR
from utils.combined_optimizer import CombinedOptimizer

import learn2learn as l2l
from utils.sinusoid import Sinusoid
from torchmeta.utils.data import BatchMetaDataLoader


class SinusoidMLP(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.hidden1 = nn.Linear(1, dim)
        self.hidden2 = nn.Linear(dim, dim)
        self.hidden3 = nn.Linear(dim, 1)
        
    def forward(self, x):
        x = nn.functional.relu(self.hidden1(x))
        x = nn.functional.relu(self.hidden2(x))
        x = self.hidden3(x)
        return x


class SinusoidBase(object):
    def __init__(self, outman, cfg, device, data_parallel):
        self.outman = outman
        self.cfg = cfg
        self.device = device
        self.data_parallel = data_parallel

        self.debug_max_iters = self.cfg['debug_max_iters']
        self.train_augmentation = self.cfg['train_augmentation']
        self.dataset_cfg = self.cfg['__other_configs__'][self.cfg['dataset.config_name']]
        self.model_cfg = self.cfg['__other_configs__'][self.cfg['model.config_name']]

        self.train_steps = self.cfg['train_steps']
        self.test_steps = self.cfg['test_steps'] if self.cfg['test_steps'] is not None else self.train_steps

        self.train_loader, self.val_loader, self.test_loader = self._get_dataloaders()
        self.model = self._get_model().to(self.device)
        self.meta_model = self._get_meta_model().to(self.device)
        self.optimizer = self._get_optimizer()
        self.criterion = self._get_criterion()
        self.scheduler = self._get_scheduler()

    def train_one_iter(self, it):
        raise NotImplementedError
        return {
                'loss': None,
                'moving_accuracy': None,
                }

    def evaluate(self, dataset_type='val'):
        raise NotImplementedError
        return {
                'loss': None,
                'accuracy': None,
                }

    def get_sparsity(self, simplified_keys=False):
        sparsity_dict = dict()
        module = self.meta_model.module

        if not hasattr(module, '_get_mask'):
            return None
        else:
            for m_name, p_name in module.all_sparse_params:
                param_name = module._get_param_name(m_name, p_name)
                m = module._get_module(m_name)
                mask = module._get_mask(m, m_name, p_name)
                if simplified_keys:
                    sparsity_dict[m_name + '.' + p_name] = 1.0-(mask.sum()/mask.flatten().size(0)).item()
                else:
                    sparsity_dict[param_name + '_before_pruned'] = 1.0-(mask.sum()/mask.flatten().size(0)).item()
            return sparsity_dict

    def _get_dataloaders(self):
        dataset_dir = self.cfg['dataset_dir']
        max_size = self.cfg['max_train_dataset_size']
        dataset_classname = self.dataset_cfg['class']

        self.train_shots = self.cfg['train_shots']
        self.test_shots = self.cfg['test_shots'] if self.cfg['test_shots'] is not None else self.train_shots

        if dataset_classname in ['sinusoid']:
            num_tasks_train = self.dataset_cfg['num_tasks_train']
            num_tasks_test = self.dataset_cfg['num_tasks_test']
            taskset_train = Sinusoid(num_samples_per_task=2*self.train_shots, num_tasks=num_tasks_train,
                                     amplitude_min=0.1,  # default
                                     amplitude_max=self.cfg['sin_amplitude_train'],
                                     phase_min=self.cfg['sin_phase_min_train'],
                                     phase_max=self.cfg['sin_phase_max_train'])
            taskset_val = Sinusoid(num_samples_per_task=2*self.test_shots, num_tasks=num_tasks_test,
                                     amplitude_min=0.1,  # default
                                     amplitude_max=self.cfg['sin_amplitude_test'],
                                     phase_min=self.cfg['sin_phase_min_test'],
                                     phase_max=self.cfg['sin_phase_max_test'])
            taskset_test = Sinusoid(num_samples_per_task=2*self.test_shots, num_tasks=num_tasks_test,
                                     amplitude_min=0.1,  # default
                                     amplitude_max=self.cfg['sin_amplitude_test'],
                                     phase_min=self.cfg['sin_phase_min_test'],
                                     phase_max=self.cfg['sin_phase_max_test'])
        else:
            raise NotImplementedError

        loader_train = BatchMetaDataLoader(taskset_train,
                                           batch_size=self.cfg['batch_size'],
                                           num_workers=self.cfg['num_workers'])
        loader_val = BatchMetaDataLoader(taskset_val, batch_size=1,
                                         num_workers=self.cfg['num_workers'])
        loader_test = BatchMetaDataLoader(taskset_test, batch_size=1,
                                          num_workers=self.cfg['num_workers'])
        return loader_train, loader_val, loader_test

    def _get_model(self, model_cfg=None):
        if model_cfg is None:
            model_cfg = self.model_cfg
        assert model_cfg['class'] == 'SinusoidMLP'
        model = SinusoidMLP(int(model_cfg['hidden_dim'] * self.cfg['factor']))

        if self.data_parallel:
            gpu_ids = list(range(self.cfg['num_gpus']))
            return DataParallel(model, gpu_ids)
        else:
            return model

    def _get_meta_model(self):
        raise NotImplementedError

    def _get_optimizer(self):
        optim_name = self.cfg['optimizer']

        lr = self.cfg['lr']
        weight_decay = self.cfg['weight_decay']
        return self._new_optimizer(optim_name,
                                   self.meta_model.meta_parameters(),
                                   lr,
                                   weight_decay)

    def _get_criterion(self):
        return nn.MSELoss(reduction='mean')

    def _new_optimizer(self, name, params, lr, weight_decay, momentum=0.9):
        if name == 'AdamW':
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        elif name == 'SGD':
            return torch.optim.SGD(params, lr=lr,
                                   momentum=self.cfg['sgd_momentum'], weight_decay=weight_decay)
        else:
            raise NotImplementedError

    def _get_scheduler(self):
        class null_scheduler(object):
            def __init__(self, *args, **kwargs):
                return
            def step(self, *args, **kwargs):
                return
            def state_dict(self):
                return {}
            def load_state_dict(self, dic):
                return

        if isinstance(self.optimizer, CombinedOptimizer):
            score_optim = self.optimizer.opts_dict['score']
        else:
            score_optim = self.optimizer

        if self.cfg['lr_scheduler'] is None:
            return null_scheduler()
        elif self.cfg['lr_scheduler'] == 'CustomCosineLR':
            total_iter = self.cfg['iteration']
            init_lr = self.cfg['lr']
            warmup_iters = self.cfg['warmup_iters']
            return CustomCosineLR(score_optim, init_lr, total_iter, warmup_iters)
        elif self.cfg['lr_scheduler'] == 'MultiStepLR':
            return MultiStepLR(score_optim, milestones=self.cfg['lr_milestones'], gamma=self.cfg['multisteplr_gamma'])
        else:
            raise NotImplementedError


