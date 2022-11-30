import torch
import torch.nn as nn
from torch.utils.data.sampler import SubsetRandomSampler
from torch.nn import DataParallel

from torch.optim.lr_scheduler import MultiStepLR
from utils.schedulers import CustomCosineLR
from utils.combined_optimizer import CombinedOptimizer

import learn2learn as l2l
from models.networks.mlp import MLP
from models.networks.networks_from_boil import ResNet12_boil
from models.networks.wrn28 import WRN28

from utils.dataset_helpers import omniglot_helper, cifarfs_helper, miniimagenet_helper, cars_helper, cub_helper, vgg_flower_helper, aircraft_helper
from torchmeta.utils.data import BatchMetaDataLoader

class MetaLearning(object):
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
        self.train_ways = self.cfg['train_ways']
        self.train_shots = self.cfg['train_shots']
        self.test_steps = self.cfg['test_steps'] if self.cfg['test_steps'] is not None else self.train_steps
        self.test_ways = self.cfg['test_ways'] if self.cfg['test_ways'] is not None else self.train_ways
        self.test_shots = self.cfg['test_shots'] if self.cfg['test_shots'] is not None else self.train_shots

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

        if dataset_classname in ['omniglot']:
            dataset_train = omniglot_helper(dataset_dir, self.train_shots, self.train_ways,
                                            seed=self.cfg['seed'], download=True,
                                            meta_train=True)
            dataset_val = omniglot_helper(dataset_dir, self.test_shots, self.test_ways,
                                          seed=self.cfg['seed'], download=True,
                                          meta_val=True)
            dataset_test = omniglot_helper(dataset_dir, self.test_shots, self.test_ways,
                                           seed=self.cfg['seed'], download=True,
                                           meta_test=True)
        elif dataset_classname in ['cifarfs']:
            dataset_train = cifarfs_helper(dataset_dir, self.train_shots, self.train_ways,
                                            seed=self.cfg['seed'], download=True,
                                            meta_train=True)
            dataset_val = cifarfs_helper(dataset_dir, self.test_shots, self.test_ways,
                                          seed=self.cfg['seed'], download=True,
                                          meta_val=True)
            dataset_test = cifarfs_helper(dataset_dir, self.test_shots, self.test_ways,
                                           seed=self.cfg['seed'], download=True,
                                           meta_test=True)
        elif dataset_classname in ['vgg_flower']:
            dataset_train = vgg_flower_helper(dataset_dir, self.train_shots, self.train_ways,
                                            seed=self.cfg['seed'], download=True,
                                            meta_train=True)
            dataset_val = vgg_flower_helper(dataset_dir, self.test_shots, self.test_ways,
                                          seed=self.cfg['seed'], download=True,
                                          meta_val=True)
            dataset_test = vgg_flower_helper(dataset_dir, self.test_shots, self.test_ways,
                                           seed=self.cfg['seed'], download=True,
                                           meta_test=True)
        elif dataset_classname in ['aircraft']:
            dataset_train = aircraft_helper(dataset_dir, self.train_shots, self.train_ways,
                                            seed=self.cfg['seed'], download=True,
                                            meta_train=True)
            dataset_val = aircraft_helper(dataset_dir, self.test_shots, self.test_ways,
                                          seed=self.cfg['seed'], download=True,
                                          meta_val=True)
            dataset_test = aircraft_helper(dataset_dir, self.test_shots, self.test_ways,
                                           seed=self.cfg['seed'], download=True,
                                           meta_test=True)
        elif dataset_classname in ['mini-imagenet']:
            dataset_train = miniimagenet_helper(dataset_dir, self.train_shots, self.train_ways,
                                            seed=self.cfg['seed'], download=True,
                                            meta_train=True)
            dataset_val = miniimagenet_helper(dataset_dir, self.test_shots, self.test_ways,
                                          seed=self.cfg['seed'], download=True,
                                          meta_val=True)
            dataset_test = miniimagenet_helper(dataset_dir, self.test_shots, self.test_ways,
                                           seed=self.cfg['seed'], download=True,
                                           meta_test=True)
        elif dataset_classname in ['cars']:
            dataset_train = cars_helper(dataset_dir, self.train_shots, self.train_ways,
                                        seed=self.cfg['seed'], download=True,
                                        meta_train=True)
            dataset_val = cars_helper(dataset_dir, self.test_shots, self.test_ways,
                                      seed=self.cfg['seed'], download=True,
                                      meta_val=True)
            dataset_test = cars_helper(dataset_dir, self.test_shots, self.test_ways,
                                       seed=self.cfg['seed'], download=True,
                                       meta_test=True)
        elif dataset_classname in ['cub']:
            dataset_train = cub_helper(dataset_dir, self.train_shots, self.train_ways,
                                        seed=self.cfg['seed'], download=True,
                                        meta_train=True)
            dataset_val = cub_helper(dataset_dir, self.test_shots, self.test_ways,
                                      seed=self.cfg['seed'], download=True,
                                      meta_val=True)
            dataset_test = cub_helper(dataset_dir, self.test_shots, self.test_ways,
                                       seed=self.cfg['seed'], download=True,
                                       meta_test=True)
        else:
            raise NotImplementedError

        loader_train = BatchMetaDataLoader(dataset_train,
                                           batch_size=self.cfg['batch_size'],
                                           shuffle=True,
                                           num_workers=self.cfg['num_workers'])
        loader_val = BatchMetaDataLoader(dataset_val,
                                         batch_size=1,
                                         shuffle=True,
                                         num_workers=self.cfg['num_workers'])
        loader_test = BatchMetaDataLoader(dataset_test,
                                          batch_size=1,
                                          shuffle=True,
                                          num_workers=self.cfg['num_workers'])
        return loader_train, loader_val, loader_test

    def _get_model(self, model_cfg=None):
        if model_cfg is None:
            model_cfg = self.model_cfg

        image_size = self.dataset_cfg['image_size']
        train_ways = self.cfg['train_ways']
        if model_cfg['class'] == 'MLP':
            sizes = model_cfg['sizes']
            if type(sizes) is list:
                pass
            elif type(sizes) is str and sizes == '3-layers':
                sizes = [self.cfg['hid1'], self.cfg['hid2']]
            elif type(sizes) is str and sizes == '4-layers':
                sizes = [self.cfg['hid1'], self.cfg['hid2'], self.cfg['hid3']]
            elif type(sizes) is str and sizes == '5-layers':
                sizes = [self.cfg['hid1'], self.cfg['hid2'], self.cfg['hid3'], self.cfg['hid4']]
            else:
                raise NotImplementedError()
            model = MLP(self.dataset_cfg['num_channels'] * self.dataset_cfg['image_size'] ** 2,
                        train_ways,
                        sizes=[int(s * self.cfg['factor']) for s in sizes],
                        bn=not model_cfg['disable_bn'],
                        bn_affine=self.cfg['bn_affine'],
                        bn_track_running_stats=self.cfg['bn_track_running_stats'],
                        init_mode=self.cfg['init_mode'],
                        init_scale=self.cfg['init_scale'])
        elif model_cfg['class'] == 'ResNet12':
            model = ResNet12_boil(out_features=train_ways,
                                  avg_pool=True,
                                  wh_size=1)
        elif model_cfg['class'] == 'WRN28':
            model = WRN28(train_ways, widen_factor=self.model_cfg['widen_factor'])
        else:
            raise NotImplementedError

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
        return nn.CrossEntropyLoss()

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


