
import os
import time
import datetime
import json
import itertools

import torch
from utils.seed import set_random_seed
from utils.output_manager import OutputManager
from utils.pd_logger import PDLogger
from torch.nn import DataParallel

from models.maml import MAML
from models.meta_ticket import MetaTicket

from pprint import PrettyPrinter
pp = PrettyPrinter()

def meta_cross_test(exp_name, cfg, gpu_id, prefix="", iteration=None, use_best=None):
    if use_best is None:
        use_best = cfg['use_best']
    set_random_seed(cfg['seed'])
    device = torch.device(f'cuda:{gpu_id}' if cfg['use_cuda'] and torch.cuda.is_available()
                                   else 'cpu')

    target_name = cfg['target_name']
    target_cfg = cfg['__other_configs__'][target_name]
    target_cfg['dataset.config_name'] = cfg['eval_dataset.config_name']

    grid_keys = []
    grid_param_cands = []
    for k in cfg['target_grid']:
        grid_keys.append(k)
        grid_param_cands.append(cfg['target_grid'][k])

    for grid_params in itertools.product(*grid_param_cands):
        job_name = ""
        for i, k in enumerate(grid_keys):
            v = grid_params[i]
            job_name += f"{k}_{v}--"
            target_cfg[k] = v
        target_prefix = prefix + job_name

        # Copy all configs in `cfg` to `target_cfg`
        for k in cfg:
            if k in target_cfg and target_cfg[k] != cfg[k]:
                print(f'Set {k}: {target_cfg[k]} -> {cfg[k]}')
            target_cfg[k] = cfg[k]

        cross_outman = OutputManager(target_cfg['output_dir'], exp_name)
        target_outman = OutputManager(target_cfg['output_dir'], target_name)
        target_outman.print('Number of available gpus: ', torch.cuda.device_count(), prefix=target_prefix)

        if target_cfg['learning_framework'] == 'MAML':
            learner = MAML(cross_outman, target_cfg, device, target_cfg['data_parallel'])
        elif target_cfg['learning_framework'] == 'MetaTicket':
            learner = MetaTicket(cross_outman, target_cfg, device, target_cfg['data_parallel'])
        else:
            raise NotImplementedError

        if use_best:
            dump_path = target_outman.get_abspath(prefix=f"best.{target_prefix}", ext="pth")
        elif iteration is not None:
            dump_path = target_outman.get_abspath(prefix=f'it{iteration}.{target_prefix}', ext="pth")
        else:
            dump_path = target_outman.get_abspath(prefix=f"dump.{target_prefix}", ext="pth")

        cross_outman.print(dump_path, prefix=target_prefix)
        if os.path.exists(dump_path):
            dump_dict = torch.load(dump_path)
            it = dump_dict['it']
            if isinstance(learner.meta_model, DataParallel):
                learner.meta_model.module.load_state_dict(dump_dict['model_state_dict'])
            else:
                learner.meta_model.load_state_dict(dump_dict['model_state_dict'])
        else:
            raise Exception(f'Not Found (Probably): {dump_path}')

        cross_outman.print('[', str(datetime.datetime.now()) , '] Evaluate on Test Dataset...' , prefix=target_prefix)

        # Test
        result = learner.evaluate(dataset_type='test')
        if use_best:
            cross_outman.print('Test Accuracy (Best):', str(result['accuracy']), prefix=target_prefix)
        else:
            cross_outman.print('Test Accuracy:', str(result['accuracy']), prefix=target_prefix)

        test_info_dict = {
                'accuracy': result['accuracy'],
                'iteration': iteration,
                'loss': result['loss'],
                'prefix': target_prefix,
                }

        if use_best:
            output_path = cross_outman.get_abspath(prefix=f"crosstest_best.{target_prefix}", ext="json")
        elif iteration is not None:
            output_path = cross_outman.get_abspath(prefix=f"crosstest_it{iteration}.{target_prefix}", ext="json")
        else:
            output_path = cross_outman.get_abspath(prefix=f"crosstest_dump.{target_prefix}", ext="json")

        with open(output_path, 'w') as f:
            json.dump(test_info_dict, f, indent=2)


