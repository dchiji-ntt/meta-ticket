
import os
import time
import datetime
import json

import torch
from utils.seed import set_random_seed
from utils.output_manager import OutputManager
from utils.pd_logger import PDLogger
from torch.nn import DataParallel

from models.maml import MAML
from models.meta_ticket import MetaTicket
from models.sinusoid_maml import SinusoidMAML

def meta_test(exp_name, cfg, gpu_id, prefix="", iteration=None, use_best=False):
    set_random_seed(cfg['seed'])
    device = torch.device(f'cuda:{gpu_id}' if cfg['use_cuda'] and torch.cuda.is_available()
                                   else 'cpu')

    outman = OutputManager(cfg['output_dir'], exp_name)
    outman.print('Number of available gpus: ', torch.cuda.device_count(), prefix=prefix)

    if cfg['learning_framework'] == 'MAML':
        learner = MAML(outman, cfg, device, cfg['data_parallel'])
    elif cfg['learning_framework'] == 'MetaTicket':
        learner = MetaTicket(outman, cfg, device, cfg['data_parallel'])
    elif cfg['learning_framework'] == 'SinusoidMAML':
        learner = SinusoidMAML(outman, cfg, device, cfg['data_parallel'])
    else:
        raise NotImplementedError


    if use_best:
        dump_path = outman.get_abspath(prefix=f"best.{prefix}", ext="pth")
    elif iteration is not None:
        dump_path = outman.get_abspath(prefix=f'it{iteration}.{prefix}', ext="pth")
    else:
        dump_path = outman.get_abspath(prefix=f"dump.{prefix}", ext="pth")

    outman.print(dump_path, prefix=prefix)
    if os.path.exists(dump_path):
        dump_dict = torch.load(dump_path)
        it = dump_dict['it']
        if isinstance(learner.meta_model, DataParallel):
            learner.meta_model.module.load_state_dict(dump_dict['model_state_dict'])
        else:
            learner.meta_model.load_state_dict(dump_dict['model_state_dict'])
    else:
        raise Exception

    outman.print('[', str(datetime.datetime.now()) , '] Evaluate on Test Dataset...' , prefix=prefix)

    # Test
    result = learner.evaluate(dataset_type='test')
    if use_best:
        outman.print('Test Accuracy (Best):', str(result['accuracy']), prefix=prefix)
    else:
        outman.print('Test Accuracy:', str(result['accuracy']), prefix=prefix)

    test_info_dict = {
            'accuracy': result['accuracy'],
            'iteration': iteration,
            'loss': result['loss'],
            'prefix': prefix,
            }

    if use_best:
        output_path = outman.get_abspath(prefix=f"test_best.{prefix}", ext="json")
    elif iteration is not None:
        output_path = outman.get_abspath(prefix=f"test_it{iteration}.{prefix}", ext="json")
    else:
        output_path = outman.get_abspath(prefix=f"test_dump.{prefix}", ext="json")

    with open(output_path, 'w') as f:
        json.dump(test_info_dict, f, indent=2)


