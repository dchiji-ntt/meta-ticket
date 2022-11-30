
import os
import time
import datetime
import json
import math

import torch
from utils.seed import set_random_seed
from utils.output_manager import OutputManager
from utils.pd_logger import PDLogger
from torch.nn import DataParallel
from commands.meta_test import meta_test

from models.maml import MAML
from models.meta_ticket import MetaTicket
from models.sinusoid_maml import SinusoidMAML
from models.sinusoid_meta_ticket import SinusoidMetaTicket

import pprint
pp = pprint.PrettyPrinter()

def meta_train(exp_name, cfg, gpu_id, prefix=""):
    if cfg['seed'] is not None:
        set_random_seed(cfg['seed'])
    elif cfg['seed_by_time']:
        set_random_seed(int(time.time() * 1000) % 1000000)
    else:
        raise Exception("Set seed value.")
    device = torch.device(f'cuda:{gpu_id}' if cfg['use_cuda'] and torch.cuda.is_available() else 'cpu')
    outman = OutputManager(cfg['output_dir'], exp_name)

    dump_path = outman.get_abspath(prefix=f"dump.{prefix}", ext="pth")

    outman.print('Number of available gpus: ', torch.cuda.device_count(), prefix=prefix)

    pd_logger = PDLogger()
    pd_logger.set_filename(outman.get_abspath(prefix=f"pd_log.{prefix}", ext="pickle"))
    if os.path.exists(pd_logger.filename) and not cfg['force_restart']:
        pd_logger.load()

    if cfg['learning_framework'] == 'MAML':
        learner = MAML(outman, cfg, device, cfg['data_parallel'])
    elif cfg['learning_framework'] == 'MetaTicket':
        learner = MetaTicket(outman, cfg, device, cfg['data_parallel'])
    elif cfg['learning_framework'] == 'SinusoidMAML':
        learner = SinusoidMAML(outman, cfg, device, cfg['data_parallel'])
    elif cfg['learning_framework'] == 'SinusoidMetaTicket':
        learner = SinusoidMetaTicket(outman, cfg, device, cfg['data_parallel'])
    else:
        raise NotImplementedError

    params_info = None
    #params_info = count_params(learner.meta_model)

    best_value = None
    best_it = 0
    start_it = 0
    total_seconds = 0.

    outman.print(dump_path, prefix=prefix)
    if os.path.exists(dump_path) and not cfg['force_restart']:
        try:
            dump_dict = torch.load(dump_path)
            start_it = dump_dict['it'] + 1
            best_value = dump_dict['best_val']
            best_it = dump_dict['best_it'] if 'best_it' in dump_dict else 0
            total_seconds = dump_dict['total_seconds'] if 'total_seconds' in dump_dict else 0.
            if isinstance(learner.meta_model, DataParallel):
                learner.meta_model.module.load_state_dict(dump_dict['model_state_dict'])
            else:
                learner.meta_model.load_state_dict(dump_dict['model_state_dict'])
            learner.optimizer.load_state_dict(dump_dict['optim_state_dict'])
            if 'sched_state_dict' in dump_dict:
                learner.scheduler.load_state_dict(dump_dict['sched_state_dict'])
        except Exception as e:
            print("[train.py] catched unexpected error in loading checkpoint:", str(e))
            print("[train.py] start training from scratch")
    elif cfg['load_checkpoint_path'] is not None:
        assert not os.path.exists(dump_path)
        assert os.path.exists(cfg['load_checkpoint_path'])
        try:
            checkpoint_dict = torch.load(cfg['load_checkpoint_path'])
            if isinstance(learner.meta_model, DataParallel):
                learner.meta_model.module.load_state_dict(checkpoint_dict['model_state_dict'])
            else:
                learner.meta_model.load_state_dict(checkpoint_dict['model_state_dict'])
            #learner.optimizer.load_state_dict(checkpoint_dict['optim_state_dict'])
            #if 'sched_state_dict' in checkpoint_dict:
            #    learner.scheduler.load_state_dict(checkpoint_dict['sched_state_dict'])
        except Exception as e:
            print("[train.py] catched unexpected error in loading checkpoint:", str(e))
            print("[train.py] start training from scratch")

    # Training loop
    assert cfg['iteration'] is not None
    assert cfg['validation_freq'] is not None
    assert cfg['iteration'] % cfg['validation_freq'] == 0

    it = start_it
    accum_accuracy = 0.0
    outman.print('[', str(datetime.datetime.now()) , '] Start Meta-Training.')
    train_loader = iter(learner.train_loader)
    for it in range(start_it, cfg['iteration']):
        start_sec = time.time()

        # Train
        batch = train_loader.next()
        results_train = learner.train_one_iter(it, batch)
        accum_accuracy += results_train['accuracy']
        loss = results_train['loss']
        grad_norms = results_train['grad_norms']

        # Save train losses per iteration
        losses = [results_train['accuracy']]
        index = [it]
        pd_logger.add('train_losses', losses, index=index)

        grads = [[grad_norms[name] for name in grad_norms]]
        names = list(grad_norms.keys())
        pd_logger.add('train_grads', grads, index=[it], columns=names)

        if (it + 1) % cfg['validation_freq'] == 0:
            outman.print('[', str(datetime.datetime.now()) , '] Iteration: ', str(it), prefix=prefix)
            moving_accuracy = accum_accuracy / cfg['validation_freq']
            pd_logger.add('train_accs', [moving_accuracy], index=[it], columns=['train-acc'])
            outman.print('Train Accuracy:', str(moving_accuracy), prefix=prefix)
            if cfg['print_train_loss']:
                outman.print('Train Loss:', str(loss), prefix=prefix)
            if math.isnan(loss):
                break
            accum_accuracy = 0.0

            # Evaluate
            results_eval = learner.evaluate(dataset_type='val')

            val_accuracy = results_eval['accuracy']
            pd_logger.add('val_accs', [val_accuracy], index=[it], columns=['val-acc'])
            outman.print('Val Accuracy:', str(val_accuracy), prefix=prefix)

            grad_norms = results_eval['grad_norms']
            grads = [[grad_norms[name] for name in grad_norms]]
            names = list(grad_norms.keys())
            pd_logger.add('val_grads', grads, index=[it], columns=names)

            # Flag if save best model
            if (best_value is None) or (best_value < val_accuracy):
                best_value = val_accuracy
                best_it = it
                save_best_model = True
            else:
                save_best_model = False

            end_sec = time.time()
            total_seconds += end_sec - start_sec

            sparsity_info = learner.get_sparsity(simplified_keys=True)
            if sparsity_info is not None:
                pd_logger.add('sparsities', [[sparsity_info[k] for k in sparsity_info]],
                                          index=[it],
                                          columns=sparsity_info.keys())
            if cfg['print_sparsity']:
                outman.print('Current sparsities:', prefix=prefix)
                outman.pprint(sparsity_info, prefix=prefix)

            if isinstance(learner.meta_model, DataParallel):
                model_state_dict = learner.meta_model.module.state_dict()
            else:
                model_state_dict = learner.meta_model.state_dict()
            dump_dict = {
                    'it': it,
                    'model_state_dict': model_state_dict,
                    'optim_state_dict': learner.optimizer.state_dict(),
                    'sched_state_dict': learner.scheduler.state_dict(),
                    'best_val': best_value,
                    'best_it': best_it,
                    'total_seconds': total_seconds,
            }
            info_dict = {
                    'last_val': val_accuracy,
                    'it': it,
                    'best_val': best_value,
                    'best_it': best_it,
                    'loss_train': loss,
                    'acc_train': moving_accuracy,
                    'total_time': str(datetime.timedelta(seconds=int(total_seconds))),
                    'total_seconds': total_seconds,
                    'prefix': prefix,
                    'params_info': params_info,
            }
            outman.save_dict(dump_dict, prefix=f"dump.{prefix}", ext="pth")
            with open(outman.get_abspath(prefix=f"info.{prefix}", ext="json"), 'w') as f:
                json.dump(info_dict, f, indent=2)
            if save_best_model and cfg['save_best_model']:
                outman.save_dict(dump_dict, prefix=f"best.{prefix}", ext="pth")
            if it in cfg['checkpoint_iterations']:
                outman.save_dict(dump_dict, prefix=f'it{it}.{prefix}', ext='pth')

            pd_logger.save()

    meta_test(exp_name, cfg, gpu_id, prefix=prefix)

