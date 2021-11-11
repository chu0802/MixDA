import configargparse
import os
import random
from pathlib import Path

# safely load from string to dict
from ast import literal_eval

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model import DANN_Model, Basic_Model
from util import config_loading, model_handler, set_seed
from dataset import load_data

from train import train_dann
from evaluation import evaluation

def arguments_parsing():
    p = configargparse.ArgumentParser(config_file_parser_class=configargparse.YAMLConfigFileParser)
    p.add('--config', is_config_file=True, default='./config.yaml')
    p.add('--device', type=str, default='1')
    p.add('--mode', type=str, default='train')
    
    # choosing strategies, and models, and datasets
    p.add('--dataset', type=str, default='OfficeHome')
    p.add('--strategy', type=str, default='fixbi')
    p.add('--model', type=literal_eval)
    
    p.add('--source', type=int, default=0)
    p.add('--target', type=int, default=1)
    
    # transfer settings
    p.add('--transfer_loss_weight', type=float, default=1.0)
    
    # training settings
    p.add('--seed', type=int, default=1126)
    p.add('--bsize', type=int, default=32)
    p.add('--num_iters', type=int, default=1000)
    p.add('--eval_interval', type=int, default=100)

    # configurations
    p.add('--strategy_cfg', type=literal_eval)
    p.add('--dataset_cfg', type=literal_eval)
    
    # optimizer
    p.add('--lr', type=float, default=1e-2)
    p.add('--momentum', type=float, default=0.9)
    p.add('--weight_decay', type=float, default=5e-4)
    
    # lr_scheduler
    p.add('--lr_gamma', type=float, default=3e-4)
    p.add('--lr_decay', type=float, default=0.75)
    
    # mdh
    p.add('--hash_table_path', type=str)
    return p.parse_args()

def get_optimizer(model, args):
    params = model.get_parameters(init_lr=1.0)
    optimizer = torch.optim.SGD(params, weight_decay=args.weight_decay, lr=args.lr, momentum=args.momentum, nesterov=False)
    return optimizer

def get_scheduler(optimizer, args):
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, 
        lambda x:  args.lr * (1. + args.lr_gamma * float(x)) ** (-args.lr_decay)
    )
    return scheduler

def get_dloaders(args):
    src_dset, src_dloader = load_data(args, domain='source', train=True)
    tgt_train_dset, tgt_train_dloader = load_data(args, domain='target', train=True)
    tgt_test_dset, tgt_test_dloader = load_data(args, domain='target', train=False)
    
    return src_dloader, tgt_train_dloader, tgt_test_dloader
        
def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    set_seed(args.seed)

    if args.mode == 'train':
        if args.strategy == 'dann':
            model = DANN_Model(args, logging=True)
            model.to()

            optimizer = get_optimizer(model, args)
            lr_scheduler = get_scheduler(optimizer, args)

            train_dann(args, get_dloaders(args), model, optimizer, lr_scheduler, logging=model.logging)
        elif args.strategy == 'fixbi':
            pass
        elif args.strategy == 'source_only':
            model = Basic_Model(args, logging=True)
            model.to()

            optimizer = get_optimizer(model, args)
            lr_scheduler = get_scheduler(optimizer, args)

            train_source_only(args, get_dloaders(args), model, optimizer, lr_scheduler, logging=model.logging)
            
    elif args.mode == 'test':
        model_cfg_list = [{'source': i, 'target': j, 'strategy': {'name': 'dann'}} for i in range(4) for j in range(4) if i != j]
        for model_cfg in model_cfg_list:
            # replace the model config to the correct one
            args.model = model_cfg

            model = DANN_Model(args, logging=False)
            model.load(model_cfg, epoch=1000)
            model.to()

            _, _, test_dloader = get_dloaders(args)
            
            print('source %d, target: %d' % (model_cfg['source'], model_cfg['target']))
            c_acc = evaluation(test_dloader, model)
            print('\tmodel acc: %.2f%%' % (100 * c_acc))
        
    
if __name__ == '__main__':
    args = arguments_parsing()
    
    # replace the configuration
    args.dataset = args.dataset_cfg[args.dataset]
    args.model['source'], args.model['target'] = args.source, args.target
    args.model['strategy'] = args.strategy_cfg[args.strategy]
    
    # mdh
    args.mdh = model_handler(
        Path(args.dataset['path']) / 'model', 
        args.hash_table_path
    )
    main(args)
