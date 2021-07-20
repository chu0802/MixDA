import argparse
import torch
import numpy as np
import random
from pathlib import Path
import sys
sys.path.append('./src/')

from dataset import load_data, load_mix_data
from train import source_train_full, source_train_val, target_train
from model import CrossEntropyLabelSmooth, Model
from evaluation import cal_acc, prediction
from util import config_loading, model_handler

def arguments_parsing():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dev', '--device', type=int, default=5)
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    parser.add_argument('-d', '--dataset', type=str, default='OfficeHome')

    # Mode Controlling
    parser.add_argument('-m', '--mode', type=str, default='source_train', choices=['train', 'test'])

    # Source training
    parser.add_argument('-se', '--source_num_epoches', type=int, default=-1)

    # Model Config
    parser.add_argument('-s', '--source', type=int, default=None)
    parser.add_argument('-t', '--target', type=int, default=None)
    parser.add_argument('-stg', '--strategy', type=str, choices=['source_only', 'mixup', 'target_unify'])

    model_args = ['source', 'target', 'strategy']

    args = parser.parse_args()
    return args, model_args

if __name__ == '__main__':
    # Argument Processing
    args, model_args = arguments_parsing()
    args.device = (
        torch.device('cpu')
        if args.device < 0
        else torch.device('cuda', args.device)
    )

    args.config = config_loading(args.config)
    args.dataset = args.config['datasets'][args.dataset]

    # allow modifying configuration from command line
    for ma in model_args:
        value = getattr(args, ma)
        if value is not None:
            args.config['model'][ma] = value

    # Initialize a modle handler
    args.mdh = model_handler(
        Path(args.dataset['path']) / 'model', 
        args.config['hash_table_path']
    )

    # Reproduction
    random.seed(args.config['seed'])
    np.random.seed(args.config['seed'])
    torch.manual_seed(args.config['seed'])
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # Data Loading
    dsets, dloaders = load_data(args)

    # Mode Choosing
    if args.mode == 'train':
        stg_config = args.config['strategy'][args.strategy]
        if args.strategy == 'source_only':
            args.config['model']['strategy_config'] = stg_config
            criterion = CrossEntropyLabelSmooth(args)

            if args.source_num_epoches < 0:
                best_iter = source_train_val(dloaders, criterion, args, logging=True)
            else:
                best_iter = args.source_num_epoches
            source_train_full(best_iter, dloaders['source'], criterion, args, logging=True)

        elif args.strategy == 'mixup':
            stg_config['init_labeler'] = args.mdh.select_config()

            # TODO: change to a more generalize way.
            print('type in the mixup ratio: ', end='')
            stg_config['mix_ratio'] = float(input())
            args.config['model']['strategy_config'] = stg_config

            criterion = CrossEntropyLabelSmooth(args)

            model = Model(args, num_classes=args.dataset['num_classes'], logging=False)
            model.load(args.config['model']['strategy_config']['init_labeler'])
            model.to()

            pred, _ = prediction(dloaders['target_train'], model, args, verbose=True)

            mixdset, mixdloader = load_mix_data(dsets['source'], dsets['target_train'], pred, args)
            dsets['mix'], dloaders['mix'] = mixdset, mixdloader

            target_train(dloaders, criterion, args, logging=True)
        elif args.strategy == 'target_unify':
            args.config['model']['strategy_config'] = stg_config
            criterion = CrossEntropyLabelSmooth(args)
    else:
        model = Model(args, num_classes=args.dataset['num_classes'], logging=False)
        args.config['model'] = args.mdh.select_config()
        model.load(args.config['model'])
        model.to()
        print('Accuracy: %.2f%%' % (100*cal_acc(dloaders['target_test'], model, args, verbose=True)))
