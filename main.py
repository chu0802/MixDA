import argparse
import torch
import numpy as np
import sys
sys.path.append('./src/')

from dataset import load_data, load_mix_data
from train import source_train_full, source_train_val, target_train
from model import CrossEntropyLabelSmooth, Model
from evaluation import cal_acc, prediction
from util import config_loading

def arguments_parsing():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dev', '--device', type=int, default=5)
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    parser.add_argument('-d', '--dataset', type=str, default='OfficeHome')

    # Mode Controlling
    parser.add_argument('-m', '--mode', type=str, default='source_train', choices=['source_train', 'target_test', 'mixup'])

    # Source training
    parser.add_argument('-se', '--source_num_epoches', type=int, default=-1)

    # Model Config
    parser.add_argument('-s', '--source', type=int, default=-1)
    parser.add_argument('-t', '--target', type=int, default=-1)
    parser.add_argument('-mr', '--mix_ratio', type=float, default=-1)
    parser.add_argument('-il', '--init_labeler', type=str, default=None)
    parser.add_argument('-ic', '--init_config', type=str, default=None)

    # Model Loading
    parser.add_argument('-lp', '--loading_path', type=str, default=None)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # Argument Processing
    args = arguments_parsing()
    args.device = (
        torch.device('cpu')
        if args.device < 0
        else torch.device('cuda', args.device)
    )

    args.config = config_loading(args.config)
    args.dataset = args.config['datasets'][args.dataset]

    # allow modifying configuration from command line
    if args.source >= 0:
        args.config['model']['config']['source'] = args.source
    if args.target >= 0:
        args.config['model']['config']['target'] = args.target
    if args.mix_ratio >= 0:
        args.config['model']['config']['mix_ratio'] = args.mix_ratio
    if args.init_labeler:
        args.config['model']['init_labeler'] = args.init_labeler
    if args.init_config:
        args.config['model']['init_config'] = args.init_config

    # Reproduction
    torch.cuda.manual_seed(args.config['seed'])
    torch.manual_seed(args.config['seed'])
    np.random.seed(args.config['seed'])
    torch.backends.cudnn.benchmark = True

    # Data Loading
    dsets, dloaders = load_data(args)
    criterion = CrossEntropyLabelSmooth(args)

    # Mode Choosing
    if args.mode == 'source_train':
        if args.source_num_epoches < 0:
            best_iter = source_train_val(dloaders, criterion, args, logging=True)
        else:
            best_iter = args.source_num_epoches
        source_train_full(best_iter, dloaders['source'], criterion, args, logging=True)
    elif args.mode == 'target_test':
        model = Model(args, logging=False)
        m_cfg = config_loading(args.loading_path)
        model.load(m_cfg=m_cfg)
        model.to()
        print('Accuracy: %.2f%%' % (100*cal_acc(dloaders['target_test'], model, args, verbose=True)))
    elif args.mode == 'mixup':
        model = Model(args, logging=False)
        args.config['model']['init_labeler'] = config_loading(args.config['model']['init_labeler'])
        model.load(m_cfg=args.config['model']['init_labeler'])
        model.to()
        pred, _ = prediction(dloaders['target_train'], model, args, verbose=True)

        mixdset, mixdloader = load_mix_data(dsets['source'], dsets['target_train'], pred, args)
        dsets['mix'], dloaders['mix'] = mixdset, mixdloader

        target_train(dloaders, criterion, args, logging=True)
