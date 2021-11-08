import argparse
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
import sys
from tqdm import tqdm

sys.path.append('./src/')

from dataset import load_data, load_mix_data, load_target, load_reduced_data, load_bi_data
from train import source_train_full, source_train_val, target_train
from model import CrossEntropyLabelSmooth, Model, get_source_model, lr_scheduler
from evaluation import cal_acc, prediction
from util import config_loading, model_handler, interactive_input, set_seed

import matplotlib.pyplot as plt

def arguments_parsing():
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-dev', '--device', type=int, default=5)
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    parser.add_argument('-d', '--dataset', type=str, default='OfficeHome')

    # Mode Controlling
    parser.add_argument('-m', '--mode', type=str, default='source_train', choices=['train', 'test', 'expe'])

    # Source training
    parser.add_argument('-se', '--source_num_epoches', type=int, default=-1)

    # Model Config
    parser.add_argument('-s', '--source', type=int, default=None)
    parser.add_argument('-t', '--target', type=int, default=None)
    parser.add_argument('-stg', '--strategy', type=str)

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

    args.config['model']['strategy_config'] = args.config['strategy'][args.strategy]

    # Reproduction

    set_seed(args.config['seed'])


    # For fast testing

    if args.strategy == 'fixbi':
        # Load model config
        args.config['model']['strategy_config']['source_ratio'] = interactive_input('type in the source ratio:', float)
        args.config['model']['strategy_config']['target_ratio'] = interactive_input('type in the target ratio:', float)
        args.config['model']['strategy_config']['init_labeler'] = args.mdh.select_config()

        sratio = args.config['model']['strategy_config']['source_ratio']
        tratio = args.config['model']['strategy_config']['target_ratio']

        scriterion = CrossEntropyLabelSmooth(args, mix=sratio)
        tcriterion = CrossEntropyLabelSmooth(args, mix=tratio)
        criterion = CrossEntropyLabelSmooth(args)

        # load initial labelers (baseline model)
        model = Model(args, num_classes=args.dataset['num_classes'], logging=False)
        model.load(args.config['model']['strategy_config']['init_labeler'])
        model.to()

        dsets, dloaders = load_data(args)
        tdsets, tdloaders = load_target(args, shuffle=False)
        _, pred, _ = prediction(tdloaders['target_train'], model, args, verbose=True)

        # Get source-dominant dataset and target-dominant dataset
        bidset, bidloader = load_bi_data(dsets['source'], tdsets['target_train'], tdsets['target_train'].imgs, pred, args)

        ## TODO: Calculate self-penalization

        smodel = Model(args, num_classes=args.dataset['num_classes'], logging=False)
        tmodel = Model(args, num_classes=args.dataset['num_classes'], logging=False)
        smodel.to()
        tmodel.to()

        num_epoches = args.config['train']['target']['num_epoches']

        for epoch in range(num_epoches):
            smodel.train()
            for i, ((sx, sy), (tx, ty)) in tqdm(enumerate(bidloader),
                                desc='Source Iter %02d/%02d' % (epoch, num_epoches),
                                total=len(bidloader)):

                sx = sx.to(args.device)
                tx = tx.to(args.device)
                lr_scheduler(smodel.optimizer, epoch, num_epoches)

                # L_fm for source
                msx = sratio * sx + (1 - sratio) * tx

                sout_m, _ = smodel.forward(msx)
                sloss = scriterion(sout_m, (sy, ty))
                
                smodel.optimizer.zero_grad()
                sloss.backward()
                smodel.optimizer.step()

                # L_sp for source
                sout_t_raw, _ = smodel.forward(tx)
                sout_t_soft = nn.Softmax(dim=1)(sout_t_raw)
                sout_t_soft = sout_t_soft.detach().cpu()
                _, sout_t_pred = torch.max(sout_t_soft, 1)
                sout_t_pred = sout_t_pred.detach().cpu()

                confidence = sout_t_soft.gather(1, sout_t_pred.reshape(-1, 1)).flatten()
                th = confidence.mean() - 2*confidence.std()

                sspx = tx[confidence < th]
                sspy = ty[confidence < th]

                print(sspx, sspy)
                
                break
            break

        exit()

    # Mode Choosing
    if args.mode == 'train':
        # Data Loading
        dsets, dloaders = load_data(args)

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
            stg_config['mix_ratio'] = interactive_input('type in the mixup ratio:', float)
            args.config['model']['strategy_config'] = stg_config

            criterion = CrossEntropyLabelSmooth(args, args.config['model']['strategy_config']['mix_ratio'])
            model = Model(args, num_classes=args.dataset['num_classes'], logging=False)
            model.load(args.config['model']['strategy_config']['init_labeler'])
            model.to()

            _, pred, _ = prediction(dloaders['target_train'], model, args, verbose=True)

            mixdset, mixdloader = load_mix_data(dsets['source'], dsets['target_train'], dsets['target_train'].imgs, pred, args)
            dsets['mix'], dloaders['mix'] = mixdset, mixdloader

            target_train(dloaders, criterion, args, logging=True)
        elif args.strategy == 'target_unify':
            args.config['model']['strategy_config'] = stg_config
            criterion = CrossEntropyLabelSmooth(args)
        elif args.strategy == 'pseudolabel':
            stg_config['mix_ratio'] = interactive_input('type in the mixup ratio:', float)
            stg_config['confidence_ratio'] = interactive_input('type in the confidence ratio:', float) 
            stg_config['ground_truth'] = interactive_input('using ground truth?', bool)

            args.config['model']['strategy_config'] = stg_config
            criterion = CrossEntropyLabelSmooth(args, args.config['model']['strategy_config']['mix_ratio'])

            init_labeler = Model(args, num_classes=args.dataset['num_classes'], logging=False)
            init_cfg = get_source_model(args.config['model']['source'])
            init_labeler.load(init_cfg)
            init_labeler.to()

            tdsets, tdloaders = load_target(args, shuffle=False)

            raw_pred, pred, true = prediction(tdloaders['target_train'], init_labeler, args, verbose=True)
            tlabels = true if args.config['model']['strategy_config']['ground_truth'] else pred

            max_prob = raw_pred[np.arange(len(raw_pred)), pred]
            th = args.config['model']['strategy_config']['confidence_ratio']
            confidence_idx = (max_prob >= th)

            raw_target = tdsets['target_train'].imgs

            reduced_tdsets = [raw_target[i] for i in range(len(confidence_idx)) if confidence_idx[i]]
            reduced_tlabels = tlabels[confidence_idx]

            mixdset, mixdloader = load_mix_data(dsets['source'], tdsets['target_train'], reduced_tdsets, reduced_tlabels, args)
            dsets['mix'], dloaders['mix'] = mixdset, mixdloader

            target_train(dloaders, criterion, args, logging=True)
        elif args.strategy == 'partial_ground_truth':
            # mixup with partial target data
            stg_config['ratio'] = interactive_input('type in the ratio:', float)
            stg_config['mix_ratio'] = interactive_input('type in the mixup ratio:', float)
            args.config['model']['strategy_config'] = stg_config
            criterion = CrossEntropyLabelSmooth(args, args.config['model']['strategy_config']['mix_ratio'])

            ground_truth = Model(args, num_classes=args.dataset['num_classes'], logging=False)
            ground_truth_cfg = get_source_model(args.config['model']['target'])
            ground_truth.load(ground_truth_cfg)
            ground_truth.to()

            tdsets, tdloaders = load_target(args, shuffle=False)
            raw_pred, pred, _ = prediction(tdloaders['target_train'], ground_truth, args, verbose=True)

            ratio = args.config['model']['strategy_config']['ratio']
            idx = np.random.choice(len(tdsets['target_train']), size=int(ratio*len(tdsets['target_train'])), replace=False)

            raw_target = tdsets['target_train'].imgs
            reduced_tdsets = [raw_target[i] for i in idx]
            reduced_tlabels = pred[idx]

            mixdset, mixdloader = load_mix_data(dsets['source'], tdsets['target_train'], reduced_tdsets, reduced_tlabels, args)
            dsets['mix'], dloaders['mix'] = mixdset, mixdloader

            target_train(dloaders, criterion, args, logging=True)
        elif args.strategy == 'partial_answer':
            #semi supervised
            stg_config['ratio'] = interactive_input('type in the ratio:', float)
            args.config['model']['strategy_config'] = stg_config
            criterion = CrossEntropyLabelSmooth(args, args.config['model']['strategy_config']['mix_ratio'])

            ground_truth = Model(args, num_classes=args.dataset['num_classes'], logging=False)
            ground_truth_cfg = get_source_model(args.config['model']['target'])
            ground_truth.load(ground_truth_cfg)
            ground_truth.to()

            tdsets, tdloaders = load_target(args, shuffle=False)
            raw_pred, pred, _ = prediction(tdloaders['target_train'], ground_truth, args, verbose=True)

            ratio = args.config['model']['strategy_config']['ratio']
            idx = np.random.choice(len(tdsets['target_train']), size=int(ratio*len(tdsets['target_train'])), replace=False)

            raw_target = tdsets['target_train'].imgs
            reduced_tdsets = [raw_target[i] for i in idx]
            reduced_tlabels = pred[idx]

            rdset, rdloader = load_reduced_data(tdsets['target_train'], reduced_tdsets, reduced_tlabels, args)
            source_train_full(200, rdloader, criterion, args, logging=True, eval_loader=tdloaders['target_test'])
        elif args.strategy == 'mix_conf_ratio':
            stg_config['init_labeler'] = args.mdh.select_config()
            stg_config['mix_ratio'] = interactive_input('type in the mixup ratio:', float)
            stg_config['confidence_ratio'] = interactive_input('type in the confidence ratio:', float)

            args.config['model']['strategy_config'] = stg_config
            criterion = CrossEntropyLabelSmooth(args, args.config['model']['strategy_config']['mix_ratio'])

            model = Model(args, num_classes=args.dataset['num_classes'], logging=False)
            model.load(args.config['model']['strategy_config']['init_labeler'])
            model.to()

            tdsets, tdloaders = load_target(args, shuffle=False)

            raw_pred, pred, true = prediction(tdloaders['target_train'], model, args, verbose=True)

            max_prob = raw_pred[np.arange(len(raw_pred)), pred]
            th = args.config['model']['strategy_config']['confidence_ratio']
            confidence_idx = (max_prob >= th)

            # If we don't extract raw data (images), the processing time of the below 3 lines would be very long.

            raw_target = tdsets['target_train'].imgs

            reduced_tdsets = [raw_target[i] for i in range(len(confidence_idx)) if confidence_idx[i]]
            reduced_tlabels = pred[confidence_idx]

            mixdset, mixdloader = load_mix_data(dsets['source'], tdsets['target_train'], reduced_tdsets, reduced_tlabels, args)
            dsets['mix'], dloaders['mix'] = mixdset, mixdloader

            target_train(dloaders, criterion, args, logging=True)

    elif args.mode == 'test':
        # Data Loading
        dsets, dloaders = load_data(args)
        model = Model(args, num_classes=args.dataset['num_classes'], logging=False)
        args.config['model'] = args.mdh.select_config()
        model.load(args.config['model'])
        model.to()
        print('Accuracy: %.2f%%' % (100*cal_acc(dloaders['target_test'], model, args, verbose=True)))
    elif args.mode == 'expe':
        if args.strategy == 'num_cls':
            num_domains = len(args.dataset['domains'])
            for source in range(num_domains):
                for target in range(num_domains):
                    if source == target:
                        continue
                    args.config['model']['source'] = source
                    args.config['model']['target'] = target
                    # Data Loading
                    dsets, dloaders = load_data(args)

                    print('-'*10, 'source: %d, target: %d' % (source, target), '-'*10)
                    model = Model(args, num_classes=args.dataset['num_classes'], logging=False)
                    args.config['model'] = get_source_model(source)
                    model.load(args.config['model'])
                    model.to()

                    raw_pred, pred, true = prediction(dloaders['target_test'], model, args, verbose=False)
                    max_prob = raw_pred[np.arange(len(raw_pred)), pred]
                    th = 0.9
                    confidence_idx = (max_prob >= th)
                    confidence_res = (pred[confidence_idx] == true[confidence_idx])

                    num_res = len(pred[confidence_idx])
                    acc_res = confidence_res.float().mean().item()

                    print('total num: %d, ratio: %.4f, accuracy: %.4f\n' % (num_res, num_res / len(pred), acc_res))
                    unique, counts = torch.unique(pred[confidence_idx], sorted=True, return_inverse=False, return_counts=True)
                    for i, cnt, in enumerate(counts):
                        print('class %d, num: %d\n' % (i, cnt))


        elif args.strategy == 'plot_acc':
            num_domains = len(args.dataset['domains'])

            for source in range(num_domains):
                for target in range(num_domains):
                    if source == target:
                        continue
                    args.config['model']['source'] = source
                    args.config['model']['target'] = target
                    # Data Loading
                    dsets, dloaders = load_data(args)

                    print('-'*10, 'source: %d, target: %d' % (source, target), '-'*10)

                    model = Model(args, num_classes=args.dataset['num_classes'], logging=False)
                    args.config['model'] = get_source_model(source)
                    model.load(args.config['model'])
                    model.to()

                    raw_pred, pred, true = prediction(dloaders['target_test'], model, args, verbose=True)
                    max_prob = raw_pred[np.arange(len(raw_pred)), pred]
                    threshold = np.linspace(0, 1, 10, endpoint=False)
                    confidence_res = [(pred[max_prob >= th] == true[max_prob >= th]) for th in threshold]
                    num_res = [len(c) for c in confidence_res]
                    acc_res = [c.float().mean().item() for c in confidence_res]

                    
                    print(num_res)

                    plotting = False

                    if plotting:
                        title = 'Source: %d, Target: %d' % (args.source, args.target)

                        fig, ax = plt.subplots()

                        ax.plot(threshold, acc_res)
                        ax.set_title(title)
                        ax.set_xlabel('confidence ratio')
                        ax.set_ylabel('accuracy')

                        fig.savefig(title.replace(' ', '') + '.png')
            
            
