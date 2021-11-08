import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import transforms
from torchvision.datasets import ImageFolder

from torch.utils.data import Dataset, DataLoader, random_split

from util import config_loading, model_handler, set_seed
from model import Model, CrossEntropyLabelSmooth

import numpy as np
import random
from tqdm import tqdm

def arguments_parsing():
    parser = argparse.ArgumentParser()
    parser.add_argument('-dev', '--device', type=str, default='1')
    parser.add_argument('-c', '--config', type=str, default='./config.yaml')
    parser.add_argument('-d', '--dataset', type=str, default='OfficeHome')
    return parser.parse_args()

def train_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    ])

def test_transform():
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                   std=[0.229, 0.224, 0.225])
    ])

# for reproducibility        
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
        
def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def load_single_data(path, seed, bsize, transform, shuffle=False, drop_last=False):
    g = get_generator(seed)
    dset = ImageFolder(path, transform=transform)

    dloader = DataLoader(dset, 
                        batch_size=bsize,
                        worker_init_fn=seed_worker,
                        generator=g,
                        shuffle=shuffle, drop_last=drop_last, num_workers=4, pin_memory=True)
    return dset, dloader

def load_data(args, domain='source', shuffle=False):
    path = Path(args.dataset['path']) / args.dataset['domains'][args.config['model'][domain]]
        
    g = get_generator(args.config['seed'])
    
    dsets, dloaders = {}, {}
    dsets['train'], dloaders['train'] = load_single_data(path,
                                                         args.config['seed'],
                                                         args.config['train']['bsize'],
                                                         train_transform(),
                                                         shuffle=shuffle,
                                                         drop_last=True)
    dsets['test'], dloaders['test'] = load_single_data(path,
                                                       args.config['seed'],
                                                       args.config['eval']['bsize'],
                                                       test_transform(),
                                                       shuffle=False,
                                                       drop_last=False)
    return dsets, dloaders

def fix_mix_loss(model, src_data, tgt_data, criterion, ratio):
    sx, sy = src_data
    tx, ty = tgt_data
    mix_x = ratio * sx + (1 - ratio) * tx
    output, _ = model.forward(mix_x)
    loss = ratio * criterion(output, sy) + (1 - ratio) * criterion(output, ty)
    return loss

def cons_reg_loss(model, sx, tx, criterion, ratio=0.5):
    s_model, t_model = model
    mix_x = ratio * sx + (1 - ratio) * tx
    s_out, _ = s_model(mix_x)
    t_out, _ = t_model(mix_x)
    return criterion(s_out, t_out)

def self_penalization_loss(x, y, temp=5):
    criterion = nn.NLLLoss()
    return criterion(torch.log(1 - F.softmax(x / temp, dim=1)), y)
    
# TODO: change the threshold to an argument
def prediction(model, x):
    pred, _ = model.forward(x)
    pred_prob = F.softmax(pred, dim=1)
    
    top_prob, top_label = torch.topk(pred_prob, k=1)
    top_prob, top_label = top_prob.flatten(), top_label.flatten()
    
    top_mean, top_std = top_prob.mean(), top_prob.std()
    threshold = top_mean - 2 * top_std

    return pred, top_label, top_prob, threshold

def evaluation(loader, model, args, verbose=False):
    pred, true = [], []
    model.eval()
    with torch.no_grad():
        if verbose:
            pbar = tqdm(desc='Evaluation', total=len(loader))
        for x, y in loader:
            x, y = x.cuda(), y.cuda()
            output, _ = model.forward(x)
            pred.append(output.float().detach())
            true.append(y.float())
            if verbose:
                pbar.update(1)
    pred, true = torch.cat(pred), torch.cat(true)
    raw_pred = nn.Softmax(dim=1)(pred)
    _, pred = torch.max(raw_pred, 1)

    return raw_pred.detach().cpu(), pred.detach().cpu(), true.cpu().int()

def cal_acc(loader, model, args, verbose=False):
    _, pred, true = evaluation(loader, model, args, verbose)
    acc = (pred == true).float().mean()
    return acc.item()

def train_fixbi(args, dloaders, models, criterion, epoch):
    s_ratio, t_ratio = 0.8, 0.2
    src_dloaders, tgt_dloaders = dloaders
    s_model, t_model = models
    cels_criterion, mse_criterion = criterion
    
    s_model.train()
    t_model.train()
    
    # Automatically drop the remaining data
    for step, (src_data, tgt_data) in enumerate(zip(src_dloaders['train'], tgt_dloaders['train'])):
        sx, sy = src_data
        tx, _ = tgt_data
        
        sx, sy = sx.cuda(), sy.cuda()
        tx = tx.cuda()
        
        # first make predictions for target data
        s_pred, s_pseudo, s_top_prob, s_threshold = prediction(s_model, tx)
        t_pred, t_pseudo, t_top_prob, t_threshold = prediction(t_model, tx)
        
        s_mask = s_top_prob < s_threshold
        t_mask = t_top_prob < t_threshold   
        
        # Fixmix loss
        # TODO: set s_ratio, t_ratio in args, or config
        s_fm_loss = fix_mix_loss(s_model, (sx, sy), (tx, t_pseudo), cels_criterion,  s_ratio)
        t_fm_loss = fix_mix_loss(t_model, (sx, sy), (tx, t_pseudo), cels_criterion, t_ratio)
        total_loss = s_fm_loss + t_fm_loss
        

        # self penalization
        # TODO: set sp_start in args or config
        if epoch > 1000:
            # TODO: set temp in args.
            if s_mask.sum().item() > 0:
                s_sp_loss = self_penalization_loss(s_pred[s_mask], s_pseudo[s_mask])
                total_loss += s_sp_loss
            if t_mask.sum().item() > 0:
                t_sp_loss = self_penalization_loss(t_pred[t_mask], t_pseudo[t_mask])
                total_loss += t_sp_loss

        # bidirectional matching
        if epoch > 100:
            if (~t_mask).sum().item() > 0:
                s_bm_loss = cels_criterion(s_pred[~t_mask], t_pseudo[~t_mask])
                total_loss += s_bm_loss
            if (~s_mask).sum().item() > 0:
                t_bm_loss = cels_criterion(t_pred[~s_mask], s_pseudo[~s_mask])
                total_loss += t_bm_loss
        
        s_model.optimizer.zero_grad()
        t_model.optimizer.zero_grad()
        total_loss.backward()
        s_model.optimizer.step()
        t_model.optimizer.step()

def main():
    args = arguments_parsing()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device
    args.config = config_loading(args.config)
    args.dataset = args.config['datasets'][args.dataset]
    
    args.mdh = model_handler(
        Path(args.dataset['path']) / 'model', 
        args.config['hash_table_path']
    )
    
    set_seed(args.config['seed'])
    
    src_dsets, src_dloaders = load_data(args, domain='source', shuffle=True)
    tgt_dsets, tgt_dloaders = load_data(args, domain='target', shuffle=True)
    
    sd_model = Model(args, logging=False)
    td_model = Model(args, logging=False)
    
    model_config = args.mdh.select_config()
    sd_model.load(model_config)
    td_model.load(model_config)
    
    sd_model.to()
    td_model.to()

    
    cels_criterion = CrossEntropyLabelSmooth(args)
#     cels_criterion = nn.CrossEntropyLoss()
    mse_criterion = nn.MSELoss()
    
    for epoch in range(args.config['train']['target']['num_epoches']):
        print('epoch %d/%d' % (epoch, args.config['train']['target']['num_epoches']))
        if epoch % 10 == 0:
            print('sd model acc: %.2f%%' % (100*cal_acc(tgt_dloaders['test'], sd_model, args, verbose=False)))
            print('td model acc: %.2f%%' % (100*cal_acc(tgt_dloaders['test'], td_model, args, verbose=False)))
          
        train_fixbi(args, (src_dloaders, tgt_dloaders), (sd_model, td_model), (cels_criterion, mse_criterion), epoch)

if __name__ == '__main__':
    main()