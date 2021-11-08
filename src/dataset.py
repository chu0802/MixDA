from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader, random_split

import torchvision
from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
import random

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

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

class BiDataset(Dataset):
    def __init__(self, source, target_loader, target_cand, target_label):
        super(BiDataset, self).__init__()
        self.source = source
        self.target_loader = target_loader
        self.target_cand = target_cand
        self.target_label = target_label
    def __len__(self):
        return len(self.source)

    def __getitem__(self, idx):
        sx, sy = self.source[idx]
        t_idx = np.random.randint(len(self.target_cand))
        path, _ = self.target_cand[t_idx]
        tx = self.target_loader.transform(self.target_loader.loader(path))
        ty = self.target_label[t_idx]
        return (sx, sy), (tx, ty)


class MixDataset(Dataset):
    def __init__(self, source, target, target_cand, target_label, mix_ratio):
        super(MixDataset, self).__init__()
        self.source = source
        self.target = target
        self.target_cand = target_cand
        self.target_label = target_label
        self.mix_ratio = mix_ratio
    def __len__(self):
        return len(self.source)
    def __getitem__(self, idx):
        sx, sy = self.source[idx]
        t_idx = np.random.randint(len(self.target_cand))
        path, _ = self.target_cand[t_idx]
        tx = self.target.transform(self.target.loader(path))
        ty = self.target_label[t_idx]
        mx = self.mix_ratio * sx + (1 - self.mix_ratio) * tx
        return mx, (sy, ty)

class ReducedDataset(Dataset):
    def __init__(self, data, cand, labels):
        super(ReducedDataset, self).__init__()
        self.data = data
        self.cand = cand
        self.labels = labels
    def __len__(self):
        return len(self.cand)
    def __getitem__(self, idx):
        path, _ = self.cand[idx]
        x = self.data.transform(self.data.loader(path))
        y = self.labels[idx]
        return x, y

def load_reduced_data(target, target_cand, target_label, args):
    g = get_generator(args.config['seed'])
    reduced_data = ReducedDataset(target, target_cand, target_label)
    reduced_loader = DataLoader(reduced_data,
            batch_size=args.config['train']['bsize'],
            worker_init_fn=seed_worker,
        generator=g,
            shuffle=True, num_workers=4, pin_memory=True)
    return reduced_data, reduced_loader

def load_mix_data(source, target, target_cand, target_label, args, ratio=None):
    # For reproducibility
    g = get_generator(args.config['seed'])
    if ratio is None:
        ratio = args.config['model']['strategy_config']['mix_ratio']

    mixdset = MixDataset(source, target, target_cand, target_label, ratio)
    mixdloader = DataLoader(mixdset, 
            batch_size=args.config['train']['bsize'],
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=True, num_workers=4, pin_memory=True)
    return mixdset, mixdloader

def load_bi_data(source, target_loader, target_cand, target_label, args):
    g = get_generator(args.config['seed'])
    bidset = BiDataset(source, target_loader, target_cand, target_label)
    bidloader = DataLoader(bidset,
            batch_size=args.config['train']['bsize'],
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=True, num_workers=4, pin_memory=True)
    return bidset, bidloader

def load_single_data(path, seed, bsize, transform, shuffle=False):
    g = get_generator(seed)
    dset = ImageFolder(path, transform=transform)

    dloader = DataLoader(dset, 
                        batch_size=bsize,
                        worker_init_fn=seed_worker,
                        generator=g,
                        shuffle=shuffle, num_workers=4, pin_memory=True)
    return dset, dloader

def load_target(args, shuffle=False):
    target_path = Path(args.dataset['path']) / args.dataset['domains'][args.config['model']['target']]
    dsets, dloaders = {}, {}
    dsets['target_train'], dloaders['target_train'] = load_single_data(target_path,
            args.config['seed'],
            args.config['train']['bsize'],
            train_transform(),
            shuffle=shuffle)
    dsets['target_test'], dloaders['target_test'] = load_single_data(target_path,
            args.config['seed'],
            args.config['eval']['bsize'],
            test_transform(),
            shuffle=False)
    return dsets, dloaders

def load_data(args):
    dsets = {}
    dloaders = {}

    source_exist = args.config['model']['source']
    target_exist = args.config['model']['target']

    # For reproducibility
    g = get_generator(args.config['seed'])

    if source_exist is not None:
        source_path = Path(args.dataset['path']) / args.dataset['domains'][args.config['model']['source']]
        dsets['source'] = ImageFolder(source_path, transform=train_transform())

        # Using the whole training dataset for training
        dloaders['source'] = DataLoader(dsets['source'], 
                                        batch_size=args.config['train']['bsize'], 
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        shuffle=True, num_workers=4, pin_memory=True)
        dsize = len(dsets['source'])
        val_size = int(dsize * args.config['train']['val_ratio'])

        dsets['source_train'], dsets['source_val'] = random_split(dsets['source'], [dsize - val_size, val_size])

        dloaders['source_train'] = DataLoader(dsets['source_train'], 
                                        batch_size=args.config['train']['bsize'], 
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        shuffle=True, num_workers=4, pin_memory=True)

        # Selecting appropriate hyperparameters
        dloaders['source_val'] = DataLoader(dsets['source_val'], 
                                        batch_size=args.config['eval']['bsize'], 
                                        worker_init_fn=seed_worker,
                                        generator=g,
                                        shuffle=False, num_workers=4, pin_memory=True)

    if target_exist is not None:
        tdsets, tdloaders = load_target(args, shuffle=True)
        dsets.update(tdsets)
        dloaders.update(tdloaders)

    return dsets, dloaders
