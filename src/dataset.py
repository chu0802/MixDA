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

class MixDataset(Dataset):
    def __init__(self, source, target, target_label, mix_ratio):
        super(MixDataset, self).__init__()
        self.source = source
        self.target = target
        self.target_label = target_label
        self.mix_ratio = mix_ratio
    def __len__(self):
        return len(self.source)
    def __getitem__(self, idx):
        sx, sy = self.source[idx]
        t_idx = np.random.randint(len(self.target))
        tx, _ = self.target[t_idx]
        ty = self.target_label[t_idx]
        mx = self.mix_ratio * sx + (1 - self.mix_ratio) * tx
        return mx, (sy, ty)

def load_mix_data(source, target, target_label, args):
    # For reproducibility
    g = torch.Generator()
    g.manual_seed(args.config['seed'])

    mixdset = MixDataset(source, target, target_label, args.config['model']['strategy_config']['mix_ratio'])
    mixdloader = DataLoader(mixdset, 
            batch_size=args.config['train']['bsize'],
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=True, num_workers=4, pin_memory=True)
    return mixdset, mixdloader

def load_data(args):
    dsets = {}
    dloaders = {}

    source_exist = args.config['model']['source']
    target_exist = args.config['model']['target']

    # For reproducibility
    g = torch.Generator()
    g.manual_seed(args.config['seed'])

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
        target_path = Path(args.dataset['path']) / args.dataset['domains'][args.config['model']['target']]
        dsets['target_train'] = ImageFolder(target_path, transform=train_transform())
        dloaders['target_train'] = DataLoader(dsets['target_train'], 
                                            batch_size=args.config['train']['bsize'], 
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            shuffle=False, num_workers=4, pin_memory=True)

        dsets['target_test'] = ImageFolder(target_path, transform=test_transform())
        dloaders['target_test'] = DataLoader(dsets['target_test'], 
                                            batch_size=args.config['eval']['bsize'], 
                                            worker_init_fn=seed_worker,
                                            generator=g,
                                            shuffle=False, num_workers=4, pin_memory=True)

    return dsets, dloaders

