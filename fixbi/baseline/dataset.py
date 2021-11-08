from pathlib import Path

import torch
from torch.utils.data import DataLoader, Sampler, BatchSampler, RandomSampler

from torchvision import transforms
from torchvision.datasets import ImageFolder

import numpy as np
import random

# for reproducibility        
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
        
def get_generator(seed):
    g = torch.Generator()
    g.manual_seed(seed)
    return g

def load_data(args, domain='source', train=True):
    transform = {
        'train': transforms.Compose(
            [transforms.Resize([256, 256]),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])]),
        'test': transforms.Compose(
            [transforms.Resize([224, 224]),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])])
    }
    
    path = Path(args.dataset['path']) / args.dataset['domains'][args.model[domain]]
    
    g = get_generator(args.seed)
    
    dset = ImageFolder(path, transform=transform['train' if train else 'test'])
    if train:
        dloader = InfiniteDataLoader(dset,
            batch_size = args.bsize,
            worker_init_fn=seed_worker, generator=g,
            drop_last=True, num_workers=4)
    else:
        dloader = DataLoader(dset, 
            batch_size = args.bsize,
            worker_init_fn=seed_worker, generator=g, 
            shuffle=False, drop_last=False, num_workers=4, pin_memory=True)
    
    return dset, dloader

class _InfiniteSampler(Sampler):
    """Wraps another Sampler to yield an infinite stream."""
    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch

class InfiniteDataLoader:
    def __init__(self, dataset, batch_size, 
                 worker_init_fn=None, 
                 generator=None, drop_last=True, 
                 num_workers=4):
        
        sampler = RandomSampler(dataset, replacement=False, generator=generator)
        batch_sampler = BatchSampler(sampler, batch_size=batch_size, drop_last=drop_last)

        self._infinite_iterator = iter(DataLoader(
            dataset,
            num_workers=num_workers,
            batch_sampler=_InfiniteSampler(batch_sampler),
            worker_init_fn=worker_init_fn
        ))

    def __iter__(self):
        while True:
            yield next(self._infinite_iterator)

    def __len__(self):
        return 0
