from torchvision import models
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
from torch.utils.tensorboard import SummaryWriter

import os
from pathlib import Path

Models = {'resnet50': models.resnet50, 'resnet101': models.resnet101}

class ResBase(nn.Module):
    def __init__(self, backbone='resnet50'):
        super(ResBase, self).__init__()
        self.res = Models[backbone](pretrained=True)
        self.in_features = self.res.fc.in_features
        self.res.fc = nn.Identity()

    def forward(self, x):
        return self.res(x)

class BottleNeck(nn.Module):
    def __init__(self, in_features, bottleneck_dim):
        super(BottleNeck, self).__init__()
        self.bottleneck = nn.Linear(in_features, bottleneck_dim)
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True)
    def forward(self, x):
        return self.bn(self.bottleneck(x))

class Classifier(nn.Module):
    def __init__(self, bottleneck_dim, num_classes):
        super(Classifier, self).__init__()
        self.fc = weightNorm(nn.Linear(bottleneck_dim, num_classes), name='weight')
    def forward(self, x):
        return self.fc(x)

def lr_scheduler(optimizer, num_iter, num_epoches, gamma=10, power=0.75):
    decay = (1 + gamma * num_iter / num_epoches) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
    return optimizer

def get_optimizer(model, init_lr):
    F, B, C = model
    param_group = []
    for k, v in F.named_parameters():
        param_group += [{'params': v, 'lr': init_lr * 0.1}]
    for k, v in B.named_parameters():
        param_group += [{'params': v, 'lr': init_lr}]
    for k, v in C.named_parameters():
        param_group += [{'params': v, 'lr': init_lr}]

    optimizer = torch.optim.SGD(param_group, weight_decay=1e-3, momentum=0.9, nesterov=True)

    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']

    return optimizer

def get_source_model(source):
    cfg = {'source': source, 'target': None, 'strategy': 'source_only', 'strategy_config': None}
    return cfg

class Model:
    def __init__(self, args, num_classes, bottleneck_dim=256, logging=True, backbone='resnet50'):
        self.F = ResBase(backbone)
        self.B = BottleNeck(self.F.in_features, bottleneck_dim)
        self.C = Classifier(bottleneck_dim, args.dataset['num_classes'])
        self.optimizer = get_optimizer((self.F, self.B, self.C), args.config['train']['lr'])

        self.args = args

        if logging:
            self.args.mdh.update(args.config['model'])
            self.ckpt_dir = self.args.mdh.get_ckpt_dir(args.config['model'])
            self.log_dir = self.args.mdh.get_log(args.config['model'])
            self.logger = SummaryWriter(self.log_dir)

    def to(self):
        self.F = self.F.to(self.args.device)
        self.B = self.B.to(self.args.device)
        self.C = self.C.to(self.args.device)

    def train(self):
        self.F.train()
        self.B.train()
        self.C.train()

    def eval(self):
        self.F.eval()
        self.B.eval()
        self.C.eval()

    def copy(self, m):
        self.F.load_state_dict(m.F.state_dict())
        self.B.load_state_dict(m.B.state_dict())
        self.C.load_state_dict(m.C.state_dict())

    def save(self, epoch=None):
        states = {
            'F': self.F.state_dict(),
            'B': self.B.state_dict(),
            'C': self.C.state_dict(),
            'epoch': epoch
        }
        ckpt_file = (str(epoch) if epoch else 'final') + '.pt'
        torch.save(states, self.ckpt_dir / ckpt_file)

    def load(self, cfg, epoch=None):
        ckpt_file = self.args.mdh.get_ckpt(cfg, epoch)
        states = torch.load(ckpt_file, map_location='cpu')

        self.F.load_state_dict(states['F'])
        self.B.load_state_dict(states['B'])
        self.C.load_state_dict(states['C'])

        if states['epoch']:
            return states['epoch']

    def forward(self, x):
        feature = self.B(self.F(x))
        return self.C(feature), feature

class CrossEntropyLabelSmooth(nn.Module):
    """Cross entropy loss with label smoothing regularizer.
    Reference:
    Szegedy et al. Rethinking the Inception Architecture for Computer Vision. CVPR 2016.
    Equation: y = (1 - epsilon) * y + epsilon / K.
    Args:
        num_classes (int): number of classes.
        epsilon (float): weight.
    """

    def __init__(self, args, epsilon=0.1, reduction=True, mix=None):
        super(CrossEntropyLabelSmooth, self).__init__()
        self.num_classes = args.dataset['num_classes']
        self.device = args.device
        self.epsilon = epsilon
        self.reduction = reduction
        self.logsoftmax = nn.LogSoftmax(dim=1)
        self.mix = mix
    def _scatter_truth(self, truth):
        truth = torch.zeros((len(truth), self.num_classes)).scatter_(1, truth.unsqueeze(1).cpu(), 1)
        truth = truth.to(self.device)
        truth = (1 - self.epsilon) * truth + self.epsilon / self.num_classes
        return truth

    def forward(self, pred, truth):
        """
        Args:
            pred: prediction matrix (before softmax) with shape (batch_size, num_classes)
            truth: ground truth labels with shape (num_classes)
        """
        log_probs = self.logsoftmax(pred)
        if self.mix:
            s_truth, t_truth = truth
            s_truth = self._scatter_truth(s_truth)
            t_truth = self._scatter_truth(t_truth)
            truth = self.mix * s_truth + (1 - self.mix) * t_truth
        else: 
            truth = self._scatter_truth(truth)

        loss = (- truth * log_probs).sum(dim=1)
        if self.reduction:
            return loss.mean()
        else:
            return loss
