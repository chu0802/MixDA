from torchvision import models
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import Function

import os
from pathlib import Path
import numpy as np

Models = {'resnet50': models.resnet50, 'resnet101': models.resnet101}

class alpha_scheduler():
    def __init__(self, gamma=1.0, max_iter=1000):
        self.gamma = gamma
        self.max_iter = max_iter
        self.curr_iter = 0
    def get_alpha(self):
        p = self.curr_iter / self.max_iter
        return 2. / (1. + np.exp(-self.gamma * p)) - 1
    def step(self):
        self.curr_iter = min(self.curr_iter + 1, self.max_iter)

class ReverseLayerF(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None

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
    
class Discriminator(nn.Module):
    def __init__(self, in_features, inner_dim=100):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=in_features, out_features=inner_dim),
            nn.ReLU(),
            nn.Linear(in_features=inner_dim, out_features=2)
        )

    def forward(self, x):
        return self.discriminator(x)

class BaseModel(nn.Module):
    def __init__(self, args, logging=True):
        super(BaseModel, self).__init__()
        self.models = {}
        self.args = args
        self.logging = logging
        if logging:
            self.args.mdh.update(args.model)
            self.ckpt_dir = self.args.mdh.get_ckpt_dir(args.model)
            self.log_dir = self.args.mdh.get_log(args.model)
            self.logger = SummaryWriter(self.log_dir)
            
    def copy_weight(self, m):
        for k in self.models.keys():
            self.models[k].load_state_dict(m.models[k].state_dict())    
        
    def get_parameters(self, init_lr=1.0):
        params = []
        for k, v in self.models.items():
            params += [{'params': v.parameters(), 'lr': (0.1 if k == 'F' else 1.0) * init_lr}]
        return params
    
    def to(self):
        for v in self.models.values():
            v.cuda()
    def train(self):
        for v in self.models.values():
            v.train()
    def eval(self):
        for v in self.models.values():
            v.eval()
            
    def save(self, epoch=None, name=None):
        states = {}
        for k, v in self.models.items():
            states[k] = v.state_dict()
        states['epoch'] = epoch
        
        ckpt_file = ('' if name is None else name) + (str(epoch) if epoch else 'final') + '.pt'
        torch.save(states, self.ckpt_dir / ckpt_file)
        
    def load(self, cfg, epoch=None, name=None):
        ckpt_file = self.args.mdh.get_ckpt(cfg, epoch, name)
        states = torch.load(ckpt_file, map_location='cpu')
        num_epoches = states['epoch']
        del states['epoch']

        for k, v in states.items():
            self.models[k].load_state_dict(v)

        if num_epoches:
            return num_epoches
        
    def forward(self, *x):
        raise NotImplementedError

class DANN_Model(BaseModel):
    def __init__(self, args, bottleneck_dim=256, logging=True, backbone='resnet50'):
        super(DANN_Model, self).__init__(args=args, logging=logging)
        self.models['F'] = ResBase(backbone)
        self.models['B'] = BottleNeck(self.models['F'].in_features, bottleneck_dim)
        self.models['C'] = Classifier(bottleneck_dim, args.dataset['num_classes'])
        self.models['D'] = Discriminator(bottleneck_dim)
                
        self.criterion = nn.CrossEntropyLoss()
        self.alpha = alpha_scheduler()
        
    def forward(self, sx, tx, sy):
        source_f = self.models['B'](self.models['F'](sx))
        target_f = self.models['B'](self.models['F'](tx))
        
        source_clf = self.models['C'](source_f)

        clf_loss = self.criterion(source_clf, sy)
        
        alpha = self.alpha.get_alpha()
        s_domain_loss = self.get_adv_result(source_f, True, alpha)
        t_domain_loss = self.get_adv_result(target_f, False, alpha)

        domain_loss = (s_domain_loss + t_domain_loss)/2
        
        self.alpha.step()
        # return class prediction, and domain prediction
        return clf_loss, domain_loss
    
    def get_adv_result(self, x, source, alpha):
        x = ReverseLayerF.apply(x, alpha)
        domain_pred = self.models['D'](x)
        domain_label = torch.zeros(x.shape[0]).long() if source else torch.ones(x.shape[0]).long()
        return self.criterion(domain_pred, domain_label.cuda())
    
    def predict(self, x):
        return self.models['C'](self.models['B'](self.models['F'](x)))

class Basic_Model(BaseModel):
    def __init__(self, args, bottleneck_dim=256, logging=True, backbone='resnet50'):
        super(DANN_Model, self).__init__(args=args, logging=logging)
        self.models['F'] = ResBase(backbone)
        self.models['B'] = BottleNeck(self.models['F'].in_features, bottleneck_dim)
        self.models['C'] = Classifier(bottleneck_dim, args.dataset['num_classes'])

        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, x, y):
        output = self.models['C'](self.models['B'](self.models['F'](x)))
        return self.criterion(output, y)
