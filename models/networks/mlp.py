
import torch
import torch.nn as nn
import learn2learn as l2l
import math
from scipy.stats import truncnorm

def truncated_normal_(tensor, mean=0.0, std=1.0):
    # PT doesn't have truncated normal.
    # https://discuss.pytorch.org/t/implementing-truncated-normal-initializer/4778/18
    values = truncnorm.rvs(-2, 2, size=tensor.shape)
    values = mean + std * values
    tensor.copy_(torch.from_numpy(values))
    return tensor

def fc_init_(module):
    if hasattr(module, 'weight') and module.weight is not None:
        truncated_normal_(module.weight.data, mean=0.0, std=0.01)
    if hasattr(module, 'bias') and module.bias is not None:
        torch.nn.init.constant_(module.bias.data, 0.0)
    return module


init_methods = [
        'kaiming_uniform',
        'kaiming_normal',
        'asymptotic_kn',
        'positive_kaiming_normal',
        'positive_kaiming_uniform',
        'keep_sign',
        'signed_constant',
        'positive_kaiming_constant',
        ]

def init_param_(param, init_mode=None, init_scale=1.0):
    if init_mode == 'kaiming_normal' or init_mode == 'asymptotic_kn':
        nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
        param.data *= init_scale
    elif init_mode == 'positive_kaiming_normal':
        nn.init.kaiming_normal_(param, mode="fan_in", nonlinearity="relu")
        param.data *= init_scale * param.data.sign()
    elif init_mode == 'uniform(-1,1)':
        nn.init.uniform_(param, a=-1, b=1)
        param.data *= init_scale
    elif init_mode == 'kaiming_uniform':
        nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        param.data *= init_scale
    elif init_mode == 'positive_kaiming_uniform':
        nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        param.data *= init_scale * param.data.sign()
    elif init_mode == 'keep_sign':
        nn.init.kaiming_uniform_(param, mode='fan_in', nonlinearity='relu')
        param.data *= init_scale
    elif init_mode == 'positive_kaiming_constant':
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('relu')
        std = gain / math.sqrt(fan)
        nn.init.constant_(param, std)
        param.data *= init_scale
    elif init_mode == 'signed_constant':
        fan = nn.init._calculate_correct_fan(param, 'fan_in')
        gain = nn.init.calculate_gain('relu')
        std = gain / math.sqrt(fan)
        nn.init.kaiming_normal_(param)    # use only its sign
        param.data = param.data.sign() * std
        param.data *= init_scale
    else:
        raise NotImplementedError

class LinearBlock(torch.nn.Module):

    def __init__(self, input_size, output_size, bn=True, bn_affine=True, bn_track_running_stats=False, init_mode=None, init_scale=None):
        super(LinearBlock, self).__init__()
        self.relu = torch.nn.ReLU()
        if bn:
            self.normalize = torch.nn.BatchNorm1d(
                output_size,
                affine=bn_affine,
                momentum=0.999,
                eps=1e-3,
                track_running_stats=bn_track_running_stats,
            )
        else:
            self.normalize = torch.nn.Sequential()
        self.linear = torch.nn.Linear(input_size, output_size)

        assert init_mode is not None
        if init_mode == 'fc_init':
            fc_init_(self.linear)
        elif init_mode in init_methods:
            fc_init_(self.linear)
            init_param_(self.linear.weight, init_mode=init_mode, init_scale=init_scale)
        else:
            raise NotImplementedError()

    def forward(self, x):
        x = self.linear(x)
        x = self.normalize(x)
        x = self.relu(x)
        return x

class MLP(torch.nn.Module):

    def __init__(self, input_size, output_size, sizes=None,
                 bn=True, bn_affine=True, bn_track_running_stats=False,
                 init_mode=None, init_scale=None):
        super(MLP, self).__init__()
        if sizes is None:
            sizes = [256, 128, 64, 64]
        layers = [LinearBlock(input_size, sizes[0], bn=bn,
                              bn_affine=bn_affine, bn_track_running_stats=bn_track_running_stats,
                              init_mode=init_mode, init_scale=init_scale), ]
        for s_i, s_o in zip(sizes[:-1], sizes[1:]):
            layers.append(LinearBlock(s_i, s_o, bn=bn, bn_affine=bn_affine,
                                    bn_track_running_stats=bn_track_running_stats, init_mode=init_mode, init_scale=init_scale))
        layers = torch.nn.Sequential(*layers)
        self.features = torch.nn.Sequential(
            l2l.nn.Flatten(),
            layers,
        )
        self.classifier = torch.nn.Linear(sizes[-1], output_size)

        assert init_mode is not None
        if init_mode == 'fc_init':
            fc_init_(self.classifier)
        elif init_mode in init_methods:
            fc_init_(self.classifier)
            init_param_(self.classifier.weight, init_mode=init_mode, init_scale=init_scale)
        else:
            raise NotImplementedError()

        self.input_size = input_size

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

