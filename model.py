'''
This script defines the network architecture.
'''

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from skimage import io
from skimage.transform import rescale, resize, downscale_local_mean
from utils import *


# model_weights = 'curr_model_weights.pth'
model_weights = None


class Darknet(nn.Module):

    def __init__(self, features, path=None):
        super(Darknet, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(1024 * S * S, 4096),
            nn.LeakyReLU(0.1),
            nn.Linear(4096, S * S * (B * 5 + C)),
        )

        if path is None:
            self._initialize_weights()
            print("Weights initialized.")
        else:
            # load checkpoints / pretrained weights
            self.load_state_dict({k.replace('module.', ''): v for k, v in torch.load(path).items()})
            print('Weights loaded from "{}"'.format(path))

    def forward(self, x):
        print('x')
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        x = x.view(x.size(0), S, S,  B * 5 + C)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)


def expand_cfg(cfg):
    cfg_expanded = []
    for v in cfg:
        if isinstance(v, list):
            times = v[-1]
            for _ in range(times):
                cfg_expanded = cfg_expanded + v[:-1]
        else:
            cfg_expanded.append(v)
    return cfg_expanded


def make_layers(cfg):
    '''
    Make layers based on configuration.
    :param cfg: expanded cfg, that is, no list as element
    :return: nn sequential module
    '''
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':  # Max pool
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        elif isinstance(v, tuple):
            if len(v) == 3:
                # Conv (kernel_size, out_channels, stride)
                layers += [nn.Conv2d(in_channels, out_channels=v[1], kernel_size=v[0], stride=2)]
            else:
                # Conv (kernel_size, out_channels)
                layers += [nn.Conv2d(in_channels, out_channels=v[1], kernel_size=v[0])]
                layers += [nn.BatchNorm2d(num_features=v[1])]  # BN
                print('[new] BN is added.')

            layers += [nn.LeakyReLU(0.1)]   # Leaky rectified linear activation
            in_channels = v[1]
    print('Make layers done.')
    return nn.Sequential(*layers)


## Config format
# M = Maxpool
# tuple = Conv(kernel_size, out_channels, stride)

cfg = [
        (7, 64, 2), 'M',  # 1
           (3, 192), 'M',   # 2
           (1, 128), (3, 256), (1, 256), (3, 512), 'M',  # 3
           [(1, 256), (3, 512), 4], (1, 512), (3, 1024), 'M',  # 4
           [(1, 512), (3, 1024), 2], (3, 1024), (3, 1024, 2),  # 5
           (3, 1024), (3, 1024)  # 6
    ]


def build_darknet(path=None, **kwargs):
    # define architecture
    extract_features = make_layers(cfg)
    model = Darknet(extract_features, path, **kwargs)
    '''
    # load weights if using pre-trained
    if path is not None:
        model.load_state_dict(path)
    '''
    return model


if __name__ == "__main__":

    # model
    yolo_model = build_darknet(path=model_weights)

    # input
    '''
    I = io.imread('000001.jpg')
    I = resize(I, (448, 448))
    Imgs = I[np.newaxis, :]
    Imgs = torch.Tensor(Imgs).permute(0, 3, 1, 2)
    print('Imgs.size = ', Imgs.size())
    '''

    Imgs = torch.randn(20, 3, 448, 448)  # test image batch
    print('Imgs.size = ', Imgs.size())

    # output
    output = yolo_model(Imgs)
    print('Done.')



