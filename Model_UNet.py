

import torch
import torch.nn as nn
from torchvision import models
from torch.nn.functional import relu




# UNet architecture
# The UNet architecture consists of an encoder and a decoder.
# code repurposed from https://towardsdatascience.com/cook-your-first-u-net-in-pytorch-b3297a844cf3
# https://github.com/Mostafa-wael/U-Net-in-PyTorch/tree/main



class UNet(nn.Module):
    def __init__(self, n_input_channels=4, n_output_channels=2):
        super().__init__()
        
        # Encoder
        # In the encoder, convolutional layers with the Conv2d function are used to extract features from the input image. 
        # Each block in the encoder consists of two convolutional layers followed by a max-pooling layer, with the exception of the last block which does not include a max-pooling layer.
        # -------
        # input: 320 x 176 x 4
        # F0: color (LAB), F1: grayscale (L)       [+later: flow (UV)]
        self.e11 = nn.Conv2d(n_input_channels, 64, kernel_size=3, padding='same') # output: 320x176x64
        self.e12 = nn.Conv2d(64, 64, kernel_size=3, padding='same') # output: 320x176x64
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 320x176x64

        # input: 320x176x64
        self.e21 = nn.Conv2d(64, 128, kernel_size=3, padding='same') # output: 160x88x128
        self.e22 = nn.Conv2d(128, 128, kernel_size=3, padding='same') # output: 160x88x128
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 80x44x128

        # input: 80x44x128
        self.e31 = nn.Conv2d(128, 256, kernel_size=3, padding='same') # output: 80x44x256
        self.e32 = nn.Conv2d(256, 256, kernel_size=3, padding='same') # output: 80x44x256
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 40x22x256

        # input: 40x22x256
        self.e41 = nn.Conv2d(256, 512, kernel_size=3, padding='same') # output: 40x22x512
        self.e42 = nn.Conv2d(512, 512, kernel_size=3, padding='same') # output: 40x22x512
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2) # output: 20x11x512

        # input: 20x11x512
        self.e51 = nn.Conv2d(512, 1024, kernel_size=3, padding='same') # output: 20x11x1024
        self.e52 = nn.Conv2d(1024, 1024, kernel_size=3, padding='same') # output: 20x11x1024


        # Decoder
        self.upconv1 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2) # output: 40x22x512
        self.d11 = nn.Conv2d(1024, 512, kernel_size=3, padding='same')
        self.d12 = nn.Conv2d(512, 512, kernel_size=3, padding='same')

        self.upconv2 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2) # output: 80x44x256
        self.d21 = nn.Conv2d(512, 256, kernel_size=3, padding='same')
        self.d22 = nn.Conv2d(256, 256, kernel_size=3, padding='same')

        self.upconv3 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2) # output: 160x88x128
        self.d31 = nn.Conv2d(256, 128, kernel_size=3, padding=1)
        self.d32 = nn.Conv2d(128, 128, kernel_size=3, padding='same')

        self.upconv4 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2) # output: 320x176x64
        self.d41 = nn.Conv2d(128, 64, kernel_size=3, padding='same')
        self.d42 = nn.Conv2d(64, 64, kernel_size=3, padding='same')

        # Output layer
        self.outconv = nn.Conv2d(64, n_output_channels, kernel_size=1) # output: 320x176x n_output_channels

    def forward(self, x, verbose=False):
        # Encoder

        if verbose:
            print('x.shape: {}'.format(x.shape))

        xe11 = relu(self.e11(x))
        xe12 = relu(self.e12(xe11))
        xp1 = self.pool1(xe12)
        if verbose:
            print('xp1.shape: {}'.format(xp1.shape))

        xe21 = relu(self.e21(xp1))
        xe22 = relu(self.e22(xe21))
        xp2 = self.pool2(xe22)
        if verbose:
            print('xp2.shape: {}'.format(xp2.shape))

        xe31 = relu(self.e31(xp2))
        xe32 = relu(self.e32(xe31))
        xp3 = self.pool3(xe32)
        if verbose:
            print('xp3.shape: {}'.format(xp3.shape))

        xe41 = relu(self.e41(xp3))
        xe42 = relu(self.e42(xe41))
        xp4 = self.pool4(xe42)

        xe51 = relu(self.e51(xp4))
        xe52 = relu(self.e52(xe51))
        
        # Decoder
        xu1 = self.upconv1(xe52)
        xu11 = torch.cat([xu1, xe42], dim=1)
        xd11 = relu(self.d11(xu11))
        xd12 = relu(self.d12(xd11))

        xu2 = self.upconv2(xd12)
        xu22 = torch.cat([xu2, xe32], dim=1)
        xd21 = relu(self.d21(xu22))
        xd22 = relu(self.d22(xd21))

        xu3 = self.upconv3(xd22)
        xu33 = torch.cat([xu3, xe22], dim=1)
        xd31 = relu(self.d31(xu33))
        xd32 = relu(self.d32(xd31))

        xu4 = self.upconv4(xd32)
        xu44 = torch.cat([xu4, xe12], dim=1)
        xd41 = relu(self.d41(xu44))
        xd42 = relu(self.d42(xd41))

        # Output layer
        out = self.outconv(xd42)

        return out
    

class FeatureLoss(nn.Module):
    def __init__(self, layer_wgts=[20, 70, 10]):
        super().__init__()

        self.m_feat = models.vgg16_bn(True).features.cuda().eval()
        requires_grad(self.m_feat, False)
        blocks = [
            i - 1
            for i, o in enumerate(children(self.m_feat))
            if isinstance(o, nn.MaxPool2d)
        ]
        layer_ids = blocks[2:5]
        self.loss_features = [self.m_feat[i] for i in layer_ids]
        self.hooks = hook_outputs(self.loss_features, detach=False)
        self.wgts = layer_wgts
        self.metric_names = ['pixel'] + [f'feat_{i}' for i in range(len(layer_ids))]
        self.base_loss = F.l1_loss

    def _make_features(self, x, clone=False):
        self.m_feat(x)
        return [(o.clone() if clone else o) for o in self.hooks.stored]

    def forward(self, input, target):
        out_feat = self._make_features(target, clone=True)
        in_feat = self._make_features(input)
        self.feat_losses = [self.base_loss(input, target)]
        self.feat_losses += [
            self.base_loss(f_in, f_out) * w
            for f_in, f_out, w in zip(in_feat, out_feat, self.wgts)
        ]

        self.metrics = dict(zip(self.metric_names, self.feat_losses))
        return sum(self.feat_losses)

    def __del__(self):
        self.hooks.remove()
 