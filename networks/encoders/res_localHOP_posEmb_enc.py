import logging
import torch
import torch.nn as nn
import torch.nn.functional as F

from   utils import CONFIG
from   networks.encoders.resnet_enc import ResNet_D
from   networks.ops import SpectralNorm, LocalHOPBlock



class ResLocalHOP_PosEmb(ResNet_D):

    def __init__(self, block, layers, norm_layer=None, late_downsample=False):
        super(ResLocalHOP_PosEmb, self).__init__(block, layers, norm_layer, late_downsample=late_downsample)
        first_inplane = 3 + CONFIG.model.trimap_channel
        self.shortcut_inplane = [first_inplane,
                                 self.midplanes,
                                 64 * block.expansion,
                                 128 * block.expansion,
                                 256 * block.expansion]
        self.shortcut_plane = [32, self.midplanes, 64, 128, 256]

        self.shortcut = nn.ModuleList()
        for stage, inplane in enumerate(self.shortcut_inplane):
            self.shortcut.append(self._make_shortcut(inplane, self.shortcut_plane[stage]))

        self.guidance_head1 = nn.Sequential( # N x 16 x 256 x 256
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(3, 16, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(16),
        )
        self.guidance_head2 = nn.Sequential( # N x 32 x 128 x 128
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(16, 32, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(32),
        )
        self.guidance_head3 = nn.Sequential( # N x 64 x 64 x 64
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(32, 64, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(64)
        )
        self.guidance_head4 = nn.Sequential( # N x 64 x 32 x 32
            nn.ReflectionPad2d(1),
            SpectralNorm(nn.Conv2d(64, 64, kernel_size=3, padding=0, stride=2, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(64)
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                if hasattr(m, "weight_bar"):
                    nn.init.xavier_uniform_(m.weight_bar)
                else:
                    nn.init.xavier_uniform_(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _make_shortcut(self, inplane, planes):
        return nn.Sequential(
            SpectralNorm(nn.Conv2d(inplane, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes),
            SpectralNorm(nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)),
            nn.ReLU(inplace=True),
            self._norm_layer(planes)
        )

    def forward(self, x):
        im_fea1 = self.guidance_head1(x[:, :3, ...])
        im_fea2 = self.guidance_head2(im_fea1)
        im_fea3 = self.guidance_head3(im_fea2)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.activation(out)
        out = self.conv2(out)
        out = self.bn2(out)
        x1 = self.activation(out) # N x 32 x 256 x 256
        # x1 = self.localgca1(im_fea1, x1)
        out = self.conv3(x1)
        out = self.bn3(out)
        out = self.activation(out) # N x 64 x 128 x 128

        im_fea4 = self.guidance_head4(im_fea3) # downsample origin image and extract features
        if CONFIG.model.trimap_channel == 3:
            trimap = F.interpolate(x[:, 3:6, ...], scale_factor=1/16, mode='nearest')
        else:
            raise NotImplementedError("Positional Embedding only support `trimap_channel == 3`")

        x2 = self.layer1(out) # N x 64 x 128 x 128
        # x2 = self.localgca2(im_fea2, x2)
        x3= self.layer2(x2) # N x 128 x 64 x 64
        # x3, offset = self.gca(im_fea3, x3, unknown) # contextual attention
        x4 = self.layer3(x3) # N x 256 x 32 x 32
        # x4, offset = self.gac(im_fea4)
        out = self.layer_bottleneck(x4) # N x 512 x 16 x 16

        fea1 = self.shortcut[0](x) # input image and trimap
        fea2 = self.shortcut[1](x1)
        fea3 = self.shortcut[2](x2)
        fea4 = self.shortcut[3](x3)
        fea5 = self.shortcut[4](x4)

        return out, {'shortcut': (fea1, fea2, fea3, fea4, fea5),
                     'image_fea': (im_fea1, im_fea2, im_fea3, im_fea4),
                     'trimap': trimap
                     # 'offset_1':offset
                     }

if __name__ == "__main__":
    from networks.encoders.resnet_enc import BasicBlock
    m = ResLocalHOP_PosEmb(BasicBlock, [3, 4, 4, 2])
    for m in m.modules():
        print(m)