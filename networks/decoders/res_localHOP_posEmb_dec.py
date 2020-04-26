import torch.nn as nn
from   networks.ops import GuidedCxtAttenEmbedding, SpectralNorm, LocalHOPBlock
from   networks.decoders.res_shortcut_dec import ResShortCut_D_Dec
from   utils import CONFIG

class ResLocalHOP_PosEmb_Dec(ResShortCut_D_Dec):

    def __init__(self, block, layers, enc_expansion=1, norm_layer=None, large_kernel=False):
        super(ResLocalHOP_PosEmb_Dec, self).__init__(block, layers, enc_expansion, norm_layer, large_kernel)
        self.gca = GuidedCxtAttenEmbedding(256 * block.expansion,
                                           64,
                                           use_trimap_embed=True,
                                           rate=CONFIG.model.arch.global_hop_downsample,
                                           scale=CONFIG.model.arch.hop_softmax_scale,
                                           learnable_scale=CONFIG.model.arch.learnable_scale)
        self.localgca1 = LocalHOPBlock(128 * block.expansion,
                                       64,
                                       ksize=CONFIG.model.arch.hop_ksize,
                                       use_pos_emb=True,
                                       scale=CONFIG.model.arch.hop_softmax_scale,
                                       learnable_scale=CONFIG.model.arch.learnable_scale)
        self.localgca2 = LocalHOPBlock(64 * block.expansion,
                                       32,
                                       ksize=CONFIG.model.arch.hop_ksize,
                                       use_pos_emb=True,
                                       scale=CONFIG.model.arch.hop_softmax_scale,
                                       learnable_scale=CONFIG.model.arch.learnable_scale)
        self.localgca3 = LocalHOPBlock(32 * block.expansion,
                                       16,
                                       ksize=CONFIG.model.arch.hop_ksize,
                                       use_pos_emb=True,
                                       scale=CONFIG.model.arch.hop_softmax_scale,
                                       learnable_scale=CONFIG.model.arch.learnable_scale)

    def forward(self, x, mid_fea):
        fea1, fea2, fea3, fea4, fea5 = mid_fea['shortcut']
        im1, im2, im3, im4 = mid_fea['image_fea']
        x = self.layer1(x) + fea5 # N x 256 x 32 x 32
        x, offset = self.gca(im4, x, mid_fea['trimap'])
        x = self.layer2(x) + fea4 # N x 128 x 64 x 64
        x = self.localgca1(im3, x)
        x = self.layer3(x) + fea3 # N x 64 x 128 x 128
        x = self.localgca2(im2, x)
        x = self.layer4(x) + fea2 # N x 32 x 256 x 256
        x = self.localgca3(im1, x)
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.leaky_relu(x) + fea1
        x = self.conv2(x)

        alpha = (self.tanh(x) + 1.0) / 2.0

        return alpha, {'offset': offset}

