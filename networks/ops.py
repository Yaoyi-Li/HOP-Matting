import torch
import warnings
from torch import nn
from torch.nn import Parameter
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np
from utils import CONFIG


def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


def gen_pos_emb(radius, emb_dim):
    inv_freq = 1 / (10000 ** (torch.arange(0.0, emb_dim, 2.0) / emb_dim))
    pos_seq = torch.arange(radius, -1, -1.0)
    sinusoid_inp = torch.ger(pos_seq, inv_freq)
    pos_emb = torch.cat([sinusoid_inp.sin(), sinusoid_inp.cos()], dim=-1)
    return pos_emb


class SpectralNorm(nn.Module):
    """
    Based on https://github.com/heykeetae/Self-Attention-GAN/blob/master/spectral.py
    and add _noupdate_u_v() for evaluation
    """

    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _noupdate_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        try:
            u = getattr(self.module, self.name + "_u")
            v = getattr(self.module, self.name + "_v")
            w = getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        # if torch.is_grad_enabled() and self.module.training:
        if self.module.training:
            self._update_u_v()
        else:
            self._noupdate_u_v()
        return self.module.forward(*args)


class LocalHOPBlock(nn.Module):
    def __init__(self, out_channels, guidance_channels, ksize, use_pos_emb=False, scale=1., learnable_scale=False):
        super(LocalHOPBlock, self).__init__()
        self.ksize = ksize
        self.use_pos_emb = use_pos_emb
        self.scale = scale
        if learnable_scale:
            self.scale = Parameter(torch.Tensor(1))
            nn.init.constant_(self.scale, 1)
        self.padding = nn.ReflectionPad2d(ksize // 2)

        self.guidance_conv = nn.Conv2d(in_channels=guidance_channels, out_channels=guidance_channels // 2,
                                       kernel_size=1, stride=1, padding=0)

        self.W = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)),
            nn.BatchNorm2d(out_channels, eps=CONFIG.model.arch.batchnorm_eps)
        )

        if self.use_pos_emb:
            self.positional_embedding = gen_pos_emb(ksize//2+1, guidance_channels // 2)
            self.pos_idx = list(range(ksize//2, -1, -1)) + list(range(1, ksize//2 + 1))
            self.positional_embedding = self.positional_embedding[self.pos_idx, :]
            self.pos_embed_linear = nn.Linear(guidance_channels // 2, guidance_channels // 2)
            nn.init.xavier_uniform_(self.pos_embed_linear.weight)
            nn.init.constant_(self.pos_embed_linear.bias, 0)

        nn.init.xavier_uniform_(self.guidance_conv.weight)
        nn.init.constant_(self.guidance_conv.bias, 0)
        nn.init.xavier_uniform_(self.W[0].module.weight_bar)
        nn.init.constant_(self.W[1].weight, 1e-1)
        nn.init.constant_(self.W[1].bias, 0)

    def forward(self, f, alpha):
        ksize = self.ksize
        f = self.guidance_conv(f)
        if self.use_pos_emb:
            pos_fea = self.pos_embed_linear(self.positional_embedding.to(f))

        if CONFIG.model.arch.hop_normalize:
            escape_NaN = Variable(torch.FloatTensor([1e-4])).to(alpha)
            f_normed = f / torch.max(torch.norm(f, dim=1, keepdim=True), escape_NaN)
        else:
            f_normed = f

        raw_int_fs = list(f.size())
        raw_int_alpha = list(alpha.size())

        r = self.padding(f_normed)
        s = torch.zeros(ksize * ksize, raw_int_fs[0], 1, raw_int_fs[2], raw_int_fs[3]).to(f)
        for i in range(ksize):
            for j in range(ksize):
                if i == ksize // 2 and j == ksize // 2:
                    s[i * ksize + j, ...] = -1e4
                elif self.use_pos_emb:
                    s[i * ksize + j, ...] = (f_normed * (r[..., i:raw_int_fs[2] + i, j:j + raw_int_fs[3]] +
                                                         (pos_fea[i] + pos_fea[j]).view(1,-1,1,1))
                                             ).sum(dim=1, keepdim=True)
                else:
                    s[i * ksize + j, ...] = (f_normed * r[..., i:raw_int_fs[2] + i, j:j + raw_int_fs[3]]).sum(dim=1, keepdim=True)
        s = s * self.scale
        s = F.softmax(s, dim=0)

        alp = self.padding(alpha)
        y = torch.zeros(*raw_int_alpha).to(alp)
        for i in range(ksize):
            for j in range(ksize):
                if i == ksize // 2 and j == ksize // 2:
                    continue
                else:
                    y = y.add(alp[..., i:raw_int_fs[2] + i, j:j + raw_int_fs[3]] * s[i * ksize + j, ...])
        alpha = self.W(y) + alpha
        return alpha


class GuidedCxtAttenEmbedding(nn.Module):
    # Modification of class GuidedCxtAtten()
    def __init__(self,
                 out_channels,
                 guidance_channels,
                 rate=2,
                 embed_radius=7,
                 use_trimap_embed=False,
                 scale=1,
                 learnable_scale=False):
        super(GuidedCxtAttenEmbedding, self).__init__()
        self.rate = int(rate)
        self.use_trimap_embed = use_trimap_embed
        self.embed_radius = embed_radius
        self.scale = scale
        if learnable_scale:
            self.scale = Parameter(torch.Tensor(1))
            nn.init.constant_(self.scale, 1)

        self.padding = nn.ReflectionPad2d(1)
        self.up_sample = nn.Upsample(scale_factor=self.rate, mode='nearest')
        self.guidance_conv = nn.Conv2d(in_channels=guidance_channels, out_channels=guidance_channels // 2,
                                       kernel_size=1, stride=1, padding=0)
        self.pos_embed_linear = nn.Linear(guidance_channels // 2, guidance_channels // 2)
        self.trimap_embed_conv = nn.Conv2d(in_channels=3, out_channels=guidance_channels // 2,
                                           kernel_size=1, stride=1, padding=0)

        self.positional_embedding = gen_pos_emb(radius=self.embed_radius, emb_dim=guidance_channels // 2)

        self.W = nn.Sequential(
            SpectralNorm(nn.Conv2d(in_channels=out_channels, out_channels=out_channels,
                                   kernel_size=1, stride=1, padding=0, bias=False)),
            nn.BatchNorm2d(out_channels, eps=CONFIG.model.arch.batchnorm_eps)
        )

        nn.init.xavier_uniform_(self.guidance_conv.weight)
        nn.init.xavier_uniform_(self.pos_embed_linear.weight)
        nn.init.constant_(self.guidance_conv.bias, 0)
        nn.init.constant_(self.pos_embed_linear.bias, 0)
        nn.init.constant_(self.W[1].weight, 1e-1)
        nn.init.constant_(self.W[1].bias, 0)
        nn.init.constant_(self.trimap_embed_conv.weight, 0)
        nn.init.constant_(self.trimap_embed_conv.bias, 0)

    def forward(self, f, alpha, trimap, ksize=3, stride=1,
                fuse_k=3, softmax_scale=1., training=True):

        """ Contextual attention layer implementation.
        Contextual attention is first introduced in publication:
            Generative Image Inpainting with Contextual Attention, Yu et al.
        Args:
            f: Input feature to match (foreground).
            mask: Input mask for b, indicating patches not available.
            ksize: Kernel size for contextual attention.
            stride: Stride for extracting patches from b.
            rate: Dilation for matching.
            softmax_scale: Scaled softmax for attention.
            training: Indicating if current graph is training or inference.
        Returns:
            tf.Tensor: output
        """
        f = self.guidance_conv(f)
        # get shapes
        raw_int_fs = list(f.size())  # N x 64 x 64 x 64
        raw_int_alpha = list(alpha.size())  # N x 128 x 64 x 64

        # extract patches from background with stride and rate
        kernel = 2 * self.rate
        # kernel = 4 if self.rate <= 2 else 2*self.rate
        alpha_w = self.extract_patches(alpha, kernel=kernel, stride=self.rate)
        alpha_w = alpha_w.permute(0, 2, 3, 4, 5, 1)
        alpha_w = alpha_w.contiguous().view(raw_int_alpha[0], raw_int_alpha[2] // self.rate,
                                            raw_int_alpha[3] // self.rate, -1)
        alpha_w = alpha_w.contiguous().view(raw_int_alpha[0], -1, kernel, kernel, raw_int_alpha[1])
        alpha_w = alpha_w.permute(0, 1, 4, 2, 3)

        f = F.interpolate(f, scale_factor=1 / self.rate, mode='nearest')

        fs = f.size()  # B x 64 x 32 x 32
        f_groups = torch.split(f, 1, dim=0)  # Split tensors by batch dimension; tuple is returned

        # from b(B*H*W*C) to w(b*k*k*c*h*w)
        int_fs = list(fs)

        trimap = F.interpolate(trimap, scale_factor=1 / self.rate, mode='nearest')
        if self.use_trimap_embed:
            trimap_embed = self.trimap_embed_conv(trimap)
            f = f + trimap_embed
        w = self.extract_patches(f)
        w = w.permute(0, 2, 3, 4, 5, 1)
        w = w.contiguous().view(raw_int_fs[0], raw_int_fs[2] // self.rate, raw_int_fs[3] // self.rate, -1)
        w = w.contiguous().view(raw_int_fs[0], -1, ksize, ksize, raw_int_fs[1])
        w = w.permute(0, 1, 4, 2, 3)

        # handcraft softmax scale
        unknown = trimap[:, 1:2, ...]
        if not self.use_trimap_embed:
            unknown = unknown.clone()
            assert unknown.size(2) == f.size(2), "mask should have same size as f at dim 2,3"
            unknown_mean = unknown.mean(dim=[2, 3])
            known_mean = 1 - unknown_mean
            unknown_scale = torch.clamp(torch.sqrt(unknown_mean / known_mean), 0.1, 10).to(alpha)
            known_scale = torch.clamp(torch.sqrt(known_mean / unknown_mean), 0.1, 10).to(alpha)
            softmax_scale = torch.cat([unknown_scale, known_scale], dim=1)
        else:
            softmax_scale = torch.FloatTensor([softmax_scale, softmax_scale]).view(1, 2).repeat(fs[0], 1).to(alpha)

        # process mask
        m = self.extract_patches(unknown)
        m = m.permute(0, 2, 3, 4, 5, 1)
        m = m.contiguous().view(raw_int_fs[0], raw_int_fs[2] // self.rate, raw_int_fs[3] // self.rate, -1)
        m = m.contiguous().view(raw_int_fs[0], -1, ksize, ksize)
        m = self.reduce_mean(m)  # smoothing, maybe
        # modified for fp16 training
        mm = m.gt(0.).to(alpha)  # (N, 32*32, 1, 1)

        # the correlation with itself should be 0
        self_mask = F.one_hot(torch.arange(fs[2] * fs[3]).view(fs[2], fs[3]).contiguous().to(alpha.device).long(),
                              num_classes=int_fs[2] * int_fs[3])
        # modified for fp16 training
        self_mask = self_mask.permute(2, 0, 1).view(1, fs[2] * fs[3], fs[2], fs[3]).to(alpha) * (-1e4)

        w_groups = torch.split(w, 1, dim=0)  # Split tensors by batch dimension; tuple is returned
        alpha_w_groups = torch.split(alpha_w, 1, dim=0)  # Split tensors by batch dimension; tuple is returned
        mm_groups = torch.split(mm, 1, dim=0)
        scale_group = torch.split(softmax_scale, 1, dim=0)
        y = []
        offsets = []
        k = fuse_k
        # scale = softmax_scale
        fuse_weight = Variable(torch.eye(k).view(1, 1, k, k)).to(alpha)  # 1 x 1 x K x K
        y_test = []
        for xi, wi, alpha_wi, mmi, scale in zip(f_groups, w_groups, alpha_w_groups, mm_groups, scale_group):
            '''
            O => output channel as a conv filter
            I => input channel as a conv filter
            xi : separated tensor along batch dimension of front; (B=1, C=128, H=32, W=32)
            wi : separated patch tensor along batch dimension of back; (B=1, O=32*32, I=128, KH=3, KW=3)
            alpha_wi : separated tensor along batch dimension of back; (B=1, I=32*32, O=128, KH=4, KW=4)
            '''
            # conv for compare
            wi = wi[0]
            if CONFIG.model.arch.hop_normalize:
                escape_NaN = Variable(torch.FloatTensor([1e-4])).to(alpha)
                wi_normed = wi / torch.max(self.l2_norm(wi), escape_NaN)
            else:
                wi_normed = wi
            # positional embedding attention
            pos_score = self.gen_pos_score(xi, (fs[2], fs[3]))
            # content attention
            xi = F.pad(xi, (1, 1, 1, 1), mode='reflect')
            yi = F.conv2d(xi, wi_normed, stride=1, padding=0)  # yi => (B=1, C=32*32, H=32, W=32)
            y_test.append(yi)
            # conv implementation for fuse scores to encourage large patches

            yi = yi.permute(0, 2, 3, 1)
            yi = yi.contiguous().view(1, fs[2], fs[3], fs[2] * fs[3])
            yi = yi.permute(0, 3, 1, 2)  # (B=1, C=32*32, H=32, W=32)

            yi = yi * (scale[0, 0] * mmi.gt(0.).to(scale) + scale[0, 1] * mmi.le(0.).to(
                scale))  # mmi => (1, 32*32, 1, 1)
            # mask itself, self-mask only applied to unknown area
            yi = yi * self.scale + self_mask * mmi  # self_mask: (1, 32*32, 32, 32)
            yi = yi + pos_score.reshape(1, -1, fs[2], fs[3])
            # for small input inference
            yi = F.softmax(yi, dim=1)
            # to avoid nan in BN
            if self.training:
                if torch.any(torch.isnan(yi)):
                    warnings.warn("Nan found in attention score")
                    yi = yi.detach()
                    yi.requires_grad = False
                    yi[torch.isnan(yi)] = 0
                else:
                    if not yi.requires_grad:
                        raise RuntimeError("Attention Score requires_grad=False")

            _, offset = torch.max(yi, dim=1)  # argmax; index
            offset = torch.stack([offset // fs[3], offset % fs[3]], dim=1)

            wi_center = alpha_wi[0]

            if self.rate == 1:
                left = (kernel) // 2
                right = (kernel - 1) // 2
                yi = F.pad(yi, (left, right, left, right), mode='reflect')
                wi_center = wi_center.permute(1, 0, 2, 3)
                yi = F.conv2d(yi, wi_center, padding=0) / 4.  # (B=1, C=128, H=64, W=64)
            else:
                yi = F.conv_transpose2d(yi, wi_center, stride=self.rate, padding=1) / 4.  # (B=1, C=128, H=64, W=64)
            y.append(yi)
            offsets.append(offset)

        y = torch.cat(y, dim=0)  # back to the mini-batch
        y.contiguous().view(raw_int_alpha)
        # wi_patched = y
        offsets = torch.cat(offsets, dim=0)
        offsets = offsets.view([int_fs[0]] + [2] + int_fs[2:])
        offsets = offsets - torch.Tensor([fs[2] // 2, fs[3] // 2]).view(1, 2, 1, 1).to(alpha.device).long()
        y = self.W(y) + alpha

        return y, (offsets, softmax_scale)

    @staticmethod
    def extract_patches(x, kernel=3, stride=1):
        # x = self.padding(x)
        left = int(kernel - stride + 1) // 2
        right = int(kernel - stride) // 2
        x = F.pad(x, (left, right, left, right), mode='reflect')
        all_patches = x.unfold(2, kernel, stride).unfold(3, kernel, stride)

        return all_patches

    @staticmethod
    def reduce_mean(x):
        for i in range(4):
            if i <= 1: continue
            x = torch.mean(x, dim=i, keepdim=True)
        return x

    @staticmethod
    def l2_norm(x):
        def reduce_sum(x):
            for i in range(4):
                if i == 0: continue
                x = torch.sum(x, dim=i, keepdim=True)
            return x

        x = x ** 2
        x = reduce_sum(x)
        return torch.sqrt(x)

    def gen_pos_score(self, q_fea, fea_size):
        h_size, w_size = fea_size
        # pos_fea_idx_h = list(range(1, self.embed_radius+1)) \
        #                 + [self.embed_radius] * (h_size - self.embed_radius * 2) \
        #                 + list(range(self.embed_radius, -1, -1))
        # pos_fea_idx_w = list(range(self.embed_radius + 1)) \
        #                 + [self.embed_radius] * (w_size - self.embed_radius * 2) \
        #                 + list(range(self.embed_radius, 0, -1))
        pos_fea_idx_h = [self.embed_radius] * (h_size - self.embed_radius * 2) \
                        + list(range(self.embed_radius, -1, -1)) + list(range(1, self.embed_radius + 1))
        pos_fea_idx_w = [self.embed_radius] * (w_size - self.embed_radius * 2) \
                        + list(range(self.embed_radius, -1, -1)) + list(range(1, self.embed_radius + 1))
        pos_fea = self.pos_embed_linear(self.positional_embedding.to(q_fea))
        pos_score = F.conv2d(q_fea, pos_fea[:, :, None, None], stride=1, padding=0)
        pos_score_h = pos_score[:, pos_fea_idx_h, ...]
        pos_score_w = pos_score[:, pos_fea_idx_w, ...]
        pos_score_h = pos_score_h.view(1, h_size + 1, 1, h_size, w_size).repeat(1, 1, w_size, 1,
                                                                                1)  # (N,k_h,k_w,q_h,q_w)
        pos_score_w = pos_score_w.view(1, 1, w_size + 1, h_size, w_size).repeat(1, h_size, 1, 1, 1)

        # reshape and treat as 1-D positional embedding
        pos_score_h = pos_score_h.permute(0, 2, 4, 3, 1).contiguous()
        # out_score_h: the position score of furthest key.
        out_score_h = pos_score_h[..., -1:]
        pos_score_h = pos_score_h.view(1, w_size, w_size, -1)
        # add zeros to the head of first line to make a circulant matrix. Similar to the Transformer XL
        pos_score_h = torch.cat([torch.zeros(1, w_size, w_size, self.embed_radius).to(pos_score_h), pos_score_h],
                                dim=-1)[..., :-self.embed_radius]
        pos_score_h = pos_score_h.view(1, w_size, w_size, h_size + 1, h_size)[..., 1:, :]
        # mask to merge pos_score_h and out_score_h
        mask = (torch.triu(torch.ones((h_size, h_size)), diagonal=1 + self.embed_radius).byte()
                + torch.tril(torch.ones((h_size, h_size)), diagonal=-1 - self.embed_radius).byte()).to(q_fea)
        pos_score_h = pos_score_h * (1 - mask).view(1, 1, 1, h_size, h_size) \
                      + out_score_h * mask.view(1, 1, 1, h_size, h_size)
        pos_score_h = pos_score_h.permute(0, 4, 1, 3, 2)

        # for width
        pos_score_w = pos_score_w.permute(0, 1, 3, 4, 2).contiguous()
        # out_score_h: the position score of furthest key.
        out_score_w = pos_score_w[..., -1:]
        pos_score_w = pos_score_w.view(1, h_size, h_size, -1)
        # add zeros to the head of first line to make a circulant matrix. Similar to the Transformer XL
        pos_score_w = torch.cat([torch.zeros(1, h_size, h_size, self.embed_radius).to(pos_score_w), pos_score_w],
                                dim=-1)[..., :-self.embed_radius]
        pos_score_w = pos_score_w.view(1, h_size, h_size, w_size + 1, w_size)[..., 1:, :]
        # mask to merge pos_score_h and out_score_h
        mask = (torch.triu(torch.ones((w_size, w_size)), diagonal=1 + self.embed_radius).byte()
                + torch.tril(torch.ones((w_size, w_size)), diagonal=-1 - self.embed_radius).byte()).to(q_fea)
        pos_score_w = pos_score_w * (1 - mask).view(1, 1, 1, w_size, w_size) \
                      + out_score_w * mask.view(1, 1, 1, w_size, w_size)
        pos_score_w = pos_score_w.permute(0, 1, 4, 2, 3)

        return pos_score_h + pos_score_w
