import logging
from .resnet_enc import ResNet_D, BasicBlock, Bottleneck
from .res_localHOP_posEmb_enc import ResLocalHOP_PosEmb


__all__ = ['resnet_localHOP_posEmb_encoder_29']


def _res_localHOP_posEmb(block, layers, **kwargs):
    model = ResLocalHOP_PosEmb(block, layers, **kwargs)
    return model


def resnet_localHOP_posEmb_encoder_29(**kwargs):
    return _res_localHOP_posEmb(BasicBlock, [3, 4, 4, 2], **kwargs)


if __name__ == "__main__":
    import torch
    logging.basicConfig(level=logging.DEBUG, format='[%(asctime)s] %(levelname)s: %(message)s',
                        datefmt='%m-%d %H:%M:%S')
    resnet_encoder = resnet_encoder_29()
    x = torch.randn(4,6,512,512)
    z = resnet_encoder(x)
    print(z[0].shape)
