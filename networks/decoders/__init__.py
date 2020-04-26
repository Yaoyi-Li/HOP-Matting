from .resnet_dec import ResNet_D_Dec, BasicBlock
from .res_localHOP_posEmb_dec import ResLocalHOP_PosEmb_Dec


__all__ = ['res_localHOP_posEmb_decoder_22']


def _res_localHOP_posEmb_dec(block, layers, **kwargs):
    model = ResLocalHOP_PosEmb_Dec(block, layers, **kwargs)
    return model


def res_localHOP_posEmb_decoder_22(**kwargs):
    return _res_localHOP_posEmb_dec(BasicBlock, [2, 3, 3, 2], **kwargs)
