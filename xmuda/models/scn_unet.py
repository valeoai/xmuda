import torch
import torch.nn as nn

import sparseconvnet as scn

DIMENSION = 3


class UNetSCN(nn.Module):
    def __init__(self,
                 in_channels,
                 m=16,  # number of unet features (multiplied in each layer)
                 block_reps=1,  # depth
                 residual_blocks=False,  # ResNet style basic blocks
                 full_scale=4096,
                 num_planes=7
                 ):
        super(UNetSCN, self).__init__()

        self.in_channels = in_channels
        self.out_channels = m
        n_planes = [(n + 1) * m for n in range(num_planes)]

        self.sparseModel = scn.Sequential().add(
            scn.InputLayer(DIMENSION, full_scale, mode=4)).add(
            scn.SubmanifoldConvolution(DIMENSION, in_channels, m, 3, False)).add(
            scn.UNet(DIMENSION, block_reps, n_planes, residual_blocks)).add(
            scn.BatchNormReLU(m)).add(
            scn.OutputLayer(DIMENSION))

    def forward(self, x):
        x = self.sparseModel(x)
        return x


def test():
    b, n = 2, 100
    coords = torch.randint(4096, [b, n, DIMENSION])
    batch_idxs = torch.arange(b).reshape(b, 1, 1).repeat(1, n, 1)
    coords = torch.cat([coords, batch_idxs], 2).reshape(-1, DIMENSION + 1)

    in_channels = 3
    feats = torch.rand(b * n, in_channels)

    x = [coords, feats.cuda()]

    net = UNetSCN(in_channels).cuda()
    out_feats = net(x)

    print('out_feats', out_feats.shape)


if __name__ == '__main__':
    test()
