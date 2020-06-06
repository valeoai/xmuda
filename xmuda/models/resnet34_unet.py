"""UNet based on ResNet34"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.resnet import resnet34


class UNetResNet34(nn.Module):
    def __init__(self, pretrained=True):
        super(UNetResNet34, self).__init__()

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        net = resnet34(pretrained)
        # Note that we do not downsample for conv1
        # self.conv1 = net.conv1
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=1, padding=3, bias=False)
        self.conv1.weight.data = net.conv1.weight.data
        self.bn1 = net.bn1
        self.relu = net.relu
        self.maxpool = net.maxpool
        self.layer1 = net.layer1
        self.layer2 = net.layer2
        self.layer3 = net.layer3
        self.layer4 = net.layer4

        # ----------------------------------------------------------------------------- #
        # Decoder
        # ----------------------------------------------------------------------------- #
        _, self.dec_t_conv_stage5 = self.dec_stage(self.layer4, num_concat=1)
        self.dec_conv_stage4, self.dec_t_conv_stage4 = self.dec_stage(self.layer3, num_concat=2)
        self.dec_conv_stage3, self.dec_t_conv_stage3 = self.dec_stage(self.layer2, num_concat=2)
        self.dec_conv_stage2, self.dec_t_conv_stage2 = self.dec_stage(self.layer1, num_concat=2)
        self.dec_conv_stage1 = nn.Conv2d(2 * 64, 64, kernel_size=3, padding=1)

        # dropout
        self.dropout = nn.Dropout(p=0.4)

    @staticmethod
    def dec_stage(enc_stage, num_concat):
        in_channels = enc_stage[0].conv1.in_channels
        out_channels = enc_stage[-1].conv2.out_channels
        conv = nn.Sequential(
            nn.Conv2d(num_concat * out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        t_conv = nn.Sequential(
            nn.ConvTranspose2d(out_channels, in_channels, kernel_size=2, stride=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        return conv, t_conv

    def forward(self, x):
        # pad input to be divisible by 16 = 2 ** 4
        h, w = x.shape[2], x.shape[3]
        min_size = 16
        pad_h = int((h + min_size - 1) / min_size) * min_size - h
        pad_w = int((w + min_size - 1) / min_size) * min_size - w
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [0, pad_w, 0, pad_h])

        # ----------------------------------------------------------------------------- #
        # Encoder
        # ----------------------------------------------------------------------------- #
        inter_features = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        inter_features.append(x)
        x = self.maxpool(x)  # downsample
        x = self.layer1(x)
        inter_features.append(x)
        x = self.layer2(x)  # downsample
        inter_features.append(x)
        x = self.layer3(x)  # downsample
        x = self.dropout(x)
        inter_features.append(x)
        x = self.layer4(x)  # downsample
        x = self.dropout(x)

        # ----------------------------------------------------------------------------- #
        # Decoder
        # ----------------------------------------------------------------------------- #
        # upsample
        x = self.dec_t_conv_stage5(x)
        x = torch.cat([inter_features[3], x], dim=1)
        x = self.dec_conv_stage4(x)

        # upsample
        x = self.dec_t_conv_stage4(x)
        x = torch.cat([inter_features[2], x], dim=1)
        x = self.dec_conv_stage3(x)

        # upsample
        x = self.dec_t_conv_stage3(x)
        x = torch.cat([inter_features[1], x], dim=1)
        x = self.dec_conv_stage2(x)

        # upsample
        x = self.dec_t_conv_stage2(x)
        x = torch.cat([inter_features[0], x], dim=1)
        x = self.dec_conv_stage1(x)

        # crop padding
        if pad_h > 0 or pad_w > 0:
            x = x[:, :, 0:h, 0:w]

        return x


def test():
    b, c, h, w = 2, 20, 120, 160
    image = torch.randn(b, 3, h, w).cuda()
    net = UNetResNet34(pretrained=True)
    net.cuda()
    feats = net(image)
    print('feats', feats.shape)


if __name__ == '__main__':
    test()
