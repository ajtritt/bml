import torch
from torch import nn
from torch.nn import functional as F  # noqa: N812

ACT = nn.ReLU6


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None) -> None:
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor, mode='trilinear')


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def resize_conv3x3(in_planes, out_planes, scale=1):
    """Upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes))


def resize_conv1x1(in_planes, out_planes, scale=1):
    """Upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes))


class EncoderBlock(nn.Module):
    """ResNet block, copied from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = ACT(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class EncoderBottleneck(nn.Module):
    """ResNet bottleneck, copied from
    https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L75."""

    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None) -> None:
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = conv3x3(width, width, stride)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = ACT(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        return self.relu(out)


class DecoderBlock(nn.Module):
    """ResNet block, but convs replaced with resize convs, and channel increase is in second conv, not first."""

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None) -> None:
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm3d(inplanes)
        self.relu = ACT(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm3d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        return self.relu(out)


class DecoderBottleneck(nn.Module):
    """ResNet bottleneck, but convs replaced with resize convs."""

    expansion = 4

    def __init__(self, inplanes, planes, scale=1, upsample=None) -> None:
        super().__init__()
        width = planes  # this needs to change if we want wide resnets
        self.conv1 = resize_conv1x1(inplanes, width)
        self.bn1 = nn.BatchNorm3d(width)
        self.conv2 = resize_conv3x3(width, width, scale)
        self.bn2 = nn.BatchNorm3d(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = ACT(inplace=True)
        self.upsample = upsample
        self.scale = scale

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        return self.relu(out)


class ResNetEncoder(nn.Module):
    def __init__(self, block, layers, first_conv=False, maxpool1=False, layer1_channels=64, in_channels=3) -> None:
        super().__init__()

        self.inplanes = 64
        self.first_conv = first_conv
        self.maxpool1 = maxpool1

        if self.first_conv:
            self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False)
        else:
            self.conv1 = nn.Conv3d(in_channels, self.inplanes, kernel_size=3, stride=1, padding=1, bias=False)

        self.bn1 = nn.BatchNorm3d(self.inplanes)
        self.relu = ACT(inplace=True)

        if self.maxpool1:
            self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)
        else:
            self.maxpool = nn.MaxPool3d(kernel_size=1, stride=1)

        self.layer1 = self._make_layer(block, layer1_channels, layers[0])
        self.layer2 = self._make_layer(block, layer1_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, layer1_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, layer1_channels * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))


    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ResNetDecoder(nn.Module):
    """Resnet in reverse order."""

    def __init__(self, block, layers, latent_dim, output_shape, first_conv=False, maxpool1=False, layer4_channels=64, out_channels=3) -> None:
        super().__init__()

        self.expansion = block.expansion
        self.inplanes = 64 * block.expansion
        self.latent_dim = latent_dim
        self.first_conv = first_conv
        self.maxpool1 = maxpool1
        self.output_shape = output_shape
        self.output_scale = tuple(i // 16 for i in self.output_shape)

        self.inplanes = latent_dim

        self.upscale_factor = 8

        self.linear = nn.Linear(self.latent_dim, self.latent_dim * torch.prod(torch.tensor(self.output_scale)))

        self.layer1 = self._make_layer(block, layer4_channels * 8, layers[0], scale=2)
        self.layer2 = self._make_layer(block, layer4_channels * 4, layers[1], scale=2)
        self.layer3 = self._make_layer(block, layer4_channels * 2, layers[2], scale=2)

        self.layer4 = self._make_layer(block, layer4_channels, layers[3], scale=2)
        self.upscale_factor *= 2

        if self.first_conv:
            self.upscale = Interpolate(size=self.output_shape)
            self.upscale_factor *= 2
        else:
            self.upscale = Interpolate(size=self.output_shape)

        # interpolate after linear layer using scale factor
        self.upscale1 = Interpolate(size=self.output_shape)

        self.conv1 = nn.Conv3d(layer4_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm3d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.linear(x)

        # NOTE: replaced this by Linear(in_channels, 514 * 4 * 4)
        # x = F.interpolate(x, scale_factor=4)

        x = x.view(x.size(0), self.latent_dim, self.output_scale[0], self.output_scale[1], self.output_scale[2])

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.upscale(x)

        return self.conv1(x)


def resnet10_encoder(first_conv, maxpool1, layer1_channels=64):
    return ResNetEncoder(EncoderBlock, [1, 1, 1, 1], first_conv, maxpool1, layer1_channels)


def resnet10_decoder(latent_dim, input_height, first_conv, maxpool1, layer4_channels=64):
    return ResNetDecoder(DecoderBlock, [1, 1, 1, 1], latent_dim, input_height, first_conv, maxpool1, layer4_channels)


def resnet18_encoder(first_conv, maxpool1, layer1_channels=64):
    return ResNetEncoder(EncoderBlock, [2, 2, 2, 2], first_conv, maxpool1, layer1_channels)


def resnet18_decoder(latent_dim, input_height, first_conv, maxpool1, layer4_channels=64):
    return ResNetDecoder(DecoderBlock, [2, 2, 2, 2], latent_dim, input_height, first_conv, maxpool1, layer4_channels)


def resnet50_encoder(first_conv, maxpool1, layer1_channels=64):
    return ResNetEncoder(EncoderBottleneck, [3, 4, 6, 3], first_conv, maxpool1, layer1_channels)


def resnet50_decoder(latent_dim, input_height, first_conv, maxpool1, layer4_channels=64):
    return ResNetDecoder(DecoderBottleneck, [3, 4, 6, 3], latent_dim, input_height, first_conv, maxpool1, layer4_channels)
