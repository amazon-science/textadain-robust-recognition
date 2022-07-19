# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import random

import torch
import torch.nn as nn


def create_padain_class(num_windows=None):
    padain_class = PermuteAdaptiveInstanceNorm2d
    if num_windows is not None and num_windows > 0:
        padain_class = TextAdaIN
    return padain_class


def adaptive_instance_normalization(content_feat, style_feat, mode=None):
    assert (content_feat.size()[:2] == style_feat.size()[:2])
    size = content_feat.size()
    if mode is None:
        mode_func = calc_channel_mean_std
    else:
        mode_func = mode
    style_mean, style_std = mode_func(style_feat.detach())
    content_mean, content_std = mode_func(content_feat)
    content_std = content_std + 1e-4  # to avoid division by 0
    normalized_feat = (content_feat - content_mean.expand(
        size)) / content_std.expand(size)

    return normalized_feat * style_std.expand(size) + style_mean.expand(size)


def calc_channel_mean_std(feat):
    """
    Calculates the mean and standard deviation for each channel
    :param feat: features post convolutional layer
    :return: mean and std for each channel
    """
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    # STD over 1 dim results in NAN
    assert (W * H != 1), f"Cannot calculate std over W, H {size} (N,C,H,W), dimensions W={W}, H={H} cannot be 1"
    feat_std = feat.view(N, C, -1).std(dim=2).view(N, C, 1, 1)
    feat_mean = feat.view(N, C, -1).mean(dim=2).view(N, C, 1, 1)
    return feat_mean, feat_std


def calc_width_mean_std(feat):
    """
    Calculates the mean and standard deviation for each C and H
    :param feat: features post convolutional layer
    :return: mean and std for each C and H
    """
    size = feat.size()
    assert (len(size) == 4)
    N, C, H, W = size
    # STD over 1 dim results in NAN
    assert (W != 1), f"Cannot calculate std over W {size} (N,C,H,W), dimensions W={W} cannot be 1"
    feat_std = torch.sqrt(feat.var(dim=3).view(N, C, H, 1) + 1e-4)
    feat_mean = feat.mean(dim=3).view(N, C, H, 1)
    return feat_mean, feat_std


def get_adain_dim(dim):
    dim2func = {
        ("C",): calc_channel_mean_std,
        ("C", "H"): calc_width_mean_std,
    }
    dim = tuple(sorted(dim))
    assert dim in dim2func, f"Please insert one of the following : {list(dim2func.keys())}"
    return dim2func[dim]


class PermuteAdaptiveInstanceNorm2d(nn.Module):

    def __init__(self, p=0.01, dim=('C',), **kwargs):
        '''
        PermuteAdaptiveInstanceNorm2d
        "Permuted AdaIN: Reducing the Bias Towards Global Statistics in Image Classification"
        :param p: the probability of applying Permuted AdaIN
        :param dim:  a tuple of either 'C' for channel as in AdaIN or 'C,H' as in TextAdaIN
        '''
        super(PermuteAdaptiveInstanceNorm2d, self).__init__()
        self.p = p
        self.mode_func = get_adain_dim(dim)

    def forward(self, x):
        permute = random.random() < self.p
        if not self.training or not permute:
            return x

        target = x

        N, C, H, W = x.size()

        target = target[torch.randperm(N)]

        x = adaptive_instance_normalization(x, target, mode=self.mode_func)

        return x

    def extra_repr(self) -> str:
        return 'p={}, mode={}'.format(self.p, self.mode_func.__name__)


class TextAdaIN(PermuteAdaptiveInstanceNorm2d):
    def __init__(self, p=0.01, dim=('C', 'H'), num_windows=5, **kwargs):
        '''
        PermuteAdaptiveInstanceNorm2d running
        "Permuted AdaIN: Reducing the Bias Towards Global Statistics in Image Classification"
        :param p: the probability of applying Permuted AdaIN
        '''
        super(TextAdaIN, self).__init__(p=p, dim=dim, **kwargs)
        self.num_windows = num_windows

    def _pad_to_k(self, x):
        N, C, H, W = x.size()
        k = min(self.num_windows, W)
        remainder = W % k
        if remainder != 0:
            x = torch.nn.functional.pad(x, (0, k - remainder), 'constant', 0)
        return x

    def forward(self, x):
        if not self.training:
            return x

        N, C, H, W = x.size()
        k = min(self.num_windows, W)
        frame_total = W // k * k

        x_without_remainder = create_windows_from_tensor(x, k)
        x_without_remainder = super().forward(x_without_remainder)
        x_without_remainder = revert_windowed_tensor(x_without_remainder, k, W)
        x = torch.cat((x_without_remainder, x[:, :, :, frame_total:]), dim=3).contiguous()
        return x

    def extra_repr(self) -> str:
        return 'p={}, num_windows={}  mode={}'.format(self.p, self.num_windows, self.mode_func.__name__)


def revert_windowed_tensor(x_without_remainder, k, W):
    """
    Reverts a windowed tensor to its original shape, placing the windows back in their place
    :param x_without_remainder: N*k x C x H x frame_size (= original W // k)
    :param k: number of windows
    :param W: Original width
    :return: tensor N x C x H x W
    """
    N, C, H, _ = x_without_remainder.size()
    N = N // k
    frame_size = W // k
    x_without_remainder = x_without_remainder.transpose(1, 3)  # N*k x frame_size x H x C
    x_without_remainder = x_without_remainder.reshape(N, k * frame_size, H,
                                                      C)  # revert the windows back to their original position
    x_without_remainder = x_without_remainder.transpose(1, 3)  # N x C x H x k*frame_size
    return x_without_remainder


def create_windows_from_tensor(x, k):
    """
    Splits the tensor into k windows ignoring the remainder
    :param x: a tensor with dims NxCxHxW
    :param k: number of windows
    :return: tensor N*k x C x H x W
    """
    N, C, H, W = x.size()
    frame_size = W // k
    frame_total = W // k * k
    x_without_remainder = x[:, :, :, : frame_total]
    x_without_remainder = x_without_remainder.transpose(1, 3)  # NxWxHxC
    x_without_remainder = x_without_remainder.reshape(N * k, frame_size, H, C)  # N*num_windows x frame_size x H x C
    x_without_remainder = x_without_remainder.transpose(1, 3).contiguous()  # N*num_windows x C x H x frame_size
    return x_without_remainder
