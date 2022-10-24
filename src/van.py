# Copyright 2022 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
import collections.abc
import math
import os
from functools import partial
from itertools import repeat

import numpy as np
from mindspore import dtype as mstype
from mindspore import ops, Parameter, Tensor, nn
from mindspore.common import initializer as weight_init
from mindspore.common.initializer import Initializer as MeInitializer

from src.layers.drop_path import DropPath2D
from src.layers.identity import Identity

if os.getenv("DEVICE_TARGET", "GPU") == "GPU" or int(os.getenv("DEVICE_NUM")) == 1:
    BatchNorm2d = nn.BatchNorm2d
elif os.getenv("DEVICE_TARGET") == "Ascend" and int(os.getenv("DEVICE_NUM")) > 1:
    BatchNorm2d = nn.SyncBatchNorm
else:
    raise ValueError(f"Model doesn't support devide_num = {int(os.getenv('DEVICE_NUM'))} "
                     f"and device_target = {os.getenv('DEVICE_TARGET')}")


def _ntuple(n):
    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def assignment(arr, num):
    """Assign the value of `num` to `arr`."""
    if arr.shape == ():
        arr = arr.reshape((1))
        arr[:] = num
        arr = arr.reshape(())
    else:
        if isinstance(num, np.ndarray):
            arr[:] = num[:]
        else:
            arr[:] = num
    return arr


class RandomNormal(MeInitializer):
    def __init__(self, std=0.001):
        super(RandomNormal, self).__init__()
        self.std = std

    def _initialize(self, arr):
        std = self.std
        tmp = np.random.normal(0, std, arr.shape)
        assignment(arr, tmp)


class Mlp(nn.Cell):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Conv2d(in_channels=in_features, out_channels=hidden_features, kernel_size=1, pad_mode='pad',
                             has_bias=True)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Conv2d(in_channels=hidden_features, out_channels=out_features, kernel_size=1, pad_mode='pad',
                             has_bias=True)
        self.drop = nn.Dropout(keep_prob=1.0 - drop)

    def construct(self, x):
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class LKA(nn.Cell):
    def __init__(self, dim):
        super().__init__()
        self.conv0 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=5, pad_mode='pad', padding=2, group=dim,
                               has_bias=True)
        self.conv_spatial = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=7, stride=1, pad_mode='pad',
                                      padding=9, dilation=3, group=dim, has_bias=True)
        self.conv1 = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=1, pad_mode='pad', has_bias=True)

    def construct(self, x):
        u = x
        attn = self.conv0(x)
        attn = self.conv_spatial(attn)
        attn = self.conv1(attn)

        return u * attn


class Attention(nn.Cell):
    def __init__(self, d_model):
        super().__init__()

        self.proj_1 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, pad_mode='pad', has_bias=True)
        self.activation = nn.GELU()
        self.spatial_gating_unit = LKA(d_model)
        self.proj_2 = nn.Conv2d(in_channels=d_model, out_channels=d_model, kernel_size=1, pad_mode='pad', has_bias=True)

    def construct(self, x):
        shorcut = x
        x = self.proj_1(x)
        x = self.activation(x)
        x = self.spatial_gating_unit(x)
        x = self.proj_2(x)
        x = x + shorcut
        return x


class Block(nn.Cell):
    def __init__(self, dim, mlp_ratio=4., drop=0., drop_path=0., act_layer=nn.GELU):
        super().__init__()
        self.norm1 = BatchNorm2d(num_features=dim, momentum=0.9)
        self.attn = Attention(dim)
        self.drop_path = DropPath2D(drop_path) if drop_path > 0. else Identity()

        self.norm2 = BatchNorm2d(num_features=dim, momentum=0.9)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        layer_scale_init_value = 1e-2
        self.layer_scale_1 = Parameter(
            Tensor(layer_scale_init_value * np.ones((dim, 1, 1)), mstype.float32), requires_grad=True,
            name="layer_scale_1")
        self.layer_scale_2 = Parameter(
            Tensor(layer_scale_init_value * np.ones((dim, 1, 1)), mstype.float32), requires_grad=True,
            name="layer_scale_2")
        self.expand_dims = ops.ExpandDims()

    def construct(self, x):
        x = x + self.drop_path(self.layer_scale_1 * self.attn(self.norm1(x)))
        x = x + self.drop_path(self.layer_scale_2 * self.mlp(self.norm2(x)))
        return x


class OverlapPatchEmbed(nn.Cell):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.proj = nn.Conv2d(in_channels=in_chans, out_channels=embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[0] // 2, patch_size[1] // 2, patch_size[1] // 2),
                              pad_mode='pad', has_bias=True)
        self.norm = BatchNorm2d(num_features=embed_dim, momentum=0.9)

    def construct(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = self.norm(x)
        return x, H, W


class VAN(nn.Cell):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 mlp_ratios=[4, 4, 4, 4], drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], num_stages=4, flag=False):
        super().__init__()
        if flag == False:
            self.num_classes = num_classes
        self.depths = depths
        self.num_stages = num_stages

        dpr = [x for x in np.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        blocks = []
        patch_embeds = []
        norms = []
        for i in range(num_stages):
            patch_embed = OverlapPatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                            patch_size=7 if i == 0 else 3,
                                            stride=4 if i == 0 else 2,
                                            in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                            embed_dim=embed_dims[i])

            block = nn.CellList([Block(
                dim=embed_dims[i], mlp_ratio=mlp_ratios[i], drop=drop_rate, drop_path=dpr[cur + j])
                for j in range(depths[i])])
            norm = norm_layer((embed_dims[i],))
            cur += depths[i]
            patch_embeds.append(patch_embed)
            blocks.append(block)
            norms.append(norm)
        self.blocks = nn.CellList(blocks)
        self.patch_embeds = nn.CellList(patch_embeds)
        self.norms = nn.CellList(norms)

        # classification head
        self.head = nn.Dense(in_channels=embed_dims[3], out_channels=num_classes) if num_classes > 0 else Identity()
        self.init_weights()

    def init_weights(self):

        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Dense):
                cell.weight.set_data(weight_init.initializer(weight_init.TruncatedNormal(sigma=0.02),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if isinstance(cell, nn.Dense) and cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))
            elif isinstance(cell, nn.LayerNorm):
                cell.gamma.set_data(weight_init.initializer(weight_init.One(),
                                                            cell.gamma.shape,
                                                            cell.gamma.dtype))
                cell.beta.set_data(weight_init.initializer(weight_init.Zero(),
                                                           cell.beta.shape,
                                                           cell.beta.dtype))
            elif isinstance(cell, nn.Conv2d):
                fan_out = cell.kernel_size[0] * cell.kernel_size[1] * cell.out_channels
                fan_out //= cell.group
                cell.weight.set_data(weight_init.initializer(weight_init.Normal(sigma=math.sqrt(2.0 / fan_out)),
                                                             cell.weight.shape,
                                                             cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(weight_init.initializer(weight_init.Zero(),
                                                               cell.bias.shape,
                                                               cell.bias.dtype))

    def forward_features(self, x):
        B = x.shape[0]
        for i in range(self.num_stages):
            patch_embed = self.patch_embeds[i]
            block = self.blocks[i]
            norm = self.norms[i]
            x, H, W = patch_embed(x)
            for blk in block:
                x = blk(x)
            x = x.reshape(B, x.shape[1], -1).transpose((0, 2, 1))
            x = norm(x)
            if i != self.num_stages - 1:
                x = ops.Transpose()(ops.Reshape()(x, (B, H, W, -1,)), (0, 3, 1, 2,))

        return ops.ReduceMean()(x, 1)

    def construct(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


class DWConv(nn.Cell):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(in_channels=dim, out_channels=dim, kernel_size=3, stride=1, pad_mode='pad', padding=1,
                                group=dim, has_bias=True)

    def construct(self, x):
        x = self.dwconv(x)
        return x


def van_tiny(**kwargs):
    model = VAN(
        embed_dims=[32, 64, 160, 256], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 3, 5, 2],
        **kwargs)
    return model


def van_small(**kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[2, 2, 4, 2],
        **kwargs)

    return model


# 82.8%
def van_base(**kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 3, 12, 3],
        **kwargs)
    return model


def van_large(**kwargs):
    model = VAN(
        embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4],
        norm_layer=partial(nn.LayerNorm, epsilon=1e-6), depths=[3, 5, 27, 3],
        **kwargs)

    return model


if __name__ == "__main__":
    from mindspore import context

    context.set_context(mode=context.PYNATIVE_MODE, device_target="CPU")
    data = Tensor(np.ones([2, 3, 224, 224]), dtype=mstype.float32)
    model = van_base()
    out = model(data)
    print(out.shape)
    params = 0.
    end = []
    for name, param in model.parameters_and_names():
        params += np.prod(param.shape)
        print(name)
    print(params, 26604328)
    print(set(end))
