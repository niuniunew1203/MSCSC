import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, matthews_corrcoef, roc_auc_score, average_precision_score
# import load_data
import sys
from load_data import extract_pept_feat,extract_prot_feat,load_pairs,load_dataset,load_sequences, combinefeature, load_dataset_hand, AAC, DPC, AAE, BE, AAI, BLOSUM62, PC6_embedding, DDE, ZS, AAT, AAP, PAAC, CTD, KMER, CKSAAGP, PairDataset, PairDataset_5, PairDataset_6_
import h5py
from einops import rearrange
from mamba_ssm import Mamba2
import torch.nn.functional as F
from transformers import get_scheduler
from functools import partial
from typing import Dict, Type, Any
from inspect import signature
from torch.cuda.amp import autocast
# register activation function here
# register activation function here
import random
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from tcn.tcn import TCN

seed = 991
# seeds = [42,43,56,64,128,256,512,1024,2048,2024,2000,6666,8888,1314,520]
# for seed in seeds:
# random.seed(seed)
torch.manual_seed(seed)
# torch.cuda.manual_seed_all(seed)
# torch.backends.cudnn.benchmark = False
seed2s = [42,43,56,64,128,256,512,1024,2048,2024,2000,6666,8888,1314,520]

# dropouts = [0.15,0.2,0.25,0.3,0.35,0.4,0.45,0,5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9]
# lrs = [0.0008,0.0005,0.0004,0.0003,0.0006,0.0007,0.009]
dropouts = [0.1]
# batch_sizes =[512,480,448,416,384,352,320,288,256,128,64,32]
# batch_sizes = list(range(127, 219, +1))
batch_sizes = [256]
#
# for seed2 in seed2s:
np.random.seed(seed2s[2])
random.seed(seed2s[1])
torch.cuda.manual_seed(seed2s[1])
torch.cuda.manual_seed_all(seed2s[1])
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = False

REGISTERED_ACT_DICT: Dict[str, Type] = {
    "relu": nn.ReLU,
    "relu6": nn.ReLU6,
    "hswish": nn.Hardswish,
    "silu": nn.SiLU,
    "gelu": partial(nn.GELU, approximate="tanh"),
}

def get_same_padding(kernel_size: int or tuple[int, ...]) -> int or tuple[int, ...]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, "kernel size should be odd number"
        return kernel_size // 2

def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)
def build_kwargs_from_config(config: dict, target_func: callable) -> Dict[str, Any]:
    valid_keys = list(signature(target_func).parameters)
    kwargs = {}
    for key in config:
        if key in valid_keys:
            kwargs[key] = config[key]
    return kwargs

def build_act(name: str, **kwargs) -> nn.Module or None:
    if name in REGISTERED_ACT_DICT:
        act_cls = REGISTERED_ACT_DICT[name]
        args = build_kwargs_from_config(kwargs, act_cls)
        return act_cls(**args)
    else:
        return None

def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

def val2list(x: list or tuple or any, repeat_time=1) -> list:
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]


def build_norm(name="bn2d", num_features=None, **kwargs) -> nn.Module or None:
        return None
class ConvLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size=3,
        stride=1,
        dilation=1,
        groups=1,
        use_bias=False,
        dropout=0,
        norm="bn2d",
        act_func="relu",
    ):
        super(ConvLayer, self).__init__()

        padding = get_same_padding(kernel_size)
        padding *= dilation

        self.dropout = nn.Dropout2d(dropout, inplace=False) if dropout > 0 else None
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(kernel_size, kernel_size),
            stride=(stride, stride),
            padding=padding,
            dilation=(dilation, dilation),
            groups=groups,
            bias=use_bias,
        )
        self.norm = build_norm(norm, num_features=out_channels)
        self.act = build_act(act_func)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.dropout is not None:
            x = self.dropout(x)
        x = self.conv(x)
        if self.norm:
            x = self.norm(x)
        if self.act:
            x = self.act(x)
        return x

class LiteMLA(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads= 8,
        heads_ratio: float = 1.0,
        dim=32,
        use_bias=False,
        norm=(None, "bn2d"),
        act_func=(None, None),
        kernel_func="relu",
        scales= (5,),
        eps=1.0e-15,
    ):
        super(LiteMLA, self).__init__()
        self.eps = eps
        heads = heads or int(in_channels // dim * heads_ratio)

        total_dim = heads * dim

        use_bias = val2tuple(use_bias, 2)
        norm = val2tuple(norm, 2)
        act_func = val2tuple(act_func, 2)

        self.dim = dim
        self.qkv = ConvLayer(
            in_channels,
            3 * total_dim,
            1,
            use_bias=use_bias[0],
            norm=norm[0],
            act_func=act_func[0],
        )
        self.aggreg = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Conv2d(
                        3 * total_dim,
                        3 * total_dim,
                        scale,
                        padding=get_same_padding(scale),
                        groups=3 * total_dim,
                        bias=use_bias[0],
                    ),
                    nn.Conv2d(3 * total_dim, 3 * total_dim, 1, groups=3 * heads, bias=use_bias[0]),
                )
                for scale in scales
            ]
        )
        self.kernel_func = build_act(kernel_func, inplace=False)

        self.proj = ConvLayer(
            total_dim * (1 + len(scales)),
            out_channels,
            1,
            use_bias=use_bias[1],
            norm=norm[1],
            act_func=act_func[1],
        )

    @autocast(enabled=False)
    def relu_linear_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        if qkv.dtype == torch.float16:
            qkv = qkv.float()

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        # lightweight linear attention
        q = self.kernel_func(q)
        k = self.kernel_func(k)

        # linear matmul
        trans_k = k.transpose(-1, -2)

        v = F.pad(v, (0, 0, 0, 1), mode="constant", value=1)
        vk = torch.matmul(v, trans_k)
        out = torch.matmul(vk, q)
        if out.dtype == torch.bfloat16:
            out = out.float()
        out = out[:, :, :-1] / (out[:, :, -1:] + self.eps)

        out = torch.reshape(out, (B, -1, H, W))
        return out

    @autocast(enabled=False)
    def relu_quadratic_att(self, qkv: torch.Tensor) -> torch.Tensor:
        B, _, H, W = list(qkv.size())

        qkv = torch.reshape(
            qkv,
            (
                B,
                -1,
                3 * self.dim,
                H * W,
            ),
        )
        q, k, v = (
            qkv[:, :, 0 : self.dim],
            qkv[:, :, self.dim : 2 * self.dim],
            qkv[:, :, 2 * self.dim :],
        )

        q = self.kernel_func(q)
        k = self.kernel_func(k)

        att_map = torch.matmul(k.transpose(-1, -2), q)  # b h n n
        original_dtype = att_map.dtype
        if original_dtype in [torch.float16, torch.bfloat16]:
            att_map = att_map.float()
        att_map = att_map / (torch.sum(att_map, dim=2, keepdim=True) + self.eps)  # b h n n
        att_map = att_map.to(original_dtype)
        out = torch.matmul(v, att_map)  # b h d n

        out = torch.reshape(out, (B, -1, H, W))
        return out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # generate multi-scale q, k, v
        input_tensor = x.permute(0, 2, 1)  # [128, 1120, 100]
        x = input_tensor.unsqueeze(2)
        qkv = self.qkv(x)
        multi_scale_qkv = [qkv]
        for op in self.aggreg:
            multi_scale_qkv.append(op(qkv))
        qkv = torch.cat(multi_scale_qkv, dim=1)

        H, W = list(qkv.size())[-2:]
        if H * W > self.dim:
            out = self.relu_linear_att(qkv)
        else:
            out = self.relu_quadratic_att(qkv)
        out = self.proj(out)

        return out

class AddNorm(nn.Module):
    def __init__(self, normalize_shape, dropout):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        self.normalize = nn.LayerNorm(normalize_shape)

    def forward(self, X, Y):
        Y = self.dropout(Y) + X
        return self.normalize(Y)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=5):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class mcam(nn.Module):
    def __init__(self, in_planes, ratio=5, kernel_size=3):
        super(mcam, self).__init__()
        self.ca = ChannelAttention(in_planes, ratio)
        self.sa = SpatialAttention(kernel_size)

    def forward(self, x):
        out = x * self.ca(x)
        result = out * self.sa(out)
        return result
class Lambda(nn.Module):
    def __init__(self, func):
        super(Lambda, self).__init__()
        self.func = func

    def forward(self, x):
        return self.func(x)

def seq_to_mamba(x, mamba_layer):
    """
    x: (B, L, C)  → 过 Mamba-2 → (B, L, C)
    内部自动处理 stride 对齐
    """
    # (B, L, C) → (B, C, L) 并保证 contiguous + 8 对齐
    x = x.transpose(1, 2).contiguous()
    if x.stride(1) % 8:  # C-stride 对齐
        x = x.clone()
    x = mamba_layer(x)  # Mamba-2 内部再转回 (B, L, C)
    return x

class MultiHeadAttention(nn.Module):
    def __init__(self, num_hiddens, num_heads, dropout=0.0, desc='enc'):
        super().__init__()
        self.num_hiddens = num_hiddens
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)
        self.normalize = AddNorm(num_hiddens, dropout)
        self.desc = desc

        # define q,k,v linear layer
        self.Wq = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.Wk = nn.Linear(self.num_hiddens, self.num_hiddens)
        self.Wv = nn.Linear(self.num_hiddens, self.num_hiddens)

        self.relu = nn.ReLU()
        self.Q = nn.Sequential(self.Wq, self.relu)
        self.K = nn.Sequential(self.Wk, self.relu)
        self.V = nn.Sequential(self.Wv, self.relu)

    def forward(self, queries, keys, values):
        # get matrices of q, k, v
        q, k, v = self.Q(queries), self.K(keys), self.V(values)
        # q_split = q.unsqueeze(1).chunk(self.num_heads, dim=-1)
        # k_split = k.unsqueeze(1).chunk(self.num_heads, dim=-1)
        # v_split = v.unsqueeze(1).chunk(self.num_heads, dim=-1)
        #
        # q_stack = torch.stack(q_split, dim=1)
        # k_stack = torch.stack(k_split, dim=1)
        # v_stack = torch.stack(v_split, dim=1)

        score = torch.matmul(q, k.permute(0, 1, 3, 2))
        score = score / (k.size()[-1] ** 0.5)
        score = F.softmax(score, dim=-1)
        a = torch.matmul(score, v)
        a = torch.reshape(a.permute(0, 1, 3, 2), shape=(q.size(0), q.size(1), q.size(2), q.size(3)))
        a += queries
        return a


class DecayPos1d(nn.Module):

    def __init__(self, embed_dim, num_heads, initial_value, heads_range):
        '''
        recurrent_chunk_size: (clh clw)
        num_chunks: (nch ncw)
        clh * clw == cl
        nch * ncw == nc

        default: clh==clw, clh != clw is not implemented
        '''
        super().__init__()
        angle = 1.0 / (10000 ** torch.linspace(0, 1, embed_dim // num_heads // 2))
        angle = angle.unsqueeze(-1).repeat(1, 2).flatten()
        self.initial_value = initial_value
        self.heads_range = heads_range
        self.num_heads = num_heads
        decay = torch.log(
            1 - 2 ** (-initial_value - heads_range * torch.arange(num_heads, dtype=torch.float) / num_heads))
        self.register_buffer('angle', angle)
        self.register_buffer('decay', decay)

    def generate_1d_decay(self, l: int):
        '''
        generate 1d decay mask, the result is l*l
        '''
        index = torch.arange(l).to(self.decay)
        mask = index[:, None] - index[None, :]  # (l l)
        mask = mask.abs()  # (l l)
        mask = mask * self.decay[:, None, None]  # (n l l)
        return mask

    def forward(self, slen):
        '''
        slen: (c)
        recurrent is not implemented
        '''
        mask_c = self.generate_1d_decay(slen)
        retention_rel_pos = mask_c

        return retention_rel_pos


class VolSelfAttention(nn.Module):
    r""" Volumetric Self-Attention"""

    def __init__(self, dim, window_size, num_heads, qkv_bias=True, qk_scale=None, attn_drop=0., proj_drop=0.):

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.temperature = nn.Parameter(torch.ones(num_heads, 1, 1))

        # define a parameter table of relative position bias
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size[0] - 1) * (2 * window_size[1] - 1), num_heads))  # 2*Wh-1 * 2*Ww-1, nH

        # get pair-wise relative position index for each token inside the window
        coords_h = torch.arange(self.window_size[0])
        coords_w = torch.arange(self.window_size[1])
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, Wh, Ww
        coords_flatten = torch.flatten(coords, 1)  # 2, Wh*Ww
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, Wh*Ww, Wh*Ww
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # Wh*Ww, Wh*Ww, 2
        relative_coords[:, :, 0] += self.window_size[0] - 1  # shift to start from 0
        relative_coords[:, :, 1] += self.window_size[1] - 1
        relative_coords[:, :, 0] *= 2 * self.window_size[1] - 1
        relative_position_index = relative_coords.sum(-1)  # Wh*Ww, Wh*Ww
        self.register_buffer("relative_position_index", relative_position_index)
        ##
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        # spectrum location prior
        # 64 128 256 512
        self.realPos = DecayPos1d(64, num_heads, 2, 4)

        ##
        self.qkv_C = nn.Conv2d(dim, dim * 3, kernel_size=1, bias=False)
        self.qkv_dwconv_C = nn.Conv2d(dim * 3, dim * 3, kernel_size=3, stride=1, padding=1, groups=dim * 3, bias=False)
        # self.conv_point_C = nn.Conv2d(dim*3, dim*3, kernel_size=1, stride=1, padding=0, groups=1)
        self.proj_C = nn.Conv2d(dim, dim, kernel_size=1)

        trunc_normal_(self.relative_position_bias_table, std=.02)
        self.softmax = nn.Softmax(dim=-1)

        self.Gao_spatial_attention = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(num_heads, 32, 3, 1, 1),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 32, 3, 1, 1),
        )

        self.Gao_channel_attention = nn.Sequential(
            # 48 30 6 30
            nn.Conv2d(dim // num_heads, 8, 3, 1, 1),
            nn.BatchNorm2d(8),
            nn.GELU(),
            nn.Conv2d(8, 1, 3, 1, 1),
        )

    def forward(self, x, mask=None):
        """
        Args:
            x: input features with shape of (num_windows*B, N, C)
            mask: (0/-inf) mask with shape of (num_windows, Wh*Ww, Wh*Ww) or None
        """
        B_, N, C = x.shape
        bs, hw, c = x.size()
        hh = int(math.sqrt(hw))
        hh1,hh2 = 4,8

        ##Spatial-wise Projection先把x成映射三维度
        qkv = self.qkv(x).reshape(B_, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)
        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        relative_position_bias = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(
            self.window_size[0] * self.window_size[1], self.window_size[0] * self.window_size[1], -1)  # Wh*Ww,Wh*Ww,nH
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # nH, Wh*Ww, Wh*Ww

        attn = attn + relative_position_bias.unsqueeze(0)  ##W-MSA
        if mask is not None:  ##W-MSA
            nW = mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = self.softmax(attn)
        else:
            attn = self.softmax(attn)

        attn = self.attn_drop(attn)  ##W-MSA

        x1 = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x1 = self.proj(x1)
        x1 = self.proj_drop(x1)

        # spectrum location prior
        realPos = self.realPos(c / self.num_heads)

        ## Spectrum-wise Projection.
        x_s = rearrange(x, ' b (h w) (c) -> b c h w ', h=hh1, w=hh2)

        qkv_c = self.qkv_dwconv_C(self.qkv_C(x_s))
        q_c, k_c, v_c = qkv_c.chunk(3, dim=1)

        q_c = rearrange(q_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        k_c = rearrange(k_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)
        v_c = rearrange(v_c, 'b (head c) h w -> b head c (h w)', head=self.num_heads)

        q_c = torch.nn.functional.normalize(q_c, dim=-1)
        k_c = torch.nn.functional.normalize(k_c, dim=-1)

        attn_c = (q_c @ k_c.transpose(-2, -1)) * self.temperature + realPos
        attn_c = attn_c.softmax(dim=-1)

        x2 = (attn_c @ v_c)

        x2 = rearrange(x2, 'b head c (h w) -> b (head c) h w', head=self.num_heads, h=hh1, w=hh2)

        x2 = self.proj_C(x2)
        x2 = rearrange(x2, ' b c h w -> b (h w) c', h=hh1, w=hh2)

        # VolAtt
        attn_spatial = attn
        attn_spatial = self.Gao_spatial_attention(attn_spatial)
        attn_b, _, _, _ = attn_spatial.shape
        attn_spatial = attn_spatial.reshape(attn_b, hh1 * hh2, 1)
        x4 = attn_spatial * x2

        x5 = x1 + x2 + x4
        return x5

    def extra_repr(self) -> str:
        return f'dim={self.dim}, window_size={self.window_size}, num_heads={self.num_heads}'


class FST(nn.Module):
    def __init__(self, block1, channels):
        super(FST, self).__init__()
        self.block1 = block1
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        x1 = self.block1(x)
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias


class FSTS(nn.Module):
    def __init__(self, channels):
        super(FSTS, self).__init__()
        self.weight1 = nn.Parameter(torch.randn(1))
        self.weight2 = nn.Parameter(torch.randn(1))
        self.bias = nn.Parameter(torch.randn((1, channels, 1, 1)))

    def forward(self, x):
        x1 = x
        weighted_block1 = self.weight1 * x1
        weighted_block2 = self.weight2 * x1
        return weighted_block1 * weighted_block2 + self.bias
from math import sqrt

class MSC(nn.Module):
    def __init__(self, dim, num_heads=8, topk=True, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3],
                 qkv_bias=False, qk_scale=None, attn_drop_ratio=0., proj_drop_ratio=0., k1=2, k2=3):
        super(MSC, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.k1 = k1
        self.k2 = k2

        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # self.attn3 = torch.nn.Parameter(torch.tensor([0.3]), requires_grad=True)

        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0], stride=s[0], padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1], stride=s[1], padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2], stride=s[2], padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)

        self.topk = topk  # False True

    def forward(self, x, y):
        # x0 = x
        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        # y = torch.cat([y1.flatten(-2,-1),y2.flatten(-2,-1),y3.flatten(-2,-1)],dim = -1)
        y = y1 + y2 + y3
        y = y.flatten(-2, -1)

        y = y.transpose(1, 2)
        y = self.layer_norm(y)
        x = rearrange(x, 'b c h w -> b (h w) c')
        # y = rearrange(y,'b c h w -> b (h w) c')
        B, N1, C = y.shape
        # print(y.shape)
        kv = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # print(self.k1,self.k2)
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1 / self.k1), dim=-1, largest=True)[1]
        # print(index[0,:,48])
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v)

        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1 / self.k2), dim=-1, largest=True)[1]
        # print(index[0,:,48])
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v)

        out = out1 * self.attn1 + out2 * self.attn2  # + out3 * self.attn3
        # out = out1 * self.attn1 + out2 * self.attn2

        x = out.transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        # hw = int(sqrt(N))
        hw1, hw2 = 10, 10
        x = rearrange(x, 'b (h w) c -> b c h w', h=hw1, w=hw2)
        # x = x + x0
        return x

import torch
from torch import nn

class LocalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"

        self.conv = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.GAP = nn.AvgPool2d(feature_map_size)

    def forward(self, x):
        N, C, H, W = x.shape
        att = self.GAP(x).reshape(N, 1, C)
        att = self.conv(att).sigmoid()
        att =  att.reshape(N, C, 1, 1)
        return (x * att) + x

class GlobalChannelAttention(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"

        self.conv_q = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.conv_k = nn.Conv1d(1, 1, kernel_size, 1, padding=(kernel_size-1)//2)
        self.GAP = nn.AvgPool2d(feature_map_size)

    def forward(self, x):
        N, C, H, W = x.shape

        query = key = self.GAP(x).reshape(N, 1, C)
        query = self.conv_q(query).sigmoid()
        key = self.conv_q(key).sigmoid().permute(0, 2, 1)
        query_key = torch.bmm(key, query).reshape(N, -1)
        query_key = query_key.softmax(-1).reshape(N, C, C)

        value = x.permute(0, 2, 3, 1).reshape(N, -1, C)
        att = torch.bmm(value, query_key).permute(0, 2, 1)
        att = att.reshape(N, C, H, W)
        return x * att


class ChannelAttention_(nn.Module):
    def __init__(self, feature_map_size, kernel_size):
        super().__init__()
        assert (kernel_size%2 == 1), "Kernel size must be odd"
        self.global_attention = GlobalChannelAttention(feature_map_size,kernel_size)
        self.local_attention = LocalChannelAttention(feature_map_size,kernel_size)


    def forward(self, x):

        input_left, input_right = x.chunk(2,dim=1)
        x1 = self.global_attention(input_left)
        x2 = self.local_attention(input_right)
        output = torch.cat((x1,x2),dim=1)

        return output + x

class DeepPepPIModel_fenkai(nn.Module):
    def __init__(self, pept_emb_shape, pept_cm_shape, pept_hand_shape, prot_emb_shape,prot_hand_shape):
        super(DeepPepPIModel_fenkai, self).__init__()
        self.pept_emb_mcam = mcam(960)
        self.pept_cm_mcam = mcam(100)
        self.pept_hand_mcam = mcam(60)

        self.prot_emb_mcam = mcam(prot_emb_shape[1])
        self.prot_hand_mcam = mcam(prot_hand_shape[1])

        self.pept_emb_input = nn.Sequential(
            nn.Conv1d(in_channels=pept_emb_shape[2], out_channels=128, kernel_size=4, padding='same'),
            nn.LeakyReLU(negative_slope=0.04),
            # nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.25),
            Lambda(lambda x: x.permute(0, 2, 1)),  # 将 (n, 128, 32) 转换为 (n, 32, 128)
            nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
        )
        self.pept_cm_input = nn.Sequential(
            nn.Conv1d(in_channels=pept_cm_shape[2], out_channels=64, kernel_size=4, padding='same'),
            nn.LeakyReLU(negative_slope=0.02),
            # nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.25),
            Lambda(lambda x: x.permute(0, 2, 1)),  # 将 (n, 128, 32) 转换为 (n, 32, 128)
            nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        )

        self.pept_hand_input = nn.Sequential(
            nn.Conv1d(in_channels=pept_hand_shape[2], out_channels=128, kernel_size=5, padding='same'),
            nn.LeakyReLU(negative_slope=0.02),
            # nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.3),
            Lambda(lambda x: x.permute(0, 2, 1)),  # 将 (n, 128, 32) 转换为 (n, 32, 128)
            nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
        )
        self.fsts_pept = FSTS(channels=128)
        self.fsts_prot = FSTS(channels=128)
        self.fsts = FSTS(channels=256)
        self.attn = VolSelfAttention(
            576, window_size=to_2tuple(7), num_heads=8,
            qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0)

        # self.prot_emb_input = nn.Sequential(
        #     nn.Conv1d(in_channels=prot_emb_shape[1], out_channels=64, kernel_size=3),
        #     nn.LeakyReLU(negative_slope=0.02),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Dropout(p=0.25),
        #     Lambda(lambda x: x.permute(0, 2, 1)),
        #     nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        # )
        # self.prot_hand_input = nn.Sequential(
        #     nn.Conv1d(in_channels=prot_hand_shape[1], out_channels=64, kernel_size=3),
        #     nn.LeakyReLU(negative_slope=0.02),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Dropout(p=0.25),
        #     Lambda(lambda x: x.permute(0, 2, 1)),
        #     nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        # )
        # self.pept_hand_feat_input = nn.Sequential(
        #     nn.LSTM(input_size=pept_hand_shape[1], hidden_size=128, bidirectional=True, batch_first=True),
        #     )
        # self.prot_emb_feat_input = TCN(nb_filters=1024, kernel_size=7, dropout_rate=0.0, nb_stacks=1, dilations=[1,2,3,4],
        #         return_sequences=True, activation='relu', padding='same', use_skip_connections=True)
        # self.prot_emb_feat_input = nn.LSTM(input_size=prot_emb_shape[1], hidden_size=128, bidirectional=True, batch_first=True)
        # self.prot_hand_feat_input = nn.Sequential(nn.LSTM(input_size=prot_hand_shape[1], hidden_size=128, bidirectional=True, batch_first=True),
        #
        self.prot_trans = nn.Linear(320,640)
        self.prot_trans2 = nn.Linear(826,826*2)
        self.channalAtt_pept_emb = ChannelAttention_((100,1), kernel_size=7)
        self.channalAtt_pept_cm = ChannelAttention_((100,1), kernel_size=3)
        self.channalAtt_pept_hand = ChannelAttention_((100,1), kernel_size=3)
        self.channalAtt_emb = ChannelAttention_(1, kernel_size=3)
        self.channalAtt_hand = ChannelAttention_(1, kernel_size=3)

        # self.proj_Q = nn.Linear(400, 64*8)
        # self.proj_K = nn.Linear(400, 64*8)
        # self.proj_V = nn.Linear(400, 64*8)

        self.Mamba_1 = Mamba2(d_model=pept_emb_shape[2], expand=2, headdim=40)
        self.Mamba_2 = Mamba2(d_model=pept_cm_shape[2], expand=4, headdim=25)
        self.Mamba_3 = Mamba2(d_model=pept_hand_shape[2], expand=4, headdim=15)
        # self.msa = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        self.msa_pep = nn.MultiheadAttention(embed_dim=pept_emb_shape[2], num_heads=4)
        self.msa_prot = nn.MultiheadAttention(embed_dim=prot_emb_shape[1], num_heads=8)
        # self.project = nn.Linear(pept_emb_shape[2],prot_emb_shape[2])
        # self.ins_mechanism = InS_mechanism(perspective_num=64)
        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(128, 2)
        )

        self.litemla_pept = LiteMLA(
            in_channels=1280,  # 输入特征维度
            out_channels=128,  # 输出特征维度
            heads=8,  # 注意力头数量
            dim=32,  # 每个头处理的特征维度
            use_bias=False,  # 是否使用偏置
            norm="bn2d",  # 归一化类型
            act_func="relu",  # 激活函数
            kernel_func="relu",  # 卷积核函数
            scales=(5,),  # 缩放因子
            eps=1.0e-15,  # 稳定性参数
        )
        self.litemla_prot = LiteMLA(
            in_channels=2292,  # 输入特征维度
            out_channels=128,  # 输出特征维度
            heads=6,  # 注意力头数量
            dim=32,  # 每个头处理的特征维度
            use_bias=False,  # 是否使用偏置
            norm="bn2d",  # 归一化类型
            act_func="relu",  # 激活函数
            kernel_func="relu",  # 卷积核函数
            scales=(5,),  # 缩放因子
            eps=1.0e-15,  # 稳定性参数
        )
        # self.cross_attetion_1 = MultiHeadAttention(num_hiddens=64, num_heads=8)
        # self.cross_attetion_2 = MultiHeadAttention(num_hiddens=64, num_heads=8)

        self.msc1 = MSC(dim=256, kernel=[3, 5, 7], pad=[1, 2, 3], k1=2, k2=3)
        self.msc2 = MSC(dim=256, kernel=[3, 5, 7], pad=[1, 2, 3], k1=2, k2=3)
        # self.msc3 = MSC(dim=60, kernel=[3, 5, 7], pad=[1, 2, 3], k1=k1, k2=k2)

        # self.MSC = MSC(dim=256, num_heads=8, topk=True, kernel=[3, 5, 7], s=[1, 1, 1], pad=[1, 2, 3])
    # 在 forward 里

    def forward(self, pept_emb, pept_cm, pept_hand, prot_emb, prot_hand):
        pept_emb = self.Mamba_1(pept_emb)
        pept_emb = self.Mamba_1(pept_emb)
        pept_emb = self.Mamba_1(pept_emb)
        # pept_emb = self.Mamba_1(pept_emb)

        pept_cm = self.Mamba_2(pept_cm)
        pept_cm = self.Mamba_2(pept_cm)
        pept_cm = self.Mamba_2(pept_cm)
        # pept_cm = self.Mamba_2(pept_cm)

        pept_hand = self.Mamba_3(pept_hand)
        pept_hand = self.Mamba_3(pept_hand)
        pept_hand = self.Mamba_3(pept_hand)
        # pept_hand = self.Mamba_3(pept_hand)

        # pept_emb = self.pept_emb_mcam(pept_emb.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)
        # pept_cm = self.pept_cm_mcam(pept_cm.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)
        # pept_hand = self.pept_hand_mcam(pept_hand.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)

        # pept_emb = self.pept_hand_feat_input(pept_emb)
        # pept_cm = self.pept_cm_mcam(pept_cm)
        # pept_hand = self.pept_hand_mcam(pept_hand)
        # pept_cm_out, _ = self.pept_cm_input(pept_cm.permute(0, 2, 1))
        # pept_hand_out, _ = self.pept_hand_input(pept_hand.permute(0, 2, 1))256*100*1*960
        pept_emb_out =self.channalAtt_pept_emb(pept_emb.unsqueeze(2).permute(0, 3, 1, 2)).squeeze(3).permute(0,2,1)
        pept_cm_out = self.channalAtt_pept_cm(pept_cm.unsqueeze(2).permute(0, 3, 1, 2)).squeeze(3).permute(0,2,1)
        pept_hand_out = self.channalAtt_pept_hand(pept_hand.unsqueeze(2).permute(0, 3, 1, 2)).squeeze(3).permute(0,2,1)
        pept_emb_out, _ = self.pept_emb_input(pept_emb_out.permute(0, 2, 1))
        pept_cm_out, _ = self.pept_cm_input(pept_cm_out.permute(0, 2, 1))
        pept_hand_out, _ = self.pept_hand_input(pept_hand_out.permute(0, 2, 1))#256*100*1*960

        pept_cm_out_ = self.msc1(pept_emb_out.unsqueeze(3).permute(0,2,1,3),pept_cm_out.unsqueeze(3).permute(0,2,1,3)).flatten(-2,-1).permute(0,2,1)
        pept_hand_out_ = self.msc2(pept_emb_out.unsqueeze(3).permute(0,2,1,3), pept_hand_out.unsqueeze(3).permute(0,2,1,3)).flatten(-2,-1).permute(0,2,1)


        pept_all = torch.cat([pept_emb_out, pept_cm_out, pept_cm_out_, pept_hand_out, pept_hand_out_], dim=-1)

        # pept_all = self.msa_pep(pept_all)
        pept_all = self.litemla_pept(pept_all).permute(0, 2, 3, 1).squeeze(1)
        # pept_all = self.attn(pept_all)
        #pept_all =self.fsts_pept(pept_all.mean(dim=1).unsqueeze(2).unsqueeze(2)).squeeze(2).squeeze(2)
        prot_emb = self.prot_trans(prot_emb.squeeze(2))
        prot_hand = self.prot_trans2(prot_hand)
        prot_emb_out = self.channalAtt_emb(prot_emb.unsqueeze(2).unsqueeze(2))
        prot_hand_out = self.channalAtt_hand(prot_hand.unsqueeze(2).unsqueeze(2))
        # prot_emb_out = self.prot_emb_mcam(prot_emb.unsqueeze(2))
        # prot_hand_out = self.prot_hand_mcam(prot_hand.unsqueeze(2).unsqueeze(2))
        prot_all = torch.cat([prot_emb_out.squeeze(2).squeeze(2), prot_hand_out.squeeze(2).squeeze(2)], dim=-1)
        prot_all = self.litemla_prot(prot_all.unsqueeze(1)).squeeze(2).squeeze(2)
        #prot_all = self.fsts_prot(prot_all).squeeze(2).squeeze(2)
        # prot_emb_out = self.litemla_prot(prot_emb_out.unsqueeze(2))
        #两个大模型再经过卷积注意力
        # pept_out = pept_out.permute(0, 3, 2, 1)  # (B, L, 1, C)
        # pept_out = pept_out.squeeze(-2)
        # prot_out = prot_out.permute(0, 3, 2, 1)  # (B, L, 1, C)
        # prot_out = prot_out.squeeze(-2)
        # pept_out = self.pept_mat_mcam(pept_emb.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)
        # prot_out = self.prot_semb_feat_input(prot_mat.squeeze(2))

        # prot_out = self.prot_mat_mcam(prot_mat.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)
        # prot_hand_out, _ = self.prot_hand_feat_input(prot_hand)
        # pept_hand_out, _ = self.pept_hand_feat_input(pept_hand)
        #手工直接lstm
        # pept_lite = self.litemla_pept()
        # prot_lite = self.litemla_prot()
        # pept_msa = self.msa_pep(pept_out, pept_out, pept_out)[0]
        # prot_msa = self.msa_prot(prot_out, prot_out, prot_out)[0]
        #大模型特征直接自注意力
        # pept_msa = self.project(pept_msa)
        # pept_out = align_for_mamba(pept_out)
        # prot_out = align_for_mamba(prot_out)
        # pept_mam = seq_to_mamba(pept_out, self.Mamba_1)
        # pept_mam = seq_to_mamba(pept_mam, self.Mamba_1)
        # pept_mam = seq_to_mamba(pept_mam, self.Mamba_1)
        #
        # prot_mam = seq_to_mamba(prot_out, self.Mamba_2)
        # prot_mam = seq_to_mamba(prot_mam, self.Mamba_2)
        # prot_mam = seq_to_mamba(prot_mam, self.Mamba_2)
        # pept_mam = self.Mamba_1(pept_out)
        # pept_mam = self.Mamba_1(pept_mam)
        # pept_mam = self.Mamba_1(pept_mam)
        # prot_mam = self.Mamba_2(prot_out)
        # prot_mam = self.Mamba_2(prot_mam)
        # prot_mam = self.Mamba_2(prot_mam)

        #pept_msa = self.proj_Q(pept_msa).view(pept_msa.size(0), 100, 8, 64).transpose(1, 2)
        # prot_msa = self.proj_K(prot_msa).view(prot_msa.size(0), 9, 8, 64).transpose(1, 2)
        # prot_msa = self.proj_V(prot_msa).view(128, 9, 8, 64).transpose(1, 2)
        # output_attn_prot = self.cross_attetion_1(pept_msa, prot_msa, prot_msa)
        # output_attn_pept = self.cross_attetion_2(prot_msa, pept_msa, pept_msa)
        # output_attn_prot = output_attn_prot.reshape(pept_msa.size(0),100, 8*64)
        # output_attn_pept = output_attn_pept.reshape(prot_msa.size(0), 9, 8 * 64)

        # ins1 = self.ins_mechanism(output_attn_pept, output_attn_prot)
        # ins2 = self.ins_mechanism(output_attn_prot, output_attn_pept)
        #joint_rep = torch.cat([pept_mam.mean(dim=1), prot_mam.mean(dim=1),prot_hand_out, pept_hand_out], dim=1)
        joint_rep = torch.cat([pept_all.mean(dim=1), prot_all], dim=1)
        # joint_rep = self.fsts(joint_rep.unsqueeze(2).unsqueeze(2))


        # joint_rep = torch.cat(
        #     [output_attn_prot.mean(dim=1), ins1.mean(dim=1), ins2.mean(dim=1), output_attn_pept.mean(dim=1),
        #      prot_hand_out, pept_hand_out], dim=1)

        output = self.fc(joint_rep)
        return output


class DeepPepPIModel_fenkai_6_(nn.Module):
    def __init__(self, pept_emb_shape, pept_cm_shape, pept_hand_2d_shape,pept_hand_3d_shape, prot_emb_shape,prot_hand_shape):
        super(DeepPepPIModel_fenkai_6_, self).__init__()
        self.pept_emb_mcam = mcam(960)
        self.pept_cm_mcam = mcam(100)
        self.pept_hand_mcam = mcam(60)

        self.prot_emb_mcam = mcam(prot_emb_shape[1])
        self.prot_hand_mcam = mcam(prot_hand_shape[1])

        self.pept_emb_input = nn.Sequential(
            nn.Conv1d(in_channels=pept_emb_shape[2], out_channels=128, kernel_size=4),
            nn.LeakyReLU(negative_slope=0.04),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.25),
            Lambda(lambda x: x.permute(0, 2, 1)),  # 将 (n, 128, 32) 转换为 (n, 32, 128)
            nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
        )
        self.pept_cm_input = nn.Sequential(
            nn.Conv1d(in_channels=pept_cm_shape[2], out_channels=64, kernel_size=4),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.25),
            Lambda(lambda x: x.permute(0, 2, 1)),  # 将 (n, 128, 32) 转换为 (n, 32, 128)
            nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        )

        self.pept_hand_input_2d = nn.Sequential(
            nn.Conv1d(in_channels=pept_hand_2d_shape[1], out_channels=128, kernel_size=3, padding='same'),
            nn.LeakyReLU(negative_slope=0.02),
            # nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.25),
            Lambda(lambda x: x.permute(0, 2, 1)),  # 将 (n, 128, 32) 转换为 (n, 32, 128)
            nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
        )

        self.pept_hand_input = nn.Sequential(
            nn.Conv1d(in_channels=pept_hand_3d_shape[2], out_channels=128, kernel_size=5),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.25),
            Lambda(lambda x: x.permute(0, 2, 1)),  # 将 (n, 128, 32) 转换为 (n, 32, 128)
            nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
        )
        self.attn = VolSelfAttention(
            768, window_size=to_2tuple((4,8)), num_heads=8,
            qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0)
        # self.attn = VolSelfAttention(
        #     768, window_size=(4,8), num_heads=8,
        #     qkv_bias=True, qk_scale=None, attn_drop=0, proj_drop=0)
        # self.prot_emb_input = nn.Sequential(
        #     nn.Conv1d(in_channels=prot_emb_shape[1], out_channels=64, kernel_size=3),
        #     nn.LeakyReLU(negative_slope=0.02),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Dropout(p=0.25),
        #     Lambda(lambda x: x.permute(0, 2, 1)),
        #     nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        # )
        # self.prot_hand_input = nn.Sequential(
        #     nn.Conv1d(in_channels=prot_hand_shape[1], out_channels=64, kernel_size=3),
        #     nn.LeakyReLU(negative_slope=0.02),
        #     nn.MaxPool1d(kernel_size=2),
        #     nn.Dropout(p=0.25),
        #     Lambda(lambda x: x.permute(0, 2, 1)),
        #     nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        # )
        # self.pept_hand_feat_input = nn.Sequential(
        #     nn.LSTM(input_size=pept_hand_shape[1], hidden_size=128, bidirectional=True, batch_first=True),
        #     )
        # self.prot_emb_feat_input = TCN(nb_filters=1024, kernel_size=7, dropout_rate=0.0, nb_stacks=1, dilations=[1,2,3,4],
        #         return_sequences=True, activation='relu', padding='same', use_skip_connections=True)
        # self.prot_emb_feat_input = nn.LSTM(input_size=prot_emb_shape[1], hidden_size=128, bidirectional=True, batch_first=True)
        # self.prot_hand_feat_input = nn.Sequential(nn.LSTM(input_size=prot_hand_shape[1], hidden_size=128, bidirectional=True, batch_first=True),
        #                                          )
        # self.proj_Q = nn.Linear(400, 64*8)
        # self.proj_K = nn.Linear(400, 64*8)
        # self.proj_V = nn.Linear(400, 64*8)

        self.Mamba_1 = Mamba2(d_model=pept_emb_shape[2], expand=2, headdim=40)
        self.Mamba_2 = Mamba2(d_model=pept_cm_shape[2], expand=4, headdim=25)
        self.Mamba_3 = Mamba2(d_model=pept_hand_3d_shape[2], expand=4, headdim=15)
        self.Mamba_4 = Mamba2(d_model=pept_hand_2d_shape[1], expand=4, headdim=20)
        # self.msa = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        # self.msa_pep = nn.MultiheadAttention(embed_dim=pept_emb_shape[2], num_heads=4)
        # self.msa_prot = nn.MultiheadAttention(embed_dim=prot_emb_shape[1], num_heads=8)
        # self.project = nn.Linear(pept_emb_shape[2],prot_emb_shape[2])
        # self.ins_mechanism = InS_mechanism(perspective_num=64)
        self.fc = nn.Sequential(
            nn.Linear(1152, 1024),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Dropout(p=0.25),
            nn.Linear(1024, 128),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(128, 2)
        )

        self.litemla_pept = LiteMLA(
            in_channels=768,  # 输入特征维度
            out_channels=128,  # 输出特征维度
            heads=8,  # 注意力头数量
            dim=32,  # 每个头处理的特征维度
            use_bias=False,  # 是否使用偏置
            norm="bn2d",  # 归一化类型
            act_func="relu",  # 激活函数
            kernel_func="relu",  # 卷积核函数
            scales=(5,),  # 缩放因子
            eps=1.0e-15,  # 稳定性参数
        )
        self.litemla_prot = LiteMLA(
            in_channels=1146,  # 输入特征维度
            out_channels=128,  # 输出特征维度
            heads=6,  # 注意力头数量
            dim=32,  # 每个头处理的特征维度
            use_bias=False,  # 是否使用偏置
            norm="bn2d",  # 归一化类型
            act_func="relu",  # 激活函数
            kernel_func="relu",  # 卷积核函数
            scales=(5,),  # 缩放因子
            eps=1.0e-15,  # 稳定性参数
        )
        # self.cross_attetion_1 = MultiHeadAttention(num_hiddens=64, num_heads=8)
        # self.cross_attetion_2 = MultiHeadAttention(num_hiddens=64, num_heads=8)


    # 在 forward 里
    " pept_emb, pept_cm, pept_hand_2d,pept_hand_3d, prot_emb, prot_hand"
    "250*100*960, 250*100*100, 256*1280, 256*100*60, 256*320*1, 256*826"
    def forward(self, pept_emb, pept_cm, pept_hand_2d,pept_hand_3d, prot_emb, prot_hand):
        pept_emb = self.Mamba_1(pept_emb)
        pept_emb = self.Mamba_1(pept_emb)
        pept_emb = self.Mamba_1(pept_emb)
        # pept_emb = self.Mamba_1(pept_emb)

        pept_cm = self.Mamba_2(pept_cm)
        pept_cm = self.Mamba_2(pept_cm)
        pept_cm = self.Mamba_2(pept_cm)
        # pept_cm = self.Mamba_2(pept_cm)

        pept_hand_2d = self.Mamba_4(pept_hand_2d.unsqueeze(1))
        pept_hand_2d = self.Mamba_4(pept_hand_2d)
        pept_hand_2d = self.Mamba_4(pept_hand_2d)

        pept_hand_3d = self.Mamba_3(pept_hand_3d)
        pept_hand_3d = self.Mamba_3(pept_hand_3d)
        pept_hand_3d = self.Mamba_3(pept_hand_3d)
        # pept_hand = self.Mamba_3(pept_hand)

        # pept_emb = self.pept_emb_mcam(pept_emb.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)
        # pept_cm = self.pept_cm_mcam(pept_cm.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)
        # pept_hand = self.pept_hand_mcam(pept_hand.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)

        # pept_emb = self.pept_hand_feat_input(pept_emb)
        # pept_cm = self.pept_cm_mcam(pept_cm)
        # pept_hand = self.pept_hand_mcam(pept_hand)

        pept_emb_out, _ = self.pept_emb_input(pept_emb.permute(0, 2, 1))
        pept_cm_out, _ = self.pept_cm_input(pept_cm.permute(0, 2, 1))
        pept_hand_3d_out, _ = self.pept_hand_input(pept_hand_3d.permute(0, 2, 1))
        pept_hand_2d_out, _ = self.pept_hand_input_2d(pept_hand_2d.permute(0, 2, 1))
        pept_hand_2d = pept_hand_2d_out.squeeze(1)
        pept_all = torch.cat([pept_emb_out, pept_cm_out, pept_hand_3d_out], dim=-1)
        # pept_all = self.litemla_pept(pept_all).permute(0, 2, 3, 1).squeeze(1)
        pept_all = self.attn(pept_all)#256*32*768


        # prot_emb_out = self.prot_emb_mcam(pept_hand_2d_out.unsqueeze(2))
        prot_emb_out = self.prot_emb_mcam(prot_emb.unsqueeze(2))
        prot_hand_out = self.prot_hand_mcam(prot_hand.unsqueeze(2).unsqueeze(2))
        prot_all = torch.cat([prot_emb_out.squeeze(2).squeeze(2), prot_hand_out.squeeze(2).squeeze(2)], dim=-1)
        prot_all = self.litemla_prot(prot_all.unsqueeze(1)).squeeze(2).squeeze(2)

        # prot_emb_out = self.litemla_prot(prot_emb_out.unsqueeze(2))
        #两个大模型再经过卷积注意力
        # pept_out = pept_out.permute(0, 3, 2, 1)  # (B, L, 1, C)
        # pept_out = pept_out.squeeze(-2)
        # prot_out = prot_out.permute(0, 3, 2, 1)  # (B, L, 1, C)
        # prot_out = prot_out.squeeze(-2)
        # pept_out = self.pept_mat_mcam(pept_emb.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)
        # prot_out = self.prot_semb_feat_input(prot_mat.squeeze(2))

        # prot_out = self.prot_mat_mcam(prot_mat.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)
        # prot_hand_out, _ = self.prot_hand_feat_input(prot_hand)
        # pept_hand_out, _ = self.pept_hand_feat_input(pept_hand)
        #手工直接lstm
        # pept_lite = self.litemla_pept()
        # prot_lite = self.litemla_prot()
        # pept_msa = self.msa_pep(pept_out, pept_out, pept_out)[0]
        # prot_msa = self.msa_prot(prot_out, prot_out, prot_out)[0]
        #大模型特征直接自注意力
        # pept_msa = self.project(pept_msa)
        # pept_out = align_for_mamba(pept_out)
        # prot_out = align_for_mamba(prot_out)
        # pept_mam = seq_to_mamba(pept_out, self.Mamba_1)
        # pept_mam = seq_to_mamba(pept_mam, self.Mamba_1)
        # pept_mam = seq_to_mamba(pept_mam, self.Mamba_1)
        #
        # prot_mam = seq_to_mamba(prot_out, self.Mamba_2)
        # prot_mam = seq_to_mamba(prot_mam, self.Mamba_2)
        # prot_mam = seq_to_mamba(prot_mam, self.Mamba_2)
        # pept_mam = self.Mamba_1(pept_out)
        # pept_mam = self.Mamba_1(pept_mam)
        # pept_mam = self.Mamba_1(pept_mam)
        # prot_mam = self.Mamba_2(prot_out)
        # prot_mam = self.Mamba_2(prot_mam)
        # prot_mam = self.Mamba_2(prot_mam)

        #pept_msa = self.proj_Q(pept_msa).view(pept_msa.size(0), 100, 8, 64).transpose(1, 2)
        # prot_msa = self.proj_K(prot_msa).view(prot_msa.size(0), 9, 8, 64).transpose(1, 2)
        # prot_msa = self.proj_V(prot_msa).view(128, 9, 8, 64).transpose(1, 2)
        # output_attn_prot = self.cross_attetion_1(pept_msa, prot_msa, prot_msa)
        # output_attn_pept = self.cross_attetion_2(prot_msa, pept_msa, pept_msa)
        # output_attn_prot = output_attn_prot.reshape(pept_msa.size(0),100, 8*64)
        # output_attn_pept = output_attn_pept.reshape(prot_msa.size(0), 9, 8 * 64)

        # ins1 = self.ins_mechanism(output_attn_pept, output_attn_prot)
        # ins2 = self.ins_mechanism(output_attn_prot, output_attn_pept)
        #joint_rep = torch.cat([pept_mam.mean(dim=1), prot_mam.mean(dim=1),prot_hand_out, pept_hand_out], dim=1)
        pept_all = pept_all.mean(dim=1)
        joint_rep = torch.cat([pept_all, pept_hand_2d,prot_all], dim=-1)
        #joint_rep = torch.cat([pept_all.mean(dim=1), prot_all], dim=1)


        # joint_rep = torch.cat(
        #     [output_attn_prot.mean(dim=1), ins1.mean(dim=1), ins2.mean(dim=1), output_attn_pept.mean(dim=1),
        #      prot_hand_out, pept_hand_out], dim=1)

        output = self.fc(joint_rep)
        return output

class DeepPepPIModel(nn.Module):
    def __init__(self, pept_emb_shape, pept_hand_shape, prot_mat_shape,prot_hand_shape):
        super(DeepPepPIModel, self).__init__()
        self.prot_mat_mcam = mcam(prot_mat_shape[1])
        self.pept_mat_mcam = mcam(pept_emb_shape[2])
        self.pept_emb_input = nn.Sequential(
            nn.Conv1d(in_channels=pept_emb_shape[2], out_channels=128, kernel_size=4),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.25),
            Lambda(lambda x: x.permute(0, 2, 1)),  # 将 (n, 128, 32) 转换为 (n, 32, 128)
            nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
        )
        self.prot_mat_input = nn.Sequential(
            nn.Conv1d(in_channels=prot_mat_shape[1], out_channels=64, kernel_size=3),
            nn.LeakyReLU(negative_slope=0.02),
            nn.MaxPool1d(kernel_size=2),
            nn.Dropout(p=0.25),
            Lambda(lambda x: x.permute(0, 2, 1)),
            nn.LSTM(input_size=64, hidden_size=128, bidirectional=True, batch_first=True)
        )
        self.pept_hand_feat_input = nn.Sequential(
            nn.LSTM(input_size=pept_hand_shape[1], hidden_size=128, bidirectional=True, batch_first=True),
            )
        # self.prot_emb_feat_input = TCN(nb_filters=1024, kernel_size=7, dropout_rate=0.0, nb_stacks=1, dilations=[1,2,3,4],
        #         return_sequences=True, activation='relu', padding='same', use_skip_connections=True)
        self.prot_emb_feat_input = nn.LSTM(input_size=prot_mat_shape[1], hidden_size=128, bidirectional=True, batch_first=True)
        self.prot_hand_feat_input = nn.Sequential(nn.LSTM(input_size=prot_hand_shape[1], hidden_size=128, bidirectional=True, batch_first=True),
                                                 )
        self.proj_Q = nn.Linear(400, 64*8)
        self.proj_K = nn.Linear(400, 64*8)
        self.proj_V = nn.Linear(400, 64*8)

        self.Mamba_1 = Mamba2(d_model=pept_emb_shape[2], expand=4, headdim=40)
        self.Mamba_2 = Mamba2(d_model=prot_mat_shape[1], expand=8, headdim=40)
        self.Mamba_3 = Mamba2(d_model=256, expand=8, headdim=64)
        self.msa = nn.MultiheadAttention(embed_dim=256, num_heads=4)
        self.msa_pep = nn.MultiheadAttention(embed_dim=pept_emb_shape[2], num_heads=4)
        self.msa_prot = nn.MultiheadAttention(embed_dim=prot_mat_shape[1], num_heads=8)
        self.project = nn.Linear(pept_emb_shape[2],prot_mat_shape[2])
        self.ins_mechanism = InS_mechanism(perspective_num=64)
        self.fc = nn.Sequential(
            nn.Linear(1536, 512),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Dropout(p=0.25),
            nn.Linear(512, 128),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(128, 2)
        )

        self.cross_attetion_1 = MultiHeadAttention(num_hiddens=64, num_heads=8)
        self.cross_attetion_2 = MultiHeadAttention(num_hiddens=64, num_heads=8)


    # 在 forward 里

    def forward(self, pept_emb, pept_hand, prot_mat, prot_hand):
        # pept_emb = pept_emb.permute(0, 2, 1)
        # prot_mat = prot_mat.permute(0, 2, 1)
        # pept_out, _ = self.pept_emb_input(pept_emb)
        # prot_out, _ = self.prot_mat_input(prot_mat)
        def align_for_mamba(x):
            """
            x: (B, L, C)  -> (B, C, L) 且 stride 8 对齐
            """
            x = x.transpose(1, 2).contiguous()
            if x.stride(1) % 8 or x.stride(2) % 8:
                x = x.clone()
            return x
        # pept_emb = pept_emb.unsqueeze(-1)  # (B, L, C, 1)
        # pept_emb = pept_emb.permute(0, 2, 3, 1)
        # # prot_mat = prot_mat.permute(0, 2, 1)
        # prot_mat = prot_mat.unsqueeze(-1)  # (B, L, C, 1)
        # prot_mat = prot_mat.permute(0, 2, 3, 1)
        # pept_out = self.pept_mat_mcam(pept_emb)
        # prot_out = self.prot_mat_mcam(prot_mat)
        # pept_out = pept_out.permute(0, 3, 2, 1)  # (B, L, 1, C)
        # pept_out = pept_out.squeeze(-2)
        # prot_out = prot_out.permute(0, 3, 2, 1)  # (B, L, 1, C)
        # prot_out = prot_out.squeeze(-2)
        pept_out = self.pept_mat_mcam(pept_emb.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)
        prot_out = self.prot_emb_feat_input(prot_mat.squeeze(2))

        # prot_out = self.prot_mat_mcam(prot_mat.unsqueeze(-1).permute(0, 2, 3, 1)).squeeze(-2).permute(0, 2, 1)
        prot_hand_out, _ = self.prot_hand_feat_input(prot_hand)
        pept_hand_out, _ = self.pept_hand_feat_input(pept_hand)
        pept_msa = self.msa_pep(pept_out, pept_out, pept_out)[0]
        prot_msa = self.msa_prot(prot_out, prot_out, prot_out)[0]
        pept_msa = self.project(pept_msa)
        # pept_out = align_for_mamba(pept_out)
        # prot_out = align_for_mamba(prot_out)
        # pept_mam = seq_to_mamba(pept_out, self.Mamba_1)
        # pept_mam = seq_to_mamba(pept_mam, self.Mamba_1)
        # pept_mam = seq_to_mamba(pept_mam, self.Mamba_1)
        #
        # prot_mam = seq_to_mamba(prot_out, self.Mamba_2)
        # prot_mam = seq_to_mamba(prot_mam, self.Mamba_2)
        # prot_mam = seq_to_mamba(prot_mam, self.Mamba_2)
        # pept_mam = self.Mamba_1(pept_out)
        # pept_mam = self.Mamba_1(pept_mam)
        # pept_mam = self.Mamba_1(pept_mam)
        # prot_mam = self.Mamba_2(prot_out)
        # prot_mam = self.Mamba_2(prot_mam)
        # prot_mam = self.Mamba_2(prot_mam)

        pept_msa = self.proj_Q(pept_msa).view(pept_msa.size(0), 100, 8, 64).transpose(1, 2)
        # prot_msa = self.proj_K(prot_msa).view(prot_msa.size(0), 9, 8, 64).transpose(1, 2)
        # prot_msa = self.proj_V(prot_msa).view(128, 9, 8, 64).transpose(1, 2)
        output_attn_prot = self.cross_attetion_1(pept_msa, prot_msa, prot_msa)
        output_attn_pept = self.cross_attetion_2(prot_msa, pept_msa, pept_msa)
        output_attn_prot = output_attn_prot.reshape(pept_msa.size(0),100, 8*64)
        output_attn_pept = output_attn_pept.reshape(prot_msa.size(0), 9, 8 * 64)

        # ins1 = self.ins_mechanism(output_attn_pept, output_attn_prot)
        # ins2 = self.ins_mechanism(output_attn_prot, output_attn_pept)
        joint_rep = torch.cat([output_attn_prot .mean(dim=1), output_attn_pept.mean(dim=1),prot_hand_out, pept_hand_out], dim=1)
        # joint_rep = torch.cat(
        #     [output_attn_prot.mean(dim=1), ins1.mean(dim=1), ins2.mean(dim=1), output_attn_pept.mean(dim=1),
        #      prot_hand_out, pept_hand_out], dim=1)

        output = self.fc(joint_rep)
        return output

class ConvsLayer(torch.nn.Module):

    def __init__(self, emb_dim):
        super(ConvsLayer, self).__init__()
        self.embedding_size = emb_dim
        self.conv1 = nn.Conv1d(in_channels=self.embedding_size, out_channels=128, kernel_size=3)
        self.mx1 = nn.MaxPool1d(3, stride=3)
        self.conv2 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.mx2 = nn.MaxPool1d(3, stride=3)
        self.conv3 = nn.Conv1d(in_channels=128, out_channels=128, kernel_size=3)
        self.mx3 = nn.MaxPool1d(130, stride=1)

    def forward(self, x):
        # x = x.squeeze(1)
        x = x.permute(0, 2, 1)
        features = self.conv1(x)
        # print(features.shape)#add
        features = self.mx1(features)
        # print(features.shape)#add
        features = self.mx2(self.conv2(features))
        # print(features.shape)#add
        features = self.conv3(features)
        # print(features.shape)#add
        features = self.mx3(features)
        # print(features.shape)#add
        features = features.squeeze(2)
        # print(features.shape)#add
        return features
class DeepPepPIModel_all(nn.Module):
    def __init__(self, pept_all_shape, prot_all_shape):
        super(DeepPepPIModel_all, self).__init__()
        self.pept_mat_mcam = mcam(128)
        self.Mamba_1 = Mamba2(d_model=pept_all_shape[2], expand=4, headdim=40)
        self.Mamba_2 = Mamba2(d_model=pept_all_shape[2], expand=8, headdim=40)
        self.Mamba_3 = Mamba2(d_model=pept_all_shape[2], expand=8, headdim=64)

        self.cross_attetion_1 = MultiHeadAttention(num_hiddens=64, num_heads=8)
        self.cross_attetion_2 = MultiHeadAttention(num_hiddens=64, num_heads=8)
        self.textcnn = ConvsLayer(prot_all_shape[1])

        self.prot_trans = nn.Sequential(
            nn.Conv1d(in_channels=prot_all_shape[1], out_channels=128, kernel_size=1),
            nn.LeakyReLU(negative_slope=0.02),
            # nn.MaxPool1d(kernel_size=3),
            nn.Dropout(p=0.25),
            Lambda(lambda x: x.permute(0, 2, 1)),  # 将 (n, 128, 32) 转换为 (n, 32, 128)
            nn.LSTM(input_size=128, hidden_size=128, bidirectional=True, batch_first=True)
        )
        self.conv1 = nn.Conv2d(128, 64, kernel_size=3)
        # self.litemla = LiteMLA(
        #     in_channels=128,
        #     out_channels=128,
        #     heads=8,
        #     dim=32)
        self.litemla = LiteMLA(
            in_channels=1120,  # 输入特征维度
            out_channels=128,  # 输出特征维度
            heads=8,  # 注意力头数量
            dim=32,  # 每个头处理的特征维度
            use_bias=False,  # 是否使用偏置
            norm="bn2d",  # 归一化类型
            act_func="relu",  # 激活函数
            kernel_func="relu",  # 卷积核函数
            scales=(5,),  # 缩放因子
            eps=1.0e-15,  # 稳定性参数
        )
        self.litemla_prot = LiteMLA(
            in_channels=256,  # 输入特征维度
            out_channels=128,  # 输出特征维度
            heads=8,  # 注意力头数量
            dim=32,  # 每个头处理的特征维度
            use_bias=False,  # 是否使用偏置
            norm="bn2d",  # 归一化类型
            act_func="relu",  # 激活函数
            kernel_func="relu",  # 卷积核函数
            scales=(5,),  # 缩放因子
            eps=1.0e-15,  # 稳定性参数
        )

        self.fc = nn.Sequential(
            nn.Linear(256, 256),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Dropout(p=0.25),
            nn.Linear(256, 128),
            nn.LeakyReLU(negative_slope=0.02),
            nn.Linear(128, 2)
        )




    # 在 forward 里

    def forward(self, pept_all, prot_all):
        pept_mam = self.Mamba_1(pept_all)
        pept_mam = self.Mamba_1(pept_mam)
        pept_mam = self.Mamba_1(pept_mam)
        pept_mla = self.litemla(pept_mam)
        pept_out = self.pept_mat_mcam(pept_mla).squeeze(-2).permute(0, 2, 1)
        prot_all = prot_all.unsqueeze(2)
        prot_cnn, _ = self.prot_trans(prot_all)
        # prot_cnn = self.textcnn(prot_all.unsqueeze(1))
        prot_mla = self.litemla_prot(prot_cnn).squeeze(2).squeeze(2) # 自动调用forward
        # prot_mla = prot_mla.mean([2, 3])

        # output_attn_prot = self.cross_attetion_1(pept_msa, prot_msa, prot_msa)
        # output_attn_pept = self.cross_attetion_2(prot_msa, pept_msa, pept_msa)
        # output_attn_prot = output_attn_prot.reshape(pept_msa.size(0),100, 8*64)
        # output_attn_pept = output_attn_pept.reshape(prot_msa.size(0), 9, 8 * 64)
        joint_rep = torch.cat([pept_out.max(dim=1)[0], prot_mla], dim=1)
        # joint_rep = torch.cat(
        #     [output_attn_prot.mean(dim=1), ins1.mean(dim=1), ins2.mean(dim=1), output_attn_pept.mean(dim=1),
        #      prot_hand_out, pept_hand_out], dim=1)

        output = self.fc(joint_rep)
        return output


class InS_mechanism(nn.Module):
    def __init__(self, perspective_num=10):
        super(InS_mechanism, self).__init__()
        self.perspective_num = perspective_num
        self.kernel = nn.Parameter(torch.randn(perspective_num, 1))

    def _cosine_similarity(self, x1, x2):
        cos = nn.CosineSimilarity(dim=-1)
        return cos(x1, x2)

    def forward(self, sent1, sent2):
        cos_matrix = self._cosine_similarity(sent1.unsqueeze(2), sent2.unsqueeze(1))
        mean_attentive_vec = self._mean_attentive_vectors(sent2, cos_matrix)
        v1 = sent1.unsqueeze(-2) * self.kernel
        mean_attentive_vec = mean_attentive_vec.unsqueeze(-2) * self.kernel
        matching = self._cosine_similarity(v1, mean_attentive_vec)
        return matching

    def _mean_attentive_vectors(self, x2, cosine_matrix):
        weighted_sum = torch.sum(cosine_matrix.unsqueeze(-1) * x2.unsqueeze(1), dim=2)
        sum_cosine = torch.sum(cosine_matrix, dim=-1).unsqueeze(-1) + 1e-8
        attentive_vector = weighted_sum / sum_cosine
        return attentive_vector

def load_data(species):
    pept_emb_file = f'./PepPI dataset/{species}/peptide1_ESMC.npy'
    pept_emb_dict = extract_pept_feat(pept_emb_file)#100*960

    pept_cm_file = f'./PepPI dataset/{species}/peptide_contactmap.h5'
    with h5py.File(pept_cm_file, 'r') as f:
        pept_cm_dict = {k: np.array(v) for k, v in f.items()}#100*100

    pept_emb_file = f'./PepPI dataset/{species}/peptide sequences.fasta'
    pept_hand_2dict,pept_hand_3dict = combinefeature(pept_emb_file, encoders={'AAC': AAC,
                                                                    'DPC': DPC, 'AAE':AAE, 'BE' :BE,
    'Blosum62' : BLOSUM62,
    'AAI' : AAI,
    'PC6' : PC6_embedding,
    'DDE' : DDE,
    'AAT' : AAT,
    'AAP' : AAP,
    #'PAAC' :PAAC,
    'KMER': KMER,
    #'CKSGGAP':CKSAAGP,
    #'ZS' : ZS,
    #'CTD' : CTD,
                                                       })
    # 以后直接往这里加)

    # pept_all_dict = {}
    # for pid in pept_emb_dict.keys() & data.keys() &pept_hand_3dict.keys():
    #     emb = pept_emb_dict[pid] # (100,960)
    #     # cmap = data[pid]  # (100,100)
    #     hand = pept_hand_3dict[pid][None,:]
    #     # pept_all_dict[pid] = np.concatenate([emb, cmap], axis=1)  # (100,1060)
    #     cmap = data[pid][None, :, :]  # (1, 100, 100)
    #     pept_all_dict[pid] = np.concatenate([emb, cmap, hand], axis=2)
    # print('合并完成：', len(pept_all_dict),'形状为：',list(pept_all_dict.values())[0].shape)


    prot_emb_file = f'./PepPI dataset/{species}/protein_esm2.npy'
    prot_emb_dict = extract_pept_feat(prot_emb_file)
    prot_seq_file = f'./PepPI dataset/{species}/protein sequences.fasta'
    # prot_str_file = f'./PepPI dataset/{species}/protein secondary structures.fasta'
    order = 2
    # prot_mat_dict = extract_prot_feat(prot_seq_file, prot_str_file, order)
    # feature_list_ml = ['OE', 'AAC', 'DPC', 'PSAAC', 'asdc', 'AAE', 'BLO2']
    prot_hand_2dict, prot_hand_3dict = combinefeature(prot_seq_file, encoders = {
    # 'OE'  : OE,
    'AAC' : AAC,
    'DPC' : DPC,
    'AAE' : AAE,
    'AAT': AAT,
    'AAP': AAP,
    # 'PAAC' :PAAC,
    'KMER': KMER,
    })

    target_length = 1280

    # 填充函数
    def pad_array(arr, target_length, pad_value=0):
        """
        将数组填充到指定长度。
        :param arr: 原始数组
        :param target_length: 目标长度
        :param pad_value: 填充值，默认为 0
        :return: 填充后的数组
        """
        if len(arr) >= target_length:
            return arr[:target_length]  # 如果数组长度大于目标长度，截断
        else:
            pad_size = target_length - len(arr)
            pad = np.full(pad_size, pad_value)  # 创建填充数组
            return np.concatenate((arr, pad))  # 拼接原始数组和填充数组

    # 对字典中的每个数组进行填充
    pept_hand_2dict = {key: pad_array(value, target_length) for key, value in pept_hand_2dict.items()}  # 拼接原始数组和填充数组


# 对字典中的每个数组进行填充
#     pept_hand_2dict = {key: pad_array(value, target_length) for key, value in pept_hand_2dict.items()}

    prot_all_dict = prot_emb_dict.copy()
    for key, value in prot_hand_2dict.items():
        if key in prot_all_dict:
            prot_all_dict[key] = np.concatenate([prot_all_dict[key].squeeze(0), value])  # 假设值为数组
        else:
            prot_all_dict[key] = value

    # protein_combined_dict = {}
    # for pid in prot_emb_dict.keys() & prot_hand_2dict.keys():
    #     emb = prot_emb_dict[pid].squeeze(0)  # (960,)
    #     hand = prot_hand_2dict[pid]  # (431,)
    #     feat = np.concatenate([emb, hand], axis=0)  # (1391,)
    #     protein_combined_dict[pid] = feat
    file_path = f'./PepPI dataset/{species}/PepPIs cv.txt'
    posi_pairs = load_pairs(file_path)

    file_path = f'./PepPI dataset/{species}/non-PepPIs cv.txt'
    nega_pairs = load_pairs(file_path)

    file_path = f'./PepPI dataset/{species}/PepPIs ind.txt'
    posi_pairs_test = load_pairs(file_path)

    file_path = f'./PepPI dataset/{species}/non-PepPIs ind.txt'
    nega_pairs_test =load_pairs(file_path)

    train_pairs = posi_pairs + nega_pairs
    # train_pairs.sort()
    train_labels = [[0, 1]] * len(posi_pairs) + [[1, 0]] * len(nega_pairs)
    train_ds = PairDataset_5(
        train_pairs,
        pept_emb_dict,
        pept_cm_dict,
        # pept_hand_2dict,
        pept_hand_3dict,
        prot_emb_dict,
        prot_hand_2dict,
        # prot_emb_dict,# 注意：这里把蛋白手工特征字典重命名避免同名
        train_labels
    )

    def worker_init_fn(worker_id):
        np.random.seed(42 + worker_id)
    train_loader = torch.utils.data.DataLoader(
        train_ds,
        batch_size=256,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(42)
    )
    # pept_emb_feat, prot_mat_feat,prot_hand_feat, label = load_dataset_hand(pept_all_dict, pept_hand_dict, prot_mat_dict,prot_hand_dict, posi_pairs, nega_pairs)
    test_pairs = posi_pairs_test + nega_pairs_test
    test_labels = [[0, 1]] * len(posi_pairs_test) + [[1, 0]] * len(nega_pairs_test)
    test_ds = PairDataset_5(
        test_pairs,
        pept_emb_dict,
        pept_cm_dict,
        # pept_hand_2dict,
        pept_hand_3dict,
        prot_emb_dict,
        prot_hand_2dict,
        # prot_emb_dict,# 注意：这里把蛋白手工特征字典重命名避免同名
        test_labels
    )

    test_loader = torch.utils.data.DataLoader(
        test_ds,
        batch_size=128,
        shuffle=True,
        worker_init_fn=worker_init_fn,
        generator=torch.Generator().manual_seed(42)
    )



    return train_loader, test_loader

def train_and_evaluate(species):
    # pept_emb_feat, pept_hand_feat, prot_mat_feat, prot_hand_feat, label, pept_emb_feat_test, pept_hand_feat_test, prot_mat_feat_test, prot_hand_feat_test, label_test = load_data(species)
    train_loader, test_loader = load_data(species)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 取形状
    sample = next(iter(train_loader))
    pept_emb_shape = sample[0].squeeze(1).shape  # (B,100,1060)
    pept_cm_shape = sample[1].shape  # (B,100,?)
    # pept_hand_2d_shape = sample[2].shape
    pept_hand_3d_shape = sample[2].shape # (B,100,?)
    prot_emb_shape = sample[3].shape
    prot_hand_shape = sample[4].shape
    print(pept_emb_shape,pept_cm_shape,pept_hand_3d_shape,prot_emb_shape,prot_hand_shape)# (B,L,C)
 # (B,L,?)
    model = DeepPepPIModel_fenkai(pept_emb_shape,pept_cm_shape,pept_hand_3d_shape,prot_emb_shape,prot_hand_shape).to(device)
    #
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # pept_emb_feat = torch.tensor(pept_emb_feat, dtype=torch.float32)
    # pept_hand_feat = torch.tensor(pept_hand_feat, dtype=torch.float32)
    # prot_mat_feat = torch.tensor(prot_mat_feat, dtype=torch.float32)
    # prot_hand_feat = torch.tensor(prot_hand_feat, dtype=torch.float32)
    # label = torch.tensor(label, dtype=torch.float32)
    #
    # pept_emb_feat_test = torch.tensor(pept_emb_feat_test, dtype=torch.float32).to(device)
    # pept_hand_feat_test = torch.tensor(pept_hand_feat_test, dtype=torch.float32).to(device)
    # prot_mat_feat_test = torch.tensor(prot_mat_feat_test, dtype=torch.float32).to(device)
    # prot_hand_feat_test = torch.tensor(prot_hand_feat_test, dtype=torch.float32).to(device)
    # label_test = torch.tensor(label_test, dtype=torch.float32).to(device)
    #
    # train_ds = PairDataset(train_pairs,
    #                        pept_all_dict, pept_hand_dict,
    #                        prot_mat_dict, prot_hand_dict2,
    #                        train_labels)
    # train_loader = torch.utils.data.DataLoader(train_ds,
    #                                            batch_size=32,  # 先小一点试
    #                                            shuffle=True)

    # dataset = TensorDataset(pept_emb_feat, pept_hand_feat, prot_mat_feat, prot_hand_feat, label)
    # dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # model = DeepPepPIModel(pept_emb_feat.shape,pept_hand_feat.shape, prot_mat_feat.shape, prot_hand_feat.shape).to(device)  # ✅ 模型搬到 GPU
    # optimizer = optim.RMSprop(model.parameters(), lr=0.0005)
    optimizer = optim.Adam(model.parameters(), lr=0.0005)
    criterion = nn.CrossEntropyLoss().to(device)
    # num_warmup_steps = 160
    # num_training_steps = 6000
    # scheduler = get_scheduler(
    #     name='cosine',
    #     optimizer=optimizer,
    #     num_warmup_steps=num_warmup_steps,
    #     num_training_steps=num_training_steps
    # )
    for epoch in range(200):
        model.train()
        epoch_loss = 0
        all_preds = []
        all_labels = []
        for batch in train_loader:
            pept_emb, pept_cm, pept_hand_3d, prot_emb, prot_hand,labels = batch
            pept_emb, pept_cm, pept_hand_3d, prot_emb, prot_hand, labels = pept_emb.squeeze(1).to(device), pept_cm.squeeze(1).to(device),pept_hand_3d.squeeze(1).to(device), prot_emb.to(device), prot_hand.to(device),labels.to(device)
            optimizer.zero_grad()
            outputs = model(pept_emb, pept_cm, pept_hand_3d, prot_emb, prot_hand)
            loss = criterion(outputs, labels.argmax(dim=1))
            loss.backward()
            optimizer.step()
            # scheduler.step()
            epoch_loss += loss.item()
        # print(f"Epoch {epoch+1}, Loss: {loss.item()}")
            y_prob = torch.softmax(outputs, dim=1)[:, 1].detach().cpu().numpy()
            y_true = labels[:, 1].cpu().numpy()
            all_preds.extend(y_prob)
            all_labels.extend(y_true)

        tp_train, fp_train, tn_train, fn_train, acc_train, prec_train, recall_train, MCC_train, f1_train, AUC_train, AUPR_train = calculate_performance(
            np.array(all_preds), np.array(all_labels))

        print(f"Epoch {epoch + 1}, Loss: {epoch_loss / len(train_loader):.4f}")
        print(f"Training Metrics - tp={tp_train}, fp={fp_train}, tn={tn_train}, fn={fn_train}")
        print(
            f"Training Metrics - Acc={acc_train:.4f}, Prec={prec_train:.4f}, Recall={recall_train:.4f}, MCC={MCC_train:.4f}, F1={f1_train:.4f}, AUC={AUPR_train:.4f}, AUPR={AUPR_train:.4f}")

    # 独立测试集评估
        model.eval()
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for pept_emb, pept_cm, pept_hand_3d, prot_emb, prot_hand,test_labels in test_loader:

                pept_emb, pept_cm, pept_hand_3d, prot_emb, prot_hand, labels = pept_emb.squeeze(1).to(device), pept_cm.squeeze(
                    1).to(device), pept_hand_3d.squeeze(1).to(device), prot_emb.to(device), prot_hand.to(device), labels.to(
                    device)

                outputs = model(pept_emb, pept_cm, pept_hand_3d, prot_emb, prot_hand)
                y_prob_test = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                all_probs.extend(y_prob_test)
                all_labels.extend(test_labels.cpu().numpy())
        y_prob_test = np.array(all_probs)
        y_true_test = np.array(all_labels)[:, 1]
        tp, fp, tn, fn, acc, prec, rec, mcc, f1, auc, aupr = calculate_performance(y_prob_test, y_true_test)

        print(f"tp={tp}, fp={fp}, tn={tn}, fn={fn}")
        print(f"Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, "
              f"MCC={mcc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, AUPR={aupr:.4f}")

    # 保存最终模型
    model_save_path = f'./Results/{species}_DeepPepPI_final_model.pth'
    torch.save(model.state_dict(), model_save_path)
    print(f"Final model saved to {model_save_path}")

    # 独立测试集评估
    model.eval()
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for pept_emb, pept_cm, pept_hand, prot_emb, prot_hand,test_labels in test_loader:

            pept_emb, pept_cm, pept_hand, prot_emb, prot_hand, labels = pept_emb.squeeze(1).to(device), pept_cm.squeeze(
                1).to(device), pept_hand.squeeze(1).to(device), prot_emb.to(device), prot_hand.to(device), labels.to(
                device)

            outputs = model(pept_emb, pept_cm, pept_hand, prot_emb, prot_hand)
            y_prob_test = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(y_prob_test)
            all_labels.extend(test_labels.cpu().numpy())
    y_prob_test = np.array(all_probs)
    y_true_test = np.array(all_labels)[:, 1]
    tp, fp, tn, fn, acc, prec, rec, mcc, f1, auc, aupr = calculate_performance(y_prob_test, y_true_test)

    print(f"tp={tp}, fp={fp}, tn={tn}, fn={fn}")
    print(f"Acc={acc:.4f}, Prec={prec:.4f}, Recall={rec:.4f}, "
          f"MCC={mcc:.4f}, F1={f1:.4f}, AUC={auc:.4f}, AUPR={aupr:.4f}")
def calculate_performance(y_prob, y_true):
    y_pred = (y_prob >= 0.5).astype(int)
    tp = np.sum((y_pred == 1) & (y_true == 1))
    fp = np.sum((y_pred == 1) & (y_true == 0))
    tn = np.sum((y_pred == 0) & (y_true == 0))
    fn = np.sum((y_pred == 0) & (y_true == 1))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    MCC = matthews_corrcoef(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    AUC = roc_auc_score(y_true, y_prob)
    AUPR = average_precision_score(y_true, y_prob)
    return tp, fp, tn, fn, acc, prec, recall, MCC, f1, AUC, AUPR

if __name__ == '__main__':
    species = 'Arabidopsis thaliana'
    train_and_evaluate(species)