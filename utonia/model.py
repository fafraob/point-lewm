"""
Point Transformer - V3 Mode3 - Utonia
Pointcept detached version

Author: Xiaoyang Wu (xiaoyang.wu.cs@gmail.com)
Please cite our work if the code is helpful to you.
"""

# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import os
import math
from packaging import version
from huggingface_hub import hf_hub_download, PyTorchModelHubMixin
from addict import Dict
import torch
import torch.nn as nn
from torch.nn.init import trunc_normal_
import spconv.pytorch as spconv
import torch_scatter
from timm.layers import DropPath

try:
    import flash_attn
except ImportError:
    flash_attn = None

from .structure import Point
from .module import PointSequential, PointModule
from .utils import offset2bincount

MODELS = [
    "utonia",
    "utonia_linear_prob_head_sc",
]


class Point3DRoPE(nn.Module):
    def __init__(self, head_dim, base=10000):
        super().__init__()
        assert (
            head_dim % 3 == 0
        ), f"Head dimension must be divisible by 3 for 3D RoPE, {head_dim}"

        self.head_dim = head_dim
        self.chunk_dim = head_dim // 3
        self.base = base
        inv_freq = 1.0 / (
            self.base ** (torch.arange(0, self.chunk_dim, 2).float() / self.chunk_dim)
        )
        self.register_buffer("inv_freq", inv_freq)

    def get_cos_sin(self, xyz):
        x = xyz[:, 0:1]
        y = xyz[:, 1:2]
        z = xyz[:, 2:3]
        freqs = self.inv_freq.unsqueeze(0)

        emb_x = x * freqs
        emb_y = y * freqs
        emb_z = z * freqs

        emb_x = torch.cat((emb_x, emb_x), dim=-1)
        emb_y = torch.cat((emb_y, emb_y), dim=-1)
        emb_z = torch.cat((emb_z, emb_z), dim=-1)
        emb_3d = torch.cat((emb_x, emb_y, emb_z), dim=-1)
        return emb_3d.cos().unsqueeze(1), emb_3d.sin().unsqueeze(1)

    def rotate_half(self, x):
        x1, x2 = x[..., : x.shape[-1] // 2], x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    def forward(self, q, k, xyz):
        cos, sin = self.get_cos_sin(xyz)
        qs = torch.split(q, self.chunk_dim, dim=-1)
        ks = torch.split(k, self.chunk_dim, dim=-1)

        coss = torch.split(cos, self.chunk_dim, dim=-1)
        sins = torch.split(sin, self.chunk_dim, dim=-1)

        q_outs = []
        k_outs = []

        for i in range(3):
            q_part = (qs[i] * coss[i]) + (self.rotate_half(qs[i]) * sins[i])
            k_part = (ks[i] * coss[i]) + (self.rotate_half(ks[i]) * sins[i])

            q_outs.append(q_part)
            k_outs.append(k_part)

        q_rot = torch.cat(q_outs, dim=-1)
        k_rot = torch.cat(k_outs, dim=-1)

        return q_rot, k_rot


class LayerScale(nn.Module):
    def __init__(
        self,
        dim: int,
        init_values: float = 1e-5,
        inplace: bool = False,
    ) -> None:
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class RPE(torch.nn.Module):
    def __init__(self, patch_size, num_heads):
        super().__init__()
        self.patch_size = patch_size
        self.num_heads = num_heads
        self.pos_bnd = int((4 * patch_size) ** (1 / 3) * 2)
        self.rpe_num = 2 * self.pos_bnd + 1
        self.rpe_table = torch.nn.Parameter(torch.zeros(3 * self.rpe_num, num_heads))
        torch.nn.init.trunc_normal_(self.rpe_table, std=0.02)

    def forward(self, coord):
        idx = (
            coord.clamp(-self.pos_bnd, self.pos_bnd)  # clamp into bnd
            + self.pos_bnd  # relative position to positive index
            + torch.arange(3, device=coord.device) * self.rpe_num  # x, y, z stride
        )
        out = self.rpe_table.index_select(0, idx.reshape(-1))
        out = out.view(idx.shape + (-1,)).sum(3)
        out = out.permute(0, 3, 1, 2)  # (N, K, K, H) -> (N, H, K, K)
        return out


class SerializedAttention(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        order_index=0,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        assert channels % num_heads == 0
        self.channels = channels
        self.num_heads = num_heads
        self.scale = qk_scale or (channels // num_heads) ** -0.5
        self.order_index = order_index
        self.upcast_attention = upcast_attention
        self.upcast_softmax = upcast_softmax
        self.enable_rpe = enable_rpe
        self.enable_flash = enable_flash
        if enable_flash:
            assert (
                enable_rpe is False
            ), "Set enable_rpe to False when enable Flash Attention"
            assert (
                upcast_attention is False
            ), "Set upcast_attention to False when enable Flash Attention"
            assert (
                upcast_softmax is False
            ), "Set upcast_softmax to False when enable Flash Attention"
            assert flash_attn is not None, "Make sure flash_attn is installed."
            self.patch_size = patch_size
            self.attn_drop = attn_drop
        else:
            # when disable flash attention, we still don't want to use mask
            # consequently, patch size will auto set to the
            # min number of patch_size_max and number of points
            self.patch_size_max = patch_size
            self.patch_size = 0
            self.attn_drop = torch.nn.Dropout(attn_drop)

        self.qkv = torch.nn.Linear(channels, channels * 3, bias=qkv_bias)
        self.proj = torch.nn.Linear(channels, channels)
        self.proj_drop = torch.nn.Dropout(proj_drop)
        self.softmax = torch.nn.Softmax(dim=-1)
        self.rpe = RPE(patch_size, num_heads) if self.enable_rpe else None
        self.rope = Point3DRoPE(head_dim=channels // num_heads, base=rope_base)
        self.shift_coords = shift_coords
        self.jitter_coords = jitter_coords
        self.rescale_coords = rescale_coords

    @torch.no_grad()
    def get_rel_pos(self, point, order):
        K = self.patch_size
        rel_pos_key = f"rel_pos_{self.order_index}"
        if rel_pos_key not in point.keys():
            grid_coord = point.grid_coord[order]
            grid_coord = grid_coord.reshape(-1, K, 3)
            point[rel_pos_key] = grid_coord.unsqueeze(2) - grid_coord.unsqueeze(1)
        return point[rel_pos_key]

    @torch.no_grad()
    def get_padding_and_inverse(self, point):
        """[le-wm vendored patch #2, see utonia/README.md]

        Upstream pads a cloud only when it is LARGER than ``patch_size``, which
        forces the non-flash path to shrink ``patch_size`` to the smallest cloud
        in the batch (``forward`` below) so that every cloud is patch-aligned.
        That couples clouds: a cloud's attention window -- and so its features --
        depends on which other clouds share the batch.

        Here EVERY cloud is padded up to a multiple of ``patch_size``, and the
        padding of a cloud that fits in a single patch is reported as invalid
        via ``patch_valid`` so ``forward`` can mask it. That reproduces what
        flash attention does with its per-cloud ``cu_seqlens`` (a ragged final
        segment for a sub-patch cloud, duplicate-padded full segments
        otherwise) while keeping ``patch_size`` fixed, so results no longer
        depend on batch composition.
        """
        pad_key = "pad"
        unpad_key = "unpad"
        cu_seqlens_key = "cu_seqlens_key"
        valid_key = "patch_valid"
        if (
            pad_key not in point.keys()
            or unpad_key not in point.keys()
            or cu_seqlens_key not in point.keys()
        ):
            offset = point.offset
            bincount = offset2bincount(offset)
            # pad EVERY cloud up to a whole number of patches (upstream skipped
            # clouds smaller than patch_size, which is what forced the shrink)
            bincount_pad = (
                torch.div(
                    bincount + self.patch_size - 1,
                    self.patch_size,
                    rounding_mode="trunc",
                )
                * self.patch_size
            )
            _offset = nn.functional.pad(offset, (1, 0))
            _offset_pad = nn.functional.pad(torch.cumsum(bincount_pad, dim=0), (1, 0))
            pad = torch.arange(_offset_pad[-1], device=offset.device)
            unpad = torch.arange(_offset[-1], device=offset.device)
            cu_seqlens = []
            patch_valid = []
            for i in range(len(offset)):
                unpad[_offset[i] : _offset[i + 1]] += _offset_pad[i] - _offset[i]
                n, n_pad = int(bincount[i]), int(bincount_pad[i])
                if n_pad > self.patch_size:
                    # more than one patch: fill the tail from the PREVIOUS patch
                    # exactly as upstream does (those duplicated points do take
                    # part in attention, matching flash's padded segments)
                    if n != n_pad:
                        pad[
                            _offset_pad[i + 1]
                            - self.patch_size
                            + (n % self.patch_size) : _offset_pad[i + 1]
                        ] = pad[
                            _offset_pad[i + 1]
                            - 2 * self.patch_size
                            + (n % self.patch_size) : _offset_pad[i + 1]
                            - self.patch_size
                        ]
                    patch_valid += [self.patch_size] * (n_pad // self.patch_size)
                else:
                    # single patch: there is no previous patch to copy from, so
                    # point the padding at the cloud's own first row (kept
                    # in-range) and mark it invalid -- flash would have used a
                    # ragged segment of n here, i.e. attention over n keys only.
                    pad[_offset_pad[i] + n : _offset_pad[i + 1]] = _offset_pad[i]
                    patch_valid.append(n)
                pad[_offset_pad[i] : _offset_pad[i + 1]] -= _offset_pad[i] - _offset[i]
                cu_seqlens.append(
                    torch.arange(
                        _offset_pad[i],
                        _offset_pad[i + 1],
                        step=self.patch_size,
                        dtype=torch.int32,
                        device=offset.device,
                    )
                )
            point[pad_key] = pad
            point[unpad_key] = unpad
            point[cu_seqlens_key] = nn.functional.pad(
                torch.concat(cu_seqlens), (0, 1), value=_offset_pad[-1]
            )
            # None whenever every patch is full: keeps the common path mask-free
            point[valid_key] = (
                None
                if all(v == self.patch_size for v in patch_valid)
                else torch.tensor(patch_valid, device=offset.device)
            )
        return (
            point[pad_key],
            point[unpad_key],
            point[cu_seqlens_key],
            point[valid_key],
        )

    def forward(self, point):
        if not self.enable_flash:
            # [le-wm vendored patch #2] Upstream shrank patch_size to the
            # SMALLEST cloud in the batch here, which made every cloud's
            # attention window (and features) depend on its batch-mates --
            # measured up to 56% relative on this data. The padding above now
            # handles sub-patch clouds, so the configured size can be used
            # unconditionally, which is also what the flash path (the one the
            # released checkpoint was trained with) does.
            self.patch_size = self.patch_size_max

        H = self.num_heads
        K = self.patch_size
        C = self.channels

        pad, unpad, cu_seqlens, patch_valid = self.get_padding_and_inverse(point)

        order = point.serialized_order[self.order_index][pad]
        inverse = unpad[point.serialized_inverse[self.order_index]]

        # padding and reshape feat and batch for serialized point patch
        qkv = self.qkv(point.feat)[order]

        rope_coord = point.coord[order].clone()

        qkv = qkv.reshape(-1, 3, H, C // H)
        q, k, v = qkv.unbind(dim=1)
        q, k = self.rope(q, k, rope_coord)

        if not self.enable_flash:
            q = q.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
            k = k.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)
            v = v.reshape(-1, K, H, C // H).permute(0, 2, 1, 3)

            if not self.enable_rpe:
                # [le-wm vendored patch, see utonia/README.md] The upstream
                # non-flash path materializes the dense (N', H, K, K) attention
                # matrix -- tens of GB at K=1024 on batched clouds. Same math
                # through torch's memory-efficient SDPA kernel instead; only
                # the RPE variant (which must add a bias onto the dense matrix)
                # keeps the original path. The released Utonia checkpoint runs
                # with enable_rpe=False, so this is the branch it takes.
                # Utonia's head_dim is 18 (channels/heads); the efficient SDPA
                # backends need a multiple of 8, else torch silently falls back
                # to the dense math backend. Zero-padding q/k/v to the next
                # multiple of 8 is exact: zero dims add nothing to q.k, and the
                # padded v dims are sliced back off the output.
                head_dim = C // H
                pad_to = -(-head_dim // 8) * 8
                if pad_to != head_dim:
                    padding = (0, pad_to - head_dim)
                    q = torch.nn.functional.pad(q, padding)
                    k = torch.nn.functional.pad(k, padding)
                    v = torch.nn.functional.pad(v, padding)
                # [le-wm vendored patch #2] mask the padding of any cloud small
                # enough to fit in one patch (see get_padding_and_inverse); None
                # -- the common case -- keeps the fast unmasked kernel.
                attn_mask = None
                if patch_valid is not None:
                    attn_mask = (
                        torch.arange(K, device=q.device)[None, :] < patch_valid[:, None]
                    )[:, None, None, :]  # (patches, 1, 1, K), broadcast over heads/queries
                feat = torch.nn.functional.scaled_dot_product_attention(
                    q,
                    k,
                    v,
                    attn_mask=attn_mask,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    scale=self.scale,
                )
                feat = feat[..., :head_dim].transpose(1, 2).reshape(-1, C)
            else:
                # [le-wm vendored patch #2] this branch adds an RPE bias onto the
                # dense matrix and has no masking, so it cannot express a
                # sub-patch cloud. The released checkpoint uses enable_rpe=False.
                assert patch_valid is None, (
                    "enable_rpe=True cannot mask sub-patch clouds; encode clouds "
                    "with at least patch_size points per stage, or use the "
                    "enable_rpe=False path"
                )
                # attn
                if self.upcast_attention:
                    q = q.float()
                    k = k.float()
                attn = (q * self.scale) @ k.transpose(-2, -1)  # (N', H, K, K)
                attn = attn + self.rpe(self.get_rel_pos(point, order))
                if self.upcast_softmax:
                    attn = attn.float()
                attn = self.softmax(attn)
                attn = self.attn_drop(attn).to(qkv.dtype)
                feat = (attn @ v).transpose(1, 2).reshape(-1, C)
        else:
            qkv_roped = torch.stack([q, k, v], dim=1)
            feat = flash_attn.flash_attn_varlen_qkvpacked_func(
                qkv_roped.to(torch.bfloat16),
                cu_seqlens,
                max_seqlen=self.patch_size,
                dropout_p=self.attn_drop if self.training else 0,
                softmax_scale=self.scale,
            ).reshape(-1, C)
            feat = feat.to(qkv.dtype)
        feat = feat[inverse]

        # ffn
        feat = self.proj(feat)
        feat = self.proj_drop(feat)
        point.feat = feat
        return point


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        hidden_channels=None,
        out_channels=None,
        act_layer=nn.GELU,
        drop=0.0,
    ):
        super().__init__()
        out_channels = out_channels or in_channels
        hidden_channels = hidden_channels or in_channels
        self.fc1 = nn.Linear(in_channels, hidden_channels)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_channels, out_channels)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Block(PointModule):
    def __init__(
        self,
        channels,
        num_heads,
        patch_size=48,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.0,
        layer_scale=None,
        norm_layer=nn.LayerNorm,
        act_layer=nn.GELU,
        pre_norm=True,
        order_index=0,
        cpe_indice_key=None,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=True,
        upcast_softmax=True,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        self.channels = channels
        self.pre_norm = pre_norm

        self.cpe = PointSequential(
            spconv.SubMConv3d(
                channels,
                channels,
                kernel_size=3,
                bias=True,
                indice_key=cpe_indice_key,
            ),
            nn.Linear(channels, channels),
            norm_layer(channels),
        )

        self.norm1 = PointSequential(norm_layer(channels))
        self.ls1 = PointSequential(
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.attn = SerializedAttention(
            channels=channels,
            patch_size=patch_size,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            order_index=order_index,
            enable_rpe=enable_rpe,
            enable_flash=enable_flash,
            upcast_attention=upcast_attention,
            upcast_softmax=upcast_softmax,
            rope_base=rope_base,
            shift_coords=shift_coords,
            jitter_coords=jitter_coords,
            rescale_coords=rescale_coords,
        )
        self.norm2 = PointSequential(norm_layer(channels))
        self.ls2 = PointSequential(
            LayerScale(channels, init_values=layer_scale)
            if layer_scale is not None
            else nn.Identity()
        )
        self.mlp = PointSequential(
            MLP(
                in_channels=channels,
                hidden_channels=int(channels * mlp_ratio),
                out_channels=channels,
                act_layer=act_layer,
                drop=proj_drop,
            )
        )
        self.drop_path = PointSequential(
            DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        )

    def forward(self, point: Point):
        shortcut = point.feat
        point = self.cpe(point)
        point.feat = shortcut + point.feat
        shortcut = point.feat
        if self.pre_norm:
            point = self.norm1(point)
        point = self.drop_path(self.ls1(self.attn(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm1(point)

        shortcut = point.feat
        if self.pre_norm:
            point = self.norm2(point)
        point = self.drop_path(self.ls2(self.mlp(point)))
        point.feat = shortcut + point.feat
        if not self.pre_norm:
            point = self.norm2(point)
        point.sparse_conv_feat = point.sparse_conv_feat.replace_feature(point.feat)
        return point


class GridPooling(PointModule):
    def __init__(
        self,
        in_channels,
        out_channels,
        stride=2,
        norm_layer=None,
        act_layer=None,
        reduce="max",
        shuffle_orders=True,
        traceable=True,  # record parent and cluster
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.stride = stride
        assert reduce in ["sum", "mean", "min", "max"]
        self.reduce = reduce
        self.shuffle_orders = shuffle_orders
        self.traceable = traceable

        self.proj = nn.Linear(in_channels, out_channels)
        if norm_layer is not None:
            self.norm = PointSequential(norm_layer(out_channels))
        if act_layer is not None:
            self.act = PointSequential(act_layer())

    def forward(self, point: Point):
        if "grid_coord" in point.keys():
            grid_coord = point.grid_coord
        elif {"coord", "grid_size"}.issubset(point.keys()):
            grid_coord = torch.div(
                point.coord - point.coord.min(0)[0],
                point.grid_size,
                rounding_mode="trunc",
            ).int()
        else:
            raise AssertionError(
                "[gird_coord] or [coord, grid_size] should be include in the Point"
            )
        grid_coord = torch.div(grid_coord, self.stride, rounding_mode="trunc")
        grid_coord = grid_coord | point.batch.view(-1, 1) << 48
        grid_coord, cluster, counts = torch.unique(
            grid_coord,
            sorted=True,
            return_inverse=True,
            return_counts=True,
            dim=0,
        )
        grid_coord = grid_coord & ((1 << 48) - 1)
        # indices of point sorted by cluster, for torch_scatter.segment_csr
        _, indices = torch.sort(cluster)
        # index pointer for sorted point, for torch_scatter.segment_csr
        idx_ptr = torch.cat([counts.new_zeros(1), torch.cumsum(counts, dim=0)])
        # head_indices of each cluster, for reduce attr e.g. code, batch
        head_indices = indices[idx_ptr[:-1]]
        point_dict = Dict(
            feat=torch_scatter.segment_csr(
                self.proj(point.feat)[indices], idx_ptr, reduce=self.reduce
            ),
            coord=torch_scatter.segment_csr(
                point.coord[indices], idx_ptr, reduce="mean"
            ),
            grid_coord=grid_coord,
            batch=point.batch[head_indices],
        )
        if "origin_coord" in point.keys():
            point_dict["origin_coord"] = torch_scatter.segment_csr(
                point.origin_coord[indices], idx_ptr, reduce="mean"
            )
        if "condition" in point.keys():
            point_dict["condition"] = point.condition
        if "context" in point.keys():
            point_dict["context"] = point.context
        if "name" in point.keys():
            point_dict["name"] = point.name
        if "split" in point.keys():
            point_dict["split"] = point.split
        if "color" in point.keys():
            point_dict["color"] = torch_scatter.segment_csr(
                point.color[indices], idx_ptr, reduce="mean"
            )
        if "grid_size" in point.keys():
            point_dict["grid_size"] = point.grid_size * self.stride

        if self.traceable:
            point_dict["pooling_inverse"] = cluster
            point_dict["pooling_parent"] = point
            point_dict["idx_ptr"] = idx_ptr
        order = point.order
        point = Point(point_dict)
        if self.norm is not None:
            point = self.norm(point)
        if self.act is not None:
            point = self.act(point)
        point.serialization(order=order, shuffle_orders=self.shuffle_orders)
        point.sparsify()
        return point


class GridUnpooling(PointModule):
    def __init__(
        self,
        in_channels,
        skip_channels,
        out_channels,
        norm_layer=None,
        act_layer=None,
        traceable=False,  # record parent and cluster
    ):
        super().__init__()
        self.proj = PointSequential(nn.Linear(in_channels, out_channels))
        self.proj_skip = PointSequential(nn.Linear(skip_channels, out_channels))

        if norm_layer is not None:
            self.proj.add(norm_layer(out_channels))
            self.proj_skip.add(norm_layer(out_channels))

        if act_layer is not None:
            self.proj.add(act_layer())
            self.proj_skip.add(act_layer())

        self.traceable = traceable

    def forward(self, point):
        assert "pooling_parent" in point.keys()
        assert "pooling_inverse" in point.keys()
        parent = point.pop("pooling_parent")
        inverse = point.pooling_inverse
        feat = point.feat

        parent = self.proj_skip(parent)
        parent.feat = parent.feat + self.proj(point).feat[inverse]
        parent.sparse_conv_feat = parent.sparse_conv_feat.replace_feature(parent.feat)

        if self.traceable:
            point.feat = feat
            parent["unpooling_parent"] = point
        return parent


class Embedding(PointModule):
    def __init__(
        self,
        in_channels,
        embed_channels,
        norm_layer=None,
        act_layer=None,
        mask_token=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.embed_channels = embed_channels

        self.stem = PointSequential(linear=nn.Linear(in_channels, embed_channels))
        if norm_layer is not None:
            self.stem.add(norm_layer(embed_channels), name="norm")
        if act_layer is not None:
            self.stem.add(act_layer(), name="act")

        if mask_token:
            self.mask_token = nn.Parameter(torch.zeros(1, embed_channels))
        else:
            self.mask_token = None

    def forward(self, point: Point):
        point = self.stem(point)
        if "mask" in point.keys():
            point.feat = torch.where(
                point.mask.unsqueeze(-1),
                self.mask_token.to(point.feat.dtype),
                point.feat,
            )
        return point


class PointTransformerV3(PointModule, PyTorchModelHubMixin):
    def __init__(
        self,
        in_channels=6,
        order=("z", "z-trans"),
        stride=(2, 2, 2, 2),
        enc_depths=(3, 3, 3, 12, 3),
        enc_channels=(48, 96, 192, 384, 512),
        enc_num_head=(3, 6, 12, 24, 32),
        enc_patch_size=(1024, 1024, 1024, 1024, 1024),
        dec_depths=(3, 3, 3, 3),
        dec_channels=(96, 96, 192, 384),
        dec_num_head=(6, 6, 12, 32),
        dec_patch_size=(1024, 1024, 1024, 1024),
        mlp_ratio=4,
        qkv_bias=True,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        drop_path=0.3,
        layer_scale=None,
        pre_norm=True,
        shuffle_orders=True,
        enable_rpe=False,
        enable_flash=True,
        upcast_attention=False,
        upcast_softmax=False,
        traceable=False,
        mask_token=False,
        enc_mode=False,
        freeze_encoder=False,
        rope_base=10,
        shift_coords=None,
        jitter_coords=None,
        rescale_coords=None,
    ):
        super().__init__()
        self.num_stages = len(enc_depths)
        self.order = [order] if isinstance(order, str) else order
        self.enc_mode = enc_mode
        self.shuffle_orders = shuffle_orders
        self.freeze_encoder = freeze_encoder

        assert self.num_stages == len(stride) + 1
        assert self.num_stages == len(enc_depths)
        assert self.num_stages == len(enc_channels)
        assert self.num_stages == len(enc_num_head)
        assert self.num_stages == len(enc_patch_size)
        assert self.enc_mode or self.num_stages == len(dec_depths) + 1
        assert self.enc_mode or self.num_stages == len(dec_channels) + 1
        assert self.enc_mode or self.num_stages == len(dec_num_head) + 1
        assert self.enc_mode or self.num_stages == len(dec_patch_size) + 1

        # normalization layer
        ln_layer = nn.LayerNorm
        # activation layers
        act_layer = nn.GELU

        self.embedding = Embedding(
            in_channels=in_channels,
            embed_channels=enc_channels[0],
            norm_layer=ln_layer,
            act_layer=act_layer,
            mask_token=mask_token,
        )

        # encoder
        enc_drop_path = [
            x.item() for x in torch.linspace(0, drop_path, sum(enc_depths))
        ]
        self.enc = PointSequential()
        for s in range(self.num_stages):
            enc_drop_path_ = enc_drop_path[
                sum(enc_depths[:s]) : sum(enc_depths[: s + 1])
            ]
            enc = PointSequential()
            if s > 0:
                enc.add(
                    GridPooling(
                        in_channels=enc_channels[s - 1],
                        out_channels=enc_channels[s],
                        stride=stride[s - 1],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                    ),
                    name="down",
                )
            for i in range(enc_depths[s]):
                enc.add(
                    Block(
                        channels=enc_channels[s],
                        num_heads=enc_num_head[s],
                        patch_size=enc_patch_size[s],
                        mlp_ratio=mlp_ratio,
                        qkv_bias=qkv_bias,
                        qk_scale=qk_scale,
                        attn_drop=attn_drop,
                        proj_drop=proj_drop,
                        drop_path=enc_drop_path_[i],
                        layer_scale=layer_scale,
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        pre_norm=pre_norm,
                        order_index=i % len(self.order),
                        cpe_indice_key=f"stage{s}",
                        enable_rpe=enable_rpe,
                        enable_flash=enable_flash,
                        upcast_attention=upcast_attention,
                        upcast_softmax=upcast_softmax,
                        rope_base=rope_base,
                        shift_coords=shift_coords,
                        jitter_coords=jitter_coords,
                        rescale_coords=rescale_coords,
                    ),
                    name=f"block{i}",
                )
            if len(enc) != 0:
                self.enc.add(module=enc, name=f"enc{s}")

        # decoder
        if not self.enc_mode:
            dec_drop_path = [
                x.item() for x in torch.linspace(0, drop_path, sum(dec_depths))
            ]
            self.dec = PointSequential()
            dec_channels = list(dec_channels) + [enc_channels[-1]]
            for s in reversed(range(self.num_stages - 1)):
                dec_drop_path_ = dec_drop_path[
                    sum(dec_depths[:s]) : sum(dec_depths[: s + 1])
                ]
                dec_drop_path_.reverse()
                dec = PointSequential()
                dec.add(
                    GridUnpooling(
                        in_channels=dec_channels[s + 1],
                        skip_channels=enc_channels[s],
                        out_channels=dec_channels[s],
                        norm_layer=ln_layer,
                        act_layer=act_layer,
                        traceable=traceable,
                    ),
                    name="up",
                )
                for i in range(dec_depths[s]):
                    dec.add(
                        Block(
                            channels=dec_channels[s],
                            num_heads=dec_num_head[s],
                            patch_size=dec_patch_size[s],
                            mlp_ratio=mlp_ratio,
                            qkv_bias=qkv_bias,
                            qk_scale=qk_scale,
                            attn_drop=attn_drop,
                            proj_drop=proj_drop,
                            drop_path=dec_drop_path_[i],
                            layer_scale=layer_scale,
                            norm_layer=ln_layer,
                            act_layer=act_layer,
                            pre_norm=pre_norm,
                            order_index=i % len(self.order),
                            cpe_indice_key=f"stage{s}",
                            enable_rpe=enable_rpe,
                            enable_flash=enable_flash,
                            upcast_attention=upcast_attention,
                            upcast_softmax=upcast_softmax,
                            rope_base=rope_base,
                            shift_coords=shift_coords,
                            jitter_coords=jitter_coords,
                            rescale_coords=rescale_coords,
                        ),
                        name=f"block{i}",
                    )
                self.dec.add(module=dec, name=f"dec{s}")
        if self.freeze_encoder:
            for p in self.embedding.parameters():
                p.requires_grad = False
            for p in self.enc.parameters():
                p.requires_grad = False
        self.apply(self._init_weights)

    @staticmethod
    def _init_weights(module):
        if isinstance(module, nn.Linear):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, spconv.SubMConv3d):
            trunc_normal_(module.weight, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, data_dict):
        point = Point(data_dict)
        point = self.embedding(point)

        point.serialization(order=self.order, shuffle_orders=self.shuffle_orders)
        point.sparsify()

        point = self.enc(point)
        if not self.enc_mode:
            point = self.dec(point)
        return point


def load(
    name: str = "utonia",
    repo_id="Pointcept/utonia",
    download_root: str = None,
    custom_config: dict = None,
    ckpt_only: bool = False,
):
    if name in MODELS:
        print(f"Loading checkpoint from HuggingFace: {name} ...")
        ckpt_path = hf_hub_download(
            repo_id=repo_id,
            filename=f"{name}.pth",
            repo_type="model",
            revision="main",
            local_dir=download_root or os.path.expanduser("~/.cache/utonia/ckpt"),
        )
    elif os.path.isfile(name):
        print(f"Loading checkpoint in local path: {name} ...")
        ckpt_path = name
    else:
        raise RuntimeError(f"Model {name} not found; available models = {MODELS}")

    if version.parse(torch.__version__) >= version.parse("2.4"):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    else:
        ckpt = torch.load(ckpt_path, map_location="cpu")
    if custom_config is not None:
        for key, value in custom_config.items():
            ckpt["config"][key] = value

    if ckpt_only:
        return ckpt

    model = PointTransformerV3(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model params: {n_parameters / 1e6:.2f}M")
    return model
