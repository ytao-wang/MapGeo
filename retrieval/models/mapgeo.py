import os
import torch
import torch.nn as nn
import math
import timm
import numpy as np
from timm.layers import get_norm_layer
from retrieval.models.DinoExtractor import DinoFeatureExtractor
import torch.nn.functional as F

from functools import partial
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from einops import repeat
from typing import Dict, Literal, Tuple, Optional, Callable
from functools import partial
from torchvision.transforms.functional import rotate
from torchvision.transforms import InterpolationMode
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

tensor_rotate = partial(rotate,interpolation=InterpolationMode.BILINEAR)
ViewType = Literal["grd", "sat"]


class NormMlpClassifierHead(nn.Module):
    def __init__(self, in_features: int, pool_type: str = 'avg',  drop_rate: float = 0.,):
        super().__init__()
        self.global_pool = nn.AdaptiveAvgPool2d(output_size=1)
        norm_layer = get_norm_layer('layernorm2d')
        self.norm = norm_layer(in_features)
        self.flatten = nn.Flatten(1) if pool_type else nn.Identity()
        self.drop = nn.Dropout(drop_rate)

    def forward(self, x):
        x = self.global_pool(x)
        x = self.norm(x)
        x = self.flatten(x)
        x = self.drop(x)
        return x

class SS2D(nn.Module):
    def __init__(self, d_model: int, d_state: int = 16, d_conv: int = 3,
        ssm_ratio: float = 2.0, dt_rank: "int | str" = "auto",
        dropout: float = 0.0, conv_bias: bool = True, bias: bool = False,
        dt_min: float = 0.001, dt_max: float = 0.1,
        dt_init: str = "random", dt_scale: float = 1.0, dt_init_floor: float = 1e-4,
        is_pos: bool = False, view_type: str = 'grd',):
        super().__init__()
        self.is_pos = is_pos
        self.view_type = view_type

        self.d_model = d_model
        self.d_state = math.ceil(d_model / 6) if d_state == "auto" else d_state
        self.d_inner = int(ssm_ratio * d_model)
        self.dt_rank = math.ceil(d_model / 16) if dt_rank == "auto" else dt_rank
        self.K = 2  # forward scan + reverse scan

        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=bias)
        self.d_conv = d_conv
        self.dwconv = nn.Conv2d(self.d_inner, self.d_inner, kernel_size=d_conv, padding=0, groups=self.d_inner, bias=conv_bias,)

        self.act = nn.SiLU()

        x_proj = [nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False) for _ in range(self.K)]
        self.x_proj_weight = nn.Parameter(torch.stack([m.weight for m in x_proj], dim=0))
        del x_proj

        dt_projs = [self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor) for _ in range(self.K)]
        self.dt_projs_weight = nn.Parameter(torch.stack([m.weight for m in dt_projs], dim=0))
        self.dt_projs_bias = nn.Parameter(torch.stack([m.bias for m in dt_projs], dim=0))
        del dt_projs

        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K, merge=True)
        self.Ds = self.D_init(self.d_inner, copies=self.K, merge=True)

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()
        self.selective_scan = selective_scan_fn

        # Scan permutations depend only on view type, H, W, and device.
        # Cache them so each resolution/device builds its index once, while keeping
        # the tensors outside state_dict/checkpoint serialization.
        self._scan_index_cache: Dict[Tuple[str, int, int, str, int], torch.Tensor] = {}

    @staticmethod
    def _sincos_pos(H, W, C, device, dtype):
        """2D sine-cosine positional encoding in original grid-flattened order: [H*W, C]."""
        yy, xx = torch.meshgrid(
            torch.linspace(0, 1, H, device=device, dtype=torch.float32),
            torch.linspace(0, 1, W, device=device, dtype=torch.float32),
            indexing="ij",)
        dim_each = max(C // 4, 1)
        inv_freq = 1.0 / (10000 ** (torch.arange(dim_each, device=device).float() / dim_each))

        px = xx[..., None] * inv_freq
        py = yy[..., None] * inv_freq
        pos = torch.cat([px.sin(), px.cos(), py.sin(), py.cos()], dim=-1).reshape(H * W, -1)

        if pos.shape[-1] > C:
            pos = pos[:, :C]
        elif pos.shape[-1] < C:
            pos = torch.cat([pos, torch.zeros(H * W, C - pos.shape[-1], device=device)], dim=-1)
        return pos.to(dtype=dtype)

    @staticmethod
    def _ground_scan_index(H, W, device):
        """
        Ground forward scan index.
        Order:
            bottom row -> top row; inside each row: left -> right.
        Returns:
            scan_idx: [H*W], where x_scan[:, t] = x_flat[:, scan_idx[t]].
        """
        rows = torch.arange(H - 1, -1, -1, device=device, dtype=torch.long)
        cols = torch.arange(0, W, device=device, dtype=torch.long)
        yy = rows[:, None].expand(H, W)
        xx = cols[None, :].expand(H, W)
        return (yy * W + xx).reshape(-1).contiguous()

    @staticmethod
    def _satellite_scan_index(H, W, device):
        """
        Satellite forward scan index.
        Order:
            center -> outer square Chebyshev rings;
            every ring starts at the north point and moves clockwise.
        For strict physical interpretation, the satellite feature grid must be
        odd and square, e.g. 27 x 27, so that one exact center token exists.
        """
        if H != W:
            raise ValueError(f"Satellite square-ring scan requires H == W, got H={H}, W={W}.")
        if H % 2 == 0:
            raise ValueError(f"Satellite square-ring scan requires odd H and W, got {H}x{W}.")

        N = H
        c = N // 2
        order = [c * W + c]  # k = 0, exact center

        for k in range(1, c + 1):
            # Start from the north point: (c-k, c), then clockwise.
            u = c - k
            for v in range(c, c + k + 1):
                order.append(u * W + v)

            # East edge: northeast -> southeast, excluding northeast.
            v = c + k
            for u in range(c - k + 1, c + k + 1):
                order.append(u * W + v)

            # South edge: southeast -> southwest, excluding southeast.
            u = c + k
            for v in range(c + k - 1, c - k - 1, -1):
                order.append(u * W + v)

            # West edge: southwest -> northwest, excluding southwest.
            v = c - k
            for u in range(c + k - 1, c - k - 1, -1):
                order.append(u * W + v)

            # Top edge remainder: northwest -> before north point.
            u = c - k
            for v in range(c - k + 1, c):
                order.append(u * W + v)

        scan_idx = torch.tensor(order, device=device, dtype=torch.long)
        L = H * W
        if scan_idx.numel() != L:
            raise RuntimeError(f"square-ring scan length {scan_idx.numel()} != H*W={L}.")
        if torch.unique(scan_idx).numel() != L:
            raise RuntimeError("square-ring scan_idx is not a permutation.")
        return scan_idx.contiguous()

    @classmethod
    def build_scan_index(cls, view_type, H, W, device):
        """Build the full-image forward scan index for ground or satellite."""
        if view_type == "grd":
            return cls._ground_scan_index(H, W, device)
        if view_type == "sat":
            return cls._satellite_scan_index(H, W, device)
        raise ValueError(f"view_type must be 'grd' or 'sat', got {view_type}.")

    def _get_scan_index(self, view_type, H, W, device):
        """Return a cached full-image forward scan index.
        The scan order is deterministic for a given (view_type, H, W, device).
        Caching avoids repeatedly rebuilding the ground/satellite permutation in
        every forward pass. The cached tensor is a non-persistent runtime cache;
        it is not part of the model parameters or buffers.
        """
        device_index = -1 if device.index is None else int(device.index)
        key = (str(view_type), int(H), int(W), device.type, device_index)
        scan_idx = self._scan_index_cache.get(key)
        if scan_idx is None:
            scan_idx = self.build_scan_index(view_type, H, W, device).contiguous()
            self._scan_index_cache[key] = scan_idx
        return scan_idx

    def clear_scan_index_cache(self):
        """Clear cached scan permutations, useful after unusual device changes."""
        self._scan_index_cache.clear()
    
    @staticmethod
    def dt_init(dt_rank: int,
        d_inner: int,
        dt_scale: float = 1.0,
        dt_init: str = "random",
        dt_min: float = 0.001,
        dt_max: float = 0.1,
        dt_init_floor: float = 1e-4,):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True)
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise ValueError(f"Unsupported dt_init: {dt_init}")

        dt = torch.exp(torch.rand(d_inner) * (math.log(dt_max) - math.log(dt_min)) + math.log(dt_min)).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        dt_proj.bias._no_reinit = True
        return dt_proj

    @staticmethod
    def A_log_init(d_state: int, d_inner: int, copies: int = 1, merge: bool = True):
        A = repeat(torch.arange(1, d_state + 1, dtype=torch.float32), "n -> d n", d=d_inner).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner: int, copies: int = 1, merge: bool = True):
        D = torch.ones(d_inner)
        if copies > 1:
            D = repeat(D, "d -> r d", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_core(self, x, scan_idx, pos=None):
        """
        Args:
            x:        [B, H, W, D], convolution-enhanced inner feature map.
            scan_idx: [L], forward scan permutation in original flattened grid indices.
            pos:      [L, D], positional encoding in original flattened grid order.
        Returns:
            y: [B, H, W, D], restored to original grid coordinates.
        The core equations are:
            X_scan[:, t, :] = X_flat[:, scan_idx[t], :]
            Y_scan = BiSSM(X_scan, Reverse(X_scan))
            Y_flat[:, scan_idx[t], :] = Y_scan[:, t, :]
        """
        if x.dim() != 4:
            raise ValueError(f"x must be [B, H, W, D], got {tuple(x.shape)}.")

        B, H, W, D = x.shape
        L = H * W
        if scan_idx.numel() != L:
            raise ValueError(f"scan_idx length {scan_idx.numel()} does not match H*W={L}.")

        x_flat = x.reshape(B, L, D).contiguous()

        # Original grid -> geometry scan order. gather is optimal for read/reorder.
        gather_idx = scan_idx.view(1, L, 1).expand(B, L, D)
        x_scan = torch.gather(x_flat, dim=1, index=gather_idx)
        if pos is not None:
            pos = pos.to(device=x.device, dtype=x.dtype)
            x_scan = x_scan + pos[scan_idx].unsqueeze(0)

        # Build xs_scan once, exactly like the standard SS2D style:
        # [B, 2, D, L] -> [B, 2D, L], where dim 1 is forward and reverse.
        x_fwd = x_scan.transpose(1, 2).contiguous()  # [B, D, L]
        x_rev = torch.flip(x_fwd, dims=[-1])         # [B, D, L]
        xs = torch.stack([x_fwd, x_rev], dim=1)      # [B, K=2, D, L]

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, self.x_proj_weight)
        dts_low, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts_low, self.dt_projs_weight)

        xs_scan = xs.float().view(B, -1, L)                    # [B, K*D, L]
        dts_scan = dts.contiguous().float().view(B, -1, L)     # [B, K*D, L]
        Bs = Bs.float().view(B, self.K, -1, L)                 # [B, K, d_state, L]
        Cs = Cs.float().view(B, self.K, -1, L)                 # [B, K, d_state, L]
        Ds = self.Ds.float().view(-1)                          # [K*D]
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)
        dt_bias = self.dt_projs_bias.float().view(-1)

        out_y = self.selective_scan(
            xs_scan, dts_scan,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, self.K, D, L)
        assert out_y.dtype == torch.float

        # Reverse branch is flipped back, so both outputs are aligned to forward scan_idx order.
        y_fwd = out_y[:, 0]                         # [B, D, L]
        y_rev = torch.flip(out_y[:, 1], dims=[-1])  # [B, D, L]
        y_scan = (y_fwd + y_rev).transpose(1, 2).contiguous()  # [B, L, D]

        # Geometry scan order -> original grid. scatter_ directly implements
        # y_flat[:, scan_idx[t], :] = y_scan[:, t, :].
        # print('yyy', x_flat.dtype, y_scan.dtype)
        y_flat =  torch.zeros(B, L, D,device=y_scan.device,dtype=y_scan.dtype)
        y_flat.scatter_(dim=1, index=gather_idx, src=y_scan)

        return y_flat.view(B, H, W, D).contiguous()

    def _dwconv_pad(self, x):
        p = (self.d_conv - 1) // 2
        if p == 0:
            return x

        if self.view_type == "grd":
            x = F.pad(x, (p, p, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, p, p), mode="constant", value=0)
        else:
            x = F.pad(x, (p, p, p, p), mode="constant", value=0)

        return x

    def forward(self, x):
        """
        Args:
            x: [B, H, W, D]
            view_type: "grd" or "sat"
        Returns:
            [B, H, W, D]
        """
        if x.dim() != 4:
            raise ValueError(f"x must be [B, H, W, D], got {tuple(x.shape)}.")

        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x_inner, z = xz.chunk(2, dim=-1)

        x_inner = x_inner.permute(0, 3, 1, 2).contiguous()
        x_inner = self._dwconv_pad(x_inner)
        x_inner = self.act(self.dwconv(x_inner)).permute(0, 2, 3, 1).contiguous()

        pos = self._sincos_pos(H, W, self.d_inner, x.device, x_inner.dtype) if self.is_pos else None
        scan_idx = self._get_scan_index(self.view_type, H, W, x.device)
        y = self.forward_core(x_inner, scan_idx=scan_idx, pos=pos)

        y = self.out_norm(y)
        y = y * F.silu(z)
        y = self.out_proj(y)
        y = self.dropout(y)
        return y


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class VSSBscan(nn.Module):
    def __init__( self, input_dim: int = 768, dropout: float = 0, d_state: int = 16, ssm_ratio: float = 2., is_pos: bool = True, view_type: str = 'grd',
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6)):
        super().__init__()

        self.ln_1 = norm_layer(input_dim)
        self.ss2d = SS2D(d_model=input_dim, d_state=d_state, ssm_ratio=ssm_ratio,dropout=dropout, is_pos=is_pos, view_type=view_type)
        self.drop_path = DropPath(dropout)
        self.skip_scale= 1.0 # nn.Parameter(torch.ones(input_dim))

        self.ln_2 = norm_layer(input_dim)
        self.skip_scale2 = 1.0 # nn.Parameter(torch.ones(input_dim))

        self.pos_drop = nn.Dropout(p=dropout)

        mlp_hidden_dim = int(input_dim * ssm_ratio)
        self.mlp = Mlp(in_features=input_dim, hidden_features=mlp_hidden_dim, drop=dropout)

    def forward(self, input):
        # x [B, H, W, C]
        input = self.pos_drop(input) # [B,H,W,C]

        x = self.ln_1(input) 
        x_ss2d = self.drop_path(self.ss2d(x)) # [B,H,W,C]

        x = input*self.skip_scale + x_ss2d
        x = x*self.skip_scale2 + self.drop_path(self.mlp(self.ln_2(x)))
      
        return x


class MSLR(nn.Module):
    """
    Multi-Scale Local Relation Adapter
    Input / Output: [B, C, H, W]
    """
    def __init__(self, dim, reduction=4, pano=False):
        super().__init__()

        self.pano = pano
        hidden = dim // reduction
        self.reduce = nn.Conv2d(dim, hidden, 1, padding=0, bias=False)

        self.local3 = nn.Conv2d(hidden, hidden, 3, padding=0, groups=hidden, bias=False)
        self.local5 = nn.Conv2d(hidden, hidden, 3, dilation=2, groups=hidden, bias=False)

        self.fuse = nn.Conv2d(hidden * 3, dim, 1, bias=False)

        self.act = nn.GELU()
        self.alpha = nn.Parameter(torch.zeros(1))

    def _pad(self, x, p):
        if self.pano:
            # panorama: horizontal circular topology
            x = F.pad(x, (p, p, 0, 0), mode="circular")
            x = F.pad(x, (0, 0, p, p), mode="constant", value=0)
        else:
            x = F.pad(x, (p, p, p, p), mode="constant", value=0)
        return x

    def _local_diff(self, x, k):
        avg = F.avg_pool2d(self._pad(x, k // 2), kernel_size=k, stride=1)
        return x - avg

    def forward(self, x):
        u = self.act(self.reduce(x))

        # multi-scale relative relations
        d3 = self._local_diff(u, 3)
        d5 = self._local_diff(u, 5)

        # scale-specific local modeling
        r3 = self.act(self.local3(self._pad(d3, 1)))
        r5 = self.act(self.local5(self._pad(d5, 2)))

        # preserve original feature + two relation scales
        s = torch.cat([u, r3, r5], dim=1)

        s = self.fuse(s)

        return x + self.alpha * s

    
class PMC(nn.Module):
    """
    Partial Matching Constraint
    """
    def __init__(self, dim, proj_dim=768, topk_grd=64, topk_ref=64, match_temp=0.1, loss_temp=0.07, proj=True):
        super().__init__()
        self.topk_grd = topk_grd
        self.topk_ref = topk_ref
        self.match_temp = match_temp
        self.loss_temp = loss_temp

        self.grd_proj = nn.Linear(dim, proj_dim, bias=False) if proj else nn.Identity()
        self.ref_proj = nn.Linear(dim, proj_dim, bias=False) if proj else nn.Identity()

    def _project(self, x, proj):
        return F.normalize(proj(x), dim=-1)

    def _partial_score(self, x, y, kx, ky):
        sim = torch.einsum('ind,jmd->ijnm', x, y)
 
        sim_t = sim / self.match_temp
        conf = F.softmax(sim_t, dim=-1) * F.softmax(sim_t, dim=-2)

        # x -> y
        row_conf, row_idx = conf.max(dim=-1)
        row_sim = sim.gather(-1, row_idx.unsqueeze(-1)).squeeze(-1)
        kx = min(kx, x.shape[1])
        conf_x, idx_x = row_conf.topk(kx, dim=-1)
        sim_x = row_sim.gather(-1, idx_x)
        weight_x = conf_x / (conf_x.sum(-1, keepdim=True) + 1e-6)
        score_x = (weight_x * sim_x).sum(-1)

        # y -> x
        col_conf, col_idx = conf.max(dim=-2)
        col_sim = sim.transpose(-1, -2).gather(-1, col_idx.unsqueeze(-1)).squeeze(-1)
        ky = min(ky, y.shape[1])
        conf_y, idx_y = col_conf.topk(ky, dim=-1)
        sim_y = col_sim.gather(-1, idx_y)
        weight_y = conf_y / (conf_y.sum(-1, keepdim=True) + 1e-6)
        score_y = (weight_y * sim_y).sum(-1)

        return 0.5 * (score_x + score_y)


    def forward(self, grd_patch, ref_patch, grd_desc=None, ref_desc=None):
        B = grd_patch.shape[0]
        g = self._project(grd_patch, self.grd_proj)
        r = self._project(ref_patch, self.ref_proj)

        scores = self._partial_score(g, r, self.topk_grd, self.topk_ref)

        logits_gr = scores / self.loss_temp
        logits_rg = scores.T / self.loss_temp
        target = torch.arange(B, dtype=torch.long, device=g.device)
        return logits_gr, logits_rg, target


class MapGeo(nn.Module):
    def __init__(self, config, model_name, sat_size, grd_size, return_layers=None, max_layer=12, freeze_until=3, final_norm=False):
        super().__init__()
        drop=0.3
        self.return_layers = return_layers

        self.patch = 16 if 'dinov3' in model_name else 14
        self.bk_dim = 768 if 'dino' in model_name else 1024

        self.h_sat, self.w_sat = sat_size[0] // self.patch, sat_size[1] // self.patch
        self.h_grd, self.w_grd = grd_size[0] // self.patch, grd_size[1] // self.patch

        ckpt = os.path.join('/home/wangyuntao/dinov2-models', model_name)
        ckpt = ckpt.replace('/dinov2-models', '/dinov3-models') if 'dinov3' in model_name else ckpt

        self.backbone = DinoFeatureExtractor(ckpt, layers=self.return_layers, max_layer=max_layer, freeze_until=freeze_until, final_norm=final_norm)

        self.grd_mslr = MSLR(dim=self.bk_dim, pano=True)
        self.sat_mslr = MSLR(dim=self.bk_dim, pano=False)
        
        self.grd_scm = VSSBscan(input_dim=self.bk_dim, dropout=drop, d_state=16, ssm_ratio=2.0, is_pos=True, view_type='grd', norm_layer=nn.LayerNorm)
        self.sat_scm = VSSBscan(input_dim=self.bk_dim, dropout=drop, d_state=16, ssm_ratio=2.0, is_pos=True, view_type='sat', norm_layer=nn.LayerNorm)

        self.alpha1 = torch.nn.Parameter(torch.tensor(1.0)) # torch.nn.Parameter(torch.tensor(1.0))
        self.alpha2 = torch.nn.Parameter(torch.tensor(1.0)) # torch.nn.Parameter(torch.tensor(1.0))
        self.logit_scale = torch.nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

        self.pmc = PMC(dim=self.bk_dim, proj_dim=768, topk_grd=64, topk_ref=64, proj=False)


    def get_config(self,):
        data_config = timm.data.resolve_model_data_config(self.backbone)
        return data_config
    
    def set_grad_checkpointing(self, enable=True):
        self.backbone.set_grad_checkpointing(enable)

    def backbone_feat(self, x):
        out = self.backbone(x)
        final_cls = out["final_cls"]
        final_patch = out["final_patch"]
        return final_cls, final_patch

    def forward_feature_grd(self, x, mask=None):
        b,c,h,w = x.shape
        global_token, patch_token = self.backbone_feat(x)

        patch = patch_token.view(b, self.h_grd, self.w_grd, self.bk_dim).contiguous()
        patch = self.grd_mslr(patch.permute(0, 3, 1, 2).contiguous())
        patch_vss = self.grd_scm(patch.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        patch = self.alpha1*patch_vss+patch
        
        patch_t = patch.view(b, self.bk_dim, -1).permute(0, 2, 1).contiguous()
        patch_desc, _ = patch_t.max(1)

        fused = torch.cat([global_token, patch_desc], dim=-1)

        return fused, patch_t

    def forward_feature_sat(self, x):
        b,c,h,w = x.shape
        global_token, patch_token = self.backbone_feat(x)

        patch = patch_token.view(b, self.h_sat, self.w_sat, self.bk_dim).contiguous()
        patch = self.sat_mslr(patch.permute(0, 3, 1, 2).contiguous())
        patch_vss = self.sat_scm(patch.permute(0, 2, 3, 1).contiguous()).permute(0, 3, 1, 2).contiguous()
        patch = self.alpha2*patch_vss+patch

        patch_t = patch.view(b, self.bk_dim, -1).permute(0, 2, 1).contiguous()
        patch_desc, _ = patch_t.max(1)

        fused = torch.cat([global_token, patch_desc], dim=-1)

        return fused, patch_t
    
    def forward(self, img1=None, img2=None, input_id=0):
        if img1 is not None and img2 is not None:
                out_grd, grd_patch = self.forward_feature_grd(img1)     
                out_sat, sat_patch = self.forward_feature_sat(img2)
                loss_out = self.pmc(grd_patch, sat_patch, out_grd, out_sat)
        
                return out_grd, out_sat, loss_out
        elif input_id == 1:
                grd_emb, _ = self.forward_feature_grd(img1)     
                return grd_emb
        elif input_id == 2:
            sat_emb, _ = self.forward_feature_sat(img1)
            return sat_emb


