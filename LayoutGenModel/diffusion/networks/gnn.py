import math

import torch
import torch_geometric as tg
import torch_geometric.nn as tgn
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .mlp import FiLM, MLP
from .vit import AttentionBlock
import networks.layers as layers

def get_conv_layer(layer_type, in_channels, out_channels, edge_features, **layer_kwargs):
    layer_fns = {
        "gcn": tgn.GCNConv,
        "sage": tgn.SAGEConv,
        "gin": tgn.GINConv,
        "transformer": tgn.TransformerConv, # TODO figure out why this doesn't work
        "custom_transformer": layers.CustomTransformerConv,
        "gated": layers.GatedGraphConv,
        "gat": tgn.GATv2Conv,
    }
    if layer_type == "gin":
        layer_params = {
            "nn": MLP(
                num_layers = layer_kwargs.get("nn_num_layer", 2), 
                model_width = layer_kwargs.get("nn_hidden_width", out_channels),
                in_size = in_channels, 
                out_size = out_channels,
            ),
            **layer_kwargs
        }
    elif layer_type == "gated":
        assert in_channels <= out_channels
        layer_params = {
            "out_channels": out_channels,
            **layer_kwargs
        }
    elif layer_type == "gat" or layer_type == "transformer":
        if layer_kwargs.get("concat", True):
            assert (out_channels % layer_kwargs.get("heads", 1) == 0), "out channels must be divisible by number of heads in GAT"
            channel_divisor = layer_kwargs.get("heads", 1)
        else:
            channel_divisor = 1
        layer_params = {
            "in_channels": in_channels,
            "out_channels": out_channels // channel_divisor,
            "edge_dim": edge_features,
            **layer_kwargs
        }
    elif layer_type == "custom_transformer":
        if layer_kwargs.get("concat", True):
            assert (out_channels % layer_kwargs.get("heads", 1) == 0), "out channels must be divisible by number of heads in GAT"
            channel_divisor = layer_kwargs.get("heads", 1)
        else:
            channel_divisor = 1
        layer_params = {
            "in_channels": in_channels,
            "out_channels": out_channels // channel_divisor,
            "edge_dim": edge_features,
            **layer_kwargs
        }
    else:
        layer_params = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            **layer_kwargs
        }
    layer = layer_fns[layer_type](**layer_params)
    if layer_type in ["gat", "transformer", "custom_transformer"]:
        # use batch wrapper since tgn does not support batched dim
        layer = layers.BatchWrapper(layer)
    return layer

def accepts_edge_attr(layer):
    # checks if layer takes edge attribute as input
    return isinstance(layer, layers.BatchWrapper)

class GConvLayer(nn.Module):
    def __init__(self, in_node_features, out_node_features):
        super().__init__()
        self._layer = tgn.GCNConv(in_node_features, out_node_features)
    
    def forward(self, x_in):
        x, data, _ = x_in
        edge_index = data.edge_index
        return self._layer(x, edge_index)

class LinearEncoderLayer(nn.Module):
    def __init__(self, in_node_features, out_node_features, input_encoding_dim=0, mask_key=None, device="cpu"):
        MAX_FREQ = 100
        super().__init__()
        assert input_encoding_dim % 2 == 0, "input encoding dimension must be even"
        mask_features = 1 if mask_key is not None else 0
        self._layer = nn.Linear(in_node_features + mask_features, out_node_features)
        self._encoding_layer = nn.Linear(in_node_features * input_encoding_dim, out_node_features) if input_encoding_dim>0 else None
        self.input_encoding_dim = input_encoding_dim
        self.input_encoding_freqs = torch.exp(
            np.log(MAX_FREQ) * torch.arange(0, self.input_encoding_dim // 2, dtype=torch.float32, device=device) / (self.input_encoding_dim // 2)
        ).view(1, 1, 1 , self.input_encoding_dim // 2)
        self.mask_key = mask_key
    
    def forward(self, x, cond_data, t_embed):
        node_data = cond_data.x
        node_data = node_data.view(1, *node_data.shape).expand(x.shape[0], -1, -1)
        spatial_input = torch.concatenate((x, node_data), dim=-1)
        if self.mask_key is not None:
            node_mask = cond_data[self.mask_key]
            node_mask = node_mask.float().view(1, *node_mask.shape, 1).expand(x.shape[0], -1, 1)
            proj_input = torch.concatenate((spatial_input, node_mask), dim=-1)
        else:
            proj_input = spatial_input
        output = self._layer(proj_input)
        if self._encoding_layer is not None:
            input_encodings = self.get_input_encoding(spatial_input)
            input_encodings_proj = self._encoding_layer(input_encodings)
            output = output + input_encodings_proj
        return output

    def get_input_encoding(self, spatial_input):
        # spatial_input: (B, V, D)
        B, V, D = spatial_input.shape
        
        theta = spatial_input.unsqueeze(dim=-1) * self.input_encoding_freqs
        embedding = torch.cat([torch.cos(theta), torch.sin(theta)], dim=-1) # (B, V, D, E)
        embedding = embedding.view(B, V, D * self.input_encoding_dim)

        return embedding


class LinearDecoderLayer(nn.Module):
    def __init__(self, in_node_features, out_node_features):
        super().__init__()
        self._layer = nn.Linear(in_node_features, out_node_features)
    
    def forward(self, x, cond_data, t_embed):
        return self._layer(x)


class ThermalMessagePassing(nn.Module):
    """
    Heat-kernel message passing over the current placement.

    This builds a dense, dynamic thermal coupling graph from node power and
    current Euclidean distance. It is intentionally separate from netlist
    message passing: thermal interaction is spatial, not only topological.
    """
    def __init__(
            self,
            hidden_size,
            power_key="node_power",
            sigma=0.35,
            topk=0,
            normalize_power="graph_max",
            dropout=0.0,
        ):
        super().__init__()
        self.power_key = power_key
        self.sigma = float(sigma)
        self.topk = int(topk)
        self.normalize_power = normalize_power
        self.msg_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.out_proj = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size),
        )
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.dropout = nn.Dropout(p=dropout)

    def _power(self, cond, batch_size, dtype):
        if self.power_key not in cond:
            raise KeyError(f"thermal power feature '{self.power_key}' not found in graph condition")
        power = cond[self.power_key].to(device=cond.x.device, dtype=dtype)
        if power.dim() > 1:
            power = power.view(power.shape[0], -1).mean(dim=-1)
        power = power.abs()
        if self.normalize_power == "graph_max":
            power = power / power.max().clamp_min(1e-6)
        elif self.normalize_power in [None, "none", "None", ""]:
            pass
        else:
            raise ValueError(f"unknown thermal normalize_power={self.normalize_power}")
        return power.view(1, 1, -1).expand(batch_size, -1, -1)

    def forward(self, h, pos, cond, mask=None):
        B, V, _ = h.shape
        power_j = self._power(cond, B, h.dtype)  # (B, 1, V)
        dist2 = torch.cdist(pos[..., :2], pos[..., :2], p=2).square()
        sigma2 = max(self.sigma * self.sigma, 1e-8)
        weights = power_j * torch.exp(-dist2 / sigma2)

        eye = torch.eye(V, dtype=torch.bool, device=h.device).view(1, V, V)
        weights = weights.masked_fill(eye, 0.0)

        if mask is not None:
            active = (~mask).view(1, V).to(device=h.device)
            weights = weights * active.view(1, 1, V).to(dtype=h.dtype)
            weights = weights * active.view(1, V, 1).to(dtype=h.dtype)

        if self.topk > 0 and self.topk < V:
            _, idx = torch.topk(weights, k=self.topk, dim=-1)
            keep = torch.zeros_like(weights, dtype=torch.bool).scatter_(-1, idx, True)
            weights = weights.masked_fill(~keep, 0.0)

        weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-6)
        msg = torch.matmul(weights, self.msg_proj(h))
        msg = self.dropout(self.out_proj(msg))
        return h + torch.tanh(self.gate) * msg


class GeometryAttentionBlock(nn.Module):
    """
    Dense geometry-aware attention over the current placement.

    This block keeps the existing netlist GNN path intact and adds a spatial
    branch whose attention logits are biased by pairwise placement features.
    It is intended for small chiplet graphs where dense VxV attention is cheap.
    """
    pair_feature_dim = 17

    def __init__(
            self,
            hidden_size,
            num_heads=4,
            pair_hidden_size=None,
            pair_num_layers=2,
            ff_num_layers=2,
            ff_size_factor=1,
            power_key="node_power",
            sigma=0.35,
            normalize_power="graph_max",
            dropout=0.0,
        ):
        super().__init__()
        assert hidden_size % num_heads == 0, "hidden_size must be divisible by geometry attention heads"
        self.hidden_size = hidden_size
        self.num_heads = int(num_heads)
        self.head_dim = hidden_size // self.num_heads
        self.power_key = power_key
        self.sigma = float(sigma)
        self.normalize_power = normalize_power

        self.q_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.k_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.v_proj = nn.Linear(hidden_size, hidden_size, bias=False)
        self.out_proj = nn.Linear(hidden_size, hidden_size, bias=False)

        pair_hidden_size = int(pair_hidden_size or hidden_size)
        pair_layers = []
        num_pair_layers = max(1, int(pair_num_layers))
        in_dim = self.pair_feature_dim
        for layer_idx in range(num_pair_layers):
            out_dim = self.num_heads if layer_idx == num_pair_layers - 1 else pair_hidden_size
            pair_layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < num_pair_layers - 1:
                pair_layers.append(nn.SiLU())
                pair_layers.append(nn.LayerNorm(out_dim))
            in_dim = out_dim
        self.pair_bias = nn.Sequential(*pair_layers)

        ff_layers = []
        num_ff_layers = max(1, int(ff_num_layers))
        ff_hidden = int(ff_size_factor * hidden_size)
        for layer_idx in range(num_ff_layers):
            in_dim = hidden_size if layer_idx == 0 else ff_hidden
            out_dim = hidden_size if layer_idx == num_ff_layers - 1 else ff_hidden
            ff_layers.append(nn.Linear(in_dim, out_dim))
            if layer_idx < num_ff_layers - 1:
                ff_layers.append(nn.SiLU())
        self.ff = nn.Sequential(*ff_layers)

        self.ln_attn = nn.LayerNorm(hidden_size)
        self.ln_ff = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(p=dropout)
        self.residual_scale = nn.Parameter(torch.tensor(0.1))

    def _power(self, cond, batch_size, dtype, device):
        if self.power_key not in cond:
            return torch.zeros((batch_size, cond.x.shape[0], 1), device=device, dtype=dtype)

        power = cond[self.power_key].to(device=device, dtype=dtype)
        if power.dim() > 1:
            power = power.view(power.shape[0], -1).mean(dim=-1)
        power = power.abs()
        if self.normalize_power == "graph_max":
            power = power / power.max().clamp_min(1e-6)
        elif self.normalize_power in [None, "none", "None", ""]:
            pass
        else:
            raise ValueError(f"unknown geometry attention normalize_power={self.normalize_power}")
        return power.view(1, -1, 1).expand(batch_size, -1, -1)

    def _active_mask(self, mask, batch_size, num_nodes, device):
        if mask is None:
            return None
        active = (~mask.bool()).to(device=device)
        if active.dim() == 1:
            active = active.view(1, num_nodes).expand(batch_size, -1)
        elif active.dim() == 2:
            if active.shape == (num_nodes, 1):
                active = active.view(1, num_nodes)
            elif active.shape[-1] == num_nodes:
                active = active.view(active.shape[0], num_nodes)
            elif active.shape[0] == num_nodes:
                active = active.t().contiguous()
            else:
                active = active.view(active.shape[0], num_nodes)
            if active.shape[0] == 1:
                active = active.expand(batch_size, -1)
        elif active.dim() == 3:
            active = active.view(active.shape[0], num_nodes)
            if active.shape[0] == 1:
                active = active.expand(batch_size, -1)
        else:
            raise ValueError("geometry attention mask must be 1D, 2D, or 3D")
        return active

    def _pair_features(self, pos, cond):
        B, V, _ = pos.shape
        dtype = pos.dtype
        device = pos.device

        sizes = cond.x.to(device=device, dtype=dtype)
        if sizes.dim() == 1:
            sizes = sizes.view(V, 1).expand(-1, 2)
        sizes = sizes[:, :2]

        pos_i = pos.unsqueeze(2)
        pos_j = pos.unsqueeze(1)
        delta = pos_j - pos_i
        abs_delta = delta.abs()
        dist2 = delta.square().sum(dim=-1, keepdim=True)
        dist = torch.sqrt(dist2 + 1e-8)

        size_i = sizes.view(1, V, 1, 2).expand(B, -1, V, -1)
        size_j = sizes.view(1, 1, V, 2).expand(B, V, -1, -1)
        area_i = (size_i[..., 0:1] * size_i[..., 1:2]).clamp_min(0.0)
        area_j = (size_j[..., 0:1] * size_j[..., 1:2]).clamp_min(0.0)

        overlap_xy = torch.relu(0.5 * (size_i + size_j) - abs_delta)
        overlap_area = overlap_xy[..., 0:1] * overlap_xy[..., 1:2]

        power = self._power(cond, B, dtype, device)
        power_i = power.view(B, V, 1, 1).expand(-1, -1, V, -1)
        power_j = power.view(B, 1, V, 1).expand(-1, V, -1, -1)
        sigma2 = max(self.sigma * self.sigma, 1e-8)
        thermal_kernel = power_j * torch.exp(-dist2 / sigma2)

        return torch.cat(
            [
                delta,
                abs_delta,
                dist,
                size_i,
                size_j,
                area_i,
                area_j,
                overlap_xy,
                overlap_area,
                power_i,
                power_j,
                thermal_kernel,
            ],
            dim=-1,
        )

    def forward(self, h, pos, cond, t_embed=None, mask=None):
        B, V, C = h.shape
        if C != self.hidden_size:
            return h

        active = self._active_mask(mask, B, V, h.device)
        if active is not None and not active.any():
            return h

        h_norm = self.ln_attn(h)
        q = self.q_proj(h_norm).view(B, V, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(h_norm).view(B, V, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(h_norm).view(B, V, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        pair_bias = self.pair_bias(self._pair_features(pos, cond)).permute(0, 3, 1, 2)
        scores = scores + pair_bias

        if active is not None:
            scores = scores.masked_fill(~active.view(B, 1, 1, V), torch.finfo(scores.dtype).min)

        attn = F.softmax(scores.float(), dim=-1).to(dtype=scores.dtype)
        attn = self.dropout(attn)
        attn_out = torch.matmul(attn, v).transpose(1, 2).reshape(B, V, C)
        attn_out = self.out_proj(attn_out)
        if active is not None:
            attn_out = torch.where(active.view(B, V, 1), attn_out, torch.zeros_like(attn_out))

        scale = self.residual_scale.to(dtype=h.dtype)
        h = h + scale * self.dropout(attn_out)
        ff_out = self.ff(self.ln_ff(h))
        if active is not None:
            ff_out = torch.where(active.view(B, V, 1), ff_out, torch.zeros_like(ff_out))
        return h + scale * self.dropout(ff_out)

class ResGNNBlock(nn.Module):
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_node_features, 
            cond_node_features, 
            edge_features, 
            num_layers, 
            encoding_dim,
            residual=True, 
            norm=True,
            dropout=0.0, 
            conv_params={"layer_type": "gcn"}, 
            device="cpu", 
            **kwargs
        ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.residual = residual
        if residual:
            assert in_node_features == out_node_features, "input and output features must be equal to perform residual connection"
        self._gconv_layers = nn.ModuleList()
        self._lnorm_layers = nn.ModuleList()
        self._linear_layers = nn.ModuleList()

        self._cond_layer = FiLM(encoding_dim, hidden_node_features, channel_axis=-1) if encoding_dim>0 else None
        for i in range(num_layers):
            in_features = in_node_features + cond_node_features if i==0 else hidden_node_features
            out_features = hidden_node_features if i<(num_layers-1) else out_node_features
            self._gconv_layers.append(get_conv_layer(
                in_channels=in_features, 
                out_channels=hidden_node_features,
                edge_features=edge_features,
                **conv_params
                ))
            self._lnorm_layers.append(nn.LayerNorm(hidden_node_features))
            self._linear_layers.append(nn.Linear(hidden_node_features, out_features))
            
        self.use_edge_attr = accepts_edge_attr(self._gconv_layers[0])
        # self.linear = nn.Linear(self.hidden_node_features, self.out_node_features)
        if norm:
            self._norm = nn.GroupNorm(1, hidden_node_features)
        else:
            self._norm = None
        self._nonlinear = nn.ReLU()
        self._dropout = nn.Dropout(p = dropout)

    def forward(self, x, data, t): # data is conditioning info
        B, V, F = x.shape
        cond_x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        cond_x = cond_x.view(1, *cond_x.shape).expand(B, -1, -1)
        x_skip = x
        x = torch.cat((x, cond_x), dim=-1)
        for i, (lnorm, linear, conv) in enumerate(zip(self._lnorm_layers[:-1], self._linear_layers[:-1], self._gconv_layers[:-1])):
            if self._norm is not None and x.shape[-1] == self.hidden_node_features:
                x = torch.movedim(x, -1, 1)
                x = self._norm(x)
                x = torch.movedim(x, 1, -1)
            x = conv(x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else conv(x, edge_index)
            x = self._nonlinear(x)
            x = lnorm(x)
            x = linear(x)
            x = self._nonlinear(x)
            x = self._dropout(x)
        x = self._gconv_layers[-1](x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else self._gconv_layers[-1](x, edge_index)
        if (not self._cond_layer is None):
            x = self._cond_layer(x, t)
        x = self._nonlinear(x)
        x = self._lnorm_layers[-1](x)
        x = self._linear_layers[-1](x)
        if self.residual:
            x = x + x_skip 
        return x 

class AttGNNBlock(nn.Module):
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_node_features, 
            cond_node_features,
            attention_extra_features, 
            edge_features, 
            num_layers, 
            encoding_dim, 
            residual=True, 
            norm=True, 
            dropout=0.0, 
            conv_params={"layer_type": "gcn"}, 
            device="cpu", 
            **kwargs
        ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.attention_extra_features = attention_extra_features
        self.edge_features = edge_features
        self.residual = residual
        if residual:
            assert in_node_features == out_node_features, "input and output features must be equal to perform residual connection"
        self._gconv_layers = []
        self._att_extra_input_embed_layers = []
        self._attention_layers = []
        self._lnorm_layers = []
        self._linear_layers = []

        self._cond_layer = FiLM(encoding_dim, hidden_node_features, channel_axis=-1) if encoding_dim>0 else None
        for i in range(num_layers):
            in_features = in_node_features + cond_node_features if i==0 else hidden_node_features
            out_features = hidden_node_features if i<(num_layers-1) else out_node_features
            self._gconv_layers.append(get_conv_layer(
                in_channels=in_features, 
                out_channels=hidden_node_features,
                edge_features=edge_features,
                **conv_params
                ))
            self._att_extra_input_embed_layers.append(nn.Linear(cond_node_features + attention_extra_features, hidden_node_features))
            self._attention_layers.append(AttentionBlock(
                kwargs["num_heads"], 
                hidden_node_features, 
                kwargs["ff_num_layers"], 
                kwargs["ff_size_factor"], 
                dropout, 
                att_implementation = kwargs["att_implementation"]
            ))
            self._lnorm_layers.append(nn.LayerNorm(hidden_node_features))
            self._linear_layers.append(nn.Linear(hidden_node_features, out_features))
            # self._linear_layers.append(MLP(mlp_num_layers, att_model_size * mlp_size_factor, att_model_size, out_features, skip = att_model_size==out_features, layernorm = True))
        
        self.use_edge_attr = accepts_edge_attr(self._gconv_layers[0])
        self._gconv_layers = nn.ModuleList(self._gconv_layers)
        self._att_extra_input_embed_layers = nn.ModuleList(self._att_extra_input_embed_layers)
        self._attention_layers = nn.ModuleList(self._attention_layers)
        self._lnorm_layers = nn.ModuleList(self._lnorm_layers)
        self._linear_layers = nn.ModuleList(self._linear_layers)
        # self.linear = nn.Linear(self.hidden_node_features, self.out_node_features)
        if norm:
            self._norm = nn.GroupNorm(1, hidden_node_features)
        else:
            self._norm = None
        self._nonlinear = nn.ReLU()
        self._dropout = nn.Dropout(p = dropout)

    def forward(self, x, data, t, att_extra_input = None): # data is conditioning info
        B, V, F = x.shape
        assert att_extra_input is None or att_extra_input.shape[-1] == self.attention_extra_features, "extra attention features must have right shape"
        cond_x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        cond_x = cond_x.view(1, *cond_x.shape).expand(B, -1, -1)
        x_skip = x
        x = torch.cat((x, cond_x), dim=-1)
        x_att_features = torch.cat((cond_x, att_extra_input), dim=-1) if att_extra_input is not None else cond_x
        for i, (lnorm, linear, conv, attention, att_input_embed_layer) in enumerate(zip(self._lnorm_layers[:-1], self._linear_layers[:-1], self._gconv_layers[:-1], self._attention_layers[:-1], self._att_extra_input_embed_layers[:-1])):
            if self._norm is not None and x.shape[-1] == self.hidden_node_features:
                x = torch.movedim(x, -1, 1)
                x = self._norm(x)
                x = torch.movedim(x, 1, -1)
            x = conv(x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else conv(x, edge_index)
            x = self._nonlinear(x)
            att_extra_embedded = att_input_embed_layer(x_att_features)
            x = x + att_extra_embedded
            # x = torch.cat((x, x_att_features), dim=-1)
            x = attention(x)
            x = lnorm(x)
            x = linear(x)
            x = self._nonlinear(x)
            x = self._dropout(x)
        x = self._gconv_layers[-1](x, edge_index, edge_attr=edge_attr) if self.use_edge_attr else self._gconv_layers[-1](x, edge_index)
        if (not self._cond_layer is None):
            x = self._cond_layer(x, t)
        x = self._nonlinear(x)
        att_extra_embedded = self._att_extra_input_embed_layers[-1](x_att_features)
        x = x + att_extra_embedded
        # x = torch.cat((x, x_att_features), dim=-1)
        x = self._attention_layers[-1](x)
        x = self._lnorm_layers[-1](x)
        x = self._linear_layers[-1](x)
        if self.residual:
            x = x + x_skip 
        return x

class ResGNN(nn.Module):
    blocks = {"res": ResGNNBlock, "att": AttGNNBlock}
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_size, 
            hidden_node_features, 
            cond_node_features, 
            edge_features, 
            layers_per_block, 
            encoding_dim, 
            dropout=0.0, 
            device="cpu", 
            block_type="res", 
            **kwargs
            ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features

        self._gnn_blocks = []
        self.use_enc = not (hidden_size == in_node_features == out_node_features)
        if self.use_enc:
            self._gnn_blocks.append(GConvLayer(in_node_features, hidden_size))
        for i, hidden_node_size in enumerate(hidden_node_features):
            self._gnn_blocks.append(ResGNN.blocks[block_type](
                in_node_features=hidden_size,
                out_node_features=hidden_size,
                hidden_node_features=hidden_node_size,
                cond_node_features=cond_node_features,
                edge_features=edge_features,
                num_layers=layers_per_block,
                encoding_dim=encoding_dim,
                residual=True,
                norm=True,
                dropout=dropout,
                device=device,
                **kwargs,
            ))
        if self.use_enc:
            self._gnn_blocks.append(GConvLayer(hidden_size, out_node_features))
        self._network = nn.Sequential(*self._gnn_blocks)
        print("ENCODER USED IN RESGNN", self.use_enc)

    def forward(self, x, cond, t_embed):
        x_skip = x
        x,_,_ = self._network((x, cond, t_embed))
        return (x + x_skip if self.use_enc else x)

class AttGNN(nn.Module):
    # Same as ResGNN, but with attention layer in between resBlocks
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_size, 
            hidden_node_features,
            attention_node_features, 
            cond_node_features,
            edge_features, 
            layers_per_block, 
            t_encoding_dim,
            conv_params,
            mlp_num_layers,
            mlp_size_factor,
            input_encoding_dim=0,
            dir_att_input=False,
            mask_key=None,
            extra_node_feature_keys=None,
            extra_node_feature_normalize="graph_max",
            thermal_mp_enabled=False,
            thermal_mp_power_key="node_power",
            thermal_mp_sigma=0.35,
            thermal_mp_topk=0,
            thermal_mp_normalize_power="graph_max",
            geometry_attention_enabled=False,
            geometry_attention_layers=0,
            geometry_attention_heads=None,
            geometry_attention_pair_hidden_size=None,
            geometry_attention_pair_layers=2,
            geometry_attention_ff_num_layers=None,
            geometry_attention_ff_size_factor=None,
            geometry_attention_power_key="node_power",
            geometry_attention_sigma=0.35,
            geometry_attention_normalize_power="graph_max",
            replace_attention_with_geometry=False,
            log_name="ATTGNN",
            auxiliary_legality_heads_enabled=False,
            auxiliary_legality_head_hidden_size=None,
            auxiliary_legality_head_layers=2,
            dropout=0.0,
            device="cpu",
            **kwargs, # should contain attention parameters
            ):
        super().__init__()
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_size = hidden_size
        self.hidden_node_features = hidden_node_features
        self.attention_node_features = attention_node_features
        self.attention_extra_features = in_node_features # TODO fix case for dir_att_input=False
        self.edge_features = edge_features
        self.dir_att_input = dir_att_input
        self.mask_key = mask_key
        self.device = device
        self.extra_node_feature_keys = list(extra_node_feature_keys or [])
        self.extra_node_feature_normalize = extra_node_feature_normalize
        self.thermal_mp_enabled = thermal_mp_enabled
        self.replace_attention_with_geometry = bool(replace_attention_with_geometry)
        self.geometry_attention_enabled = bool(geometry_attention_enabled or self.replace_attention_with_geometry)
        self.auxiliary_legality_heads_enabled = bool(auxiliary_legality_heads_enabled)
        self.last_aux_outputs = None

        gnn_blocks = []
        self.use_enc = not (hidden_size == in_node_features == out_node_features)
        if self.use_enc:
            gnn_blocks.append(LinearEncoderLayer(
                in_node_features + cond_node_features, 
                hidden_size, 
                mask_key=mask_key, 
                input_encoding_dim=input_encoding_dim,
                device=device,
            ))
        self._extra_node_feature_layer = (
            nn.Linear(len(self.extra_node_feature_keys), hidden_size)
            if self.extra_node_feature_keys else None
        )
        self._thermal_mp = (
            ThermalMessagePassing(
                hidden_size=hidden_size,
                power_key=thermal_mp_power_key,
                sigma=thermal_mp_sigma,
                topk=thermal_mp_topk,
                normalize_power=thermal_mp_normalize_power,
                dropout=dropout,
            )
            if thermal_mp_enabled else None
        )
        self._overlap_head = None
        self._boundary_head = None
        if self.auxiliary_legality_heads_enabled:
            aux_hidden = int(auxiliary_legality_head_hidden_size or hidden_size)
            aux_layers = max(1, int(auxiliary_legality_head_layers))
            self._overlap_head = self._make_auxiliary_legality_head(hidden_size, aux_hidden, aux_layers)
            self._boundary_head = self._make_auxiliary_legality_head(hidden_size, aux_hidden, aux_layers)
        geometry_attention_layers = int(geometry_attention_layers or 0)
        geometry_attention_heads = int(geometry_attention_heads or kwargs.get("num_heads", 4))
        geometry_attention_ff_num_layers = int(
            geometry_attention_ff_num_layers
            if geometry_attention_ff_num_layers is not None
            else kwargs.get("ff_num_layers", 2)
        )
        geometry_attention_ff_size_factor = int(
            geometry_attention_ff_size_factor
            if geometry_attention_ff_size_factor is not None
            else kwargs.get("ff_size_factor", 1)
        )

        def make_geometry_attention_block():
            return GeometryAttentionBlock(
                hidden_size=hidden_size,
                num_heads=geometry_attention_heads,
                pair_hidden_size=geometry_attention_pair_hidden_size,
                pair_num_layers=geometry_attention_pair_layers,
                ff_num_layers=geometry_attention_ff_num_layers,
                ff_size_factor=geometry_attention_ff_size_factor,
                power_key=geometry_attention_power_key,
                sigma=geometry_attention_sigma,
                normalize_power=geometry_attention_normalize_power,
                dropout=dropout,
            )

        for i, (hidden_node_size, attention_node_size) in enumerate(zip(hidden_node_features, attention_node_features)):
            gnn_blocks.append(ResGNNBlock(
                in_node_features=hidden_size,
                out_node_features=hidden_size,
                hidden_node_features=hidden_node_size,
                cond_node_features=cond_node_features,
                edge_features=edge_features,
                num_layers=layers_per_block,
                encoding_dim=t_encoding_dim,
                conv_params=conv_params,
                residual=True,
                norm=True,
                dropout=dropout,
                device=device,
            ))
            if mlp_num_layers > 0 and mlp_size_factor > 0:
                gnn_blocks.append(MLP(
                    mlp_num_layers, 
                    mlp_size_factor * hidden_size, 
                    hidden_size, 
                    hidden_size, 
                    skip = True, 
                    layernorm = True,
                ))
            use_geometry_for_stage = self.geometry_attention_enabled and (
                geometry_attention_layers <= 0 or i < geometry_attention_layers
            )
            if self.replace_attention_with_geometry:
                if use_geometry_for_stage:
                    gnn_blocks.append(make_geometry_attention_block())
            elif attention_node_size > 0:
                gnn_blocks.append(AttGNNBlock(
                    in_node_features=hidden_size,
                    out_node_features=hidden_size,
                    hidden_node_features=attention_node_size,
                    cond_node_features=cond_node_features,
                    attention_extra_features=self.attention_extra_features,
                    edge_features=edge_features,
                    num_layers=1,
                    encoding_dim=t_encoding_dim,
                    conv_params=conv_params,
                    residual=True,
                    norm=True,
                    dropout=dropout,
                    device=device,
                    **kwargs,
                ))
            else:
                gnn_blocks.append(ResGNNBlock(
                    in_node_features=hidden_size,
                    out_node_features=hidden_size,
                    hidden_node_features=hidden_node_size,
                    cond_node_features=cond_node_features,
                    edge_features=edge_features,
                    num_layers=1,
                    encoding_dim=t_encoding_dim,
                    conv_params=conv_params,
                    residual=True,
                    norm=True,
                    dropout=dropout,
                    device=device,
                ))
            if (not self.replace_attention_with_geometry) and use_geometry_for_stage:
                gnn_blocks.append(make_geometry_attention_block())
            if mlp_num_layers > 0 and mlp_size_factor > 0:
                gnn_blocks.append(MLP(
                    mlp_num_layers, 
                    mlp_size_factor * hidden_size, 
                    hidden_size, 
                    hidden_size, 
                    skip = True, 
                    layernorm = True,
                ))
        if self.use_enc:
            gnn_blocks.append(LinearDecoderLayer(hidden_size, out_node_features))
            if self.in_node_features != self.out_node_features:
                self._skip_linear = nn.Linear(in_node_features, self.out_node_features)
        self._gnn_blocks = nn.ModuleList(gnn_blocks)
        print(f"ENCODER USED IN {log_name}", self.use_enc)

    @staticmethod
    def _make_auxiliary_legality_head(in_size, hidden_size, num_layers):
        layers = []
        for layer_idx in range(num_layers):
            layer_in = in_size if layer_idx == 0 else hidden_size
            layer_out = 1 if layer_idx == num_layers - 1 else hidden_size
            layers.append(nn.Linear(layer_in, layer_out))
            if layer_idx < num_layers - 1:
                layers.append(nn.SiLU())
                layers.append(nn.LayerNorm(layer_out))
        return nn.Sequential(*layers)

    def _set_auxiliary_legality_outputs(self, h):
        if not self.auxiliary_legality_heads_enabled or self._overlap_head is None or self._boundary_head is None:
            self.last_aux_outputs = None
            return
        self.last_aux_outputs = {
            "overlap": F.softplus(self._overlap_head(h)).squeeze(-1),
            "boundary": F.softplus(self._boundary_head(h)).squeeze(-1),
        }

    def _mask(self, cond):
        if self.mask_key is None or self.mask_key not in cond:
            return None
        return cond[self.mask_key]

    def _extra_node_features(self, cond, batch_size, dtype):
        if not self.extra_node_feature_keys:
            return None

        features = []
        for key in self.extra_node_feature_keys:
            if key not in cond:
                raise KeyError(f"extra node feature '{key}' not found in graph condition")
            value = cond[key].to(device=cond.x.device, dtype=dtype)
            if value.dim() == 1:
                value = value.unsqueeze(-1)
            elif value.dim() > 2:
                value = value.view(value.shape[0], -1)
            features.append(value)

        x = torch.cat(features, dim=-1)
        if self.extra_node_feature_normalize == "graph_max":
            scale = x.abs().amax(dim=0, keepdim=True).clamp_min(1e-6)
            x = x / scale
        elif self.extra_node_feature_normalize in [None, "none", "None", ""]:
            pass
        else:
            raise ValueError(f"unknown extra_node_feature_normalize={self.extra_node_feature_normalize}")
        return x.view(1, *x.shape).expand(batch_size, -1, -1)

    def _inject_thermal_features(self, x, x_pos, cond, mask, extra_node_features, add_extra_node_features):
        if x.shape[-1] != self.hidden_size:
            return x
        if add_extra_node_features and self._extra_node_feature_layer is not None:
            x = x + self._extra_node_feature_layer(extra_node_features)
        if self._thermal_mp is not None:
            x = self._thermal_mp(x, x_pos, cond, mask=mask)
        return x

    def forward(self, x, cond, t_embed):
        with torch.autocast(device_type=self.device):
            x_skip = x
            self.last_aux_outputs = None
            thermal_mask = self._mask(cond)
            extra_node_features = self._extra_node_features(cond, x.shape[0], x.dtype)
            for block in self._gnn_blocks:
                add_extra_node_features = False
                if isinstance(block, LinearDecoderLayer):
                    self._set_auxiliary_legality_outputs(x)
                    x = block(x, cond, t_embed)
                elif isinstance(block, AttGNNBlock): # include attention conditioning
                    att_input = x_skip if self.dir_att_input else x
                    x = block(x, cond, t_embed, att_extra_input=att_input)
                elif isinstance(block, GeometryAttentionBlock):
                    x = block(x, x_skip, cond, t_embed=t_embed, mask=thermal_mask)
                elif isinstance(block, MLP):
                    x = block(x)
                else:
                    x = block(x, cond, t_embed)
                    add_extra_node_features = True
                x = self._inject_thermal_features(
                    x,
                    x_skip,
                    cond,
                    thermal_mask,
                    extra_node_features,
                    add_extra_node_features,
                )
            if self.last_aux_outputs is None and x.shape[-1] == self.hidden_size:
                self._set_auxiliary_legality_outputs(x)
            if self.use_enc:
                if self.in_node_features != self.out_node_features:
                    x_skip = self._skip_linear(x_skip)
                x = x + x_skip
        return x


class GeometryAttGNN(AttGNN):
    """
    AttGNN variant that replaces the full-node AttGNNBlock with
    GeometryAttentionBlock while keeping the netlist GNN branch.
    """

    def __init__(self, *args, **kwargs):
        kwargs["geometry_attention_enabled"] = True
        kwargs["replace_attention_with_geometry"] = True
        kwargs["log_name"] = "GEOMETRY_ATTGNN"
        super().__init__(*args, **kwargs)

class GraphUNet(nn.Module):
    def __init__(
            self, 
            in_node_features, 
            out_node_features, 
            hidden_node_features, # list
            cond_node_features, 
            edge_features, 
            blocks_per_level, # list
            layers_per_block,
            level_block="res", 
            device="cpu", 
            **kwargs
        ):
        # length of CNN_depths determines how many levels u-net has
        super().__init__()
        self._down_conv_blocks = []
        self._up_conv_blocks = []
        self.in_node_features = in_node_features
        self.out_node_features = out_node_features
        self.hidden_node_features = hidden_node_features
        self.edge_features = edge_features
        self.cond_node_features = cond_node_features
        self.blocks_per_level = blocks_per_level
        self.layers_per_block = layers_per_block
        self.level_block = level_block
        self.device=device

        # create downward branch
        for i, (hidden_size, num_blocks) in enumerate(zip(hidden_node_features, blocks_per_level)):
            level_in_size = in_node_features if i==0 else hidden_node_features[i-1]
            if self.level_block == "res":
                level_in_layer = GConvLayer(level_in_size, hidden_size)
                level_blocks = [ResGNNBlock(
                    in_node_features = hidden_size, 
                    out_node_features = hidden_size, 
                    hidden_node_features = hidden_size, 
                    cond_node_features = cond_node_features, 
                    edge_features = edge_features, 
                    num_layers = layers_per_block,
                    device = device,
                    **kwargs
                    ) for _ in range(num_blocks)]
                if i == len(hidden_node_features)-1:
                    level_blocks.append(GConvLayer(hidden_size, level_in_size))
                level_net = nn.Sequential(level_in_layer, *level_blocks)
            else:
                raise NotImplementedError
            
            self._down_conv_blocks.append(level_net)
        self._down_conv_blocks = nn.ModuleList(self._down_conv_blocks)

        # create upsampling branch
        for i in range(len(hidden_node_features)-2, -1, -1):
            level_in_size = 2 * hidden_node_features[i]
            level_out_size = hidden_node_features[i-1] if i>0 else out_node_features
            hidden_size = hidden_node_features[i]
            num_blocks = blocks_per_level[i]
            if self.level_block == "res":
                level_in_layer = GConvLayer(level_in_size, hidden_size)
                level_blocks = [ResGNNBlock(
                    in_node_features = hidden_size, 
                    out_node_features = hidden_size, 
                    hidden_node_features = hidden_size, 
                    cond_node_features = cond_node_features, 
                    edge_features = edge_features, 
                    num_layers = layers_per_block,
                    device = device,
                    **kwargs
                    ) for _ in range(layers_per_block)]
                level_out_layer = GConvLayer(hidden_size, level_out_size)
                level_net = nn.Sequential(level_in_layer, *level_blocks, level_out_layer)
            else:
                raise NotImplementedError
            self._up_conv_blocks.append(level_net)
        self._up_conv_blocks = nn.ModuleList(self._up_conv_blocks)

    def __call__(self, x, data, t_enc):
        # x is (B, V, F)
        B, _, _ = x.shape
        assert t_enc.shape[0] == B and len(t_enc.shape) == 2, "t has to have shape (B, E)"
        x_skip = x

        # downward branch
        skip_images = []
        for down_block in self._down_conv_blocks[:-1]:
            x, _, _ = down_block((x, data, t_enc))
            skip_images.append(x)

        x, _, _ = self._down_conv_blocks[-1]((x, data, t_enc))

        # upward branch
        for i, up_block in enumerate(self._up_conv_blocks):
            x = torch.cat((x, skip_images[-(i+1)]), dim = -1)
            x, _, _ = up_block((x, data, t_enc))
        
        return x + x_skip
