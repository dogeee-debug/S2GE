"""Graph encoder backbones available to S2GE and graph-only baselines."""

import torch
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GCNConv, TransformerConv
from torch_scatter import scatter_add
from torch.nn import MultiheadAttention


def _reset_multihead_attention(module):
    if hasattr(module, "_reset_parameters"):
        module._reset_parameters()
        return
    if hasattr(module, "in_proj_weight") and module.in_proj_weight is not None:
        torch.nn.init.xavier_uniform_(module.in_proj_weight)
    if hasattr(module, "in_proj_bias") and module.in_proj_bias is not None:
        torch.nn.init.zeros_(module.in_proj_bias)
    if hasattr(module, "out_proj") and hasattr(module.out_proj, "reset_parameters"):
        module.out_proj.reset_parameters()


class GCN(torch.nn.Module):
    """Multi-layer graph convolution network with optional normalization."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(in_channels, hidden_channels))
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_channels, hidden_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(GCNConv(hidden_channels, out_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index)
        return x, edge_attr


class GraphTransformer(torch.nn.Module):
    """TransformerConv stack that propagates node and edge representations."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=-1):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TransformerConv(
                in_channels=in_channels,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                edge_dim=in_channels,
                dropout=dropout,
            )
        )
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    edge_dim=in_channels,
                    dropout=dropout,
                )
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            TransformerConv(
                in_channels=hidden_channels,
                out_channels=out_channels // num_heads,
                heads=num_heads,
                edge_dim=in_channels,
                dropout=dropout,
            )
        )
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr


class _AnyGraphFeedForward(torch.nn.Module):
    def __init__(self, dim, dropout):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(dim, dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(dropout),
            torch.nn.Linear(dim, dim),
        )

    def forward(self, x):
        return self.net(x)


class _AnyGraphGTLayer(torch.nn.Module):
    def __init__(self, dim, num_heads, dropout, anchor_count):
        super().__init__()
        self.anchor_count = anchor_count
        self.query_to_anchor = MultiheadAttention(dim, num_heads, dropout=dropout, bias=False, batch_first=True)
        self.anchor_to_query = MultiheadAttention(dim, num_heads, dropout=dropout, bias=False, batch_first=True)
        self.ffn = _AnyGraphFeedForward(dim, dropout)
        self.norm1 = torch.nn.LayerNorm(dim)
        self.norm2 = torch.nn.LayerNorm(dim)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):
        if x.size(0) == 0:
            return x
        anchor_count = min(self.anchor_count, x.size(0))
        anchor_idx = torch.randperm(x.size(0), device=x.device)[:anchor_count]
        anchors = x[anchor_idx].unsqueeze(0)
        tokens = x.unsqueeze(0)
        anchor_ctx, _ = self.query_to_anchor(anchors, tokens, tokens, need_weights=False)
        token_ctx, _ = self.anchor_to_query(tokens, anchor_ctx, anchor_ctx, need_weights=False)
        x = self.norm1(x + self.dropout(token_ctx.squeeze(0)))
        x = self.norm2(x + self.dropout(self.ffn(x)))
        return x


class AnyGraphBackbone(torch.nn.Module):
    """Transformer-style global/local backbone retained for experiments."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super().__init__()
        if hidden_channels != out_channels:
            self.input_proj = torch.nn.Linear(in_channels, hidden_channels)
            self.output_proj = torch.nn.Linear(hidden_channels, out_channels)
            work_dim = hidden_channels
        else:
            self.input_proj = torch.nn.Linear(in_channels, out_channels)
            self.output_proj = torch.nn.Identity()
            work_dim = out_channels
        self.topo_norm = torch.nn.LayerNorm(work_dim, elementwise_affine=False)
        self.topo_steps = max(0, int(num_layers))
        self.anchor_count = 256
        self.gt_layers = torch.nn.ModuleList(
            [_AnyGraphGTLayer(work_dim, max(1, num_heads), dropout, self.anchor_count) for _ in range(2)]
        )

    def reset_parameters(self):
        self.input_proj.reset_parameters()
        if hasattr(self.output_proj, "reset_parameters"):
            self.output_proj.reset_parameters()
        for layer in self.gt_layers:
            _reset_multihead_attention(layer.query_to_anchor)
            _reset_multihead_attention(layer.anchor_to_query)
            for mod in layer.ffn.modules():
                if hasattr(mod, "reset_parameters"):
                    mod.reset_parameters()
            layer.norm1.reset_parameters()
            layer.norm2.reset_parameters()

    def _make_norm_adj(self, edge_index, num_nodes, device, dtype):
        row, col = edge_index
        if row.numel() == 0:
            diag = torch.arange(num_nodes, device=device)
            idx = torch.stack([diag, diag], dim=0)
            vals = torch.ones(num_nodes, device=device, dtype=dtype)
            return torch.sparse_coo_tensor(idx, vals, (num_nodes, num_nodes)).coalesce()
        rev = torch.stack([col, row], dim=0)
        diag = torch.arange(num_nodes, device=device)
        self_loops = torch.stack([diag, diag], dim=0)
        full_index = torch.cat([edge_index, rev, self_loops], dim=1)
        values = torch.ones(full_index.size(1), device=device, dtype=dtype)
        adj = torch.sparse_coo_tensor(full_index, values, (num_nodes, num_nodes)).coalesce()
        deg = torch.sparse.sum(adj, dim=1).to_dense().clamp(min=1.0)
        norm_vals = adj.values() * deg[adj.indices()[0]].pow(-0.5) * deg[adj.indices()[1]].pow(-0.5)
        return torch.sparse_coo_tensor(adj.indices(), norm_vals, adj.shape).coalesce()

    def forward(self, x, edge_index, edge_attr):
        x = self.input_proj(x)
        x = self.topo_norm(x)
        adj = self._make_norm_adj(edge_index, x.size(0), x.device, x.dtype)
        topo_sum = x if self.topo_steps == 0 else 0.0
        h = x
        for _ in range(self.topo_steps):
            h = torch.sparse.mm(adj, h)
            topo_sum = topo_sum + h
        x = topo_sum
        for layer in self.gt_layers:
            x = layer(x)
        x = self.output_proj(x)
        return x, edge_attr


class GAT(torch.nn.Module):
    """Graph attention network with a valid single-layer execution path."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"gat requires at least 1 layer, got {num_layers}")
        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        if num_layers == 1:
            self.convs.append(GATConv(in_channels, out_channels, heads=num_heads, concat=False, edge_dim=in_channels))
        else:
            self.convs.append(GATConv(in_channels, hidden_channels, heads=num_heads, concat=False, edge_dim=in_channels))
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            for _ in range(num_layers - 2):
                self.convs.append(GATConv(hidden_channels, hidden_channels, heads=num_heads, concat=False, edge_dim=in_channels))
                self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
            self.convs.append(GATConv(hidden_channels, out_channels, heads=num_heads, concat=False, edge_dim=in_channels))
        self.dropout = dropout

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()

    def forward(self, x, edge_index, edge_attr):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index=edge_index, edge_attr=edge_attr)
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.convs[-1](x, edge_index=edge_index, edge_attr=edge_attr)
        return x, edge_attr


class VPAAEGFTransformer(torch.nn.Module):
    """VPA-AEGF Transformer with entropy-gated structural updates."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super().__init__()
        self.edge_dim = in_channels
        self.convs = torch.nn.ModuleList()
        self.convs.append(
            TransformerConv(
                in_channels=in_channels,
                out_channels=hidden_channels // num_heads,
                heads=num_heads,
                edge_dim=in_channels,
                dropout=dropout,
            )
        )
        self.bns = torch.nn.ModuleList()
        self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(
                TransformerConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels // num_heads,
                    heads=num_heads,
                    edge_dim=in_channels,
                    dropout=dropout,
                )
            )
            self.bns.append(torch.nn.BatchNorm1d(hidden_channels))
        self.convs.append(
            TransformerConv(
                in_channels=hidden_channels,
                out_channels=out_channels // num_heads,
                heads=num_heads,
                edge_dim=in_channels,
                dropout=dropout,
            )
        )
        self.entropy_gates = torch.nn.ModuleList([torch.nn.Linear(1, 1) for _ in range(num_layers)])
        self.dropout = dropout
        self.num_layers = num_layers

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for gate in self.entropy_gates:
            gate.reset_parameters()

    def _entropy_gate(self, edge_index, alpha, num_nodes, gate_layer):
        eps = 1e-12
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        alpha = alpha.clamp(min=eps)
        entropy = scatter_add(
            torch.special.entr(alpha),
            edge_index[1],
            dim=0,
            dim_size=num_nodes,
        )
        if entropy.dim() > 1:
            entropy = entropy.mean(dim=1)
        deg = scatter_add(
            torch.ones(edge_index.size(1), device=edge_index.device, dtype=entropy.dtype),
            edge_index[1],
            dim=0,
            dim_size=num_nodes,
        ).clamp(min=1.0)
        entropy_norm = torch.zeros_like(entropy)
        valid = deg > 1
        entropy_norm[valid] = entropy[valid] / deg[valid].log().clamp(min=eps)
        # Convert to match gate_layer dtype
        return torch.sigmoid(gate_layer(entropy_norm.unsqueeze(-1).to(gate_layer.weight.dtype)))

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=x.device)
        for i, conv in enumerate(self.convs[:-1]):
            h_prev = x
            out, (ei, alpha) = conv(
                x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
            deg = scatter_add(
                torch.ones(ei.size(1), device=ei.device),
                ei[1],
                dim=0,
                dim_size=out.size(0),
            )
            scale = deg.clamp(min=1.0).pow(-0.5).unsqueeze(-1)
            message = out * scale
            g = self._entropy_gate(ei, alpha, out.size(0), self.entropy_gates[i])
            x = g * h_prev + (1.0 - g) * message
            x = self.bns[i](x)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        h_prev = x
        out, (ei, alpha) = self.convs[-1](
            x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            return_attention_weights=True,
        )
        deg = scatter_add(
            torch.ones(ei.size(1), device=ei.device),
            ei[1],
            dim=0,
            dim_size=out.size(0),
        )
        scale = deg.clamp(min=1.0).pow(-0.5).unsqueeze(-1)
        message = out * scale
        g = self._entropy_gate(ei, alpha, out.size(0), self.entropy_gates[-1])
        x = g * h_prev + (1.0 - g) * message
        return x, edge_attr


class VPAAEGFGATLight(torch.nn.Module):
    """Lightweight VPA-AEGF GAT used by released S2GE configurations."""
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers, dropout, num_heads=4):
        super().__init__()
        if num_layers < 1:
            raise ValueError(f"vpa_aegf_gat_light requires at least 1 layer, got {num_layers}")
        self.edge_dim = in_channels
        self.dropout = dropout
        self.num_layers = num_layers

        self.convs = torch.nn.ModuleList()
        self.bns = torch.nn.ModuleList()
        self.entropy_gates = torch.nn.ModuleList()
        self.last_node_importance = None

        if num_layers == 1:
            layer_dims = [in_channels, out_channels]
        else:
            layer_dims = [in_channels] + [hidden_channels] * (num_layers - 1) + [out_channels]
        for layer_idx in range(num_layers):
            in_dim = layer_dims[layer_idx]
            out_dim = layer_dims[layer_idx + 1]
            self.convs.append(GATConv(in_dim, out_dim, heads=num_heads, concat=False, edge_dim=in_channels))
            if layer_idx < num_layers - 1:
                self.bns.append(torch.nn.BatchNorm1d(out_dim))
            self.entropy_gates.append(torch.nn.Linear(1, 1))

    def reset_parameters(self):
        for conv in self.convs:
            conv.reset_parameters()
        for bn in self.bns:
            bn.reset_parameters()
        for gate in self.entropy_gates:
            gate.reset_parameters()

    def _entropy_gate(self, edge_index, alpha, num_nodes, gate_layer):
        eps = 1e-12
        if alpha.dim() == 1:
            alpha = alpha.unsqueeze(-1)
        alpha = alpha.clamp(min=eps)
        entropy = scatter_add(
            torch.special.entr(alpha),
            edge_index[1],
            dim=0,
            dim_size=num_nodes,
        )
        if entropy.dim() > 1:
            entropy = entropy.mean(dim=1)
        deg = scatter_add(
            torch.ones(edge_index.size(1), device=edge_index.device, dtype=entropy.dtype),
            edge_index[1],
            dim=0,
            dim_size=num_nodes,
        ).clamp(min=1.0)
        entropy_norm = torch.zeros_like(entropy)
        valid = deg > 1
        entropy_norm[valid] = entropy[valid] / deg[valid].log().clamp(min=eps)
        return torch.sigmoid(gate_layer(entropy_norm.unsqueeze(-1).to(gate_layer.weight.dtype)))

    def _mix_residual(self, residual, message, gate):
        if residual.size(-1) != message.size(-1):
            return message
        return gate * residual + (1.0 - gate) * message

    def forward(self, x, edge_index, edge_attr):
        if edge_attr is None:
            edge_attr = torch.zeros((edge_index.size(1), self.edge_dim), device=x.device)
        layer_scores = []

        for layer_idx, conv in enumerate(self.convs):
            h_prev = x
            out, (ei, alpha) = conv(
                x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                return_attention_weights=True,
            )
            deg = scatter_add(
                torch.ones(ei.size(1), device=ei.device),
                ei[1],
                dim=0,
                dim_size=out.size(0),
            )
            scale = deg.clamp(min=1.0).pow(-0.5).unsqueeze(-1)
            message = out * scale
            gate = self._entropy_gate(ei, alpha, out.size(0), self.entropy_gates[layer_idx])
            if h_prev.size(-1) == message.size(-1):
                delta = torch.norm(message - h_prev, dim=-1)
            else:
                delta = torch.norm(message, dim=-1)
            layer_scores.append(((1.0 - gate).squeeze(-1) * delta).detach())
            x = self._mix_residual(h_prev, message, gate)
            if layer_idx < len(self.bns):
                x = self.bns[layer_idx](x)
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
        self.last_node_importance = torch.stack(layer_scores, dim=0).mean(dim=0) if layer_scores else None
        return x, edge_attr


load_gnn_model = {
    "gcn": GCN,
    "gat": GAT,
    "gt": GraphTransformer,
    "anygraph": AnyGraphBackbone,
    "vpa_aegf_gt": VPAAEGFTransformer,
    "vpa_aegf_gat_light": VPAAEGFGATLight,
}
