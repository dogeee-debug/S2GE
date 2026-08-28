import torch

from s2ge.models.encoders.gnn_backbones import GAT, VPAAEGFGATLight


def test_gat_true_single_layer_has_one_conv_and_no_bn():
    model = GAT(
        in_channels=8,
        hidden_channels=16,
        out_channels=8,
        num_layers=1,
        dropout=0.0,
        num_heads=2,
    )

    assert len(model.convs) == 1
    assert len(model.bns) == 0


def test_vpa_aegf_light_true_single_layer_runs_forward():
    model = VPAAEGFGATLight(
        in_channels=8,
        hidden_channels=16,
        out_channels=8,
        num_layers=1,
        dropout=0.0,
        num_heads=2,
    )
    x = torch.randn(3, 8)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_attr = torch.randn(edge_index.size(1), 8)

    out, returned_edge_attr = model(x, edge_index, edge_attr)

    assert len(model.convs) == 1
    assert len(model.bns) == 0
    assert len(model.entropy_gates) == 1
    assert out.shape == (3, 8)
    assert returned_edge_attr.shape == edge_attr.shape
    assert model.last_node_importance.shape == (3,)
