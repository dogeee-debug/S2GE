"""Factory translating training arguments into a graph encoder backbone."""

from s2ge.models.encoders.gnn_backbones import load_gnn_model


class DeepS2GEEncoderFactory:
    """Construct the configured GNN with a uniform dimensional contract."""
    @staticmethod
    def build(args):
        return load_gnn_model[args.gnn_model_name](
            in_channels=args.gnn_in_dim,
            out_channels=args.gnn_hidden_dim,
            hidden_channels=args.gnn_hidden_dim,
            num_layers=args.gnn_num_layers,
            dropout=args.gnn_dropout,
            num_heads=args.gnn_num_heads,
        )
