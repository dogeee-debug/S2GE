__all__ = ["GRBenchDataset"]


def __getattr__(name):
    if name == "GRBenchDataset":
        from s2ge.data.datasets.grbench import GRBenchDataset

        return GRBenchDataset
    raise AttributeError(name)
