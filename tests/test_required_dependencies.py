import torch


def test_gensim_keyed_vectors_smoke():
    from gensim.models import KeyedVectors

    vectors = KeyedVectors(vector_size=4)
    vectors.add_vectors(["node"], [[1.0, 0.0, 0.0, 0.0]])

    assert vectors["node"].shape == (4,)


def test_torch_scatter_extension_smoke():
    from torch_scatter import scatter_add

    values = torch.tensor([1.0, 2.0, 3.0])
    index = torch.tensor([0, 1, 0])
    result = scatter_add(values, index, dim=0, dim_size=2)

    assert torch.equal(result, torch.tensor([4.0, 2.0]))
