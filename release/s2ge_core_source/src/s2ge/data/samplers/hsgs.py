import math


def hsgs_fanouts(num_layers: int, k_init: int = 25, gamma: float = 0.7, min_fanout: int = 1):
    fanouts = []
    for layer in range(num_layers):
        k_l = int(math.floor(k_init * (gamma ** layer)))
        fanouts.append(max(min_fanout, k_l))
    return fanouts
