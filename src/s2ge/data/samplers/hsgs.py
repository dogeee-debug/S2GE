"""Hop-sensitive graph sampling schedules."""

import math


def hsgs_fanouts(num_layers: int, k_init: int = 25, gamma: float = 0.7, min_fanout: int = 1):
    """Return geometrically decaying neighbor budgets for successive hops.

    Early hops receive the largest budget because they are closest to the
    query endpoints; later hops are progressively compressed by ``gamma``.
    """
    fanouts = []
    for layer in range(num_layers):
        k_l = int(math.floor(k_init * (gamma ** layer)))
        fanouts.append(max(min_fanout, k_l))
    return fanouts
