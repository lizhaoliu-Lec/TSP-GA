from random import sample


def get_k_idx(min_val, max_val, k=2, sort=False):
    """
    interval: [min_val, max_val)
    """
    idx = sample(range(min_val, max_val), k)
    if sort:
        idx = sorted(idx)
    return idx
