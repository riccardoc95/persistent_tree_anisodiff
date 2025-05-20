import numpy as np

# http://wscg.zcu.cz/wscg2019/2019-papers/!_2019_JWSCG-1-5.pdf

def compute_c(p, a):
    """
    Computes c(x, y) element-wise between prediction p and reference a.
    Both inputs must be 2D numpy arrays of the same shape.
    """
    c = np.zeros_like(p, dtype=np.float64)

    same_sign = np.sign(p) == np.sign(a)
    neg_pos = (p < 0) & (a > 0)
    pos_neg = (p > 0) & (a < 0)

    # Case 1: same sign
    c[same_sign] = np.abs(p[same_sign] - a[same_sign]) / (np.abs(p[same_sign]) + np.abs(a[same_sign]) + 1e-12)

    # Case 2: p < 0 and a > 0
    a2 = a[neg_pos]
    p2 = p[neg_pos]
    c[neg_pos] = np.abs((a2 + (a2 - p2)) - a2) / (np.abs(p2 + (a2 - p2)) + 1e-12)

    # Case 3: p > 0 and a < 0
    a3 = a[pos_neg]
    p3 = p[pos_neg]
    c[pos_neg] = np.abs((a3 + (a3 - p3)) + a3) / (np.abs(p3 + (p3 - a3)) + 1e-12)

    return c


def compute_CIRR(p, a, chat):
    """
    Computes CIRR between predicted (p), reference (a), and estimated (chat).
    All inputs must be 2D numpy arrays of the same shape.
    """
    c_true = compute_c(p, a)
    c_hat = compute_c(chat, a)

    numerator = np.sum((c_true - c_hat) ** 2)
    denominator = np.sum(c_true ** 2) + 1e-12  # avoid division by zero

    return numerator / denominator
