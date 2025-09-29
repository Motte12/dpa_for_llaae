import torch

def pearsonr_cols(x, y, dim=0, eps=1e-12):
    """
    Pearson correlation per feature along `dim` (default: samples axis).
    x, y: tensors of the same shape, e.g. (n_samples, n_features)
    Returns: tensor of shape equal to the non-reduced dims (e.g. (n_features,))
    """
    # center
    x_mean = x.mean(dim=dim, keepdim=True)
    y_mean = y.mean(dim=dim, keepdim=True)
    x_c = x - x_mean
    y_c = y - y_mean

    # numerator: covariance (without / (n-1) since it cancels in correlation)
    num = (x_c * y_c).sum(dim=dim)

    # denominator: product of std devs
    x_ss = (x_c * x_c).sum(dim=dim)
    y_ss = (y_c * y_c).sum(dim=dim)
    den = (x_ss * y_ss).sqrt().clamp_min(eps)

    return num / den

def r2_score(y_true, y_pred, dim=0):
    """
    Compute R² per feature along `dim`.
    y_true, y_pred: torch tensors of same shape
    """
    ss_res = torch.sum((y_true - y_pred) ** 2, dim=dim)
    ss_tot = torch.sum((y_true - y_true.mean(dim=dim, keepdim=True)) ** 2, dim=dim)
    r2 = 1 - ss_res / ss_tot
    return r2


def reliability_index(counts, normalize=True):
    """
    Compute Reliability Index (RI) for a rank histogram.

    Parameters
    ----------
    counts : array-like, shape (B,)
        Bin counts of the rank histogram.
    normalize : bool, default=True
        If True, scale RI into [0, 1].

    Returns
    -------
    RI : float
        Reliability Index (0 = flat, higher = less reliable).
    """
    counts = np.asarray(counts, dtype=float)
    N = counts.sum()
    B = counts.size
    if N == 0:
        return np.nan  # undefined if no counts
    
    freqs = counts / N
    RI_raw = np.abs(freqs - 1.0/B).sum()

    if normalize:
        RI = RI_raw / (2.0 * (1 - 1.0/B))
        return RI
    else:
        return RI_raw
    