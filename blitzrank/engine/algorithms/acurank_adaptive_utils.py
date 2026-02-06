"""
Adaptive utilities for AcuRank. (https://github.com/soyoung97/AcuRank/blob/main/adaptive_utils.py)
"""

import torch
import math

def normal_cdf(x):
    # CDF of the standard normal distribution
    # torch.erf(x / math.sqrt(2)) ranges from -1 to 1. 0.5*(1 + erf(...)) converts to [0,1].
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2)))

def mixture_cdf(x, mu, std):
    """
    Compute the CDF of the mixture at point x.
    x: scalar (float) or 0-dim tensor
    mu, std: 1D tensors of shape [n]
    Returns a scalar tensor.
    """
    # Expand x to the same shape as mu, std for elementwise operation
    x_expanded = x.unsqueeze(-1)  # shape [1]
    # Standardize and compute each Normal's CDF
    z = (x_expanded - mu) / std  # shape [n]
    cdfs = normal_cdf(z)         # shape [n]
    return cdfs.mean()           # average over the n Gaussians

def _find_threshold_binary_search(q, mu, std, tol=1e-7, max_steps=5000):
    # == mixture_icdf
    #print(f"q: {q}")
    if q >= 1: # total len(candidates) is less than self.args.R/
        # k = math.sqrt(2) * torch.special.erfinv(torch.tensor(2*p - 1)) = 3.0902
        p = 0.999
        k = 3.0902
        t  = (mu - k * std).min()
        return t
    """
    Find t such that P(X > t) = q for a mixture of Gaussians (equal weights).
    Equivalently, F(t) = 1 - q, where F is the mixture CDF.

    q: scalar in [0,1].
    mu, std: 1D tensors of shape [n].
    tol: tolerance for the final bracket size in bisection.
    max_steps: max number of bisection steps.
    K: controls initial bracket width in terms of standard deviations.

    Returns a scalar (float) t.
    """
    # Convert right-tail probability q to left-tail = (1 - q).
    target_cdf_value = 1.0 - q

    # 1) Bracket [left, right]
    left = mu.min() - 5 * std.max()
    right = mu.max() + 5 * std.max()

    # Evaluate g(left) and g(right)
    def g(x):
        return mixture_cdf(x, mu, std) - target_cdf_value

    g_left = g(left)
    g_right = g(right)

    # If these don't bracket zero, widen further or raise an error
    if g_left > 0:
        # Move left outward until g_left < 0
        while g_left > 0:
            left -= (right - left)
            g_left = g(left)
    if g_right < 0:
        # Move right outward until g_right > 0
        while g_right < 0:
            right += (right - left)
            g_right = g(right)

    # 2) Bisection loop
    for step in range(max_steps):
        mid = 0.5 * (left + right)
        g_mid = g(mid)

        if g_mid == 0.0:
            return mid  # found exact root (unlikely in floating-point)
        if g_mid < 0.0:
            # This means mixture_cdf(mid) < target_cdf_value
            # so we need a bigger x
            left = mid
        else:
            # mixture_cdf(mid) > target_cdf_value
            # we need a smaller x
            right = mid

        if (right - left) < tol:
            break
    #if (right - left) >= tol:
        #print(f"Failed to close gap, but step size {max_steps} ended!!!!")

    return 0.5 * (left + right)  # final estimate


if __name__ == '__main__':
    # Example usage
    torch.manual_seed(0)
    n = 3
    mu = torch.tensor([0.0, 1.0, 2.0])
    std = torch.tensor([0.5, 1.0, 1.5])
    q = 0.05  # we want P(X > t) = 0.05

    t = _find_threshold_binary_search(q, mu, std)
    #t = mixture_icdf(q, mu, std)
    print(f"Quantile t such that P(X > t) = {q:.2f} is approximately {t.item()}")

    # Test
    print(f"For the given t, P(X > t) = {1 - mixture_cdf(t, mu, std):.2f}")