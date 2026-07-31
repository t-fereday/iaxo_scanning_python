"""
hpd_hist.py — HPD from a histogram via symmetric encircled-fraction expansion.

Ported from IDL hpd_hist.pro.

Computes the half-width at a given encircled fraction percent by expanding
a symmetric window from the centre of the histogram outward until the
enclosed fraction exceeds the target.

Modification History:
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL hpd_hist.pro
"""
import numpy as np


def hpd_hist(hist_x, hist_y, percent):
    """
    Compute the half-width at *percent* percent encircled from a histogram.

    Mirrors IDL hpd_hist.pro exactly:
      1. Build enc_y (NhistY/2 elements) of enclosed fractions.
      2. Upsample enc_y to 2*NhistY points via linear interpolation
         (IDL's interpol overwrites enc_y — case-insensitive variable).
      3. Find first upsampled index k where enc_y >= target.
      4. Return hpd = ((2*k+1)/2.) * (max-min) / NhistY.

    Parameters
    ----------
    hist_x : array-like
        Bin centres (uniformly spaced).
    hist_y : array-like
        Bin counts / weights.
    percent : float
        Target encircled fraction in percent (e.g. 50 for HPD).

    Returns
    -------
    float
        Half-width at the requested encircled fraction.
    """
    hist_x = np.asarray(hist_x, dtype=float)
    hist_y = np.asarray(hist_y, dtype=float)
    n = len(hist_y)
    total = np.sum(hist_y)

    if total == 0:
        return 0.0

    half = n // 2

    # Build NhistY/2-element enclosed-fraction array
    enc_y_orig = np.zeros(half)
    for i in range(half):
        lo = half - i
        hi = half + i + 1
        enc_y_orig[i] = np.sum(hist_y[lo:hi]) / total

    # Upsample to 2*NhistY points — mirrors IDL interpol(enc_y, 2*NhistY)
    # which reassigns enc_y (IDL is case-insensitive).
    n_up = 2 * n
    x_orig = np.linspace(0, 1, half)
    x_up   = np.linspace(0, 1, n_up)
    enc_y_up = np.interp(x_up, x_orig, enc_y_orig)

    target = percent * 0.01
    k_arr = np.where(enc_y_up >= target)[0]
    if len(k_arr) == 0:
        return float(np.max(np.abs(hist_x)))

    k = k_arr[0]
    x_range = float(np.max(hist_x) - np.min(hist_x))
    hpd = ((2 * k + 1) / 2.0) * x_range / n
    return hpd
