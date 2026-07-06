"""
hist1d.py — 1-D weighted histogram with optional King/Gaussian fit and 2-bounce convolution.

Ported from IDL hist1d.pro / hist1dfit (combined here as hist1dfit).

Returns a one-dimensional histogram of the input data with optional:
  - Normalisation to unit area (/norm)
  - Gaussian overlay fitted with scipy.optimize.curve_fit
  - King profile overlay: A0 / (1 + ((x-x0)/r)^2)^alpha
  - Self-convolution (2-bounce PSF simulation) before computing W50/W70
  - HPD at 50% and 70% via hpd_hist

The bin assignment loop faithfully ports the IDL histogram loop (including
the asymmetric boundary condition: bin i receives events in [x_i - dx/2,
x_i + dx/2)).  The first and last bins are left empty to match IDL behaviour.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
    Dec 2004: added scale keyword
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL hist1d.pro; IDL mpfitfun replaced with
    scipy.optimize.curve_fit; IDL PLOT replaced with matplotlib
"""
import numpy as np
from scipy.optimize import curve_fit
from .hpd_hist import hpd_hist
from .fit_functions import fit_king


def _fit_gauss(x, A0, x0, sigma):
    return A0 * np.exp(-0.5 * ((x - x0) / sigma) ** 2)


def _fit_king_scipy(x, A0, x2, r2, alpha2):
    return fit_king(x, [A0, x2, r2, alpha2])


def hist1dfit(hist, min1=None, max1=None, nbins1=None,
              norm=False, gauss=False, king=False,
              noplot=False, nsigma=5.0, zero=False,
              convol=False, weight=None, scale=1.0,
              noinfo=False,
              title=None, xtit=None,
              ax=None,
              x_hist=None):
    """
    1-D histogram with optional King or Gaussian fit and 2-bounce convolution.

    Equivalent to IDL hist1dfit.

    Parameters
    ----------
    hist : array-like  1-D data to histogram
    min1, max1 : float, optional  histogram range
    nbins1 : int, optional  number of bins
    norm : bool  normalise histogram to unit area
    gauss : bool  fit and overplot a Gaussian
    king : bool   fit and overplot a King profile
    noplot : bool  suppress plotting
    nsigma : float  set range to ±nsigma*std (when zero=False)
    zero : bool    centre range about zero
    convol : bool  convolve histogram with itself (2-bounce PSF)
    weight : array-like, optional  per-bin weights
    scale : float  uniform weight scale factor
    noinfo : bool  suppress text annotation on plot
    title : str    plot title
    xtit : str     x-axis label
    ax : matplotlib.axes.Axes, optional  axes to plot into

    Returns
    -------
    x_hist : ndarray  bin centres
    y_hist : ndarray  bin counts (or weighted sums)
    fit_output : ndarray or None  fit parameters
    W50convol : float  50 % encircled half-width (after convolution)
    W70convol : float  70 % encircled half-width (after convolution)
    """
    hist1 = np.asarray(hist, dtype=float).ravel()

    avg  = np.mean(hist1)
    std  = np.std(hist1)

    if x_hist is not None:
        # Use provided bin centres directly (matches IDL behaviour when x_hist
        # is passed as an already-set output variable from a previous call).
        x_hist = np.asarray(x_hist, dtype=float)
        nbins1 = len(x_hist)
        bin1   = x_hist[1] - x_hist[0]
    else:
        if nbins1 is None:
            nbins1 = max(int(len(hist1) / 10.0), 10)

        if min1 is None:
            if nsigma:
                min1 = -nsigma * std if zero else avg - nsigma * std
            else:
                min1 = np.min(hist1)
        if max1 is None:
            if nsigma:
                max1 = nsigma * std if zero else avg + nsigma * std
            else:
                max1 = np.max(hist1)

        if zero:
            bound = max(abs(min1), abs(max1))
            min1, max1 = -bound, bound

        if min1 >= max1:
            min1 = avg - std
            max1 = avg + std

        bin1   = (max1 - min1) / nbins1
        x_hist = np.arange(nbins1) * bin1 + min1
        nbins1 = len(x_hist)

    if weight is None:
        w = np.full(len(hist1), scale)
    else:
        w = np.asarray(weight, dtype=float).ravel()

    y_hist = np.zeros(nbins1)
    for i in range(1, nbins1 - 1):
        mask = (hist1 - x_hist[i] < bin1 / 2.0) & (hist1 - x_hist[i] >= -bin1 / 2.0)
        if np.any(mask):
            y_hist[i] = np.sum(w[mask] if len(w) == len(hist1) else w[mask])

    if norm:
        total = np.sum(y_hist)
        if total > 0:
            y_hist = y_hist / total

    if convol:
        kernel = y_hist / np.sum(y_hist) if np.sum(y_hist) > 0 else y_hist
        y_hist = np.convolve(y_hist, kernel[::-1], mode='same')

    W50convol = hpd_hist(x_hist, y_hist, 50)
    W70convol = hpd_hist(x_hist, y_hist, 70)

    # PSF fitting — weighted exactly like IDL hist1dfit.pro (mpfitfun with
    # y_err = sqrt(y_hist) floored at 1). Without the weighting, an unweighted fit
    # over-weights the peak and gives a too-narrow king (wrong r/alpha). Fit runs
    # regardless of noplot (as in IDL), so fit_output is always available.
    fit_output = None
    fit_curve  = None
    chi_sq     = None
    y_err = np.maximum(np.sqrt(np.abs(y_hist)), 1.0)

    if king:
        try:
            p0 = [np.max(y_hist), 0.0, 20.0, 1.0]
            popt, _ = curve_fit(_fit_king_scipy, x_hist, y_hist, p0=p0,
                                 sigma=y_err, absolute_sigma=True, maxfev=5000,
                                 bounds=([-np.inf, -np.inf, 0.0, 0.0],
                                         [np.inf, np.inf, np.inf, 10.0]))
            fit_output = popt
            fit_curve  = _fit_king_scipy(x_hist, *popt)
            chi_sq = np.sum(((y_hist - fit_curve) / y_err) ** 2) / (len(x_hist) - len(p0))
        except Exception:
            pass

    if gauss:
        try:
            p0 = [np.max(y_hist), 0.0, std if std > 0 else 1.0]
            popt, _ = curve_fit(_fit_gauss, x_hist, y_hist, p0=p0,
                                 sigma=y_err, absolute_sigma=True, maxfev=5000)
            fit_output = popt
            fit_curve  = _fit_gauss(x_hist, *popt)
            chi_sq = np.sum(((y_hist - fit_curve) / y_err) ** 2) / (len(x_hist) - len(p0))
        except Exception:
            pass

    if not noplot:
        import matplotlib.pyplot as plt
        import scipy.stats as _stats
        if ax is None:
            fig, ax = plt.subplots()

        ax.step(x_hist, y_hist, where='mid')
        if fit_curve is not None:
            ax.plot(x_hist, fit_curve, 'r-')

        if xtit:
            ax.set_xlabel(xtit)
        if title:
            ax.set_title(title)

        if not noinfo:
            # Annotation matches IDL hist1dfit.pro:211-252 (Events/Nbins, then
            # skewness/kurtosis for non-convol or W50/W70 for convol, then the
            # king (r/alpha/Chisq) or gauss (x0/sigma/A0/Chisq) fit params).
            info = [f"Events: {int(round(np.sum(y_hist)))}", f"Nbins: {len(x_hist)}"]
            if convol:
                info += [f'W50 = {W50convol:.1f}"', f'W70 = {W70convol:.1f}"']
            else:
                info += [f"skewness = {_stats.skew(hist1):.2f}",
                         f"kurtosis = {_stats.kurtosis(hist1):.2f}"]
            if king and fit_output is not None:
                info += [f"r = {fit_output[2]:.2f}",
                         f"alpha = {fit_output[3]:.1f}",
                         f"Chisq = {chi_sq:.1f}"]
            if gauss and fit_output is not None:
                info += [f"x0 = {fit_output[1]:.2f}",
                         f"sigma = {fit_output[2]:.2f}",
                         f"A0 = {fit_output[0]:.1f}",
                         f"Chisq = {chi_sq:.1f}"]
            ax.text(0.97, 0.97, '\n'.join(info),
                    transform=ax.transAxes, ha='right', va='top', fontsize=7,
                    family='monospace',
                    bbox=dict(boxstyle='square,pad=0.3', facecolor='white',
                              edgecolor='black', linewidth=0.6))

    return x_hist, y_hist, fit_output, W50convol, W70convol
