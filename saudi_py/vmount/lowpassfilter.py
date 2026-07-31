"""
lowpassfilter.py — Frequency-domain low-pass filter (LVDT/mount-path variant).

Ported from IDL lowpassfilter.pro (Jason Koglin, CAL, Feb 2001; endpoint-fit
Apr 2006; sigma-weeding added later).  This is the *fuller* standalone routine
used by the virtual-mounting analysis — NOT the simpler LowpassFilter_Old
already in saudi_py/filters.py.

A 10th-order Butterworth low-pass (5th order in power) is applied in the
frequency domain with cutoff ``Fcutoff`` = number of cycles allowed over the
data length.  The series is padded to a power of two with a point-reflected
mirror about fitted end values (for continuity across the FFT wrap), a linear
"stepper" is removed and restored, and — by default — the two end values are
optimized (IDL mpfitfun -> scipy.optimize.least_squares) so the filtered curve
best matches the data.  An optional sigma-weeding pass removes outliers before
filtering.

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001 / Apr 2006
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL lowpassfilter.pro; IDL FFT convention reproduced
    explicitly, IDL SMOOTH via a boxcar with unchanged ends, IDL mpfitfun via
    scipy.optimize.least_squares, interpol(/quad) via scipy interp1d.
"""
import numpy as np
from scipy.optimize import least_squares
from scipy.interpolate import interp1d


def _smooth(a, width):
    """IDL SMOOTH: centered boxcar average, ends (width//2) left unchanged."""
    a = np.asarray(a, dtype=float)
    w = int(width)
    if w < 2 or w > len(a):
        return a.copy()
    half = w // 2
    out = a.copy()
    kernel = np.ones(w) / w
    conv = np.convolve(a, kernel, mode='same')
    out[half:len(a) - half] = conv[half:len(a) - half]
    return out


def _idl_fft(x, direction):
    """IDL FFT(x, direction): -1 forward (1/N normalized), +1 inverse (un-normalized)."""
    x = np.asarray(x, dtype=complex)
    n = len(x)
    if direction > 0:
        return n * np.fft.ifft(x)
    return np.fft.fft(x) / n


def _butterworth_freq(ns, fcutoff):
    """Symmetric 10th-order Butterworth frequency response of length ns."""
    ffilter = np.arange(ns, dtype=float)
    ib = int(ns / 2.0 - 1.0)
    ic = int(ns / 2.0 + 0.5)
    idd = int(ns - 1)
    ia = int(ib - (idd - ic))
    # Ffilter(ic:id) = -reverse(Ffilter(ia:ib))   (mirror to negative frequencies)
    ffilter[ic:idd + 1] = -ffilter[ia:ib + 1][::-1]
    return 1.0 / (1.0 + (ffilter / fcutoff) ** 10)


def _apply_filter_endpoints(data0, fcutoff, endpoints):
    """Padded-to-2^n, mirror-about-endpoints filter (Fcutoff doubled per IDL)."""
    data0 = np.asarray(data0, dtype=float)
    ns0 = len(data0)
    ns = 2 ** int(np.ceil(np.log(ns0) / np.log(2.0)))
    if ns <= ns0 * 1.3:
        ns *= 2
    ns0a = int((ns - ns0) / 2.0)
    ns0b = int(ns - ns0 - ns0a)

    front = -data0[0:ns0a][::-1] + 2.0 * endpoints[0]
    back = -data0[ns0 - ns0b:ns0][::-1] + 2.0 * endpoints[1]
    data = np.concatenate([front, data0, back])
    ns = len(data)

    data_stepper = data[0] + np.arange(ns) * (data[ns - 1] - data[0]) / ns
    data_temp = data - data_stepper

    ffilter = _butterworth_freq(ns, fcutoff * 2.0)
    data_temp = np.real(_idl_fft(_idl_fft(data_temp, 1) * ffilter, -1))

    data_filt = data_temp[0:ns] + data_stepper[0:ns]
    return data_filt[ns0a:ns - ns0b]


def _apply_filter_simple(data0, fcutoff):
    """Un-padded filter (Fcutoff not doubled), used when no endpoints supplied."""
    data = np.asarray(data0, dtype=float)
    ns = len(data)
    data_stepper = data[0] + np.arange(ns) * (data[ns - 1] - data[0]) / ns
    data_temp = data - data_stepper
    ffilter = _butterworth_freq(ns, fcutoff)
    data_temp = np.real(_idl_fft(_idl_fft(data_temp, 1) * ffilter, -1))
    return data_temp[0:ns] + data_stepper[0:ns]


def _weed(data_in, weedsigma, weedsmoothscale):
    """Sigma-weeding: interpolate over outliers before filtering (IDL weed branch)."""
    data_in = np.asarray(data_in, dtype=float)
    n = len(data_in)
    if not weedsmoothscale:
        weedsmoothscale = 5
    time = np.arange(n, dtype=float)

    def good_indices(datanew):
        ddata0 = datanew - _smooth(datanew, weedsmoothscale * 10)
        ddata0 = np.where(np.isfinite(ddata0), ddata0, 0.0)
        std_val = np.std(ddata0, ddof=1) if n > 1 else 1.0
        if std_val <= 0 or not np.isfinite(std_val):
            std_val = 1.0
        cond = ((ddata0 / std_val <= weedsigma)
                & (np.abs(np.cumsum(ddata0 / std_val)) <= weedsigma * 0.8)
                & (time > weedsmoothscale / 2)
                & (time < time.max() - weedsmoothscale / 2))
        return np.where(cond)[0]

    gind = good_indices(data_in)
    thresh = n - weedsmoothscale * 10
    if len(gind) > thresh:
        # remove a parabola estimated from the double derivative, then re-weed
        d2 = np.mean(np.gradient(np.gradient(data_in[gind])))
        datanew = data_in - d2 * (time - np.mean(time)) ** 2 / 4.0
        gind = good_indices(datanew)

    if len(gind) > thresh and len(gind) >= 3:
        f = interp1d(time[gind], data_in[gind], kind='quadratic',
                     fill_value='extrapolate', bounds_error=False)
        return f(time)
    return data_in


def lowpassfilter(data_in, fcutoff=10.0, endpoints=None, no_fit_endpoints=False,
                  errors=None, weedsigma=None, weedsmoothscale=None):
    """Low-pass filter a 1-D series.

    Equivalent to IDL ``lowpassfilter(data, Fcutoff, ...)``.

    Parameters
    ----------
    data_in : array-like        the series to filter
    fcutoff : float             cycles allowed over the data length
    endpoints : (2,) array-like, optional  end values for continuity
    no_fit_endpoints : bool     if True, do not optimize the end values
    errors : array-like, optional  per-point errors for the endpoint fit
    weedsigma, weedsmoothscale : optional  enable outlier weeding

    Returns
    -------
    ndarray  filtered series (same length as ``data_in``)
    """
    data_in = np.asarray(data_in, dtype=float)
    if not fcutoff:
        fcutoff = 10.0

    if weedsigma:
        data0 = _weed(data_in, weedsigma, weedsmoothscale)
    else:
        data0 = data_in

    ns = len(data0)

    if not no_fit_endpoints:
        if endpoints is None:
            endpoints = np.array([data0[0], data0[ns - 1]], dtype=float)
        if errors is None:
            resid = data0 - _smooth(data0, min(10, ns // 2))
            err = np.std(resid, ddof=1) if ns > 1 else 0.0
            errors = np.full(ns, max(err, 0.05))

        def residual(ep):
            model = _apply_filter_endpoints(data0, fcutoff, ep)
            return (data0 - model) / errors

        sol = least_squares(residual, np.asarray(endpoints, dtype=float))
        return _apply_filter_endpoints(data0, fcutoff, sol.x)

    if endpoints is not None:
        return _apply_filter_endpoints(data0, fcutoff, endpoints)

    return _apply_filter_simple(data0, fcutoff)
