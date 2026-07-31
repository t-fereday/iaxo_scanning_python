"""
filters.py — Low-pass FFT filter for laser-scanner axial scan data.

Ported from IDL LowpassFilter_Old (described in surf_create_v7.pro).

The Frequency Domain Filtering technique is described in the Getting Started
with IDL Guide.  A low-pass Butterworth filter is created with a frequency
cutoff of Fcutoff (cycles per scan length).  For N data points:

  1. Build an index array Y = [0, 1, ..., N/2-1, -(N/2-1), ..., -1, 0].
  2. Create the filter: filter = 1 / (1 + (Y/Fcutoff)^10).
  3. Apply: result = IFFT(FFT(data) * filter).

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL lowpassfilter_old.pro
"""
import numpy as np


def lowpass_filter_old(data, fcutoff):
    """
    Low-pass Butterworth-style FFT filter (faithful port of IDL LowpassFilter_Old).

    The linear trend is removed before filtering and restored afterward.
    fcutoff is in units of FFT frequency bins (cycles per scan length).

    Parameters
    ----------
    data : array-like, 1-D
    fcutoff : float
        Cut-off frequency in FFT bin units.

    Returns
    -------
    numpy.ndarray
        Filtered data, same length as input.
    """
    data = np.array(data, dtype=float)
    ns = len(data)

    data_stepper = data[0] + np.arange(ns) * (data[-1] - data[0]) / ns
    data_temp = data - data_stepper

    Ffilter = np.arange(ns, dtype=float)
    ib = int(ns / 2.0 - 1.0)
    ic = int(ns / 2.0 + 0.5)
    id_val = ns - 1
    ia = max(ib - (id_val - ic), 0)
    # Negative-frequency side: mirror and negate the positive-frequency values
    Ffilter[ic: id_val + 1] = -Ffilter[ia: ib + 1][::-1]
    Ffilter = 1.0 / (1.0 + (Ffilter / fcutoff) ** 10)

    data_temp = np.real(np.fft.ifft(np.fft.fft(data_temp) * Ffilter))

    return data_temp + data_stepper
