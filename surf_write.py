"""
surf_write.py — Save reconstructed laser-scanner surface data to disk.

Ported from IDL surf_write.pro.

The original IDL procedure used the IDL SAVE command to write a proprietary
binary XDR file (.sxd).  This Python port uses numpy.savez to write a
compressed NumPy archive (.sxd.npz), which is readable with numpy.load.

To read back in Python:
    data = np.load('output.sxd.npz')
    dr = data['dr']

To read the original IDL .sxd files for comparison:
    import scipy.io
    data = scipy.io.readsav('output.sxd')

Modification History:
  Jason Koglin, Columbia Astrophysics Laboratory, Feb 2001
    koglin@astro.columbia.edu
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Python port from IDL surf_write.pro; IDL SAVE (.xdr/.sxd) replaced
    with numpy.savez (.sxd.npz)
"""
import os
import numpy as np


def surf_write(filename, drdtheta, drdz, signal, theta, z, dr, r0):
    """
    Save reconstructed surface data to a compressed NumPy archive (.npz).

    Equivalent to IDL surf_write (which uses IDL SAVE → .xdr binary).
    The output file will have the extension ``.npz`` appended if the caller
    passes a bare ``.sxd`` path.

    To read the saved file:
        data = np.load('output.sxd.npz')
        dr = data['dr']

    Parameters
    ----------
    filename : str
        Output file path (e.g. ``'output/shell.sxd'``).
    drdtheta : ndarray  (ntheta, nz)  azimuthal slopes
    drdz     : ndarray  (ntheta, nz)  axial slopes
    signal   : ndarray  (ntheta, nz)  laser signal
    theta    : ndarray  (ntheta,)     azimuthal coordinates [rad]
    z        : ndarray  (nz,)         axial coordinates [mm]
    dr       : ndarray  (ntheta, nz)  surface height residuals [mm]
    r0       : float                  nominal shell radius [mm]
    """
    dirpath = os.path.dirname(filename)
    if dirpath:
        os.makedirs(dirpath, exist_ok=True)

    np.savez(filename,
             drdtheta=drdtheta,
             drdz=drdz,
             signal=signal,
             theta=theta,
             z=z,
             dr=dr,
             r0=np.array(r0))

    print(f"save to {filename}.npz")
