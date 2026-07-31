"""
sxd_io.py — Read a laser-scanned shell surface from disk.

Ports the IDL ``restore`` of a ``.sxd`` file (written by surf_write.pro via
the IDL ``SAVE`` command in XDR format) so the virtual-mounting simulation can
load the same input the lab uses.

Two formats are auto-detected:
  - ``.sxd``      — the lab's IDL XDR SAVE file, read with scipy.io.readsav.
  - ``.sxd.npz``  — the Python port's own output (surf_write.py -> numpy.savez),
                    e.g. the VMfile export from surf_create_v7.py.

Both store the same seven variables (surf_write signature):
    drdtheta, drdz, signal, theta, z, dr, r0

Modification History:
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    New helper for the virtual-mounting port; mirrors IDL restore of a .sxd.
"""
import os
import numpy as np


class ShellSurface(dict):
    """Attribute-accessible container for a shell surface.

    Keys/attributes: ``theta`` [rad], ``z`` [mm], ``dr`` [mm] (ntheta, nz),
    ``drdtheta``, ``drdz``, ``signal`` (ntheta, nz), ``r0`` [mm].
    """

    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc


def _as_2d(arr):
    a = np.asarray(arr, dtype=float)
    return a


def surf_read(path):
    """Read a shell surface from a ``.sxd`` or ``.sxd.npz`` file.

    Parameters
    ----------
    path : str
        Path to a ``.sxd`` (IDL XDR) or ``.sxd.npz`` (numpy) file.  If the
        bare ``.sxd`` path is given but only ``.sxd.npz`` exists, the ``.npz``
        is used (mirrors surf_write.py appending ``.npz``).

    Returns
    -------
    ShellSurface
        dict/attribute container with theta, z, dr, drdtheta, drdz, signal, r0.
    """
    npz_path = path if path.endswith('.npz') else path + '.npz'

    if path.endswith('.npz') or (not os.path.exists(path) and os.path.exists(npz_path)):
        data = np.load(npz_path if not path.endswith('.npz') else path)
        return ShellSurface(
            theta=_as_2d(data['theta']).ravel(),
            z=_as_2d(data['z']).ravel(),
            dr=_as_2d(data['dr']),
            drdtheta=_as_2d(data['drdtheta']),
            drdz=_as_2d(data['drdz']),
            signal=_as_2d(data['signal']),
            r0=float(np.asarray(data['r0']).ravel()[0]),
        )

    # IDL XDR SAVE file — read with scipy.io.readsav (variable names lower-cased).
    from scipy.io import readsav
    sav = readsav(path, verbose=False)

    def get(name):
        # readsav exposes variables as lower-case attributes/keys.
        return sav[name] if name in sav else sav[name.lower()]

    theta = _as_2d(get('theta')).ravel()
    z = _as_2d(get('z')).ravel()
    dr = _as_2d(get('dr'))
    drdtheta = _as_2d(get('drdtheta'))
    drdz = _as_2d(get('drdz'))
    signal = _as_2d(get('signal'))
    r0 = float(np.asarray(get('r0')).ravel()[0])

    # IDL arrays restore in column-major order; surf_write stored dr as
    # (ntheta, nz) so that dr[itheta, iz] with theta the first index.  readsav
    # returns arrays with IDL's fastest-varying axis last, i.e. shape (nz,
    # ntheta) in numpy.  Transpose so the Python convention dr[theta, z] holds.
    if dr.ndim == 2 and dr.shape == (len(z), len(theta)) and len(z) != len(theta):
        dr = dr.T
        drdtheta = drdtheta.T
        drdz = drdz.T
        signal = signal.T

    return ShellSurface(theta=theta, z=z, dr=dr, drdtheta=drdtheta,
                        drdz=drdz, signal=signal, r0=r0)
