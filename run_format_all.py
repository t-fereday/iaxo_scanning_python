"""
run_format_all.py — Stub for the IDL run_format_all procedure.

The original IDL procedure calls format_psdmotor to convert raw `.psd.*` and
`.mot.*` hardware files from the laser-scanner motor controller into the
5-column `.txt.*` format expected by read_psd.  That conversion is
hardware-specific to the NuSTAR/IAXO laser-scanner system and is not ported.

This stub assumes the `.txt.*` files already exist in filedir.

Modification History:
  Thomas Fereday, Nevis Labs REU Student, July 2026
    thomas@fereday.org
    Stub — format_psdmotor (hardware-specific) not ported to Python
"""
import warnings


def run_format_all(filedir=None, shellname=None, nsnum=None,
                   mincounts=None, scanrange=None, extrascans=False, **kwargs):
    """
    Stub for the IDL run_format_all procedure.

    The original IDL version converts raw `.psd.*` / `.mot.*` hardware files
    into the `.txt.*` format expected by read_psd.  That conversion is
    hardware-specific (format_psdmotor) and is not ported here.

    This stub assumes the `.txt.*` files already exist in *filedir*.
    """
    warnings.warn(
        "run_format_all: stub — assuming .txt.* files already exist in "
        f"{filedir!r}.  The raw-format conversion (format_psdmotor) is "
        "not implemented in saudi_py.",
        stacklevel=2,
    )
