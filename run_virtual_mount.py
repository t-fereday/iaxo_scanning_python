"""
Runner for the virtual-mounting (PVM) simulation.

Python port of the IDL command:
    LVDTanalysis, module="PVM", vlayer=indgen(1)+N, /HPDsummary, /reloadlaser
reduced to its mount-simulation core: one laser-scanned shell (.sxd or .sxd.npz)
-> a six-panel Mount figure (Raw / Raw-LP / Raw-HP / Mount-LP / Mount-HP /
Mounted) with 2-bounce HPDs. By default three figures are produced from the one
simulation run — idl (original-style contours), heatmap, and labeled (values on
the contours) — saved to py_out/ (pdf+png); pick one with --style. The
interactive matplotlib window opens after the run (suppress with --noshow).

Usage:
    python3 run_virtual_mount.py <path.sxd|.sxd.npz> [options]

Example:
    python3 run_virtual_mount.py \
        "vmexample/A115AD13-001P0-U307162026-260618.sxd" --focal 5600

Notes:
  * r0 defaults to the shell diameter parsed from the filename (per shell).
  * The layer defaults to the value parsed from the filename; --layer overrides.
  * The spacer figure error is RANDOM and unseeded by default (matching IDL), so
    the Mount-High-Pass and Mounted HPDs fluctuate run-to-run.  Pass --seed for a
    reproducible realization.
"""
import argparse
import sys

from saudi_py.vmount import virtual_mount

parser = argparse.ArgumentParser(description='Virtual-mounting (PVM) simulation')
parser.add_argument('sxd', help='Path to shell surface (.sxd or .sxd.npz)')
parser.add_argument('--layer', type=int, help='Layer number (default: from filename)')
parser.add_argument('--r0', type=float, help='Shell radius [mm] (default: filename diameter/2)')
parser.add_argument('--focal', type=float, default=5600.0, help='Focal length [mm] (default 5600)')
parser.add_argument('--out', help='Output figure path (default: py_out/<name>_Mount.pdf)')
parser.add_argument('--seed', type=int, help='RNG seed for the spacer error (default: unseeded)')
parser.add_argument('--geometry', help='Path to BabyIAXO_geometry.txt (optional; feeds unused dr_conic)')
parser.add_argument('--noplot', action='store_true', help='Skip figure; print HPDs only')
parser.add_argument('--noshow', action='store_true',
                    help='Save the figure without opening the matplotlib window')
parser.add_argument('--style', default='all',
                    choices=['all', 'idl', 'heatmap', 'labeled'],
                    help='Panel style: idl (original contours), heatmap, '
                         'labeled (values on contours), or all three (default)')
parser.add_argument('--quiet', action='store_true', help='Suppress verbose output')

args = parser.parse_args()

result = virtual_mount(
    args.sxd,
    layer=args.layer,
    r0=args.r0,
    focal=args.focal,
    out=args.out,
    seed=args.seed,
    geometry_file=args.geometry,
    make_plot=not args.noplot,
    show=not args.noshow,
    style=args.style,
    quiet=args.quiet,
)

if result is None:
    print('virtual_mount returned None — check messages above.')
    sys.exit(1)
