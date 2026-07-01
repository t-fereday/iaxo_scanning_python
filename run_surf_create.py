"""
Runner script for saudi_py.surf_create.

Pass the base path to your scan folder (the folder name and scan name are
usually the same).  All .txt.* files in that folder are used automatically.

Usage:
    python3 run_surf_create.py /path/to/folder/scanname [options]

Example:
    python3 run_surf_create.py \
        /Users/Thomas/Downloads/scandata_06252026D18-260624-2/scandata_06252026D18-260624-2 \
        --r0 54 --zmin -76 --zmax 36 --scanrange 3 33 --zfix --nofilter --nocal
"""

import argparse
import os
import sys
from saudi_py import surf_create

parser = argparse.ArgumentParser(description='Run surf_create pipeline')
parser.add_argument('filename',
                    help='Base path to scan (without .txt.N suffix)')

# Shell / instrument geometry
parser.add_argument('--r0',     type=float, help='Shell radius [mm]')
parser.add_argument('--focal',  type=float, default=5200.0, help='Focal length [mm] (default: 5200)')
parser.add_argument('--glass',  default='gsfc', help='Glass type: gsfc or other (default: gsfc)')
parser.add_argument('--D0',     type=float, help='PSD distance [mm]')
parser.add_argument('--C0',     type=float, help='Cylindrical lens to beamsplitter distance [mm]')

# Axial range
parser.add_argument('--zmin',   type=float, help='Minimum axial value [mm]')
parser.add_argument('--zmax',   type=float, help='Maximum axial value [mm]')
parser.add_argument('--zstep',  type=float, help='Axial step size [mm]')
parser.add_argument('--zfix',   action='store_true', help='Fix the axial range')
parser.add_argument('--zlength',type=float, help='Axial length cut [mm]')
parser.add_argument('--zcenter',type=float, help='Axial centre offset [mm]')

# Azimuthal range
parser.add_argument('--thetamin',      type=float, help='Min azimuthal value [deg]')
parser.add_argument('--thetamax',      type=float, help='Max azimuthal value [deg]')
parser.add_argument('--alignthetacut', type=float, help='Theta cut for alignment [deg]')

# Scan selection
parser.add_argument('--scanrange', type=int, nargs=2, metavar=('START', 'END'),
                    help='Range of scan numbers to use, e.g. --scanrange 3 33')
parser.add_argument('--nscans',    type=int, help='Number of scans to use')

# Processing flags
parser.add_argument('--nocal',    action='store_true', help='Skip calibration')
parser.add_argument('--nofilter', action='store_true', help='Skip low-pass filter')
parser.add_argument('--noalign',  action='store_true', help='Skip alignment')
parser.add_argument('--noplot',   action='store_true', help='Skip 3D surface plot')
parser.add_argument('--nowrite',  action='store_true', help='Skip .npz output')
parser.add_argument('--convol',   action='store_true', help='Use 2-bounce convolution for HPD')
parser.add_argument('--fitdeg',   type=int, default=2, help='Axial polynomial degree (default: 2)')
parser.add_argument('--VMfile',   action='store_true', help='VMfile mode')
parser.add_argument('--quiet',    action='store_true', help='Suppress verbose output')

args = parser.parse_args()

# Strip .txt.N suffix if the user accidentally passed the full filename
filename = args.filename
for _ in range(2):
    root, ext = os.path.splitext(filename)
    if ext.startswith('.txt') or (ext.startswith('.') and ext[1:].isdigit()):
        filename = root
    else:
        break

result = surf_create(
    filename,
    r0=args.r0,
    focal=args.focal,
    glass=args.glass,
    D0=args.D0,
    C0=args.C0,
    zmin=args.zmin,
    zmax=args.zmax,
    zstep=args.zstep,
    zfix=args.zfix,
    zlength=args.zlength,
    zcenter=args.zcenter,
    thetamin=args.thetamin,
    thetamax=args.thetamax,
    alignthetacut=args.alignthetacut,
    scanrange=args.scanrange,
    nscans=args.nscans,
    nocal=args.nocal,
    nofilter=args.nofilter,
    noalign=args.noalign,
    noplot=args.noplot,
    nowrite=args.nowrite,
    convol=args.convol,
    fitdeg=args.fitdeg,
    VMfile=args.VMfile,
    quiet=args.quiet,
    noformat=True,
)

if result is None:
    print("surf_create returned None — check data error messages above.")
    sys.exit(1)
