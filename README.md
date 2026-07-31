# iaxo_scanning_python

A Python port of the IDL laser-scanner analysis used for IAXO / BabyIAXO X-ray
optics. It reconstructs a mirror shell's surface from laser-scanner data and
simulates how that shell deforms once it's mounted on spacers.

Original IDL by Jason Koglin (Columbia Astrophysics Laboratory).

## Requirements

Python 3 with `numpy`, `scipy`, `matplotlib`.

## Layout

    run_surf_create.py     raw laser scan  ->  surface file
    run_virtual_mount.py   surface file    ->  mount simulation + HPDs
    saudi_py/              the package
      └── vmount/          virtual-mounting subpackage
    vmexample/             a worked example you can run straight away

## 1. Surface reconstruction

The scanner measures the surface *slope*, not its height, so the pipeline
integrates the slopes back into a height map, then removes the per-scan offsets
and misalignments that integration exposes.

Point it at a scan folder (the folder and scan name are usually the same — all
`.txt.*` files in it are picked up automatically):

```bash
python3 run_surf_create.py ~/scans/scandata_06252026D18-260624-2/scandata_06252026D18-260624-2 \
    --r0 54 --zmin -76 --zmax 36 --scanrange 3 33 --zfix --nofilter --nocal
```

Add `--VMfile` to also export the surface for the mount simulation. Output goes
to `SXD/` as `.sxd.npz`. Useful flags: `--r0` (shell radius, mm), `--zmin/--zmax`,
`--nocal`, `--noplot`. Run with `-h` for the full list.

Calibration files, if you have them, go in `calibration/`. Without them the run
still completes — it warns and uses a zero baseline.

## 2. Virtual mounting

Simulates gluing the shell onto five graphite spacers: the glass flexes to meet
them, and the resulting figure error sets the optic's Half-Power Diameter
(HPD, in arcsec — smaller is sharper).

```bash
python3 run_virtual_mount.py vmexample/A115AD13-001P0-U307162026-260618.sxd
```

Produces a six-panel figure (Raw / Raw-LP / Raw-HP / Mount-LP / Mount-HP /
Mounted) with 2-bounce HPDs, saved to `py_out/` as PDF + PNG, and opens it
interactively. Options:

- `--style {idl,heatmap,labeled,all}` — panel rendering; `all` (default) writes
  one figure of each from a single run
- `--noshow` — save without opening a window
- `--r0`, `--layer`, `--focal` — override values parsed from the filename
- `--seed` — reproducible spacer error

### A note on randomness

The spacers aren't perfectly flat, so the simulation draws a random figure error
for each one. It's **unseeded by default**, matching the IDL, so the
*Mount High Pass* and *Mounted* panels vary slightly run to run. The other four
panels are deterministic. Use `--seed` for a repeatable result.

## Comparing against the original IDL

`idl_out/` holds a harness that runs the original IDL routines and diffs them
against this port. It needs IDL and a copy of the lab's `user_contrib` tree
(not included — not ours to redistribute); point `IAXO_IDL` at it and `IDL_BIN`
at your IDL binary.

Agreement on the shells tested is ~0.7% mean difference in HPD, the residual
coming from the beam-deflection solve (IDL `SVDC/SVSOL` vs numpy `lstsq`).
