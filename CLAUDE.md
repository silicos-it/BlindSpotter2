# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

BindSpotter locates the preferred binding hotspot of a ligand on a protein from a LiGaMD
(Ligand Gaussian accelerated MD) AMBER simulation. The method (see README.md):

1. For a set of reference residues (CA atoms), compute the ligand–residue distance at every
   frame of the trajectory using `cpptraj`.
2. Reweight each residue's distance histogram with the LiGaMD boost-potential reweighting
   method (McLaurin/cumulant-expansion reweighting, `amdweight_CE`) to get an unbiased PMF
   along that distance.
3. Take the distance at the PMF minimum as the radius of a sphere centered on that residue's
   CA atom — the ligand's most probable position lies on that sphere's surface.
4. With many residues (many spheres), do a nonlinear least-squares fit (Levenberg–Marquardt,
   falling back to a hand-rolled damped Gauss-Newton if SciPy is unavailable) for the 3D point
   that best satisfies all `||x - center_i|| ≈ radius_i` constraints simultaneously. That point
   is the hotspot. A bootstrap routine (`bootstrap_uncertainty`) is available for resampling-based
   uncertainty estimation.

`PyReweighting-1D.py` is Miao/Sinko's third-party reweighting script (the reference
implementation this project builds on, cited in README). `BlindSpotter2.py` currently shells out
to it per-residue as a subprocess rather than reimporting its reweighting math directly.
**Do not treat `PROMPT` as a current spec** — it is the original design brief and describes an
earlier, since-abandoned clustering-based approach (cpptraj spatial clustering of the ligand +
per-cluster PMF) that does not match what `BlindSpotter2.py` actually does today (per-residue
sphere fitting). When in doubt, trust the code over `PROMPT`.

## Running it

```bash
python BlindSpotter2.py -c <centers.pdb> -p <parm.prmtop> -l <ligand_resid> \
    -b <boosts.txt> -t <trajectories.txt> [-o hotspot.pdb] [--pmf_plot pmf.png] [-f fraction]
```

- `-c/--centers`: a PDB file of reference atoms (typically CA atoms of candidate residues).
  To build one, grab a random subset of CA cards from a reference PDB, e.g.:
  `cat pdb_file | grep " CA " | shuf -n N > centers.pdb`
- `-p/--parm`: AMBER parameter/topology file for the system.
- `-l/--ligand`: ligand residue id (integer) as it appears in the parm file.
- `-b/--boosts`: a text file listing paths to one or more LiGaMD boost-potential log files
  (e.g. `gamd.log`), one per line; `#`-prefixed lines are treated as comments.
- `-t/--trajectories`: a text file listing paths to one or more trajectory files, one per line
  (same comment convention).
- `-f/--fraction`: randomly subsample this fraction of the centers (default 1.0 = use all).
- At least 4 spheres (centers) are required for `IntersectSpheres` to constrain a 3D point.

`cpptraj` is invoked via a hardcoded absolute path (`CPPTRAJ` constant at the top of
`BlindSpotter2.py`, currently `/usr/local/amber24/bin/cpptraj`) — update that constant if AMBER
is installed elsewhere.

Dependencies: `numpy`, `matplotlib`, and optionally `scipy` (used for the LM least-squares fit
in `IntersectSpheres`; without it, the script falls back to its own Gauss-Newton optimizer).
There is no requirements file or virtualenv setup in the repo — dependencies are assumed to be
available in the active Python environment. No test suite, linter, or build system is present.

### Test data

`TestSystems/1T49/` contains a full LiGaMD production run for one real system: `gamd.log`
(boost potentials, ~150MB), `NPT_prod.nc` (trajectory, large), and
`stripped.complex.solvated.prmtop` (topology with waters/ions already stripped). These files are
large (the trajectory is tens of GB) — avoid reading/catting them wholesale; use `cpptraj` or
targeted byte/line ranges when inspecting. `ExternalDocumentation/CpptrajManual.pdf` is the
cpptraj reference for selector syntax, `trajin`/`distance`/`cluster` commands, etc., when working
on anything that generates or edits `cpptraj.inp`.

## Code structure (BlindSpotter2.py, single file)

- CLI parsing: `ParseCommandline()` — validates all file arguments exist up front and exits with
  code 1 and a message on any problem (no exceptions are raised for user-facing errors).
- I/O helpers: `ReadAtomCenters` (parses PDB ATOM/HETATM records, columns 21-26 for resid,
  30-38/38-46/46-54 for x/y/z — fixed-width PDB columns, not whitespace-split), `ReadTrajectoryFiles`,
  `ProcessBoostFiles` (converts each boost log's 7th whitespace field into the
  `dV/kT COUNTER dV` triplet format that `PyReweighting-1D.py`'s `-weight` option expects, using
  `kT = 0.001987 * 300.0`).
- `CalculateDistances`: writes and runs a `cpptraj.inp` script computing one `distance` command
  per residue center against the ligand's CA.
- Per-residue PMF extraction (in `__main__`): for each center, slices its distance column out of
  `distances.txt`, shells out to `PyReweighting-1D.py` with `-job amdweight_CE`, parses the
  resulting `pmf-c2-tmp.txt.xvg`, and takes the first distance bin whose cumulant-corrected PMF is
  ~0 (the PMF minimum) as that residue's sphere radius. Intermediate `*.xvg` files are cleaned up
  after each residue.
- Sphere-intersection fit: `_residuals`/`_jacobian` define `f_i(x) = ||x - c_i|| - r_i` and its
  Jacobian; `IntersectSpheres` runs SciPy's `least_squares(method="lm")` when available (falling
  back to the local `_gauss_newton`), then derives a covariance matrix and per-axis 1-sigma via
  `_covariance_from_jacobian`. `bootstrap_uncertainty` resamples spheres with replacement to assess
  fit stability.
- Output: the fitted hotspot coordinate is written as a single-atom PDB (`ATOM 1 O POC ...`) to
  `-o/--output`, and the per-residue PMF curves are all plotted onto one figure saved to
  `--pmf_plot`.

Note the gap against the README/PROMPT's stated goal of writing cluster centers *and* the protein
structure into the output PDB with PMF in the B-factor/temperature column — the current
implementation instead writes only the single fitted hotspot point, with no PMF or protein atoms
in the PDB. Be aware of this when extending the output-writing code.
