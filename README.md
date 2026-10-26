# BlindSpotter

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

BlindSpotter locates the preferred binding hotspot of a ligand on a protein from a LiGaMD
(Ligand Gaussian accelerated MD) AMBER simulation, by spatially clustering the ligand's
trajectory and reweighting each cluster's population with the LiGaMD boost potential.

## BlindSpotter: From Clusters to Binding Hotspot
BlindSpotter is a ```Python``` tool that drives `cpptraj` and a boost-potential reweighting
scheme to identify the preferred binding hotspot of a ligand on a protein target from LiGaMD
simulations.

#### 1. Trajectory preparation and ligand tracking
- Water/ion residues are stripped (auto-detected from the topology's residue names), the
  trajectory is RMS-fit onto the protein, and the ligand's center of mass is extracted at every
  processed frame (`cpptraj`'s `rms`/`vector center` commands).
- Frames are strided (automatically, or via `--stride`) so that clustering stays tractable on
  long trajectories.

#### 2. Spatial clustering of the ligand position
- The ligand center-of-mass positions are clustered with DBSCAN (`cpptraj`'s `cluster dbscan`),
  using `--epsilon` as the neighborhood radius and `--minpoints` as the minimum population for a
  cluster to be considered statistically significant.
- Each resulting cluster represents a distinct region of space the ligand occupied during the
  simulation; its center is the mean of the ligand COM positions assigned to it.

#### 3. LiGaMD reweighting and PMF per cluster
- For each frame, the total boost potential is read from the LiGaMD log (`gamd.log`, summed
  over `--dv-columns`).
- Each cluster's population is reweighted with the 2nd-order cumulant-expansion method
  (`amdweight_CE`-style correction) to turn raw frame counts into an unbiased relative free
  energy (PMF, kcal/mol) for that cluster.
- The cluster with the lowest PMF is reported as the predicted binding hotspot.

#### 4. Error estimation
- The PMF error is estimated with a moving-block bootstrap over the per-frame series (cluster
  assignment and boost potential). Blocks of consecutive frames are resampled with replacement
  and the PMF of every cluster is recomputed for each resample. When several runs are given,
  they are concatenated and treated as one series.
- Errors are reported relative to the lowest-PMF cluster (whose error is therefore 0), as a
  standard error and a 95% interval. For each cluster, `P(low)` is the fraction of resamples in
  which it has the lowest PMF, i.e. the confidence in the predicted hotspot.
- The block length must exceed the time over which frames are correlated. By default it is
  derived automatically by block averaging (Flyvbjerg-Petersen) of the boost potential and the
  cluster-membership signals.
- Each cluster is also checked for the validity of the 2nd-order cumulant expansion, and flagged:
  - `S`: the boost-potential spread (`dV_std`) exceeds 6 kcal/mol.
  - `C`: the neglected 3rd-order cumulant term, relative to the hotspot cluster, exceeds
    0.5 kcal/mol and is more than twice its bootstrap error.
  - `E`: the cluster is absent from more than 5% of the resamples (undersampled).
- An error summary is always printed: the time correlation and effective number of independent
  pieces of data, a hotspot-vs-runner-up comparison with a verdict on whether the ranking is
  robust, the status of the minor clusters, and the `S`/`C` flags.

| Option | Default | Description |
|--------|---------|-------------|
| `--error` | `block` | `block` for the moving-block bootstrap, `none` to skip error estimation |
| `--nboot` | `200` | number of bootstrap resamples |
| `--block-frames` | `0` | block length in processed (post-stride) frames; `0` derives it automatically |
| `--seed` | none | random seed, for reproducible error estimates |

#### 5. Output
- A PDB file (`-o/--output`, default `hotspots.pdb`) is written containing one `HETATM` per
  cluster center (ranked by PMF, PMF value in the B-factor column, PMF standard error in the
  occupancy column), followed by a protein reference structure (first frame, protein atoms only)
  for visualization. With `--error none` the occupancy column is 1.00.
- An optional PMF plot (`--pmf-plot`) shows PMF vs. cluster rank, with standard-error bars.
