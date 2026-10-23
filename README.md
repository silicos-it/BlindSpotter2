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

#### 4. Output
- A PDB file (`-o/--output`, default `hotspots.pdb`) is written containing one `HETATM` per
  cluster center (ranked by PMF, PMF value in the B-factor column), followed by a protein
  reference structure (first frame, protein atoms only) for visualization.
- An optional PMF plot (`--pmf-plot`) shows PMF vs. cluster rank.
