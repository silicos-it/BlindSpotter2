# BindSpotter
BindSpotter triangulates the ligand’s hotspot by turning residue-wise distance preferences into intersecting geometric constraints, yielding a precise location for the binding site.

## BindSpotter: From PMF to Binding Hotspot
BindSpotter is a ```Python``` tool designed to identify the preferred binding hotspot of a ligand to a protein target from LiGaMD simulations. The workflow combines statistical reweighting, distance mapping, and geometric intersection analysis to pinpoint the most favorable ligand location.
#### 1. LiGaMD reweighting and PMF calculation
- For each MD frame, the distances between the ligand and every protein residue are computed.
- Using the LiGaMD boost potential applied on ligand–protein interactions, the probability distribution along each distance axis is reweighted.
- From these distributions, the minimum of the potential of mean force (PMF) is identified, corresponding to the distance where the ligand most prefers to reside relative to that residue.
#### 2. Sphere representation of residue–ligand preferences
- Each residue defines a sphere in 3D space:
  - Center = C-alpha position of the residue.
  - Radius = preferred ligand distance (from the PMF minimum).
- For example, if ASP-145 shows the lowest PMF at 25 Å, the ligand’s most favorable position lies somewhere on the surface of a sphere with center at ASP-145 and radius 25 Å.
- This procedure is repeated for all residues in the protein.
#### 3. Geometric intersection to localize the hotspot
- By computing the intersection of these spheres:
  - 2 spheres intersect in a circle.
  - 3 spheres intersect in two points.
  - 4 or more spheres converge to a single point.
- With many residues contributing (hundreds of spheres), their intersection collapses to a unique 3D coordinate, representing the binding hotspot where the ligand most favorably resides.
