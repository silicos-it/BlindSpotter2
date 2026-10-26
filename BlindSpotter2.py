#!/usr/bin/env python

import argparse
import sys
import os
import math
import re
import shutil
import glob
import subprocess
import numpy as np




VERSION = "2.0"
CPPTRAJ = "/usr/local/amber24/bin/cpptraj"  # last-resort fallback if nothing better is found
KB = 0.001987  # Boltzmann constant, kcal/mol/K

# Heuristic thresholds for flagging clusters whose cumulant-expansion PMF may be unreliable
SIGMA_DV_MAX = 6.0       # kcal/mol; 2nd-order cumulant expansion is accurate below roughly this dV spread
C3_TERM_MAX = 0.5        # kcal/mol; neglected 3rd-order cumulant term, relative to the reference cluster
EMPTY_FRACTION_MAX = 0.05  # fraction of bootstrap resamples in which a cluster may be empty
MIN_BOOTSTRAP_BLOCKS = 20  # auto block size is capped so at least this many blocks remain
MINOR_CLUSTER_FRACTION = 0.01  # clusters below this fraction of frames are "minor" in the error summary

# Canonical AMBER water/ion residue names. Deliberately excludes ambiguous
# bare-element tokens (e.g. plain "CA") that could collide with a real residue.
CANONICAL_SOLVENT_ION_NAMES = {
    "WAT", "HOH", "TIP3", "TIP4", "TIP4EW", "TIP5", "TIP5P", "SPC", "SPCE", "OPC", "OPC3",
    "Na+", "NA", "SOD", "Cl-", "CL", "CLA", "K+", "K", "POT", "Li+", "LI",
    "Rb+", "RB", "Cs+", "CS", "Mg2+", "MG", "Ca2+", "CAL", "Zn2+", "ZN",
    "F-", "I-", "IOD", "Br-", "BR",
}



# Locate a usable cpptraj executable
# #######################################
#
# Search order: $AMBERHOME/bin/cpptraj, 'cpptraj' on $PATH, the hardcoded CPPTRAJ
# fallback, then any /usr/local/amber*/bin/cpptraj (newest version first). Returns
# the first existing, executable path found, or None if nothing was found.

def FindCpptraj():

	candidates = []

	amberhome = os.environ.get("AMBERHOME")
	if amberhome:
		candidates.append(os.path.join(amberhome, "bin", "cpptraj"))

	onPath = shutil.which("cpptraj")
	if onPath:
		candidates.append(onPath)

	candidates.append(CPPTRAJ)
	candidates.extend(sorted(glob.glob("/usr/local/amber*/bin/cpptraj"), reverse=True))

	for path in candidates:
		if path and os.path.isfile(path) and os.access(path, os.X_OK):
			return path

	return None



# Detect whether a file is a plain list of file paths (one per line) rather than
# the data file itself
# ###################################################################################
#
# A listing line is a bare path with no internal whitespace; a real trajectory
# (binary NetCDF, or whitespace-separated columns in a formatted/ASCII trajectory)
# or boost log (whitespace-separated numeric columns) line never satisfies that, so
# this safely tells the two apart without relying on file extensions.

def IsListFile(path):

	try:
		with open(path, "r") as f:
			LINES = f.readlines()
	except (OSError, UnicodeDecodeError):
		return False

	sawEntry = False
	for LINE in LINES:
		LINE = LINE.strip()
		if LINE == "" or LINE.startswith("#"):
			continue
		if len(LINE.split()) != 1:
			return False
		sawEntry = True

	return sawEntry



# Read the file paths listed in a list file ('#'-prefixed lines are comments)
# ##################################################################################
#
# A relative entry is resolved against the current directory first, then against
# the list file's own directory, so lists can be written either cwd-relative or
# self-contained next to the files they reference.

def ReadFileList(path):

	baseDir = os.path.dirname(os.path.abspath(path))
	entries = []
	with open(path, "r") as f:
		for LINE in f:
			LINE = LINE.strip()
			if LINE == "" or LINE.startswith("#"):
				continue
			entries.append(LINE)

	resolved = []
	for entry in entries:
		if os.path.isfile(entry):
			resolved.append(entry)
		else:
			resolved.append(os.path.join(baseDir, entry))

	return resolved



# Expand a -t/--trajectory or -b/--boost argument into one or more file paths
# ################################################################################

def ExpandFileArgument(path):

	if IsListFile(path):
		return ReadFileList(path)
	return [path]



# Parse command-line arguments
# #############################

def ParseCommandline():

	parser = argparse.ArgumentParser(description="Identify preferred ligand binding hotspots from a LiGaMD trajectory "
												   "via spatial clustering of the ligand and LiGaMD cumulant-expansion PMF reweighting.",
									  formatter_class=argparse.ArgumentDefaultsHelpFormatter)

	parser.add_argument("-v",
						"--version",
						action="store_true",
						help="returns the version of the script",
						required=False)

	parser.add_argument("-p",
						"--parm",
						type=str,
						help="AMBER parameter/topology file",
						required=True,
						default=argparse.SUPPRESS)

	parser.add_argument("-t",
						"--trajectory",
						type=str,
						help="MD trajectory file (e.g. NetCDF), or a plain-text file listing several "
							 "trajectory files, one per line ('#'-prefixed lines are comments) - e.g. "
							 "to combine multiple independent LiGaMD runs into one combined analysis. "
							 "Must list the same number of files, in the same order, as --boost.",
						required=True,
						default=argparse.SUPPRESS)

	parser.add_argument("-b",
						"--boost",
						type=str,
						help="LiGaMD boost potential log file (gamd.log), or a plain-text file listing "
							 "several boost log files (same one-per-line convention as --trajectory), "
							 "paired in order with the files listed under --trajectory",
						required=True,
						default=argparse.SUPPRESS)

	parser.add_argument("--ligand",
						type=str,
						help="cpptraj atom mask selecting the ligand, e.g. ':283' (residue number). "
							 "Note: a purely-numeric mask token (e.g. ':892') is always parsed by cpptraj "
							 "as a residue NUMBER, never as a residue NAME, even if the residue's actual "
							 "name happens to be the string '892' - use the residue's number in that case.",
						required=True,
						default=argparse.SUPPRESS)

	parser.add_argument("--protein",
						type=str,
						help="cpptraj atom mask selecting the protein, e.g. ':1-282'",
						required=True,
						default=argparse.SUPPRESS)

	parser.add_argument("-o",
						"--output",
						type=str,
						help="output PDB file with cluster centers (PMF in the B-factor column) and the protein as reference",
						required=False,
						default="hotspots.pdb")

	parser.add_argument("--minpoints",
						type=int,
						help="DBSCAN minimum cluster population, i.e. the minimum number of frames "
							 "required for a cluster to be considered statistically significant",
						required=False,
						default=50)

	parser.add_argument("--epsilon",
						type=float,
						help="DBSCAN neighborhood radius in Angstrom, i.e. the minimum spatial resolution between clusters",
						required=False,
						default=2.5)

	parser.add_argument("-T",
						"--temperature",
						type=float,
						help="simulation temperature in Kelvin",
						required=False,
						default=300.0)

	parser.add_argument("--cb-max",
						dest="cb_max",
						type=float,
						help="PMF ceiling (kcal/mol) used for undersampled/infinite values",
						required=False,
						default=8.0)

	parser.add_argument("--dv-columns",
						dest="dv_columns",
						type=str,
						help="comma-separated 0-based field indices in the boost log that are summed "
							 "into the total boost potential per frame (the default sums the LiGaMD "
							 "total-potential and dihedral boost columns for a dual-boost run)",
						required=False,
						default="6,7")

	parser.add_argument("--stride",
						type=int,
						help="frame stride applied uniformly to the trajectory, ligand COM extraction, "
							 "clustering and the boost log (0 = auto, derived from --max-cluster-frames)",
						required=False,
						default=0)

	parser.add_argument("--max-cluster-frames",
						dest="max_cluster_frames",
						type=int,
						help="target number of frames used to auto-derive --stride when --stride is 0",
						required=False,
						default=20000)

	parser.add_argument("--extra-strip-residues",
						dest="extra_strip_residues",
						type=str,
						help="comma-separated extra residue names to strip beyond the canonical water/ion list",
						required=False,
						default="")

	parser.add_argument("--pmf-plot",
						dest="pmf_plot",
						type=str,
						help="optional PNG file name for a per-cluster PMF plot",
						required=False,
						default=None)

	parser.add_argument("--error",
						type=str,
						choices=["block", "none"],
						help="PMF error estimation: 'block' = moving-block bootstrap over the concatenated "
							 "frame series, 'none' = skip error estimation",
						required=False,
						default="block")

	parser.add_argument("--nboot",
						type=int,
						help="number of bootstrap resamples used for the PMF error estimate",
						required=False,
						default=200)

	parser.add_argument("--block-frames",
						dest="block_frames",
						type=int,
						help="bootstrap block length in processed (post-stride) frames "
							 "(0 = auto, derived from block averaging of the per-frame signals)",
						required=False,
						default=0)

	parser.add_argument("--seed",
						type=int,
						help="random seed for the bootstrap (default: non-reproducible)",
						required=False,
						default=None)

	parser.add_argument("--cpptraj",
						type=str,
						help="path to the cpptraj executable (default: auto-detected from $AMBERHOME, "
							 "$PATH, or common Amber install locations)",
						required=False,
						default=None)

	args = parser.parse_args()

	# --version
	if args.version:
		print("Version %s" % (VERSION))
		sys.exit(0)

	# Required files
	for (LABEL, PATH) in (("parm", args.parm), ("trajectory", args.trajectory), ("boost", args.boost)):
		if not os.path.exists(PATH):
			print("Can not find the %s file %s - quitting with error code 1" % (LABEL, PATH))
			sys.exit(1)
		if not os.path.isfile(PATH):
			print("The %s file %s is errorneous - quitting with error code 1" % (LABEL, PATH))
			sys.exit(1)

	# -t/--trajectory and -b/--boost may each be a single file or a list file -
	# expand to the concrete file(s) to process, and check those exist too
	args.trajectory_files = ExpandFileArgument(args.trajectory)
	args.boost_files = ExpandFileArgument(args.boost)

	for (LABEL, SOURCE, PATHS) in (("trajectory", args.trajectory, args.trajectory_files),
									("boost", args.boost, args.boost_files)):
		for PATH in PATHS:
			if not os.path.isfile(PATH):
				print("Can not find the %s file %s (listed in %s) - quitting with error code 1" %
					  (LABEL, PATH, SOURCE))
				sys.exit(1)

	if len(args.trajectory_files) != len(args.boost_files):
		print("The number of trajectory files (%d) must match the number of boost files (%d) - "
			  "each trajectory is paired with its own boost log, in the same order - "
			  "quitting with error code 1" % (len(args.trajectory_files), len(args.boost_files)))
		sys.exit(1)

	# Numeric bounds
	if args.minpoints < 2:
		print("--minpoints should be >= 2 - quitting with error code 1")
		sys.exit(1)
	if args.epsilon <= 0:
		print("--epsilon should be > 0 - quitting with error code 1")
		sys.exit(1)
	if args.temperature <= 0:
		print("--temperature should be > 0 - quitting with error code 1")
		sys.exit(1)
	if args.cb_max <= 0:
		print("--cb-max should be > 0 - quitting with error code 1")
		sys.exit(1)
	if args.max_cluster_frames < 1:
		print("--max-cluster-frames should be >= 1 - quitting with error code 1")
		sys.exit(1)
	if args.nboot < 10:
		print("--nboot should be >= 10 - quitting with error code 1")
		sys.exit(1)
	if args.block_frames < 0:
		print("--block-frames should be >= 0 - quitting with error code 1")
		sys.exit(1)

	# --dv-columns
	try:
		args.dv_columns = [int(x) for x in args.dv_columns.split(",") if x.strip() != ""]
	except ValueError:
		print("Could not parse --dv-columns '%s' - quitting with error code 1" % (args.dv_columns))
		sys.exit(1)
	if len(args.dv_columns) == 0:
		print("--dv-columns must specify at least one column index - quitting with error code 1")
		sys.exit(1)

	# --extra-strip-residues
	args.extra_strip_residues = [x.strip() for x in args.extra_strip_residues.split(",") if x.strip() != ""]

	# --cpptraj
	if args.cpptraj is None:
		found = FindCpptraj()
		if found is None:
			print("Could not automatically locate a cpptraj executable (checked $AMBERHOME/bin, "
				  "$PATH, %s, and /usr/local/amber*/bin) - specify one explicitly with --cpptraj "
				  "- quitting with error code 1" % (CPPTRAJ))
			sys.exit(1)
		args.cpptraj = found
		args.cpptraj_autodetected = True
	else:
		if not os.path.isfile(args.cpptraj) or not os.access(args.cpptraj, os.X_OK):
			print("The cpptraj executable %s does not exist or is not executable - quitting with error code 1" % (args.cpptraj))
			sys.exit(1)
		args.cpptraj_autodetected = False

	return args



# Read the residue labels from an AMBER prmtop file
# ##################################################

def ReadTopologyResidueLabels(parmFile):

	try:
		f = open(parmFile, "r")
	except OSError:
		print("Could not open the parm file %s - quitting with error code 1" % (parmFile))
		sys.exit(1)

	with f:
		LINES = f.readlines()

	LABELS = []
	i = 0
	n = len(LINES)
	while i < n:
		LINE = LINES[i].rstrip("\n")
		if LINE.startswith("%FLAG") and "RESIDUE_LABEL" in LINE:
			i += 1
			if i < n and LINES[i].rstrip("\n").startswith("%FORMAT"):
				i += 1
			# Lines must be stripped of their newline BEFORE concatenation, otherwise
			# the embedded '\n' characters shift every subsequent 4-character field.
			CHUNK = ""
			while i < n and not LINES[i].rstrip("\n").startswith("%FLAG"):
				CHUNK += LINES[i].rstrip("\n")
				i += 1
			for j in range(0, len(CHUNK), 4):
				TOKEN = CHUNK[j:j + 4].strip()
				if TOKEN != "":
					LABELS.append(TOKEN)
			break
		i += 1

	return LABELS



# Build the cpptraj strip mask for the water/ion residues actually present
# ##########################################################################

def BuildSolventIonStripMask(residueLabels, extra=()):

	present = set(residueLabels) & (CANONICAL_SOLVENT_ION_NAMES | set(extra))
	if len(present) == 0:
		return None
	return "strip :%s" % (",".join(sorted(present)))



# Count the number of data frames in the boost log
# #################################################

def CountBoostLogFrames(boostFile):

	count = 0
	with open(boostFile, "r") as f:
		for LINE in f:
			LINE = LINE.strip()
			if LINE == "" or LINE[0] == "#":
				continue
			count += 1
	return count



# Run a cpptraj input script
# ###########################

def RunCpptraj(cpptrajExe, scriptPath):

	CMD = "%s -i %s" % (cpptrajExe, scriptPath)
	rc = os.system(CMD)
	if rc != 0:
		print("cpptraj failed (return code %d) while running %s - quitting with error code 1" % (rc, scriptPath))
		sys.exit(1)



# Count the frames cpptraj will read from each trajectory file
# ################################################################
#
# Uses cpptraj's "list trajin", which reports "(reading N of M)" per trajectory
# without loading any coordinates. Returns a list of per-file frame counts (after
# striding), or None if cpptraj's output could not be parsed.

def CountTrajectoryFrames(cpptrajExe, scriptPath, parmFile, trajFiles, stride):

	f = open(scriptPath, "w")
	f.write("parm %s\n" % (parmFile))
	for trajFile in trajFiles:
		f.write("trajin %s 1 last %d\n" % (trajFile, stride))
	f.write("list trajin\n")
	f.write("quit\n")
	f.close()

	try:
		result = subprocess.run([cpptrajExe, "-i", scriptPath], stdout=subprocess.PIPE,
								stderr=subprocess.STDOUT, universal_newlines=True)
	except OSError:
		return None
	if result.returncode != 0:
		return None
	counts = [int(m.group(1)) for m in re.finditer(r"\(reading\s+(\d+)\s+of\s+\d+\)", result.stdout)]
	if len(counts) != len(trajFiles):
		return None
	return counts



# Write the main cpptraj analysis script (strip, align, ligand COM, cluster)
# ############################################################################

def WriteMainCpptrajScript(scriptPath, parmFile, trajFiles, stride, stripMask, proteinMask, ligandMask,
							minpoints, epsilon, comFile, cnumFile, summaryFile, infoFile):

	f = open(scriptPath, "w")
	f.write("parm %s\n" % (parmFile))
	for trajFile in trajFiles:
		f.write("trajin %s 1 last %d\n" % (trajFile, stride))
	if stripMask is not None:
		f.write("%s\n" % (stripMask))
	f.write("autoimage\n")
	f.write("rms RMSFIT first %s mass\n" % (proteinMask))
	f.write("vector LIGCOM center %s out %s\n" % (ligandMask, comFile))
	f.write("cluster C0 dbscan minpoints %d epsilon %.5f rms %s mass nofit out %s summary %s info %s\n" %
			(minpoints, epsilon, ligandMask, cnumFile, summaryFile, infoFile))
	f.write("run\n")
	f.close()



# Write the small cpptraj script that extracts the protein-only first frame
# #############################################################################

def WriteProteinReferenceCpptrajScript(scriptPath, parmFile, trajFile, proteinMask, outPdb):

	f = open(scriptPath, "w")
	f.write("parm %s\n" % (parmFile))
	f.write("trajin %s 1 1\n" % (trajFile))
	f.write("strip !(%s)\n" % (proteinMask))
	f.write("autoimage\n")
	f.write("rms RMSFIT first %s mass\n" % (proteinMask)) 
	f.write("trajout %s pdb\n" % (outPdb))
	f.write("run\n")
	f.close()



# Parse the per-frame cluster assignment written by cpptraj's cluster command
# ###############################################################################

def ParseClusterAssignments(path):

	IDS = []
	with open(path, "r") as f:
		for LINE in f:
			LINE = LINE.strip()
			if LINE == "" or LINE[0] == "#":
				continue
			FIELDS = LINE.split()
			if len(FIELDS) < 2:
				continue
			IDS.append(int(float(FIELDS[1])))
	return np.asarray(IDS, dtype=int)



# Parse the per-frame ligand center-of-mass written by cpptraj's vector command
# #################################################################################
#
# cpptraj's "vector ... center ... out <file>" writes rows of
# "frame  VX VY VZ  DX DY DZ" - the position (VX,VY,VZ) immediately after the frame
# index, followed by a direction vector that is always zero for "center" mode. The
# position fields must be taken, not the (zero) trailing direction fields.

def ParseLigandCOM(path):

	COORDS = []
	with open(path, "r") as f:
		for LINE in f:
			LINE = LINE.strip()
			if LINE == "" or LINE[0] == "#":
				continue
			FIELDS = LINE.split()
			if len(FIELDS) < 4:
				continue
			try:
				XYZ = [float(v) for v in FIELDS[1:4]]
			except ValueError:
				continue
			COORDS.append(XYZ)
	return np.asarray(COORDS, dtype=float)



# Parse the LiGaMD boost potential log file
# ##############################################

def ParseBoostLog(path, columns):

	VALUES = []
	maxcol = max(columns)
	with open(path, "r") as f:
		for LINE in f:
			LINE = LINE.strip()
			if LINE == "" or LINE[0] == "#":
				continue
			FIELDS = LINE.split()
			if len(FIELDS) <= maxcol:
				continue
			try:
				v = sum(float(FIELDS[c]) for c in columns)
			except ValueError:
				continue
			VALUES.append(v)
	return np.asarray(VALUES, dtype=float)



# Truncate a set of per-frame arrays to a common length
# ##########################################################

def AlignFrameCounts(*arrays):

	lengths = [a.shape[0] for a in arrays]
	n = min(lengths)
	if len(set(lengths)) > 1:
		print("Warning: cpptraj/boost output lengths disagree %s - truncating all to %d frames" % (lengths, n))
	return tuple(a[:n] for a in arrays)



# Align the cpptraj per-frame output with the per-run boost logs, run by run
# ##############################################################################
#
# cpptraj processes all trajectories as one concatenated series. If a trajectory and its
# boost log disagree in length, truncating only the tail of the concatenated arrays would
# pair every later frame with the wrong dV. Instead, each run is truncated to the shorter
# of its trajectory/boost-log frame counts before concatenation. Falls back to tail
# truncation (with a warning) if the per-trajectory frame counts are unknown or
# inconsistent with the cpptraj output.

def AlignRunFrames(clusterIds, comCoords, dVRuns, trajCounts):

	nCpptraj = (clusterIds.shape[0], comCoords.shape[0])
	if trajCounts is None or nCpptraj[0] != sum(trajCounts) or nCpptraj[1] != sum(trajCounts):
		print("Warning: could not determine per-trajectory frame counts consistent with the cpptraj "
			  "output %s - falling back to tail truncation; frames may be mispaired with dV if any "
			  "trajectory/boost log pair other than the last differs in length" % (list(nCpptraj),))
		return AlignFrameCounts(clusterIds, comCoords, np.concatenate(dVRuns))

	keep = []
	dVParts = []
	offset = 0
	for i, (nTraj, dVi) in enumerate(zip(trajCounts, dVRuns), start=1):
		n = min(nTraj, dVi.shape[0])
		if nTraj != dVi.shape[0]:
			print("Warning: run [%d] has %d trajectory frames but %d boost-log frames (after stride) - "
				  "using the first %d of both" % (i, nTraj, dVi.shape[0], n))
		keep.append(np.arange(offset, offset + n))
		dVParts.append(dVi[:n])
		offset += nTraj
	keep = np.concatenate(keep)
	return clusterIds[keep], comCoords[keep], np.concatenate(dVParts)



# Effective acceleration factor (Gaussian approximation)
# ###########################################################

def AccelAMD(dV, T):

	kT = KB * T
	avg = np.mean(dV)
	std = np.std(dV)
	return float(np.exp(avg / kT + (std / kT) ** 2 / 2.0))



# Compute the per-cluster PMF via 2nd-order cumulant-expansion reweighting
# #############################################################################
#
# Ported from PyReweighting-1D.py's amdweight_CE job (reweight_CE/prephist/normalize),
# generalized from "per continuous distance bin" to "per discrete DBSCAN cluster".

def ComputeClusterPMF(clusterIds, dV, T, cbMax):

	beta = 1.0 / (KB * T)

	clusters = sorted(int(c) for c in set(clusterIds.tolist()) if c >= 0)
	nNoise = int(np.sum(clusterIds < 0))
	nTotal = int(clusterIds.shape[0])

	population = {}
	dVAvg = {}
	dVStd = {}
	logpop = {}
	c12 = {}

	for cid in clusters:
		dVj = dV[clusterIds == cid]
		avg = float(np.mean(dVj))
		std = float(np.std(dVj))
		avg2 = float(np.mean(dVj ** 2))
		avg3 = float(np.mean(dVj ** 3))

		c1 = beta * avg
		c2 = 0.5 * beta ** 2 * std ** 2

		c1_energy = -(1.0 / beta) * c1
		c2_energy = -(1.0 / beta) * c2

		population[cid] = dVj.shape[0]
		dVAvg[cid] = avg
		dVStd[cid] = std
		logpop[cid] = KB * T * math.log(population[cid] + 1e-18)
		c12[cid] = c1_energy + c2_energy

	maxLogpop = max(logpop.values())
	pmfC2Raw = {}
	for cid in clusters:
		pmfBiased = maxLogpop - logpop[cid]  # zero-shifted so the most-populated cluster is 0
		pmfC2Raw[cid] = pmfBiased + c12[cid]

	minPmf = min(pmfC2Raw.values())
	pmfC2 = {}
	for cid in clusters:
		v = pmfC2Raw[cid] - minPmf
		if math.isinf(v):
			v = cbMax
		pmfC2[cid] = v

	return {
		"clusters": clusters,
		"population": population,
		"dV_avg": dVAvg,
		"dV_std": dVStd,
		"pmf_c2": pmfC2,
		"n_noise": nNoise,
		"n_total": nTotal,
	}



# Per-cluster diagnostics on the validity of the 2nd-order cumulant expansion
# ###############################################################################
#
# The 2nd-order cumulant expansion is exact only for a Gaussian dV distribution. Reported
# per cluster: skewness and excess kurtosis of dV, and the neglected 3rd-order cumulant
# contribution to the PMF, -beta^2 * kappa3 / 6 (kcal/mol). Since only PMF differences are
# meaningful, the truncation error of a cluster's relative PMF is the difference of this
# term with that of the reference cluster (a dV skew shared by all clusters cancels). The
# term is itself a noisy estimate, but gives the order of magnitude of the truncation error.

def ComputeDVDiagnostics(clusterIds, dV, T, clusters):

	beta = 1.0 / (KB * T)
	diag = {}
	for cid in clusters:
		dVj = dV[clusterIds == cid]
		d = dVj - np.mean(dVj)
		var = float(np.mean(d ** 2))
		kappa3 = float(np.mean(d ** 3))
		kappa4 = float(np.mean(d ** 4)) - 3.0 * var ** 2
		if var > 0:
			skew = kappa3 / var ** 1.5
			kurt = kappa4 / var ** 2
		else:
			skew = 0.0
			kurt = 0.0
		diag[cid] = {
			"skew": skew,
			"kurt": kurt,
			"c3_term": -beta ** 2 * kappa3 / 6.0,
		}
	return diag



# Auto-derive the bootstrap block length by block averaging (Flyvbjerg-Petersen)
# ###################################################################################
#
# For each per-frame signal (dV, and the membership indicator of each cluster), the
# statistical inefficiency g(B) = B * var(block means) / var(x) is computed for block
# lengths B = 1, 2, 4, ... while at least MIN_BOOTSTRAP_BLOCKS blocks remain. g(B) rises
# with B until the blocks are longer than the correlation time, then plateaus; the block
# length for a signal is the smallest B reaching 90% of its largest g(B). The returned
# block length is the largest over all signals.

def EstimateBlockFrames(clusterIds, dV, clusters):

	n = clusterIds.shape[0]
	signals = [dV] + [(clusterIds == cid).astype(float) for cid in clusters]
	blockFrames = 1
	for x in signals:
		var = float(np.var(x))
		if var <= 0:
			continue
		sizes = []
		gs = []
		B = 1
		while n // B >= MIN_BOOTSTRAP_BLOCKS:
			nb = n // B
			means = x[:nb * B].reshape(nb, B).mean(axis=1)
			sizes.append(B)
			gs.append(B * float(np.var(means)) / var)
			B *= 2
		if not gs:
			continue
		gMax = max(gs)
		for B, g in zip(sizes, gs):
			if g >= 0.9 * gMax:
				blockFrames = max(blockFrames, B)
				break
	return blockFrames



# Moving-block bootstrap estimate of the per-cluster PMF error
# ##################################################################
#
# The concatenated frame series is resampled as ceil(n/B) overlapping blocks of B
# consecutive frames, drawn with replacement. For each resample the cumulant-expansion
# PMF of every cluster is recomputed (same formula as ComputeClusterPMF) and expressed
# relative to a fixed reference cluster (the full-data PMF minimum), so the reported
# errors are on the same zero point as the reported PMFs. Per-block sufficient statistics
# (count, sum dV, sum dV^2 per cluster) come from cumulative sums, so each resample costs
# only O(n/B * nClusters).
#
# Returns per cluster: standard error, 2.5/97.5 percentiles, the fraction of resamples in
# which the cluster has the lowest PMF, the number of resamples in which it was empty, and
# the standard error of the 3rd-order cumulant term relative to the reference cluster (see
# ComputeDVDiagnostics), needed to tell a real difference in non-Gaussianity apart from noise.

def BlockBootstrapPMF(clusterIds, dV, T, clusters, refCluster, blockFrames, nBoot, seed):

	beta = 1.0 / (KB * T)
	kT = KB * T
	n = clusterIds.shape[0]
	K = len(clusters)
	B = max(1, min(blockFrames, n))
	nBlocks = int(math.ceil(n / float(B)))
	ref = clusters.index(refCluster)

	member = np.zeros((n, K))
	for k, cid in enumerate(clusters):
		member[:, k] = (clusterIds == cid)
	csN = np.vstack([np.zeros(K), np.cumsum(member, axis=0)])
	csS1 = np.vstack([np.zeros(K), np.cumsum(member * dV[:, None], axis=0)])
	csS2 = np.vstack([np.zeros(K), np.cumsum(member * (dV ** 2)[:, None], axis=0)])
	csS3 = np.vstack([np.zeros(K), np.cumsum(member * (dV ** 3)[:, None], axis=0)])

	rng = np.random.default_rng(seed)
	samples = np.full((nBoot, K), np.nan)
	c3Samples = np.full((nBoot, K), np.nan)
	isLowest = np.zeros(K)
	nEmpty = np.zeros(K, dtype=int)
	for i in range(nBoot):
		starts = rng.integers(0, n - B + 1, size=nBlocks)
		N = (csN[starts + B] - csN[starts]).sum(axis=0)
		S1 = (csS1[starts + B] - csS1[starts]).sum(axis=0)
		S2 = (csS2[starts + B] - csS2[starts]).sum(axis=0)
		S3 = (csS3[starts + B] - csS3[starts]).sum(axis=0)
		present = N > 0
		nEmpty += ~present
		avg = np.where(present, S1 / np.maximum(N, 1), 0.0)
		avg2 = np.where(present, S2 / np.maximum(N, 1), 0.0)
		avg3 = np.where(present, S3 / np.maximum(N, 1), 0.0)
		var = avg2 - avg ** 2
		kappa3 = avg3 - 3.0 * avg * avg2 + 2.0 * avg ** 3
		if present[ref]:
			c3 = -beta ** 2 * kappa3 / 6.0
			c3Samples[i, present] = c3[present] - c3[ref]
		pmf = np.full(K, np.inf)
		pmf[present] = -kT * np.log(N[present]) - avg[present] - 0.5 * beta * var[present]
		isLowest[int(np.argmin(pmf))] += 1
		if present[ref]:
			samples[i, present] = pmf[present] - pmf[ref]

	result = {"block_frames": B, "n_boot": nBoot, "se": {}, "ci_low": {}, "ci_high": {},
			  "p_lowest": {}, "n_empty": {}, "c3_se": {}}
	for k, cid in enumerate(clusters):
		s = samples[:, k]
		s = s[~np.isnan(s)]
		if s.shape[0] >= 2:
			result["se"][cid] = float(np.std(s, ddof=1))
			result["ci_low"][cid] = float(np.percentile(s, 2.5))
			result["ci_high"][cid] = float(np.percentile(s, 97.5))
		else:
			result["se"][cid] = float("nan")
			result["ci_low"][cid] = float("nan")
			result["ci_high"][cid] = float("nan")
		c3 = c3Samples[:, k]
		c3 = c3[~np.isnan(c3)]
		result["c3_se"][cid] = float(np.std(c3, ddof=1)) if c3.shape[0] >= 2 else float("nan")
		result["p_lowest"][cid] = isLowest[k] / float(nBoot)
		result["n_empty"][cid] = int(nEmpty[k])
	return result



# Compute the Cartesian center of each cluster (mean ligand COM)
# ####################################################################

def ComputeClusterCenters(clusterIds, comCoords):

	centers = {}
	for cid in sorted(int(c) for c in set(clusterIds.tolist()) if c >= 0):
		centers[cid] = comCoords[clusterIds == cid].mean(axis=0)
	return centers



# Best-effort parse of the cpptraj cluster quality metrics (DBI/pSF/SSR-SST)
# ###############################################################################

def ParseClusterInfoStats(path):

	stats = {}
	try:
		with open(path, "r") as f:
			text = f.read()
		for key in ("DBI", "pSF", "SSR/SST"):
			m = re.search(re.escape(key) + r"\s*[:=]?\s*([-+0-9.eE]+)", text)
			if m:
				stats[key] = float(m.group(1))
	except Exception:
		pass
	return stats



# Write the output PDB: cluster centers (PMF in the B-factor column, its bootstrap
# standard error - if available - in the occupancy column) + protein reference
# ############################################################################################

def WriteOutputPDB(outFile, centers, pmf, referencePdb, pmfSE=None):

	fo = open(outFile, "w")

	orderedIds = sorted(centers.keys(), key=lambda cid: pmf[cid])
	serial = 0
	for cid in orderedIds:
		serial += 1
		x, y, z = centers[cid]
		occupancy = 1.00
		if pmfSE is not None and not math.isnan(pmfSE[cid]):
			occupancy = pmfSE[cid]
		fo.write("%-6s%5d %-4s%1s%-3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f\n" %
				 ("HETATM", serial, "CA", " ", "HOT", "X", cid, " ", x, y, z, occupancy, pmf[cid]))
	fo.write("TER\n")

	if os.path.exists(referencePdb):
		fi = open(referencePdb, "r")
		for LINE in fi:
			if LINE.startswith("ATOM") or LINE.startswith("HETATM"):
				serial += 1
				NEWLINE = LINE[0:6] + ("%5d" % serial) + LINE[11:]
				if not NEWLINE.endswith("\n"):
					NEWLINE += "\n"
				fo.write(NEWLINE)
		fi.close()

	fo.write("END\n")
	fo.close()



# Print a plain-language summary of the PMF error estimate and the reweighting diagnostics
# ##############################################################################################
#
# Always printed. Covers: the time correlation (block length and effective number of
# independent pieces of data), the predicted hotspot versus the runner-up, the
# undersampled minor clusters, and the S/C flags on the cumulant-expansion validity.
# Parts that need the bootstrap are replaced by a note when --error is 'none'.

def PrintErrorSummary(pmfResult, boot, flagsById, orderedIds, stride, blockSource):

	nProcessed = pmfResult["n_total"]
	frac = dict((cid, pmfResult["population"][cid] / float(nProcessed)) for cid in orderedIds)
	best = orderedIds[0]

	print("")
	print("Error summary")
	print("-------------")

	if boot is None:
		print("- PMF error estimation was disabled (--error none): no correlation, hotspot-confidence "
			  "or undersampling assessment is available.")
	else:
		# Time correlation
		B = boot["block_frames"]
		nPieces = nProcessed / float(B)
		label = "automatic" if blockSource.startswith("auto") else "user-specified"
		if B == 1:
			print("- No time correlation detected. The %s block length is 1 processed frame (%d MD frames), "
				  "so all %d processed frames count as independent data." % (label, stride, nProcessed))
		else:
			if nPieces < 100:
				heading = "Long correlation"
				meaning = ("The ligand stays in one region for long stretches, so there are effectively "
						   "only about %d independent pieces of data." % (round(nPieces)))
			else:
				heading = "Time correlation"
				meaning = ("There are effectively about %d independent pieces of data." % (round(nPieces)))
			print("- %s. The %s block length is %d processed frames, which is %d MD frames. %s" %
				  (heading, label, B, B * stride, meaning))

		# Hotspot versus runner-up
		if len(orderedIds) < 2:
			print("- Hotspot vs runner-up: only one cluster was found (cluster %d), so there is no "
				  "runner-up to compare with." % (best))
		else:
			second = orderedIds[1]
			gap = pmfResult["pmf_c2"][second]
			se = boot["se"][second]
			colA = "Cluster %d (predicted hotspot)" % (best)
			colB = "Cluster %d (runner-up)" % (second)
			rows = [
				("Share of frames", "%.0f%%" % (100.0 * frac[best]), "%.0f%%" % (100.0 * frac[second])),
				("Lowest in % of resamples", "%.0f%%" % (100.0 * boot["p_lowest"][best]),
				 "%.0f%%" % (100.0 * boot["p_lowest"][second])),
				("PMF gap to cluster %d" % (best), "-", "%.2f +/- %.2f kcal/mol" % (gap, se)),
			]
			w0 = max(len(r[0]) for r in rows)
			w1 = max([len(colA)] + [len(r[1]) for r in rows])
			w2 = max([len(colB)] + [len(r[2]) for r in rows])
			print("- Hotspot vs runner-up:")
			print("")
			print("  | %-*s | %-*s | %-*s |" % (w0, "", w1, colA, w2, colB))
			print("  |%s|%s|%s|" % ("-" * (w0 + 2), "-" * (w1 + 2), "-" * (w2 + 2)))
			for r in rows:
				print("  | %-*s | %-*s | %-*s |" % (w0, r[0], w1, r[1], w2, r[2]))
			print("")
			if math.isnan(se):
				print("  The error of the gap could not be estimated (the runner-up is too often absent "
					  "from the resamples).")
			elif gap > 2.0 * se:
				print("  The gap exceeds 2x its error, so the hotspot ranking is statistically robust.")
			elif gap > se:
				print("  The hotspot is favoured, but the gap is less than 2x its error, so the ranking "
					  "is not fully certain.")
			else:
				print("  Within the error, the ranking of these two is uncertain.")

		# Minor clusters
		minor = [cid for cid in orderedIds if frac[cid] < MINOR_CLUSTER_FRACTION]
		if not minor:
			print("- Minor clusters. No cluster is below %.0f%% of frames." % (100.0 * MINOR_CLUSTER_FRACTION))
		else:
			minorE = [cid for cid in minor if "E" in flagsById[cid]]
			ses = [boot["se"][cid] for cid in minor if not math.isnan(boot["se"][cid])]
			text = ("- Minor clusters. %d of the %d cluster%s below %.0f%% of frames get the E flag" %
					(len(minorE), len(minor), "s" if len(minor) > 1 else "", 100.0 * MINOR_CLUSTER_FRACTION))
			if minorE:
				text += ": they come from only a few visits"
			if len(ses) == 1 or (ses and "%.1f" % min(ses) == "%.1f" % max(ses)):
				text += "; their PMF SE is %.1f kcal/mol" % (ses[0])
			elif ses:
				text += "; their PMF SEs are %.1f-%.1f kcal/mol" % (min(ses), max(ses))
			print(text + ".")
		majorE = [cid for cid in orderedIds if frac[cid] >= MINOR_CLUSTER_FRACTION and "E" in flagsById[cid]]
		if majorE:
			print("  Larger clusters with the E flag: %s." % (", ".join(str(c) for c in majorE)))

	# Cumulant-expansion validity (S and C flags)
	sigmas = [pmfResult["dV_std"][cid] for cid in orderedIds]
	sClusters = [cid for cid in orderedIds if "S" in flagsById[cid]]
	cClusters = [cid for cid in orderedIds if "C" in flagsById[cid]]
	cQualifier = "significant " if boot is not None else ""
	if not sClusters:
		text = ("- No S flags. sigma_dV is about %.1f kcal/mol (range %.1f-%.1f), so the 2nd-order expansion "
				"is not in trouble because of dV spread." % (float(np.median(sigmas)), min(sigmas), max(sigmas)))
	else:
		text = ("- S flags on cluster(s) %s: sigma_dV exceeds %.1f kcal/mol there, so their 2nd-order PMFs "
				"may be inaccurate." % (", ".join(str(c) for c in sClusters), SIGMA_DV_MAX))
	if not cClusters:
		text += " There are also no %sC flags." % (cQualifier)
	else:
		text += (" %sC flags on cluster(s) %s: their dV distribution differs in non-Gaussianity from the "
				 "hotspot's, so their relative PMFs may carry a 3rd-order truncation error." %
				 (cQualifier.capitalize(), ", ".join(str(c) for c in cClusters)))
		if len(orderedIds) > 1 and orderedIds[1] in cClusters:
			text += " This includes the runner-up, which affects the hotspot ranking."
		if boot is None:
			text += (" Without the bootstrap these flags are not tested for significance, and the 3rd-order "
					 "estimate is noisy, so many may be false alarms.")
	print(text)



# Write the per-cluster PMF plot
# ###################################
#
# Clusters are shown in PMF order and labeled by their cluster ID (as in the table and the
# output PDB). Error bars span the bootstrap 95% interval, which is generally asymmetric.
# Flagged clusters (S/C/E) are drawn as open markers with their flag letters alongside.

def WritePMFPlot(plt, path, orderedIds, pmf, boot, flagsById):

	x = list(range(len(orderedIds)))
	values = [pmf[cid] for cid in orderedIds]

	fig, ax = plt.subplots(figsize=(max(6.4, 0.3 * len(orderedIds)), 4.8))
	ax.plot(x, values, "-", color="0.75", zorder=1)

	if boot is not None:
		lower = []
		upper = []
		for cid, v in zip(orderedIds, values):
			lo = boot["ci_low"][cid]
			hi = boot["ci_high"][cid]
			lower.append(0.0 if math.isnan(lo) else max(0.0, v - lo))
			upper.append(0.0 if math.isnan(hi) else max(0.0, hi - v))
		ax.errorbar(x, values, yerr=[lower, upper], fmt="none", ecolor="C0", capsize=3, zorder=2,
					label="bootstrap 95% interval")

	plain = [i for i, cid in enumerate(orderedIds) if not flagsById[cid]]
	flagged = [i for i, cid in enumerate(orderedIds) if flagsById[cid]]
	if plain:
		ax.plot([x[i] for i in plain], [values[i] for i in plain], "o", color="C0", zorder=3,
				label="cluster")
	if flagged:
		ax.plot([x[i] for i in flagged], [values[i] for i in flagged], "o", markerfacecolor="white",
				markeredgecolor="C3", markeredgewidth=1.5, zorder=3, label="flagged cluster (S/C/E)")
		for i in flagged:
			ax.annotate(flagsById[orderedIds[i]], (x[i], values[i]), textcoords="offset points",
						xytext=(5, -3), fontsize=7, color="C3")

	ax.set_xticks(x)
	ax.set_xticklabels([str(cid) for cid in orderedIds], rotation=90 if len(orderedIds) > 20 else 0,
					   fontsize=8 if len(orderedIds) > 20 else 10)
	ax.set_xlabel("Cluster ID (sorted by PMF)")
	ax.set_ylabel("PMF (kcal/mol)")
	ax.legend(fontsize=8, loc="upper left")
	fig.tight_layout()
	fig.savefig(path, dpi=300)
	plt.close(fig)



# Main
# ####

if __name__ == "__main__":

	args = ParseCommandline()

	print("=" * 70)
	print("BlindSpotter2 version %s" % (VERSION))
	print("=" * 70)

	if args.cpptraj_autodetected:
		print("Auto-detected cpptraj executable: %s" % (args.cpptraj))
	else:
		print("Using cpptraj executable: %s" % (args.cpptraj))

	# Water/ion stripping
	# ###################

	residueLabels = ReadTopologyResidueLabels(args.parm)
	stripMask = BuildSolventIonStripMask(residueLabels, args.extra_strip_residues)
	if stripMask is None:
		print("No water/ion residues detected in %s - skipping strip step" % (args.parm))
	else:
		print("Detected solvent/ion residues to strip: %s" % (stripMask))

	# Runs to process
	# ################

	print("Processing %d run(s):" % (len(args.trajectory_files)))
	for i, (trajFile, boostFile) in enumerate(zip(args.trajectory_files, args.boost_files), start=1):
		print("  [%d] trajectory=%s  boost=%s" % (i, trajFile, boostFile))

	# Frame stride
	# ############

	nFramesTotal = sum(CountBoostLogFrames(b) for b in args.boost_files)
	if args.stride > 0:
		stride = args.stride
	else:
		stride = max(1, int(math.ceil(nFramesTotal / float(args.max_cluster_frames))))
	print("Boost log(s) report %d frames total; using stride %d (~%d frames to be processed)" %
		  (nFramesTotal, stride, (nFramesTotal + stride - 1) // stride))

	# Per-trajectory frame counts (for run-by-run alignment with the boost logs)
	# ##########################################################################

	countScript = "cpptraj_count.inp"
	trajCounts = CountTrajectoryFrames(args.cpptraj, countScript, args.parm, args.trajectory_files, stride)

	# Main cpptraj analysis: strip, align, ligand COM, cluster
	# #########################################################

	comFile = "ligand_com.dat"
	cnumFile = "cnumvtime.dat"
	summaryFile = "cluster_summary.dat"
	infoFile = "cluster_info.dat"
	mainScript = "cpptraj_cluster.inp"

	WriteMainCpptrajScript(mainScript, args.parm, args.trajectory_files, stride, stripMask,
							args.protein, args.ligand, args.minpoints, args.epsilon,
							comFile, cnumFile, summaryFile, infoFile)
	RunCpptraj(args.cpptraj, mainScript)

	# Protein-only first-frame reference structure
	# #############################################

	referencePdb = "reference_protein.pdb"
	refScript = "cpptraj_refpdb.inp"
	WriteProteinReferenceCpptrajScript(refScript, args.parm, args.trajectory_files[0], args.protein, referencePdb)
	RunCpptraj(args.cpptraj, refScript)

	# Parse cpptraj outputs and the boost log(s)
	# ###########################################

	clusterIds = ParseClusterAssignments(cnumFile)
	comCoords = ParseLigandCOM(comFile)
	dVRuns = [ParseBoostLog(b, args.dv_columns)[::stride] for b in args.boost_files]

	clusterIds, comCoords, dV = AlignRunFrames(clusterIds, comCoords, dVRuns, trajCounts)

	# Reweighting
	# ###########

	accel = AccelAMD(dV, args.temperature)
	print("")
	print("Effective acceleration factor with Gaussian approximation: %.6g" % (accel))

	pmfResult = ComputeClusterPMF(clusterIds, dV, args.temperature, args.cb_max)
	centers = ComputeClusterCenters(clusterIds, comCoords)

	clusters = pmfResult["clusters"]
	if len(clusters) == 0:
		print("No clusters were found - try relaxing --minpoints/--epsilon - quitting with error code 1")
		sys.exit(1)

	nProcessed = pmfResult["n_total"]
	nNoise = pmfResult["n_noise"]
	print("Frames processed       : %d" % (nProcessed))
	print("Noise (unassigned)     : %d (%.1f%%)" % (nNoise, 100.0 * nNoise / nProcessed))
	print("Number of clusters     : %d" % (len(clusters)))

	infoStats = ParseClusterInfoStats(infoFile)
	if infoStats:
		print("Clustering quality     : " + ", ".join("%s=%.4g" % (k, v) for k, v in infoStats.items()))
	else:
		print("Could not parse DBI/pSF/SSR-SST from %s (format not recognized) - see file directly." % (infoFile))

	orderedIds = sorted(clusters, key=lambda cid: pmfResult["pmf_c2"][cid])
	best = orderedIds[0]

	# Error estimation and cumulant-expansion diagnostics
	# ####################################################

	diag = ComputeDVDiagnostics(clusterIds, dV, args.temperature, clusters)

	boot = None
	if args.error == "block":
		if args.block_frames > 0:
			blockFrames = args.block_frames
			blockSource = "user-specified"
		else:
			blockFrames = EstimateBlockFrames(clusterIds, dV, clusters)
			blockSource = "auto, block averaging"
		boot = BlockBootstrapPMF(clusterIds, dV, args.temperature, clusters, best,
								 blockFrames, args.nboot, args.seed)
		print("PMF error estimate     : moving-block bootstrap, %d resamples, block length %d frames (%s)" %
			  (boot["n_boot"], boot["block_frames"], blockSource))

	print("")
	print("%6s %10s %8s %10s %10s %10s %8s %17s %8s %7s %7s %7s %5s %30s" %
		  ("Cluster", "Population", "Frac(%)", "dV_avg", "dV_std", "PMF", "PMF_SE", "PMF_95%CI", "P(low)",
		   "Skew", "ExKurt", "dC3", "Flags", "Center (x,y,z)"))
	flagsById = {}
	for cid in orderedIds:
		pop = pmfResult["population"][cid]
		frac = 100.0 * pop / nProcessed
		x, y, z = centers[cid]
		flags = ""
		if pmfResult["dV_std"][cid] > SIGMA_DV_MAX:
			flags += "S"
		dC3 = diag[cid]["c3_term"] - diag[best]["c3_term"]
		if abs(dC3) > C3_TERM_MAX:
			# With a bootstrap available, only flag a 3rd-order term that is clearly non-zero
			if boot is None or not (abs(dC3) <= 2.0 * boot["c3_se"][cid]):
				flags += "C"
		if boot is not None:
			ci = "[%7.3f,%7.3f]" % (boot["ci_low"][cid], boot["ci_high"][cid])
			if boot["n_empty"][cid] > EMPTY_FRACTION_MAX * boot["n_boot"]:
				flags += "E"
			errCols = "%8.4f %17s %8.3f" % (boot["se"][cid], ci, boot["p_lowest"][cid])
		else:
			errCols = "%8s %17s %8s" % ("-", "-", "-")
		flagsById[cid] = flags
		print("%6d %10d %8.2f %10.4f %10.4f %10.4f %s %7.3f %7.3f %7.3f %5s %10.3f%10.3f%10.3f" %
			  (cid, pop, frac, pmfResult["dV_avg"][cid], pmfResult["dV_std"][cid], pmfResult["pmf_c2"][cid],
			   errCols, diag[cid]["skew"], diag[cid]["kurt"], dC3, flags or "-", x, y, z))
	print("")
	print("PMF_SE/PMF_95%CI are relative to the lowest-PMF cluster (whose error is therefore 0); "
		  "P(low) is the fraction of bootstrap resamples in which the cluster has the lowest PMF; "
		  "dC3 is the neglected 3rd-order cumulant PMF term (kcal/mol), relative to the lowest-PMF cluster.")
	print("Flags: S = dV_std > %.1f kcal/mol, C = |dC3| > %.2f kcal/mol "
		  "%s(2nd-order cumulant expansion may be inaccurate), E = cluster empty in > %.0f%% of resamples "
		  "(undersampled)." % (SIGMA_DV_MAX, C3_TERM_MAX,
							"and > 2x its bootstrap SE " if boot is not None else "",
							100.0 * EMPTY_FRACTION_MAX))

	print("")
	print("Predicted binding hotspot: cluster %d" % (best))
	print("  Population : %d (%.2f%%)" % (pmfResult["population"][best], 100.0 * pmfResult["population"][best] / nProcessed))
	print("  PMF        : %.4f kcal/mol" % (pmfResult["pmf_c2"][best]))
	if boot is not None:
		print("  P(lowest)  : %.3f" % (boot["p_lowest"][best]))
		if len(orderedIds) > 1:
			second = orderedIds[1]
			print("  Gap to next: %.4f +/- %.4f kcal/mol (cluster %d)" %
				  (pmfResult["pmf_c2"][second], boot["se"][second], second))
	print("  Center     : [%.3f, %.3f, %.3f]" % tuple(centers[best]))

	PrintErrorSummary(pmfResult, boot, flagsById, orderedIds, stride, blockSource if boot is not None else None)

	# Output PDB
	# ##########

	WriteOutputPDB(args.output, centers, pmfResult["pmf_c2"], referencePdb,
				   boot["se"] if boot is not None else None)
	print("")
	if boot is not None:
		print("Wrote cluster centers (PMF in the B-factor column, PMF standard error in the occupancy column) "
			  "and the protein reference to %s" % (args.output))
	else:
		print("Wrote cluster centers (PMF in the B-factor column) and the protein reference to %s" % (args.output))

	# Optional PMF plot
	# #################

	if args.pmf_plot:
		try:
			import matplotlib.pyplot as plt
		except Exception:
			print("matplotlib not available - skipping PMF plot")
		else:
			WritePMFPlot(plt, args.pmf_plot, orderedIds, pmfResult["pmf_c2"], boot, flagsById)
			print("Wrote PMF plot to %s" % (args.pmf_plot))
