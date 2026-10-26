#!/usr/bin/env python

import argparse
import sys
import os
import math
import re
import shutil
import glob
import numpy as np

try:
    import matplotlib.pyplot as plt
    _HAVE_MATPLOTLIB = True
except Exception:
    _HAVE_MATPLOTLIB = False




VERSION = "2.0"
CPPTRAJ = "/usr/local/amber24/bin/cpptraj"  # last-resort fallback if nothing better is found
KB = 0.001987  # Boltzmann constant, kcal/mol/K

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



# Write the output PDB: cluster centers (PMF in the B-factor column) + protein reference
# ############################################################################################

def WriteOutputPDB(outFile, centers, pmf, referencePdb):

	fo = open(outFile, "w")

	orderedIds = sorted(centers.keys(), key=lambda cid: pmf[cid])
	serial = 0
	for cid in orderedIds:
		serial += 1
		x, y, z = centers[cid]
		fo.write("%-6s%5d %-4s%1s%-3s %1s%4d%1s   %8.3f%8.3f%8.3f%6.2f%6.2f\n" %
				 ("HETATM", serial, "CA", " ", "HOT", "X", cid, " ", x, y, z, 1.00, pmf[cid]))
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
	dV = np.concatenate([ParseBoostLog(b, args.dv_columns)[::stride] for b in args.boost_files])

	clusterIds, comCoords, dV = AlignFrameCounts(clusterIds, comCoords, dV)

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

	print("")
	print("%6s %10s %8s %10s %10s %10s %30s" % ("Cluster", "Population", "Frac(%)", "dV_avg", "dV_std", "PMF", "Center (x,y,z)"))
	orderedIds = sorted(clusters, key=lambda cid: pmfResult["pmf_c2"][cid])
	for cid in orderedIds:
		pop = pmfResult["population"][cid]
		frac = 100.0 * pop / nProcessed
		x, y, z = centers[cid]
		print("%6d %10d %8.2f %10.4f %10.4f %10.4f %10.3f%10.3f%10.3f" %
			  (cid, pop, frac, pmfResult["dV_avg"][cid], pmfResult["dV_std"][cid], pmfResult["pmf_c2"][cid], x, y, z))

	best = orderedIds[0]
	print("")
	print("Predicted binding hotspot: cluster %d" % (best))
	print("  Population : %d (%.2f%%)" % (pmfResult["population"][best], 100.0 * pmfResult["population"][best] / nProcessed))
	print("  PMF        : %.4f kcal/mol" % (pmfResult["pmf_c2"][best]))
	print("  Center     : [%.3f, %.3f, %.3f]" % tuple(centers[best]))

	# Output PDB
	# ##########

	WriteOutputPDB(args.output, centers, pmfResult["pmf_c2"], referencePdb)
	print("")
	print("Wrote cluster centers (PMF in the B-factor column) and the protein reference to %s" % (args.output))

	# Optional PMF plot
	# #################

	if args.pmf_plot:
		if _HAVE_MATPLOTLIB:
			values = [pmfResult["pmf_c2"][cid] for cid in orderedIds]
			plt.figure()
			plt.plot(range(len(values)), values, "o-")
			plt.xlabel("Cluster rank (sorted by PMF)")
			plt.ylabel("PMF (kcal/mol)")
			plt.savefig(args.pmf_plot, dpi=300)
			print("Wrote PMF plot to %s" % (args.pmf_plot))
		else:
			print("matplotlib not available - skipping PMF plot")
