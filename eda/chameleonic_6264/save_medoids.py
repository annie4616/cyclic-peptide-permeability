"""Write medoid frames as PDB files for visualization (PyMOL/ChimeraX)."""
import warnings; warnings.filterwarnings("ignore")
import MDAnalysis as mda
ROOT = "/ssd0/sohyun/cyclic_peptide/cyclic_peptide_permeability/CycPeptMPDB-4D"
OUT  = "/ssd0/sohyun/cyclic_peptide/cyclic_peptide_permeability/eda/chameleonic_6264"
PEP  = "2021_Kelly_6264"

# medoid frame indices computed in analyze.py
MED = {"water": (28, f"{ROOT}/Water/Trajectories/{PEP}_H2O_Traj.pdb"),
       "hexane": (41, f"{ROOT}/Hexane/Trajectories/{PEP}_Hexane_Traj.pdb")}
for tag, (idx, path) in MED.items():
    u = mda.Universe(path, path)
    u.trajectory[idx]
    u.atoms.write(f"{OUT}/medoid_{tag}_frame{idx}.pdb")
    print(f"saved {tag} medoid frame {idx} -> {OUT}/medoid_{tag}_frame{idx}.pdb")
