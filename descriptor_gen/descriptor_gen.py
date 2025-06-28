import dbstep.Dbstep as db
import os, sys
import subprocess
import re
import warnings
from openbabel.pybel import readfile

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import numpy as np
import rdkit
from rdkit import Chem
from rdkit.Chem import rdDetermineBonds, rdFingerprintGenerator
from rdkit.Chem import Descriptors, Descriptors3D
from functools import partial
from utils.molFF import get_olefin_attack_idx

import json

command_names = ["basics", "mol_geom", "multipole",
                  "FMO_delocalization_rocshell", "FMO_delocalization_uroshell",
                  "Surface_ELES", "Surface_ALIE", "Surface_LEA"]

multiwfn_descriptors = {
    # Basic Molecular Properties
    "AtomNum": r"Atoms:\s*(\d+)",
    "Weight": r"Molecule weight:\s*([\d.]+)\s*Da",

    # Orbital Delocalization Indices, !! Multiple ouptut shares this pattern
    "ODI_pattern": r"Orbital delocalization index:\s*([\d.]+)",
    
    # Molecular Geometry
    "Farthest_Distance": r"Farthest distance:.*?---\s+([\d.]+)",
    "Mol_Radius": r"Radius of the system:\s+([\d.]+)",
    "Mol_Sizes": r"Length of the three sides:.*?([\d.]+).*?([\d.]+).*?([\d.]+)",
    "MPP": r"Molecular planarity parameter \(MPP\) is\s+([\d.]+)",
    "SDP": r"Span of deviation from plane \(SDP\) is\s+([\d.]+)",
    
    # Multipole Moments
    "Dipole_Moment": r"Magnitude of dipole moment:.*?a\.u\.\s+([\d.]+)",
    "Quadrupole_Moment": r"\|Q_2\|=\s+([\d.]+)",
    "Octopole_Moment": r"\|Q_3\|=\s+([\d.]+)",
    
    # Surface Analysis Properties
    "Volume": r"Volume:\s*([\d.]+)\s*Bohr\^3",
    "Density": r"\(M/V\):\s*([\d.]+)\s*g/cm\^3",
    "ESPmin": r"Minimal value:\s*([\d.-]+)\s*kcal/mol\s*Maximal",
    "ESPmax": r"Maximal value:\s*([\d.-]+)\s*kcal/mol",
    "Overall_Surface_Area": r"Overall surface area:\s*([\d.]+)\s*Bohr\^2",
    "Pos_Surface_Area": r"Positive surface area:\s*([\d.]+)\s*Bohr\^2",
    "Neg_Surface_Area": r"Negative surface area:\s*([\d.]+)\s*Bohr\^2",
    "Overall_Average": r"Overall average value:\s*([\d.-]+)\s*a\.u\.",
    "Pos_Average": r"Positive average value:\s*([\d.-]+)\s*a\.u\.",
    "Neg_Average": r"Negative average value:\s*([\d.-]+)\s*a\.u\.",
    "Overall_Variance": r"Overall variance:\s*([\d.]+)\s*a\.u\.",
    "Nu": r"Balance of charges \(nu\):\s*([\d.]+)",
    "Pi": r"Internal charge separation \(Pi\):\s*([\d.]+)\s*a\.u\.",
    "MPI": r"Molecular polarity index \(MPI\):\s*([\d.]+)\s*eV",
    "Nonpolar_Area": r"Nonpolar surface area \(\|ESP\| <= 10 kcal/mol\):\s*([\d.]+)\s*Angstrom\^2\s*\(\s*([\d.]+)\s*%\)",
    "Polar surface area": r"Polar surface area \(\|ESP\| > 10 kcal/mol\):\s*([\d.]+)\s*Angstrom\^2\s*\(\s*([\d.]+)\s*%\)",
    
    # ALIE Analysis Properties
    "ALIEmin": r"Minimal value:\s*([\d.-]+)\s*eV,\s*Maximal",
    "ALIEmax": r"Maximal value:\s*([\d.-]+)\s*eV",
    "ALIE_Ave": r"Average value:\s*([\d.-]+)\s*a\.u\.",
    "ALIE_Var": r"Variance:\s*([\d.]+)\s*a\.u\.",
    
    # LEA Analysis Properties
    "LEAmin": r"Minimal value:\s*([\d.-]+)\s*eV,\s*Maximal",
    "LEAmax": r"Maximal value:\s*([\d.-]+)\s*eV",
    "LEA_Ave": r"Average value:\s*([\d.-]+)\s*a\.u\.",
    "LEA_Var": r"Variance:\s*([\d.]+)\s*a\.u\.",
    
    # FMO Energy
    "Orbital_Energy_Section": r"Orbital:\s+\d+\s+Energy\(a\.u\.\):\s+([\d.-]+)",
    
    # Section Headers for Detection
    "Surface_Analysis_Section": r"================= Summary of surface analysis =================\s*(.*?)(?===================)",
    "ALIE_Section": r"================= Summary of ALIE analysis =================\s*(.*?)(?===================)",
    "LEA_Section": r"================= Summary of LEA analysis =================\s*(.*?)(?===================)"
}

def get_multiwfn_path():
    """
    Get the path to the Multiwfn executable.
    
    This function first tries to find Multiwfn in the system PATH using 'which' command.
    If not found, it returns None and issues a warning.
    
    Returns:
    --------
    str or None
        Path to Multiwfn executable if found, None otherwise
        
    Note:
    -----
    DEPENDENCY: Requires Multiwfn to be installed and accessible.
    If Multiwfn is not in PATH, uncomment and modify the hardcoded path example.
    """
    # return "/home/xxx/apps/Multiwfn_3.8_dev/Multiwfn_noGUI" # Hardcoded path example
    try:
        result = subprocess.run(["which", "Multiwfn"], 
                               capture_output=True, 
                               text=True, 
                               check=False)
        if result.returncode == 0:
            return result.stdout.strip()
    except FileNotFoundError:
        warnings.warn("Multiwfn executable not found in PATH. Please set the path manually.")
        return None

def _run_single_multiwfn_call(target_file, input_stream, 
                              multiwfn_path=get_multiwfn_path(),
                              output_name="", silent=False):
    """
    Helper function to run a single Multiwfn call on one file.
    Returns the path to the result file produced.
    """
    if not input_stream.endswith('q\n'):
        input_stream += 'q\n'
    
    assert isinstance(target_file, str), "target_file must be a string."
    assert os.path.isfile(target_file), f"File {target_file} does not exist."
    assert os.path.isfile(multiwfn_path), f"Multiwfn executable {multiwfn_path} does not exist."
    assert isinstance(input_stream, str), "input_stream must be a string."
    assert isinstance(output_name, str), "output_name must be a string."
    
    if not silent:    
        if len(output_name) > 0:
            result_file = os.path.join(os.path.dirname(target_file), output_name)
        else:
            result_file = os.path.splitext(target_file)[0] + ".txt"
    else:
        result_file = "/dev/null"
        
    # Build the shell command
    arg = (
        f'{multiwfn_path} {target_file} << EOF > {result_file}\n'
        f'{input_stream}'
        'EOF\n'
    )
    # Execute the command
    subprocess.run(arg, shell=True)
    
    if not silent:
        return result_file
    else:
        return None
    
def gen_cube(molden_path, grid_qual=1, info_only=False):
    command_dict = {
    "den_cube_exp": ((5, 1, grid_qual, 2), (0,), 
            "Exporting electron densitygrid data to density.cub"),
    }
    command_stream = "\n".join(map(str, command_dict["den_cube_exp"][0] + command_dict["den_cube_exp"][1])) + "\nq\n"
    
    
    if info_only:
        for key, value in command_dict.items():
            print(f"<{key}>: {value[-1]}")
        return
    else:
        if not os.path.isfile(molden_path):
            raise ValueError(f"{molden_path} not exist")
        else:
            dir_path = os.path.dirname(molden_path)
            basename = os.path.basename(molden_path).split('.')[0]
            cube_name = f"{basename}.cube"
            cube_path = os.path.join(dir_path, cube_name)
            
            # record current dir
            current_dir = os.getcwd()
            
            os.chdir(dir_path)
            _run_single_multiwfn_call(f"{basename}.molden.input", command_stream,
                                     silent=True, multiwfn_path=get_multiwfn_path())
            os.system(f"mv density.cub {cube_name}")
            os.chdir(current_dir)
            return cube_path
        
def gen_command_stream(command_names, grid_qual=1, info_only=False, is_os=False):
    """
    Generate the command stream for Multiwfn.
    """
    
    os_sel = "FMO_delocalization_rocshell" if is_os else "FMO_delocalization_uroshell"
    
    command_dict = {
    "basics": ((0,), (), 
                        "Basic information of the molecule"),
    "mol_geom": ((100, 21, 'size', 0, 'MPP', 'a', 'n'), ("q", 0), 
                "Molcular geometry, including size and molecular planarity parameter (MPP) and span of deviation from plane (SDP)"),
    "multipole": ((300, 5), (0,), "dipole/multipole moments"),
    "FMO_delocalization_rocshell": ((8, 8, 1, 'h-1', 'h', 'l', 'l+1'), (0, -10), 
                                "Delocalization index of FMO, restricted open- or closed-shell molecule"),
    "FMO_delocalization_uroshell": ((8, 8, 1, 'ha', 'hb' ,'la', 'lb'), 
                                (0, -10), "Delocalization index of FMO, unrestricted open-shell molecule"),
    "Surface_ELES": ((12, 0), (-1, -1), "Qantitative ESP analysis on molecular surface"),
    "Surface_ALIE": ((12, 2, 2, 0), (-1, -1), "Qantitative Average local ionization energy (ALIE) analysis on molecular surface"),
    "Surface_LEA": ((12, 2, 4, 0), (-1, -1), "Qantitative Local electron affinity (LEA) analysis on molecular surface"),
    }
    
    if info_only:
        for key, value in command_dict.items():
            print(f"<{key}>: {value[-1]}")
        return
    
    assert grid_qual in [1, 2, 3], "Grid quality must be 1 (low), 2 (medium), or 3 (high)."
    
    error_names = []
    res_stream = ""
    
    for command_name in command_names:
        if command_name not in command_dict:
            error_names.append(command_name)
        elif command_name == os_sel:
            pass
        else:
            command = command_dict[command_name]
            commands = command[0] + command[1]
            res_stream += "\n".join(map(str, commands)) 
            if not res_stream.endswith('\n'):
                res_stream = res_stream + '\n'
    if len(error_names) > 0:
        raise ValueError(f"Command(s) '{', '.join(error_names)}' not recognized.")
    return res_stream
        
def get_chg_mul(molden_path, input_stream="q\n", 
                multiwfn_path=get_multiwfn_path()):
    """
    Get the charge and multiplicity from the Molden file.
    """
    tmp_name = "tmp.txt"
    tmp_path = os.path.join(os.path.dirname(molden_path), tmp_name)
    _run_single_multiwfn_call(molden_path, input_stream, multiwfn_path, output_name=tmp_name)
    with open(tmp_path, "r") as f:
        cont = f.read()
        
    chg_mul_re = r'\sNet charge:\s+([+-]?\d+\.\d+)\s+Expected multiplicity:\s+(\d+)'
    chg_mul = re.findall(chg_mul_re, cont)
    result = chg_mul[0] if len(chg_mul) > 0 else None
    
    if result is not None:
        result = (round(float(result[0])), int(result[1]))
    os.remove(tmp_path)
    return result

def orca2xyz(orca_out_path):
    """
    Convert ORCA output file to XYZ format.
    """
    xyz_path = os.path.splitext(orca_out_path)[0] + ".xyz"
    if not os.path.exists(xyz_path):
        path = os.path.splitext(orca_out_path)[0]
        in_, out_ = "orca", "xyz"
        for mymol in readfile(in_, orca_out_path):
            mymol.write(out_, path + '.' + out_, overwrite=True)
    return xyz_path

def GetMolfromXYZ(xyz_path, chrg=0, multi=1):
    chrg_sign = -1 if chrg >=0 else 1
    uhf = multi - 1
    rev_chrg = chrg + chrg_sign * uhf
    
    raw_mol = Chem.MolFromXYZFile(xyz_path)
    mol = Chem.Mol(raw_mol)
    rdDetermineBonds.DetermineBonds(mol, charge=rev_chrg)
    
    for atom in mol.GetAtoms():
        if uhf>0:
            if atom.GetFormalCharge() == chrg_sign:
                atom.SetFormalCharge(0)  # Remove the negative charge
                atom.SetNumRadicalElectrons(1)  # Set it as a radical
                uhf -= 1
        else:
            break
    
    return mol

def Getfp_generator(fp_name="morgan", fpSize=2048, r=2, info_only=False):
    """
    Get the fingerprint generator based on the name.
    https://greglandrum.github.io/rdkit-blog/posts/2023-01-18-fingerprint-generator-tutorial.html
    """
    fingerprint_generators = {
        "morgan": (rdFingerprintGenerator.GetMorganGenerator, "Morgan Fingerprint"),
        "rdkfp": (rdFingerprintGenerator.GetRDKitFPGenerator, "RDKit Fingerprint"),
        "apfp": (rdFingerprintGenerator.GetAtomPairGenerator, "Atom Pairs Fingerprint"),
        "ttgen": (rdFingerprintGenerator.GetTopologicalTorsionGenerator, "Topological Torsions Fingerprint"),
        "fmgen": (partial(rdFingerprintGenerator.GetMorganGenerator, 
                        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen()), 
                        "Feature Morgan Fingerprint")
    }
    
    if info_only:
        for key, value in fingerprint_generators.items():
            print(f"{key}: {value[1]}")
        return
    else:
        if fp_name not in fingerprint_generators:
            print("Fingerprints:")
            Getfp_generator(info_only=True)
            raise ValueError(f"{fp_name} not a valid fingerprint name")
        else:
            return fingerprint_generators[fp_name][0](fpSize=fpSize, radius=r)

def Get_ReactAtom(mol, rad_flag):
    # if it is radical, we choose the atom with radical electron
    # if it is olefin, we choose the less connected C=C carbon ("Markovnikov's rule")
    if rad_flag:
        for atom in mol.GetAtoms():
            if atom.GetNumRadicalElectrons() > 0:
                return atom.GetIdx() + 1 # 1-based index
    else:
        return get_olefin_attack_idx(mol, mol_flag=True) + 1 # 1-based index
    

def main(gbw_path, BV_flag=True, rdkit_flag=True, fp_flag=True):
    if not os.path.isfile(gbw_path):
        raise ValueError(f"{gbw_path} not exist")
    
    # Path Information
    base_path = os.path.splitext(gbw_path)[0]
    orca_out_path = base_path + ".out"
    molden_path = base_path + ".molden.input"
    multiwfn_output = os.path.splitext(molden_path)[0] + ".txt"
    res_json_path = os.path.splitext(molden_path)[0] + ".json"
    idx_path = base_path + ".idx.txt"

    if not os.path.exists(molden_path):
        # Convert GBW to Molden format
        subprocess.run(f"orca_2mkl {base_path} -molden", shell=True)
        print(f"Converted {gbw_path} to {molden_path}")
        

        
    # Run Multiwfn for charge and multiplicity
    chrg, mul = get_chg_mul(molden_path, multiwfn_path=get_multiwfn_path())
    rad_flag = True if mul > 1 else False
    
    if not os.path.exists(multiwfn_output):
        # Run Multiwfn for various properties, assuming open-shell == unrestricted, controlled by is_os
        _run_single_multiwfn_call(molden_path, 
                            input_stream=gen_command_stream(command_names, grid_qual=1, is_os=rad_flag),
                            multiwfn_path=get_multiwfn_path(), output_name='')

    def extract_value(pattern, content):
        match = re.search(pattern, content)
        if match:
            return match.group(1)
        return None
        
    with open(multiwfn_output, "r") as f:
        content = f.read()
        
    result_dict = {}

    # Round 1 Extraction: Basic Properties in Multiwfn output
    for descriptor, pattern in multiwfn_descriptors.items():
        if "Section" not in descriptor:  # Skip section headers
            value = extract_value(pattern, content)
            if value:
                result_dict[descriptor] = float(value) 
                
    # Round 2 Extraction: Molecular Geometry in Multiwfn output
    mol_sizes_match = re.search(multiwfn_descriptors["Mol_Sizes"], content)
    if mol_sizes_match:
        mol_size = [float(mol_sizes_match.group(1)), float(mol_sizes_match.group(2)), float(mol_sizes_match.group(3))]
        sorted_mol_size = sorted(mol_size)
        result_dict["Mol_Size_Short"] = sorted_mol_size[0]
        result_dict["Mol_Size_2"] = sorted_mol_size[1]
        result_dict["Mol_Size_L"] = sorted_mol_size[2]
        result_dict["Length_Ratio"] = sorted_mol_size[2] / sum(mol_size)
        result_dict["Len_Div_Diameter"] = sorted_mol_size[2] / (2 * result_dict["Mol_Radius"]) if "Mol_Radius" in result_dict else None
        
    # Round 3 Extraction: FMO Information in Multiwfn output
    orbitalE_values = re.findall(multiwfn_descriptors['Orbital_Energy_Section'], content)
    odi_values = re.findall(multiwfn_descriptors["ODI_pattern"], content)

    if not rad_flag:
        odi_names = ["ODI_HOMO_1", "ODI_HOMO", "ODI_LUMO", "ODI_LUMO_Add1", "ODI_Mean", "ODI_Std"] 
        orbitalE_names = ["HOMO_1", "HOMO", "LUMO", "LUMO_Add1"]
    else:
        # 'ha', 'hb' ,'la', 'lb'
        odi_names = ["ODI_HOMO_a", "ODI_HOMO_b", "ODI_LUMO_a", "ODI_LUMO_b", "ODI_Mean", "ODI_Std"] 
        orbitalE_names = ["HOMO_a", "HOMO_b", "LUMO_a", "LUMO_b"]


    if odi_values and orbitalE_values:
        odi_values = [float(x) for x in odi_values]
        orbitalE_values = [float(x) for x in orbitalE_values]
        
        assert len(odi_values) == len(orbitalE_values), f"\
            length of FMO Energy values {len(orbitalE_values)} != ODI values {len(odi_values)}"
        
        for i in range(max(len(odi_values), len(orbitalE_values))):
            if i >= len(odi_names):
                warnings.warn("Catching redundant FMO orbital information, could causing error")
                break
            result_dict[odi_names[i]] = odi_values[i] if len(odi_values) > i else None
            result_dict[orbitalE_names[i]] = orbitalE_values[i] if len(orbitalE_values) > i else None
            
        result_dict["ODI_Mean"] = sum(odi_values) / len(odi_values) if odi_values else None
        result_dict["ODI_Std"] = (sum((x - result_dict["ODI_Mean"]) ** 2 for x in odi_values) / len(odi_values)) ** 0.5 if odi_values else None

    if rdkit_flag:
        # Round 4 Extraction: Rdkit Descriptors
        xyz_path = orca2xyz(orca_out_path)
        mol = GetMolfromXYZ(xyz_path, chrg, mul)
        
        descriptors3D = Descriptors3D.CalcMolDescriptors3D(mol)
        smiles = Chem.MolToSmiles(Chem.RemoveHs(mol))
        # using mol directly will cause error in Descriptors.BCUT2D in rdkit 2024.3.2
        descriptors2D = Descriptors.CalcMolDescriptors(Chem.MolFromSmiles(smiles))
        ## The following command could make more sense, works properly for rdkit  2024.9.5
        # descriptors2D = Descriptors.CalcMolDescriptors(mol)
        
        result_dict.update(descriptors2D)
        result_dict.update(descriptors3D)
        result_dict['smiles'] = smiles
    
    if BV_flag:
        # Round 5 Extraction: Buried Volume
        cube_path = gen_cube(molden_path, grid_qual=1, info_only=False)
        
        if os.path.isfile(idx_path):
            with open(idx_path, "r") as f:
                idx_content = f.read()
            reactA_idx = idx_content.strip().split()[0]
        else:
            reactA_idx = Get_ReactAtom(mol, rad_flag)
        result_dict['react_atom_serial'] = reactA_idx
        
        if cube_path is not None:
            # Calculate buried volume around reactive center using dbstep
            # DEPENDENCY: Requires dbstep package for steric analysis
            # db_mol = db.dbstep(cube_path, atom1=result_dict['react_atom_serial'],volume=True,
            #                    scan="1.0:4.0:0.1")  # Optional: custom radial scan
            db_mol = db.dbstep(cube_path, atom1=result_dict['react_atom_serial'],volume=True)
            result_dict['Buried_Volume'] = db_mol.bur_vol
        else:
            warnings.warn("Buried Volume calculation failed.")
    
    if fp_flag:
        # Round 6 Extraction: Molecular fingerprints for similarity analysis
        # Generate Morgan fingerprints (ECFP-like) for machine learning applications
        fp_generator = Getfp_generator(fp_name="morgan", fpSize=2048, r=2, info_only=False)
        fp = fp_generator.GetFingerprint(mol)
        fp_list = fp.ToList()
        result_dict['Morgan_Fingerprint'] = fp_list
    
    # Save comprehensive descriptor dictionary to JSON file
    # This file contains all calculated molecular properties and descriptors
    json.dump(result_dict, open(res_json_path, "w"), indent=4)
    print(f"Results saved to {res_json_path}")
    
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run Multiwfn and extract properties.")
    parser.add_argument("--gbw_path", type=str, help="Path to the GBW file.")
    parser.add_argument("--no_BV_flag", action="store_true", help="Calculate Buried Volume.")
    parser.add_argument("--no_rdkit_flag", action="store_true", help="Calculate Rdkit Descriptors.")
    parser.add_argument("--no_fp_flag", action="store_true", help="Calculate Fingerprint.")
    args = parser.parse_args()
    
    main(args.gbw_path, 
         BV_flag=not args.no_BV_flag, 
         rdkit_flag=not args.no_rdkit_flag, 
         fp_flag=not args.no_fp_flag)
    ## Debug
    # main("ABSHBZODGOHLFR-UHFFFAOYSA-N/ABSHBZODGOHLFR-UHFFFAOYSA-N.gbw")