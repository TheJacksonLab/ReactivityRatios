import rdkit 
from rdkit import Chem
from rdkit.Chem import AllChem  
import numpy as np
import os
# Import from chem_data might be needed depending on usage patterns
# from chem_data import conformation, gen_conform
from .smarts_manipulation import get_radical_idx, get_olefin_attack_idx, count_double_bonds, count_num_radical_atoms
from .bel_trans import convert_molecule_file

def transform_mol_xyz(mol, translation=(0, 0, 0), rotation_matrix=None, move_list=None):
    """
    Transform a molecule's coordinates by translation and/or rotation.
    
    Parameters:
    -----------
    mol : rdkit.Chem.Mol
        Input molecule with 3D coordinates
    translation : tuple, default=(0, 0, 0)
        (dx, dy, dz) translation to be applied to the coordinates
    rotation_matrix : np.array, optional
        3x3 matrix for rotation transformation
    move_list : list, optional
        List of atom indices to transform. If None, all atoms are transformed
    
    Returns:
    --------
    rdkit.Chem.Mol
        Transformed molecule with updated coordinates
        
    Example:
    --------
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> AllChem.EmbedMolecule(mol)
    >>> transformed_mol = transform_mol_xyz(mol, translation=(1, 0, 0))
    """
    
    # Make a copy of the molecule
    mol_copy = Chem.Mol(mol)
    conf = mol_copy.GetConformer()
    
    move_list = list(range(mol_copy.GetNumAtoms())) if move_list is None else move_list
    # Apply translation
    for atom_idx in move_list:
        pos = np.array(conf.GetAtomPosition(atom_idx))
        new_pos = pos + np.array(translation)
        
        # Apply rotation if a rotation matrix is provided
        if rotation_matrix is not None:
            new_pos = np.dot(rotation_matrix, new_pos)
        
        conf.SetAtomPosition(atom_idx, tuple(new_pos))
    
    return mol_copy

def UFF_cons_opt(smiles, mol_flag=False, filename="", atom_pair_idx=[], distance_constraint=[], forceConstant=100, maxIts=5000, molout_flag=False):
    """
    Perform constrained geometry optimization using Universal Force Field (UFF).
    
    This function optimizes molecular geometry while maintaining specified distance
    constraints between atom pairs. Useful for generating transition states or
    constrained conformations.
    
    Parameters:
    -----------
    smiles : str or rdkit.Chem.Mol
        Input SMILES string or RDKit molecule object
    mol_flag : bool, default=False
        Whether input is already a molecule object
    filename : str, default=""
        Output filename for optimized structure (optional)
    atom_pair_idx : list, default=[]
        List of atom pairs [(idx1, idx2), ...] for distance constraints
    distance_constraint : list, default=[]
        List of target distances corresponding to atom pairs
    forceConstant : float, default=100
        Force constant for distance constraints
    maxIts : int, default=5000
        Maximum number of optimization iterations
    molout_flag : bool, default=False
        Whether to output as MOL file format instead of XYZ
        
    Returns:
    --------
    tuple or None
        If successful and no filename specified: (atom_list, xyz_array)
        Otherwise: None (writes to file)
        
    Example:
    --------
    >>> atoms, coords = UFF_cons_opt('CCO', atom_pair_idx=[(0,2)], distance_constraint=[3.0])
    """
    mol = Chem.MolFromSmiles(smiles) if not mol_flag else smiles
    mol = Chem.AddHs(mol)  # Add hydrogens for a complete structure
    AllChem.EmbedMolecule(mol)  # Generate a 3D conformation

    fragments = Chem.GetMolFrags(mol, asMols=False, sanitizeFrags=False)
    assert len(fragments) <= 2, "The molecule has more than 2 fragments"
    
    
    # Set up the force field (UFF in this case)
    forcefield = AllChem.UFFGetMoleculeForceField(mol)

    assert isinstance(atom_pair_idx, list), "atom_pair_idx must be a list"
    assert isinstance(distance_constraint, list), "distance_constraint must be a list"
    assert len(atom_pair_idx) == len(distance_constraint), "atom_pair_idx and distance_constraint must have the same length"
    
    for pair_idx, distance in zip(atom_pair_idx, distance_constraint):
        atom1_idx, atom2_idx = pair_idx
        forcefield.AddDistanceConstraint(atom1_idx, atom2_idx, distance, distance, forceConstant)

    # Optimize the molecule with the constraint applied
    forcefield.Initialize()
    success = forcefield.Minimize(maxIts=maxIts)  # Optimize with the force field

    # Check results and write to .xyz file if successful
    if success == 0:
        # print("Optimization completed successfully.")
        conf = mol.GetConformer()
        if filename and not molout_flag:
            # Write the optimized geometry to an .xyz file
            with open(filename, "w") as f:
                num_atoms = mol.GetNumAtoms()
                f.write(f"{num_atoms}\n")
                f.write("Optimized molecule\n")
                
                # Loop through atoms and write coordinates
                for atom in mol.GetAtoms():
                    pos = conf.GetAtomPosition(atom.GetIdx())
                    symbol = atom.GetSymbol()
                    f.write(f"{symbol} {pos.x:.4f} {pos.y:.4f} {pos.z:.4f}\n")
        elif filename and molout_flag:
            Chem.MolToMolFile(mol, filename)
        else:
            atom_list = []
            xyz_array = []
            for atom in mol.GetAtoms():
                pos = conf.GetAtomPosition(atom.GetIdx())
                symbol = atom.GetSymbol()
                atom_list.append(symbol)
                xyz_array.append([pos.x, pos.y, pos.z])
            return atom_list, np.array(xyz_array)
    else:
        # print("Optimization failed to converge.")
        return
    
def gen_rotate_mat(vec1, vec2):
    # Ensure vectors are normalized
    vec1 = vec1 / np.linalg.norm(vec1)
    vec2 = vec2 / np.linalg.norm(vec2)
    
    # Check if vec1 and vec2 are identical
    if np.allclose(vec1, vec2):
        # Return identity matrix if vec1 and vec2 are identical
        return np.eye(3)
    else:
        # Calculate cross product and angle if vec1 != vec2
        v = np.cross(vec1, vec2)
        s = np.linalg.norm(v)
        c = np.dot(vec1, vec2)
        
        # Skew-symmetric cross-product matrix of v
        vx = np.array([[ 0,   -v[2],  v[1]],
                       [ v[2],  0,   -v[0]],
                       [-v[1],  v[0],  0  ]])
        
        # Calculate rotation matrix using Rodrigues' rotation formula
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / s**2)
        
        return R

def translate_and_rotate(xyz_array, translation=np.array([0.0, 0.0, 0.0]), rotation_matrix=np.eye(3), replace_idx=[]):
    # geom center of replace_idx
    if len(replace_idx) == 0:
        center = np.zeros(3)
    else:
        center = np.mean(xyz_array[replace_idx], axis=0)
    new_xyz_array = np.dot((xyz_array - center), rotation_matrix) + translation
    return new_xyz_array

def get_unit_vector(v):
    return v/np.linalg.norm(v)

def smarts2prets_xyz(smarts, out_dir="./", bond_length=2.5):
    """
    Generate pre-transition state XYZ coordinates from reaction SMARTS pattern.
    
    This function processes a reaction SMARTS string to generate 3D coordinates
    for a pre-transition state geometry by:
    1. Separating reactants and identifying radical and olefin fragments
    2. Optimizing individual fragment geometries
    3. Aligning fragments with appropriate separation distance
    4. Writing combined structure to XYZ file
    
    Parameters:
    -----------
    smarts : str
        Reaction SMARTS string in format "reactant1.reactant2>>product"
    out_dir : str, default="./"
        Output directory for XYZ file
    bond_length : float, default=2.5
        Separation distance between reactive centers in Angstroms
        
    Returns:
    --------
    tuple
        (radical_atom_idx, olefin_atom_idx, filename, reactant_smiles)
        - radical_atom_idx: Index of radical atom in combined structure
        - olefin_atom_idx: Index of olefin attack site in combined structure  
        - filename: Path to generated XYZ file
        - reactant_smiles: Original reactant SMILES string
        
    Example:
    --------
    >>> r_idx, o_idx, file, smi = smarts2prets_xyz('[CH3].C=C>>[CH3]CC')
    >>> print(f"Radical at atom {r_idx}, olefin at atom {o_idx}")
    """
    sm_smiles = smarts.split(">>")[0]
    mol = Chem.MolFromSmiles(sm_smiles, sanitize=False)
    sm_inchikey = Chem.MolToInchiKey(mol)

    sm_smiles_list = sm_smiles.split(".")

    # Identify fragments and get atom indices for each fragment
    fragments = [Chem.AddHs(Chem.MolFromSmiles(sm, sanitize=True)) for sm in sm_smiles_list]
    
    # fragments_smiles = list(map(Chem, fragments))
    
    assert len(fragments)==2, f"sm fragment num:{len(fragments)}, not 2"
    
    if count_num_radical_atoms(fragments[0])==1:
        radical_idx, olefin_idx = (0,1)
    else:
        radical_idx, olefin_idx = (1,0)
        
    assert count_double_bonds(fragments[olefin_idx], mol_flag=True)>0, f"olefin fragment with no C=C"
    
    radical_idx = get_radical_idx(sm_smiles_list[radical_idx])
    olefin_idx = get_olefin_attack_idx(sm_smiles_list[olefin_idx])
    
    
    if count_num_radical_atoms(fragments[0])==1: # radical is the first fragment
        radical_atom_lis, radical_xyz_arr = UFF_cons_opt(sm_smiles_list[0])
        olefin_atom_lis, olefin_xyz_arr = UFF_cons_opt(sm_smiles_list[1])
        
    else:
        olefin_atom_lis, olefin_xyz_arr = UFF_cons_opt(sm_smiles_list[0])
        radical_atom_lis, radical_xyz_arr = UFF_cons_opt(sm_smiles_list[1])
    
    # Generate the .xyz file for the reactants
    filename = os.path.join(out_dir, f"{sm_inchikey}.xyz")

    olefin_vec = get_unit_vector(np.mean(olefin_xyz_arr, axis=0) - olefin_xyz_arr[olefin_idx])
    olefin_xyz_arr = translate_and_rotate(olefin_xyz_arr,  replace_idx=[olefin_idx])
    radical_vec = get_unit_vector(np.mean(radical_xyz_arr, axis=0) - radical_xyz_arr[radical_idx])
    radical_xyz_arr = translate_and_rotate(radical_xyz_arr, replace_idx=[radical_idx])
    
    rotation_matrix = gen_rotate_mat(olefin_vec, -radical_vec)
    
    olefin_xyz_arr = translate_and_rotate(olefin_xyz_arr, rotation_matrix=rotation_matrix, translation= -radical_vec*bond_length)
    
    all_xyz_arr = np.concatenate([radical_xyz_arr, olefin_xyz_arr], axis=0)
    atom_list = radical_atom_lis + olefin_atom_lis
    
    with open(filename, "w") as f:
        num_atoms = len(atom_list)
        f.write(f"{num_atoms}\n")
        f.write("Placed TS\n")
        
        # Loop through atoms and write coordinates
        for atom, xyz in zip(atom_list, all_xyz_arr):
            symbol = atom
            f.write(f"{symbol} {xyz[0]:.4f} {xyz[1]:.4f} {xyz[2]:.4f}\n")
    
    new_olefin_idx = olefin_idx + len(radical_atom_lis)
    new_radical_idx = radical_idx
    
    return new_radical_idx+1, new_olefin_idx+1, filename, sm_smiles

def smi_to_optimized_xyz(smi, min_search=100, out_dir="./temp/", pruneRmsThresh=0.2, verbose=False, randomSeed=42, cover=False,
                         useExpTorsionAnglePrefs=True, useBasicKnowledge=True):
    """
    Convert SMILES to optimized XYZ coordinates using conformational search and XTB optimization.
    
    This function provides a complete workflow from SMILES to optimized 3D coordinates:
    1. Generate multiple conformations using RDKit
    2. Optimize with MMFF force field
    3. Further optimize with XTB semi-empirical method
    4. Output XYZ and Gaussian input files
    
    Parameters:
    -----------
    smi : str
        Input SMILES string
    min_search : int, default=100
        Minimum number of conformations to generate
    out_dir : str, default="./temp/"
        Output directory for generated files
    pruneRmsThresh : float, default=0.2
        RMSD threshold for conformation pruning
    verbose : bool, default=False
        Whether to print detailed output from XTB
    randomSeed : int, default=42
        Random seed for reproducible conformation generation
    cover : bool, default=False
        Whether to overwrite existing files
    useExpTorsionAnglePrefs : bool, default=True
        Use experimental torsion angle preferences in embedding
    useBasicKnowledge : bool, default=True
        Use basic chemical knowledge in embedding
        
    Returns:
    --------
    None
        Writes optimized structure files to output directory
        
    Note:
    -----
    Requires XTB program to be installed and accessible in system PATH
    
    Example:
    --------
    >>> smi_to_optimized_xyz('CCO', out_dir='./structures/', verbose=True)
    """
    mol = AllChem.MolFromSmiles(smi, sanitize=True)
    output_name = Chem.MolToInchiKey(AllChem.MolFromSmiles(smi, sanitize=False)) + ".xyz"
    
    # Ensure output directory exists and has trailing slash
    if not out_dir.endswith("/"):
        out_dir += "/"
    original_dir = os.getcwd()
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    os.chdir(out_dir.rstrip("/"))
    
    if os.path.exists(output_name) and not cover:
        convert_molecule_file(output_name, "gjf")
        os.chdir(original_dir)
        return
    
    mol = Chem.AddHs(mol)
    conformers = AllChem.EmbedMultipleConfs(
        mol,
        numConfs = max(min_search, 500-6*len(mol.GetAtoms())),
        pruneRmsThresh = pruneRmsThresh,
        randomSeed=randomSeed,
        useExpTorsionAnglePrefs=useExpTorsionAnglePrefs,
        useBasicKnowledge=useBasicKnowledge,
    )
    def opt_conform(conformer):
        prop = AllChem.MMFFGetMoleculeProperties(mol,mmffVariant = "MMFF94s")
        ff = AllChem.MMFFGetMoleculeForceField(mol,prop,confId = conformer)
        ff.Minimize()
        return float(ff.CalcEnergy())
    
    opt_res = np.array(
    [opt_conform(conformer) for conformer in conformers]
    )
    most_stable_energy = int(opt_res.argmin()) 
    temp_name = ".ffout_xyz.xyz"
    Chem.rdmolfiles.MolToXYZFile(mol,temp_name,confId=most_stable_energy)
    
    redirect = "" if verbose else "> /dev/null 2>&1"
    os.system("xtb "+temp_name+f" --opt {redirect}")
    del_lis = [".xtboptok","charges","wbo","xtbrestart","xtbopt.log","xtbtopo.mol",temp_name]
    for item in del_lis:
        os.system("rm -f "+item)
    
    os.system(f"mv \"xtbopt.xyz\" {output_name}")
    # os.system(f"bel_trans {output_name} gjf {redirect}")
    convert_molecule_file(output_name, "gjf")
    
    os.chdir(original_dir)
    return 
        
    

def do_smarts(smarts, out_dir="./temp/", prefix="PreTS", blength=2.24, verbose=False):
    """
    Process a SMARTS reaction string through a series of geometry optimizations with constraints.
    
    This function takes a SMARTS pattern, generates an initial 3D geometry using smarts2prets_xyz,
    and then performs a series of constrained optimizations using the XTB program with increasing
    force constants to gradually refine the geometry.
    
    Parameters:
    -----------
    smarts : str
        A reaction SMARTS string in the format "reactant1.reactant2>>product"
    out_dir : str, default="./temp/"
        Output directory for temporary and final files
    prefix : str, default="PreTS"
        Prefix for the output filename
    blength : float, default=2.24
        Target bond length for the constraint in Angstroms
        Common values: TS=2.24, product=1.8, pre-TS=3.0
    verbose : bool, default=False
        If True, print detailed progress information during processing
    
    Returns:
    --------
    tuple
        - basename: str, Base name of the output file without extension
        - sm_smiles: str, The original SMARTS reactants as SMILES
    
    Notes:
    ------
    - Requires XTB program to be installed and accessible in the system PATH
    - Uses three stages of optimization with increasing force constants (0.0005, 0.05, 0.5)
    - Creates constraint files (.cons*.inp) in the output directory
    - Automatically cleans up temporary files after optimization
    - The working directory is temporarily changed during execution
    
    Example:
    --------
    >>> basename, reactants = do_smarts('[CH3].C=C>>[CH3]CC', verbose=True)
    >>> print(f"Generated {basename} from {reactants}")
    """
    # Helper function for logging based on verbosity
    def log(message):
        if verbose:
            print(message)
        
    # Ensure output directory exists and has trailing slash
    if not out_dir.endswith("/"):
        out_dir += "/"
    
    # Create output directory if it doesn't exist
    os.makedirs(out_dir, exist_ok=True)
    log(f"Using output directory: {out_dir}")
    

    
    # Generate initial geometry
    log("Generating initial geometry...")
    idx1, idx2, filename, sm_smiles = smarts2prets_xyz(smarts, out_dir=out_dir, bond_length=10)
    redirect = "" if verbose else "> /dev/null 2>&1"
    
    basename = os.path.basename(filename)
    output_name = f"{prefix}_{basename}" if len(prefix)>0 else f"{basename}"
    if os.path.exists(os.path.join(out_dir, output_name)):
        log(f"File {output_name} already exists in {out_dir}. Skipping optimization.")
        return output_name, sm_smiles
    
    log(f"Initial geometry generated. Constraint will be applied between atoms {idx1} and {idx2}")
    
    # Create constraint files with different force constants
    log(f"Creating constraint files with target bond length: {blength:.2f} Å")
    cons_file1 = ["$constrain\n", f"force constant=0.0005 \n", f"distance: {idx1}, {idx2}, {blength:.2} \n", "$end\n"]
    cons_file2 = ["$constrain\n", f"force constant=0.05 \n", f"distance: {idx1}, {idx2}, {blength:.2} \n", "$end\n"]
    cons_file3 = ["$constrain\n", f"force constant=0.5 \n", f"distance: {idx1}, {idx2}, {blength:.2} \n", "$end\n"]
    
    # Write constraint files
    with open(f"{out_dir}.cons1.inp", "w") as f:
        f.writelines(cons_file1)
    with open(f"{out_dir}.cons2.inp", "w") as f:
        f.writelines(cons_file2)
    with open(f"{out_dir}.cons3.inp", "w") as f:
        f.writelines(cons_file3)
    log("Constraint files created")

    # Store original directory to return to it later
    original_dir = os.getcwd()
    log(f"Current directory: {original_dir}")
    
    try:
        # Change to output directory for processing
        os.chdir(out_dir.rstrip("/"))
        log(f"Changed to directory: {os.getcwd()}")
        
        # Get just the basename from the full path
        
        log(f"Processing file: {basename}")
        
        # Run XTB optimizations with increasing constraint strength
        log("Starting first optimization with weak constraint (force constant=0.0005)...")
        
        os.system(f"xtb {basename} --input .cons1.inp --opt --uhf 1 --gfnff {redirect}")
        log("First optimization complete")
        
        log("Starting second optimization with medium constraint (force constant=0.05)...")
        os.system(f"xtb \"xtbopt.xyz\" --input .cons2.inp --opt --uhf 1 --gfnff {redirect}")
        log("Second optimization complete")
        
        log("Starting final optimization with strong constraint (force constant=0.5)...")
        os.system(f"xtb \"xtbopt.xyz\" --input .cons3.inp --opt --uhf 1 --gfn1 {redirect}")
        log("Final optimization complete")
        
        # Rename and convert output file
        
        log(f"Renaming output file to {output_name}")
        os.system(f"mv \"xtbopt.xyz\" {output_name}")
        
        log(f"Converting {output_name} to Gaussian input format")
        # trans_redirect = "" if verbose else "> /dev/null 2>&1"
        # os.system(f"bel_trans {output_name} gjf {trans_redirect}")
        convert_molecule_file(output_name, "gjf")
        
        # Clean up temporary files
        log("Cleaning up temporary files")
        os.system(f"rm -fr xtbopt.* gfnff* .cons*.inp .xtboptok")
        os.system(f"rm -f wbo xtblast.xyz xtbrestart xtbtopo.mol charges")
        log("Cleanup complete")
        
        # Return base filename (without extension) and original SMILES
        return basename.split(".")[0], sm_smiles
    except Exception as e:
        log(f"An error occurred: {e}")
        # Optionally, you can raise the exception or handle it as needed
        raise e
    finally:
        # Always return to the original directory
        log(f"Returning to original directory: {original_dir}")
        os.chdir(original_dir)