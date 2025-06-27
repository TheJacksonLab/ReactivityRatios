from rdkit.Chem import AllChem
from rdkit import Chem
from rdkit.Chem import Draw
import rdkit
import numpy as np
import matplotlib.pyplot as plt
import py3Dmol
import torch
import cairosvg
# from copy import deepcopy

class conformation:
    """
    A class for managing molecular conformations with 3D coordinates.
    
    This class stores molecular conformations as a collection of atomic coordinates
    and provides methods for visualization and file output.
    
    Attributes:
        atom_lis (list): List of atomic symbols
        conf_lis (list): List of conformations, each as a numpy array of shape (n_atoms, 3)
        mol (rdkit.Chem.Mol): RDKit molecule object (optional)
        smi (str): SMILES string representation (optional)
    """
    def __init__(self, atom_lis=None, xyz=None, smi=None, mol=None):
        """
        Initialize a conformation object.
        
        Parameters:
        -----------
        atom_lis : list, optional
            List of atomic symbols (e.g., ['C', 'H', 'H', 'H', 'O'])
        xyz : array-like or list of array-like, optional
            3D coordinates as numpy array(s) of shape (n_atoms, 3)
        smi : str, optional
            SMILES string representation of the molecule
        mol : rdkit.Chem.Mol, optional
            RDKit molecule object
        """
        self.atom_lis = atom_lis
        self.conf_lis = list()
        self.mol= mol
        self.smi=smi
        if xyz is not None:
            if isinstance(xyz, list):
                for i in range(len(xyz)):
                    self.add_xyz(xyz[i]) 
            else:
                self.add_xyz(xyz) 
            pass
    
    def add_xyz(self, xyz):
        """
        Add a new conformation to the conformation list.
        
        Parameters:
        -----------
        xyz : array-like
            3D coordinates as list, torch.Tensor, or numpy array
            Will be reshaped to (n_atoms, 3) format
        """
        if isinstance(xyz, (list, torch.Tensor)):
            xyz = np.array(xyz)
        assert isinstance(xyz, np.ndarray), "xyz should be a list, torch.Tensor, or equivalent that can be converted to a numpy array"
        xyz = np.reshape(xyz, (-1, 3))
        self.conf_lis.append(xyz)
        return
        
    def conf2xyzf(self, id=0, save_flag=False, filename="conf.xyz"):
        """
        Convert a conformation to XYZ file format.
        
        Parameters:
        -----------
        id : int, default=0
            Index of the conformation to convert
        save_flag : bool, default=False
            Whether to save the XYZ data to a file
        filename : str, default="conf.xyz"
            Output filename if save_flag is True
            
        Returns:
        --------
        str
            XYZ format string representation of the conformation
        """
        assert isinstance(id, int), "id should be an int"
        assert len(self.conf_lis) > id, "id exceed length of conformation space"
        cont = f"{len(self.atom_lis)}\nconformer{id}\n"
        xyz = self.conf_lis[id]
        for i in range(len(self.atom_lis)):
            cont = cont + f"{self.atom_lis[i]}\t{xyz[i,:][0]}\t{xyz[i,:][1]}\t{xyz[i,:][2]}\n"
        
        if save_flag:
            with open(filename, "w") as f:
                f.write(cont)
        
        return cont
    
    def show3D(self, id=0, filename=None):
        """
        Display a 3D visualization of the molecular conformation.
        
        Parameters:
        -----------
        id : int, default=0
            Index of the conformation to visualize
        filename : str, optional
            If provided, read XYZ data from this file instead of using stored conformation
            
        Note:
        -----
        Requires py3Dmol for 3D visualization in Jupyter notebooks
        """
        if filename is not None:
            with open(".temp_xyz", "r") as f:
                xyz = f.read()
        else:
            xyz = self.conf2xyzf(id=id)
        xyzview = py3Dmol.view(width=400,height=400)
        xyzview.addModel(xyz,'xyz',{'vibrate': {'frames':10,'amplitude':1}})
        xyzview.setStyle({'stick':{}})
        xyzview.setBackgroundColor('0xeeeeee')
        xyzview.animate({'loop': 'backAndForth'})
        xyzview.zoomTo()
        xyzview.show()
        return
    
    
def gen_conform(mol, conf_window=1, smi_flag=False, ff_name="MMFF94s"):
    """
    Generate optimized molecular conformations using force field methods.
    
    This function generates multiple conformations of a molecule using RDKit's embedding
    algorithm, optimizes them with MMFF force field, and returns the lowest energy conformations.
    
    Parameters:
    -----------
    mol : rdkit.Chem.Mol or str
        Input molecule object or SMILES string (if smi_flag=True)
    conf_window : int, default=1
        Number of lowest energy conformations to return
    smi_flag : bool, default=False
        Whether the input mol is a SMILES string
    ff_name : str, default="MMFF94s"
        Force field variant to use for optimization
        
    Returns:
    --------
    conformation
        Conformation object containing optimized geometries
        
    Example:
    --------
    >>> mol = Chem.MolFromSmiles('CCO')
    >>> conf = gen_conform(mol, conf_window=3)
    >>> print(len(conf.conf_lis))  # Number of conformations
    3
    """ 
    if smi_flag:
        mol = Chem.MolFromSmiles(mol)
    smi = Chem.MolToSmiles(mol)

    assert isinstance(mol, rdkit.Chem.rdchem.Mol), "input should be rdkit.Chem.rdchem.Mol, or use smi_flag==True for smiles input"
    # generate conformations using smiles, conf_window=1 for only the best
    mol = Chem.AddHs(mol)
    # print("Doing conformation Generating!")
    # t1 = time()
    conformers = AllChem.EmbedMultipleConfs(
        mol,
        numConfs = max(100,500-6*len(mol.GetAtoms())),
        pruneRmsThresh = 0.2,
        randomSeed=1,
        useExpTorsionAnglePrefs=True,
        useBasicKnowledge=True,
    )

    print(f"Using generic algorithm for conformation search:\n\tmax_search=100, rmsd_thresh=0.2, opt={ff_name}, num_conf={conf_window}")
    def opt_conform(conformer):
        prop = AllChem.MMFFGetMoleculeProperties(mol,mmffVariant = f"{ff_name}")
        ff = AllChem.MMFFGetMoleculeForceField(mol,prop,confId = conformer)
        ff.Minimize()
        return float(ff.CalcEnergy())

    # print("Doing MMFF opt!")
    opt_res = np.array(
        [opt_conform(conformer) for conformer in conformers]
    )

    energies = np.array(opt_res)
    sorted_indices = np.argsort(energies)[:conf_window]  # Get the indices of the top 10 lowest energy values

    atom_list = [atom.GetSymbol() for atom in mol.GetAtoms()]  # Get the atom symbols
    xyz_lis = []
    for i in sorted_indices:
        most_stable_energy = int(i)
        conf = mol.GetConformer(id=most_stable_energy)
        # dist_matrix = Chem.rdmolops.Get3DDistanceMatrix(mol, confId=most_stable_energy)
        # xyz_tensor = np.asarray(dist_matrix)
        xyz_coords = conf.GetPositions()
        # Print the atom list and XYZ tensor
        # print("Atom List:", atom_list)
        # print("XYZ Tensor:")
        # print(xyz_coords)
        xyz_lis.append(xyz_coords)
    return conformation(atom_lis=atom_list, xyz=xyz_lis, smi=smi, mol=mol)
  
def smiles2graph(smi, show=True, size=(75,75)):
    """
    Convert SMILES string to molecular graph and optionally display it.
    
    Parameters:
    -----------
    smi : str
        SMILES string representation of the molecule
    show : bool, default=True
        Whether to display the molecular graph
    size : tuple, default=(75, 75)
        Size of the displayed image in pixels
        
    Returns:
    --------
    rdkit.Chem.Mol
        RDKit molecule object
        
    Example:
    --------
    >>> mol = smiles2graph('CCO', show=True)
    >>> # Displays ethanol structure
    """ # display smiles: smi in, graph out
    mol = Chem.MolFromSmiles(smi)
    if show:
        # Generate the image of the molecule graph
        img = Draw.MolToMPL(mol, size=size)

        # Display the image
        # plt.imshow(img)
        plt.axis("off")
        # plt.show()
    return mol

def smile_addH(smi):
    """
    Add explicit hydrogens to a SMILES string.
    
    Parameters:
    -----------
    smi : str
        Input SMILES string
        
    Returns:
    --------
    str
        SMILES string with explicit hydrogens
        
    Example:
    --------
    >>> smile_addH('CCO')
    '[H]C([H])([H])C([H])([H])O[H]'
    """
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        mol = None
    assert isinstance(mol, rdkit.Chem.rdchem.Mol), "input should be a valid smiles"
    mol = Chem.AddHs(mol)
    new_smi = Chem.MolToSmiles(mol)
    return new_smi

def smile_rmH(smi):
    """
    Remove explicit hydrogens from a SMILES string.
    
    Parameters:
    -----------
    smi : str
        Input SMILES string with explicit hydrogens
        
    Returns:
    --------
    str
        SMILES string without explicit hydrogens
        
    Example:
    --------
    >>> smile_rmH('[H]C([H])([H])C([H])([H])O[H]')
    'CCO'
    """
    try:
        mol = Chem.MolFromSmiles(smi)
    except:
        mol = None
    assert isinstance(mol, rdkit.Chem.rdchem.Mol), "input should be a valid smiles"
    mol = Chem.RemoveHs(mol)
    new_smi = Chem.MolToSmiles(mol)
    return new_smi

def get_morganFP(datast, smi_flag=True, r=0, process=False, save=False, save_to=None):
    """
    Generate Morgan fingerprints for molecular similarity analysis.
    
    Parameters:
    -----------
    datast : str or list
        SMILES string or list of dictionaries containing molecular data
    smi_flag : bool, default=True
        Whether input is a single SMILES string (True) or dataset (False)
    r : int, default=0
        Radius for Morgan fingerprint calculation
    process : bool, default=False
        Whether to print progress for large datasets
    save : bool, default=False
        Whether to save fingerprint visualizations
    save_to : str, optional
        Path to save fingerprint images
        
    Returns:
    --------
    tuple or set
        If smi_flag=True: (fingerprint, bit_info)
        If smi_flag=False: (unique_bits, fingerprint_list, bit_info_list)
        
    Example:
    --------
    >>> fp, bi = get_morganFP('CCO', r=2)
    >>> print(len(fp))  # Fingerprint length
    """
    res = set()
    if not smi_flag:
        fp_lis = []
        bi_lis = []
        for i in range(len(datast)):
            smi = datast[i]['smi']
            mol = Chem.MolFromSmiles(smi)
            bi = {}
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=r, bitInfo=bi)
            for key in bi.keys():
                if save and (key not in res):
                    assert save_to is not None, "please input a valid save path, see: 'save_to=xxx'"
                    svg_data = Draw.DrawMorganBit(mol, key, bi, useSVG=True)
                    cairosvg.svg2png(bytestring=svg_data.data, write_to=save_to)
                    
            res = res | bi.keys()
            fp_lis.append(fp)
            bi_lis.append(bi)
            if process and i%5000==0:
                print(f"{i//1000}k")
        return res, fp_lis, bi_lis
    else:
        smi = datast
        mol = Chem.MolFromSmiles(smi)
        bi = {}
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=r, bitInfo=bi)
        return fp, bi
    
def enhance_fp(fp_arr, bi):
    """
    Enhance fingerprint array with frequency information.
    
    Parameters:
    -----------
    fp_arr : array-like
        Binary fingerprint array
    bi : dict
        Bit information dictionary from Morgan fingerprint
        
    Returns:
    --------
    numpy.ndarray
        Enhanced fingerprint with frequency counts
    """
    temp = [0] * len(fp_arr)
    for key in iter(bi.keys()):
        temp[key] = len(bi[key])
    return np.array(temp).astype(int)

def is_valid_smi(smi):
    """
    Check if a SMILES string is valid.
    
    Parameters:
    -----------
    smi : str
        SMILES string to validate
        
    Returns:
    --------
    bool
        True if SMILES is valid, False otherwise
        
    Example:
    --------
    >>> is_valid_smi('CCO')
    True
    >>> is_valid_smi('invalid_smiles')
    False
    """
    assert isinstance(smi, str), "smile should be a string obj"
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            return True
        return False
    except:
        return False
    
def get_atom_type(smi):
    """
    Get unique atom types present in a molecule.
    
    Parameters:
    -----------
    smi : str
        SMILES string
        
    Returns:
    --------
    list
        Sorted list of unique atomic symbols
        
    Example:
    --------
    >>> get_atom_type('CCO')
    ['C', 'O']
    """
    mol = Chem.MolFromSmiles(smi, sanitize=False)
    res = set()
    for atom in mol.GetAtoms():
        res.add(atom.GetSymbol())
    return sorted(list(res))