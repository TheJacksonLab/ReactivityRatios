import rdkit 
from rdkit import Chem 
from rdkit.Chem import AllChem 
from rdkit.Chem import Draw 
from rdkit import RDLogger 
RDLogger.DisableLog('rdApp.*')    # Disable rdkit warnings in force

import pandas as pd 
import numpy as np

# No direct import from chem_data to avoid circular imports

def find_radical_atom_indices(mol):
    """
    Find the indices of atoms in a molecule that have a single electron (radicals).
    
    Parameters:
    mol (rdkit.Chem.Mol): The input molecule.
    
    Returns:
    list: A list of atom indices with single electrons.
    """
    radical_indices = []
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() == 1:
            radical_indices.append(atom.GetIdx())
    return radical_indices

def count_num_radical_atoms(mol):
    return len(find_radical_atom_indices(mol))

def GetId_by_Symbol(mol, symbol):
    """ 
    Get the Index of atom with element str(symbol) from mol
    :return: the list of matched atom index
    """
    idx_ls = []
    for atom in mol.GetAtoms():
        if atom.GetSymbol()==symbol: 
            idx_ls.append(atom.GetIdx())
    if len(idx_ls) <= 0: 
        print('No matched atom.')
        return []
    else: 
        return idx_ls
    
def GetId_neighbor_by_ID(mol, idx):
    """
    Get the neighboring atom's index of atom(idx)
    :return: the list of matched atom index 
    """
    nei_ls = []
    center = mol.GetAtomWithIdx(idx)
    neighbors = center.GetNeighbors()
    for atom in neighbors:
        nei_ls.append(atom.GetIdx())
    if len(nei_ls) <= 0: print('No neighbor.')
    else: return nei_ls
    
def CombineFrag(mol, frag, identifier_symbol='*', default_symbol='H'):
    mol, frag = Chem.RemoveHs(mol), Chem.RemoveHs(frag)
    # Get the index of the identifier atom
    mol_iden_ids = GetId_by_Symbol(mol, identifier_symbol)
    frag_iden_ids = GetId_by_Symbol(frag, identifier_symbol)
    assert len(mol_iden_ids) > 0, f'No identifier symbol {identifier_symbol} in mol'
    assert len(frag_iden_ids) > 0, f'No identifier symbol {identifier_symbol} in frag'
    
    combo = Chem.CombineMols(mol, frag)
    combo_iden_ids = GetId_by_Symbol(combo, identifier_symbol)
    combo_lis = [combo for _ in range(len(mol_iden_ids) * len(frag_iden_ids) +1)]
    
    mol_idens = [ i for i in combo_iden_ids if i < len(mol.GetAtoms()) ]
    frag_idens = [ i for i in combo_iden_ids if i >= len(mol.GetAtoms()) ]
    

    # make combination
    k=1
    for i in range(len(mol_idens)):
        for j in range(len(frag_idens)):
            
            targ_id1 = mol_idens[i]
            targ_id2 = frag_idens[j]
            # get the neighbor of the target atom
            mol_nei = GetId_neighbor_by_ID(combo, targ_id1)
            frag_nei = GetId_neighbor_by_ID(combo, targ_id2)
            
            # identifier should be 1-valent
            assert len(mol_nei) == 1, f'atom {targ_id1} in mol is not 1-valent'
            assert len(frag_nei) == 1, f'atom {targ_id2} in frag is not 1-valent'
            # get the id of the neighbor
            mol_nei_id = mol_nei[0]
            frag_nei_id = frag_nei[0]
            
            combo_lis[k] = Chem.RWMol(combo_lis[k])
            combo_lis[k].AddBond(mol_nei_id, frag_nei_id, order=Chem.rdchem.BondType.SINGLE)
            
                
            # remove the larger index first to avoid index change
            combo_lis[k].RemoveAtom(targ_id2)
            combo_lis[k].RemoveAtom(targ_id1)
            
            # replace rest of the identifier with default_symbol
            for atom in combo_lis[k].GetAtoms():
                if atom.GetSymbol() == identifier_symbol:
                    combo_lis[k].GetAtomWithIdx(atom.GetIdx()).SetAtomicNum(Chem.GetPeriodicTable().GetAtomicNumber(default_symbol))
            
            k+=1
    return combo_lis



def get_smi_prod(smiles, capping_frag='[*][H]', identifier_symbol='*'):
    if identifier_symbol not in smiles:
        return smiles
    mol = Chem.MolFromSmiles(smiles)
    capping_frag = Chem.MolFromSmiles(capping_frag)
    assert mol is not None, f"smiles: {smiles} invalid"
    return Chem.MolToSmiles(CombineFrag(mol, capping_frag)[1])

def get_sm_prod(smarts):
    
    reactants_smiles = smarts.split(">>")[0].split(".")
    reactants_smiles = [reactants_smiles] if not isinstance(reactants_smiles, list) else reactants_smiles
    
    products_smiles = smarts.split(">>")[1].split(".")
    products_smiles = [products_smiles] if not isinstance(products_smiles, list) else products_smiles
    return reactants_smiles, products_smiles 


def unpack_all(nested_iterable):
    """
    Recursively unpack all elements from nested iterables into a flat list.
    
    Parameters:
    nested_iterable (iterable): The nested iterable to unpack.
    
    Returns:
    list: A flat list with all elements unpacked.
    """
    flat_list = []
    for item in nested_iterable:
        if isinstance(item, (list, tuple, set)):
            flat_list.extend(unpack_all(item))
        else:
            flat_list.append(item)
    return flat_list


def capping_smarts(smarts, capping_frag='[*][H]'):
    reactants_smiles, products_smiles = get_sm_prod(smarts)
    reactants_smiles = [get_smi_prod(reactant, capping_frag) for reactant in reactants_smiles]
    products_smiles = [get_smi_prod(product, capping_frag) for product in products_smiles]
    
    new_smarts = ".".join(reactants_smiles) + ">>" + ".".join(products_smiles)
    return new_smarts

def count_double_bonds(smiles, mol_flag=False):
    mol = Chem.MolFromSmiles(smiles) if not mol_flag else smiles
    assert mol is not None, f"Smiles:{smiles} not valid"
    double_bonds = 0
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if begin_atom.GetSymbol() == 'C' and end_atom.GetSymbol() == 'C':
                double_bonds += 1
    return double_bonds

def get_olefin_attack_idx(smiles, mol_flag=False):
    mol = Chem.MolFromSmiles(smiles) if not mol_flag else smiles
    mol = Chem.RemoveHs(mol)
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Find the first C=C bond
    double_bond = None
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            # Get the two carbon atoms involved in the double bond
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C':
                double_bond = bond
                break

    if double_bond is None:
        raise ValueError("No C=C bond found in the molecule")

    # Count non-H connections for each carbon
    non_h_connections1 = atom1.GetExplicitValence()
    non_h_connections2 = atom2.GetExplicitValence()

    # Determine which carbon is less connected
    if non_h_connections1 < non_h_connections2:
        less_connected = atom1
        more_connected = atom2
    else:
        less_connected = atom2
        more_connected = atom1
    
    return less_connected.GetIdx()

def get_radical_idx(smiles, mol_flag=False):
    mol = Chem.MolFromSmiles(smiles) if not mol_flag else smiles
    if mol is None:
        raise ValueError("Invalid SMILES string")
    
    radical_idx_lis = find_radical_atom_indices(mol)
    
    assert len(radical_idx_lis)==1, f"radical num:{len(radical_idx_lis)}, not 1"
    return radical_idx_lis[0]

def smarts2dicts(smarts):
    """
    Convert a SMARTS reaction pattern into dictionaries mapping InChIKeys to SMILES strings
    and reaction components.
    
    This function processes a reaction SMARTS string (reactants>>products) and creates two
    dictionaries:
    1. A mapping from InChIKeys to their corresponding SMILES strings for all components
       (individual reactants, combined reactants, and product)
    2. A reaction dictionary that maps the reactants' InChIKey to a tuple containing
       (product InChIKey, first reactant InChIKey, second reactant InChIKey)
    
    The function handles both sanitized and unsanitized molecule conversions to ensure
    robust InChIKey generation for potentially complex or unusual structures.
    
    Parameters:
    -----------
    smarts : str
        A reaction SMARTS string in the format "reactant1.reactant2>>product"
    
    Returns:
    --------
    tuple
        - inchikey2smiles_dic: Dict mapping InChIKeys to their corresponding SMILES strings
        - inchikey_reaction_dic: Dict mapping reactants InChIKey to tuple of 
          !!!(product(TS) InChIKey, reactant1 InChIKey, reactant2 InChIKey)!!!
    
    Notes:
    ------
    - Creates InChIKeys from both sanitized and unsanitized molecule objects to ensure
      maximum coverage and robustness
    - Assumes the reaction has exactly two reactants (separated by '.') and one product
    - Used for tracking and indexing reactions in chemical reaction databases
    """
    sm_smiles, prod_smiles = smarts.split(">>")
    sm_1, sm_2 = sm_smiles.split(".")
    
    # Create a dictionary mapping InChIKeys to their corresponding SMILES strings
    # First try with unsanitized molecules (preserves original structure exactly)
    inchikey2smiles_dic = {Chem.MolToInchiKey(Chem.MolFromSmiles(sm, sanitize=False)): 
        sm for sm in (sm_1, sm_2, prod_smiles, sm_smiles)}
    
    # Then update with sanitized molecules (standard representation, more canonical)
    inchikey2smiles_dic.update({Chem.MolToInchiKey(Chem.MolFromSmiles(sm)): 
        sm for sm in (sm_1, sm_2, prod_smiles, sm_smiles)})
    
    # Create a dictionary mapping the combined reactants InChIKey to a tuple of
    # (product InChIKey, reactant1 InChIKey, reactant2 InChIKey)
    inchikey_reaction_dic = {Chem.MolToInchiKey(Chem.MolFromSmiles(sm_smiles, sanitize=False)): 
        (Chem.MolToInchiKey(Chem.MolFromSmiles(prod_smiles)), 
         Chem.MolToInchiKey(Chem.MolFromSmiles(sm_1)), 
         Chem.MolToInchiKey(Chem.MolFromSmiles(sm_2)))}
    
    return inchikey2smiles_dic, inchikey_reaction_dic

def substitute_smiles(smiles1, smiles2):
    """
    function: take in smiles1 and smiles2, identify the radical atom in smiles1 and anonymous group (*) in smiles2. And substitute anonymous group with smiles1 part
    Step by step: (0) concatenate smiles1 and smiles2 -> "smi1.smi2"
    (1) Identify the radical atom in smiles1, and the anonymous group (*) & single bond connecting anonymous group in smiles2. 
    (2) Remove the anonymous group (*) and the single bond in smiles2, connect the radical atom in smiles1 to the rest of smiles2 (to previous anonymous connected atom).
    (3) set the radical atom in smiles1 to neutral. 
    (4) return the new smiles.
    
    smi1: radical smiles, smi2: olefin smiles(anonymous group *)
    """

    # Step 1: Identify the radical atom in smiles1 and the anonymous group (*) in smiles2
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    combined_mol = Chem.CombineMols(mol1, mol2)
    
    radical_atom_idx = None
    for atom in mol1.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            radical_atom_idx = atom.GetIdx()
            break
    
    anonymous_group_idx = None
    for atom in mol2.GetAtoms():
        if atom.GetSymbol() == '*':
            anonymous_group_idx = atom.GetIdx()
            break

    if radical_atom_idx is None or anonymous_group_idx is None:
        raise ValueError("Radical atom or anonymous group not found in the provided SMILES strings.")
    else:
        anonymous_group_idx += mol1.GetNumAtoms()  # Adjust index for the combined molecule
    
    # Step 2: Connect the radical atom in smiles1 to the rest of smiles2, 
    # Remove the anonymous group (*) and the single bond in smiles2
    editable_mol = Chem.EditableMol(combined_mol)
    neighbors = combined_mol.GetAtomWithIdx(anonymous_group_idx).GetNeighbors()
    if len(neighbors) != 1:
        raise ValueError("Anonymous group should be connected to exactly one other atom.")
    
    connected_atom_idx = neighbors[0].GetIdx()
    
    editable_mol.AddBond(radical_atom_idx, connected_atom_idx, Chem.BondType.SINGLE)
    editable_mol.RemoveAtom(anonymous_group_idx)
    
    # Step 3: Set the radical atom in smiles1 to neutral
    combined_mol = editable_mol.GetMol()
    atom = combined_mol.GetAtomWithIdx(radical_atom_idx)
    atom.SetNumRadicalElectrons(0)
    
    # Return the new SMILES
    new_smiles = Chem.MolToSmiles(combined_mol)
    return new_smiles


def olefin_to_radical(smiles):
    """
    Convert an olefin (alkene) to a radical species by breaking the C=C double bond.
    
    This function:
    1. Finds the first C=C bond in the molecule
    2. Determines which carbon is less substituted
    3. Adds an anonymous group (*) to the less substituted carbon
    4. Converts the more substituted carbon to a radical
    5. Changes the double bond to a single bond
    
    Parameters:
    -----------
    smiles : str
        SMILES string containing at least one C=C double bond
        
    Returns:
    --------
    str
        SMILES string with the olefin converted to a radical species
        
    Raises:
    -------
    ValueError
        If no C=C bond is found in the molecule
        
    Example:
    --------
    >>> olefin_to_radical('C=CC')
    '[CH2*][CH*]C'  # Propene converted to radical
    """
    mol = Chem.RemoveHs(Chem.MolFromSmiles(smiles))
    if mol is None:
        raise ValueError("Invalid SMILES string")

    # Find the first C=C bond
    double_bond = None
    for bond in mol.GetBonds():
        if bond.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            # Get the two carbon atoms involved in the double bond
            atom1 = bond.GetBeginAtom()
            atom2 = bond.GetEndAtom()
            if atom1.GetSymbol() == 'C' and atom2.GetSymbol() == 'C':
                double_bond = bond
                break

    if double_bond is None:
        raise ValueError("No C=C bond found in the molecule")

    # Count non-H connections for each carbon
    non_h_connections1 = atom1.GetExplicitValence()
    non_h_connections2 = atom2.GetExplicitValence()

    # Determine which carbon is less connected
    if non_h_connections1 < non_h_connections2:
        less_connected = atom1
        more_connected = atom2
    else:
        less_connected = atom2
        more_connected = atom1
    
    # Make the more connected carbon a radical
    more_connected.SetNumRadicalElectrons(1)
    
    # Add anonymous group (*) to the less connected carbon
    less_connected.SetAtomMapNum(1)  # Mark the atom for later identification
    mol = Chem.RWMol(mol)
    
    mol.AddAtom(Chem.Atom('*'))
    mol.AddBond(less_connected.GetIdx(), mol.GetNumAtoms() - 1, Chem.rdchem.BondType.SINGLE)

    # Remove the double bond, get a single bond
    mol.GetBondBetweenAtoms(less_connected.GetIdx(), more_connected.GetIdx()).SetBondType(Chem.rdchem.BondType.SINGLE)
    
    # Convert back to SMILES
    new_smiles = Chem.MolToSmiles(mol)
    new_smiles = new_smiles.replace('[*]', '*')  # Replace atom map number with *
    return new_smiles


