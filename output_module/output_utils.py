import re
import json
import time
from rdkit.Chem import PandasTools
import numpy as np
import os
import pandas as pd
from rdkit.Chem import AllChem
import rdkit.Chem as Chem
from rdkit.Chem.rdChemReactions import SanitizeRxn, PreprocessReaction, ReactionFromSmarts, SanitizeFlags
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
import random
from rdkit.Chem import Descriptors

def clear_mapping(smiles):
  product = Chem.MolFromSmiles(smiles)
  product_without_mapping = product
  for atom in product_without_mapping.GetAtoms():
      atom.ClearProp('molAtomMapNumber')
  smiles_without_mapping = Chem.MolToSmiles(product_without_mapping, isomericSmiles=False)
  return smiles_without_mapping

def get_bond_changes(reaction):
    """from dgllife.data.uspto import get_bond_changes"""
    """Get the bond changes in a reaction.

    Parameters
    ----------
    reaction : str
        SMILES for a reaction, e.g. [CH3:14][NH2:15].[N+:1](=[O:2])([O-:3])[c:4]1[cH:5][c:6]([C:7]
        (=[O:8])[OH:9])[cH:10][cH:11][c:12]1[Cl:13].[OH2:16]>>[N+:1](=[O:2])([O-:3])[c:4]1[cH:5]
        [c:6]([C:7](=[O:8])[OH:9])[cH:10][cH:11][c:12]1[NH:15][CH3:14]. It consists of reactants,
        products and the atom mapping.

    Returns
    -------
    bond_changes : set of 3-tuples
        Each tuple consists of (atom1, atom2, change type)
        There are 5 possible values for change type. 0 for losing the bond, and 1, 2, 3, 1.5
        separately for forming a single, double, triple or aromatic bond.
    """
    reactants = Chem.MolFromSmiles(reaction.split('>')[0])
    products  = Chem.MolFromSmiles(reaction.split('>')[2])

    conserved_maps = [
        a.GetProp('molAtomMapNumber')
        for a in products.GetAtoms() if a.HasProp('molAtomMapNumber')]
    bond_changes = set() # keep track of bond changes

    # Look at changed bonds
    bonds_prev = {}
    for bond in reactants.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        if (nums[0] not in conserved_maps) and (nums[1] not in conserved_maps):
            continue
        bonds_prev['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()
    bonds_new = {}
    for bond in products.GetBonds():
        nums = sorted(
            [bond.GetBeginAtom().GetProp('molAtomMapNumber'),
             bond.GetEndAtom().GetProp('molAtomMapNumber')])
        bonds_new['{}~{}'.format(nums[0], nums[1])] = bond.GetBondTypeAsDouble()

    for bond in bonds_prev:
        if bond not in bonds_new:
            # lost bond
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], 0.0))
        else:
            if bonds_prev[bond] != bonds_new[bond]:
                # changed bond
                bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))
    for bond in bonds_new:
        if bond not in bonds_prev:
            # new bond
            bond_changes.add((bond.split('~')[0], bond.split('~')[1], bonds_new[bond]))

    return bond_changes

def find_atoms_with_map_number(molecule, neighbor_map_number, neighbor_atomic_number):
    matching_neighbors_atoms = [atom for atom in molecule.GetAtoms() if
                      atom.HasProp('molAtomMapNumber') and atom.GetProp('molAtomMapNumber') == str(neighbor_map_number)
                                and atom.GetAtomicNum() == neighbor_atomic_number]
    target_atom_canidates = matching_neighbors_atoms[0].GetNeighbors()
    target_candiate_numbers = [(int(atom.GetProp('molAtomMapNumber')), atom.GetAtomicNum()) for atom in target_atom_canidates if
                       atom.HasProp('molAtomMapNumber')]
    return target_candiate_numbers

def get_atom_neighbors(mol, atom_idx):
    atom = mol.GetAtomWithIdx(atom_idx)
    neighbors = atom.GetNeighbors()
    return neighbors

def numbering_atoms(product_s, m):
    atoms_without_map_number = [atom.GetIdx() for atom in product_s.GetAtoms() if not atom.HasProp('molAtomMapNumber')]
    for atom_idx in atoms_without_map_number:
        target_atomic_number = product_s.GetAtomWithIdx(atom_idx).GetAtomicNum()
        all_map_numbers = [atom.GetProp('molAtomMapNumber') for atom in product_s.GetAtoms() if
                           atom.HasProp('molAtomMapNumber')]
        neighbors = get_atom_neighbors(product_s, atom_idx)
        neighbor_map_numbers = [(int(atom.GetProp('molAtomMapNumber')), atom.GetAtomicNum()) for atom in neighbors if
                       atom.HasProp('molAtomMapNumber')]
        for neighbor_map_number, neighbor_atomic_number in neighbor_map_numbers:
            try:
                target_candidate_numbers = find_atoms_with_map_number(m, neighbor_map_number, neighbor_atomic_number)
                for target_candidate_number, target_candidate_atomic_number in target_candidate_numbers:
                    if (str(target_candidate_number) not in all_map_numbers) and (target_atomic_number == target_candidate_atomic_number):
                        product_s.GetAtomWithIdx(atom_idx).SetProp('molAtomMapNumber', str(target_candidate_number))
            except Exception as e:
                # print(f"An error occurred: {e}")
                pass
    return len(atoms_without_map_number)

def numbering_new_oxygen(product_s, m, max_number):
    oxygen_index = []
    for atom in product_s.GetAtoms():
        if atom.GetAtomicNum() == 8 and not atom.HasProp('molAtomMapNumber'):
            max_number += 1
            oxygen_index.append(max_number)
            atom.SetProp('molAtomMapNumber', str(max_number))
    return oxygen_index

def numbering_metabolite(metabolite, reactant):
    if metabolite is not None:
        reactant_map_numbers = [int(atom.GetProp('molAtomMapNumber')) for atom in reactant.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        max_number_sb = max(reactant_map_numbers) if reactant_map_numbers else 0

        metabolite_map_numbers = [int(atom.GetProp('molAtomMapNumber')) for atom in metabolite.GetAtoms() if atom.HasProp('molAtomMapNumber')]
        max_number_pr = max(metabolite_map_numbers) if metabolite_map_numbers else 0

        max_number = max(max_number_sb, max_number_pr)

        count_non = numbering_atoms(metabolite, reactant)
        counter = 0
        while count_non > 0 and counter < 10:
            count_non = numbering_atoms(metabolite, reactant)
            counter += 1

        oxygen_index = numbering_new_oxygen(metabolite, reactant, max_number)

    return metabolite

def fix_missing_map_numbers(smirks):
    reactants, products = smirks.split(">>")
    reactant_mols = [Chem.MolFromSmiles(reactant) for reactant in reactants.split('.')]
    product_mols = [Chem.MolFromSmiles(product) for product in products.split('.')]

    reactant_map_numbers = set()
    for mol in reactant_mols:
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num:
                reactant_map_numbers.add(map_num)

    product_map_numbers = set()
    for mol in product_mols:
        for atom in mol.GetAtoms():
            map_num = atom.GetAtomMapNum()
            if map_num:
                product_map_numbers.add(map_num)

    missing_in_reactants = product_map_numbers - reactant_map_numbers
    missing_in_products = reactant_map_numbers - product_map_numbers

    map_number_offset = max(reactant_map_numbers | product_map_numbers) + 1

    for mol in reactant_mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0 and missing_in_reactants:
                atom.SetAtomMapNum(missing_in_reactants.pop())

    for mol in product_mols:
        for atom in mol.GetAtoms():
            if atom.GetAtomMapNum() == 0 and missing_in_products:
                atom.SetAtomMapNum(missing_in_products.pop())

    new_reactants = '.'.join([Chem.MolToSmiles(mol) for mol in reactant_mols])
    new_products = '.'.join([Chem.MolToSmiles(mol) for mol in product_mols])

    return f"{new_reactants}>>{new_products}"

def comparing_oxygen(smiles1, smiles2):
    mol1 = Chem.MolFromSmiles(smiles1)
    mol2 = Chem.MolFromSmiles(smiles2)
    mol1_oxygen_map = [atom.GetProp('molAtomMapNumber') for atom in mol1.GetAtoms() if
                       atom.HasProp('molAtomMapNumber') and atom.GetAtomicNum() == 8]
    mol2_oxygen_map = [atom.GetProp('molAtomMapNumber') for atom in mol2.GetAtoms() if
                       atom.HasProp('molAtomMapNumber') and atom.GetAtomicNum() == 8]
    additional_oxygen_map = list(set(mol2_oxygen_map) - set(mol1_oxygen_map))
    return additional_oxygen_map


def convert_to_smirks(substrate_smiles, metabolite_smiles):
    smirks = ''
    if metabolite_smiles != '':
        additional_oxygen_map = comparing_oxygen(substrate_smiles, metabolite_smiles)
        if len(additional_oxygen_map) > 0:
            for ox_id in additional_oxygen_map:
                substrate_smiles += f".[OH2:{ox_id}]"
        reaction = AllChem.ReactionFromSmarts(f"{substrate_smiles}>>{metabolite_smiles}")
        smirks = AllChem.ReactionToSmiles(reaction)
    return smirks, additional_oxygen_map

def remove_last_hyphen(s):
    if s.endswith('-'):
        return s[:-1]
    return s


def find_equivalent_atoms(atom, mol):
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    atom_rank = ranks[atom.GetIdx()]
    equivalent_atoms = []

    for other_atom in mol.GetAtoms():
        if atom.GetIdx() == other_atom.GetIdx():
            continue

        other_atom_rank = ranks[other_atom.GetIdx()]

        if atom_rank == other_atom_rank:
            equivalent_atoms.append(other_atom)

    return equivalent_atoms

def find_equivalent_bonds(bond, mol):
    ranks = Chem.CanonicalRankAtoms(mol, breakTies=False)
    begin_rank = ranks[bond.GetBeginAtom().GetIdx()]
    end_rank = ranks[bond.GetEndAtom().GetIdx()]
    equivalent_bonds = []
    for other_bond in mol.GetBonds():
        if bond.GetIdx() == other_bond.GetIdx():
            continue

        other_begin_rank = ranks[other_bond.GetBeginAtom().GetIdx()]
        other_end_rank = ranks[other_bond.GetEndAtom().GetIdx()]

        if (begin_rank == other_begin_rank and end_rank == other_end_rank) or (
                begin_rank == other_end_rank and end_rank == other_begin_rank):
            equivalent_bonds.append(other_bond)

    return equivalent_bonds

def extract_numbers(string_list):
    numbers = []
    for item in string_list:
        # Convert tuples to strings if necessary
        if isinstance(item, tuple):
            item = ''.join(str(i) for i in item)
        elif not isinstance(item, str):
            print(f"Skipping non-string item: {item}")
            continue

        nums = re.findall(r'\d+', item)
        numbers.extend(nums)
    return numbers

def convert_string_to_csv_format(input_string, reaction):
    lines = input_string.strip().split('\n')
    output_lines = []

    for line in lines:
        subtype = line.split(']')[0].replace('[', '').strip()
        remaining_part = line.split('] ')[1]
        scores = remaining_part.split(',')
        som_score = scores[0].split(': ')[1].strip()
        reaction_score = scores[1].split(': ')[1].strip()
        score = scores[2].split(': ')[1].strip()

        output_lines.append([subtype, som_score, reaction_score, score])
    df = pd.DataFrame(output_lines, columns=['Subtype', 'SOM_score', 'Reaction_score', 'Score'])
    df['Reaction'] = reaction.replace('_', ' ')
    return  df