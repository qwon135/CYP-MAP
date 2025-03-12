from rdkit import Chem
from rdkit.Chem import PandasTools
from rdkit.Chem import AllChem
import re
import copy

def get_equivalent_dict(mol):
    equivalent_dict = {}
    ranked = list(Chem.CanonicalRankAtoms(mol, breakTies=False))

    for atom_idx, rank_value in enumerate(ranked):
        equivalent_dict[atom_idx] = [idx for idx, rv in enumerate(ranked) if (rv == rank_value)]#and (atom_idx != idx)]
    return equivalent_dict

def atom_pair2bondidx(mol):
    atom_pair2bond = {}

    for bond_idx, bond in enumerate(mol.GetBonds()):
        # 결합의 시작 원자와 끝 원자 인덱스 얻기
        begin_atom_idx, end_atom_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()         
        atom_pair2bond[(begin_atom_idx, end_atom_idx)] = bond_idx
        atom_pair2bond[(end_atom_idx, begin_atom_idx)] = bond_idx

    return atom_pair2bond

def get_equivalent_bonds(mol):
    # mol = AllChem.AddHs( mol, addCoords=True)
    equivalent_dict = get_equivalent_dict(mol)        
    
    atom_pair2bond = atom_pair2bondidx(mol)
    eq_atoms = []

    for atom_idx in range(mol.GetNumAtoms()):
        for pair_atom_idx in equivalent_dict[atom_idx ]:
            if atom_idx == pair_atom_idx:
                continue
            eq_atoms.append((atom_idx, pair_atom_idx))

    eq_bonds = []

    for bond_idx, bond in enumerate(mol.GetBonds()):
        # 결합의 시작 원자와 끝 원자 인덱스 얻기
        s_idx, e_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx() #
        
        for eq_s_idx in equivalent_dict[s_idx]:
            for eq_e_idx in equivalent_dict[e_idx]:
                if (eq_s_idx, eq_e_idx) in atom_pair2bond:
                    if eq_s_idx == eq_e_idx:
                        continue
                    eq_bond_idx = atom_pair2bond[(eq_s_idx, eq_e_idx)]

                    if bond_idx == eq_bond_idx:
                        continue
                    
                    eq_bonds.append(
                        tuple(sorted([bond_idx, eq_bond_idx]))
                        )                    
    eq_bonds = list(set(eq_bonds))
    return eq_atoms, eq_bonds