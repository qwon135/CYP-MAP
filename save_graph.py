import torch
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from matplotlib import pyplot as plt
import inspect, random, os
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from multiprocessing import Pool
from rdkit.Chem import Draw, PandasTools
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score, average_precision_score

from sklearn.model_selection import train_test_split
from torch_geometric.nn.aggr import AttentionalAggregation
import torch
from modules.som_dataset import CustomDataset
from modules.som_dataset import mol2bond_label, mol2graph, get_equivalent_bonds, check_ns_oxidation
from tqdm import tqdm

import warnings
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

cyp_list = ['BOM_1A2', 'BOM_2A6', 'BOM_2B6', 'BOM_2C8', 'BOM_2C9', 'BOM_2C19', 'BOM_2D6', 'BOM_2E1', 'BOM_3A4', 'CYP_REACTION']

df = PandasTools.LoadSDF('../data/train_nonreact_0611.sdf')
test_df = PandasTools.LoadSDF('../data/test_0611.sdf')

df.loc[df['PubChem_CID'] == '2078','BOM_3A4' ] = '<2,6;Cleavage;R1>\n<6,H;Hydroxylation;R1>'
df.loc[df['PubChem_CID'] == '45279963','BOM_2B6' ] = '<2,31;Cleavage;R1>'
df.loc[df['PubChem_CID'] == '134821691', 'BOM_1A2'] = '<30,9;Cleavage;R1>'
df.loc[df['PubChem_CID'] == '134821691', 'BOM_2B6'] = ''
df.loc[df['PubChem_CID'] == '134821691', 'BOM_2D6'] = ''

def CYP_REACTION(x):
    cyp_col = ['BOM_1A2', 'BOM_2A6', 'BOM_2B6', 'BOM_2C8', 'BOM_2C9', 'BOM_2C19', 'BOM_2D6', 'BOM_2E1', 'BOM_3A4',]
    cyp_reactions = x[cyp_col].tolist()
    cyp_reactions = [i for i in cyp_reactions if i] 
    return '\n'.join( cyp_reactions )

def is_has_H(atom_idx, bonds_idx_h, atoms_h):
    return any([(atoms_h[i] == 'H' or atoms_h[j] == 'H') for i,j in bonds_idx_h if atom_idx in (i,j)])

df['CYP_REACTION'] = df.apply(CYP_REACTION, axis=1)
test_df['CYP_REACTION'] = test_df.apply(CYP_REACTION, axis=1)

df['POS_ID'] = 'TRAIN' + df.index.astype(str).str.zfill(4)
test_df['POS_ID'] = 'TEST' + test_df.index.astype(str).str.zfill(4)

data = pd.concat([df, test_df]).reset_index(drop=True)

for i in tqdm(range(data.shape[0])):
    pos_id = data.loc[i, 'POS_ID']
    mol = data.loc[i, 'ROMol']

    mol_H = AllChem.AddHs( mol, addCoords=True)
    atoms_h = [i.GetSymbol() for i in mol_H.GetAtoms()]
    bonds_idx_h = [(i.GetBeginAtomIdx(),i.GetEndAtomIdx()) for i in mol_H.GetBonds()]
    bonds_h = [(i.GetBeginAtom().GetSymbol(),i.GetEndAtom().GetSymbol()) for i in mol_H.GetBonds()]
    
    graph_h = mol2graph(mol_H)
    graph_h.x[:, 4] = torch.LongTensor([i.GetTotalNumHs() for i in mol.GetAtoms()] + [0] * (mol_H.GetNumAtoms() - mol.GetNumAtoms()) )
    
    eq_atoms, eq_bonds = get_equivalent_bonds(mol_H)
    
    graph_h.equivalent_atoms = eq_atoms
    graph_h.equivalent_bonds = eq_bonds
    graph_h.pe = torch.Tensor([0])
    graph_h.smile =  Chem.MolToSmiles(mol_H)
    graph_h.spn_atom = torch.BoolTensor([i in ['S', 'N'] for i in atoms_h ])
    # graph_h.has_H_atom = torch.BoolTensor([i.GetTotalNumHs() for i in mol.GetAtoms()] + [False] * (len(atoms_h) - mol.GetNumAtoms()))
    has_h_atom = [is_has_H(atom_idx, bonds_idx_h, atoms_h) for atom_idx in range(len(atoms_h))][:mol.GetNumAtoms()]
    graph_h.is_H = torch.BoolTensor([i == 'H' for i in atoms_h])
    graph_h.has_H_atom =   torch.BoolTensor( has_h_atom + [False] * (len(atoms_h) - mol.GetNumAtoms()))
    graph_h.not_has_H_bond = torch.BoolTensor([('H' not in i) for i in bonds_h])
    graph_h.atoms = atoms_h
    graph_h.bonds_idx_h = bonds_idx_h

    H_atom_idx = torch.where(graph_h.has_H_atom)[0]
    first_H_bond_idx = []
    for h_atom_idx in H_atom_idx:
        first_H_bond_idx.append([n for n, (i,j) in enumerate(bonds_idx_h) if h_atom_idx in (i,j) and 'H' in bonds_h[n]][0])
    first_H_bond_idx = [i in first_H_bond_idx for i in range((len(bonds_idx_h)))]

    graph_h.first_H_bond_idx = torch.BoolTensor(first_H_bond_idx)
    y, y_substrate, y_atom = {}, {}, {}        
    y_bond_hydroxylation, y_bond_oxidation, y_bond_reduction, y_bond_cleavage = {}, {}, {}, {}
    y_atom_hydroxylation, y_atom_oxidation, y_atom_reduction, y_atom_cleavage= {}, {}, {}, {}
    y_spn = {}

    for cyp in cyp_list:                
        reactions = data.loc[i, cyp].split('\n')
        reactions = [check_ns_oxidation(i) for i in reactions]
        
        (bond_label,
        bond_cleavage,
        bond_reduction,
        bond_hydroxylation,
        bond_oxidation,
        atom_label,
        atom_cleavage,
        atom_reduction,
        atom_hydroxylation,
        atom_oxidation,
        atom_spn) = mol2bond_label(atoms=atoms_h, bonds=bonds_h, bonds_idx=bonds_idx_h, reactions=reactions)    

        y[cyp] = torch.FloatTensor(bond_label)
        y_substrate[cyp] = torch.FloatTensor([1 if data.loc[i, cyp] != '' else 0])
        y_atom[cyp] = torch.FloatTensor(atom_label)        

        y_spn[cyp] = torch.FloatTensor(atom_spn)

        y_bond_cleavage[cyp] = torch.FloatTensor(bond_cleavage)
        y_bond_hydroxylation[cyp] = torch.FloatTensor(bond_hydroxylation)
        y_bond_reduction[cyp] = torch.FloatTensor(bond_reduction)
        y_bond_oxidation[cyp] = torch.FloatTensor(bond_oxidation)

        y_atom_hydroxylation[cyp] = torch.FloatTensor(atom_hydroxylation)
        y_atom_oxidation[cyp] = torch.FloatTensor(atom_oxidation)
        y_atom_reduction[cyp] = torch.FloatTensor(atom_reduction)
        y_atom_cleavage[cyp] = torch.FloatTensor(atom_cleavage)
        
    
    graph_h.y = y
    graph_h.y_substrate = y_substrate
    graph_h.y_atom = y_atom
        
    graph_h.y_spn = y_spn

    graph_h.y_bond_cleavage = y_bond_cleavage
    graph_h.y_bond_hydroxylation = y_bond_hydroxylation
    graph_h.y_bond_oxidation = y_bond_oxidation    
    graph_h.y_bond_reduction = y_bond_reduction

    graph_h.y_atom_cleavage = y_atom_cleavage
    graph_h.y_atom_hydroxylation = y_atom_hydroxylation
    graph_h.y_atom_oxidation = y_atom_oxidation    
    graph_h.y_atom_reduction = y_atom_reduction
    

    graph_h.mid = data.loc[i, 'POS_ID']
    torch.save(graph_h, f'graph_pt/{pos_id}_addh.pt')