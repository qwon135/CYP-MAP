from modules.dualgraph.mol import smiles2graphwithface
from modules.dualgraph.dataset import DGData
import torch, random
import numpy as np
import torch_geometric.transforms as T
from rdkit.Chem import AllChem
from rdkit import Chem
from torch_geometric.data import InMemoryDataset
from modules.dualgraph.dataset import DGData
from modules.add_equevalent import get_equivalent_bonds
from torch_geometric.utils import to_dense_batch, dense_to_sparse
from torch_geometric.utils import shuffle_node, mask_feature, dropout_node, dropout_edge
from torch_geometric.transforms import RandomNodeSplit
from copy import deepcopy
# import warnings
# warnings.filterwarnings('ignore')

cyp_col = ['BOM_1A2', 'BOM_2A6', 'BOM_2B6', 'BOM_2C8', 'BOM_2C9', 'BOM_2C19', 'BOM_2D6', 'BOM_2E1', 'BOM_3A4',]
cyp2label = {
    'BOM_1A2' : 0,
    'BOM_2A6' : 1,
    'BOM_2B6' : 2,
    'BOM_2C8' : 3,
    'BOM_2C9' : 4,
    'BOM_2C19' : 5,
    'BOM_2D6' : 6,
    'BOM_2E1' : 7,
    'BOM_3A4' : 8
    }


def mol2graph(mol):
    data = DGData()
    graph = smiles2graphwithface(mol)

    data.__num_nodes__ = int(graph["num_nodes"])
    data.edge_index = torch.from_numpy(graph["edge_index"]).to(torch.int64)
    data.edge_attr = torch.from_numpy(graph["edge_feat"]).to(torch.int64)
    data.x = torch.from_numpy(graph["node_feat"]).to(torch.int64)    

    data.ring_mask = torch.from_numpy(graph["ring_mask"]).to(torch.bool)
    data.ring_index = torch.from_numpy(graph["ring_index"]).to(torch.int64)
    data.nf_node = torch.from_numpy(graph["nf_node"]).to(torch.int64)
    data.nf_ring = torch.from_numpy(graph["nf_ring"]).to(torch.int64)
    data.num_rings = int(graph["num_rings"])
    data.n_edges = int(graph["n_edges"])
    data.n_nodes = int(graph["n_nodes"])
    data.n_nfs = int(graph["n_nfs"])

    return data

# To DO
# 갯수는 어떻게 처리할지?
#  S랑 N 을 따로 할지 같이 할지?

def check_ns_oxidation(reaction):
    if not reaction:
        return reaction
    
    atom = reaction.split(',')[1].split(';')[0]
    reac_type = reaction.split(',')[1].split(';')[1]
    if atom in ['N', 'S'] and reac_type == 'Oxidation':
        reaction = reaction.replace('Oxidation', f'{atom}-Oxidation')
    return reaction

def check_reaction(bond_atom_sidx, bond_atom_eidx, bond_atom_s_symbol, bond_atom_e_symbol, reaction):
    bond_atom_sidx, bond_atom_eidx = bond_atom_sidx+1, bond_atom_eidx+1 # 0부터 시작해서 맞춰 줘야함
    
    reaction_atoms, reaction_type, _ = reaction[1:-1].split(';')
    reaction_atom_s, reaction_atom_e = reaction_atoms.split(',')
    
    reaction_atom_s = int(reaction_atom_s)

    if reaction_type in ['S-Oxidation', 'N-Oxidation', 'P-Oxidation', 'SPN-Oxidation', 'spn']:
        return 0

    # 둘다 숫자인 경우는 해당 atom 인덱스 일치 여부 확인
    if reaction_atom_e == 'H': # Dealkylation or Oxidation
        if ((reaction_atom_s == bond_atom_sidx) and (bond_atom_e_symbol == 'H')) or ((reaction_atom_s == bond_atom_eidx) and (bond_atom_s_symbol ==  'H')): 
            return 1
    else:
        reaction_atom_e = int(reaction_atom_e)
        if ((bond_atom_sidx == reaction_atom_s) and (bond_atom_eidx == reaction_atom_e)) or ((bond_atom_sidx == reaction_atom_e) and (bond_atom_eidx == reaction_atom_s)):                    
            return 1        
    return 0

def is_has_H(atom_idx, bonds_idx_h, atoms_h):
    return any([(atoms_h[i] == 'H' or atoms_h[j] == 'H') for i,j in bonds_idx_h if atom_idx in (i,j)])
  
def mol2bond_label(atoms, bonds, bonds_idx, reactions, return_type=False):
    type_collect = {
            '' : '',
            # 1. OxidativeCleavage
            'Dealkylation' : 'oxc', 
            'Ox-Ring-opening' : 'oxc', 
            'Decarboxylation' : 'oxc',
            'S-N-Cleavage' : 'oxc', 
            'Decarboxylation' : 'oxc', 
            'Ox-Ring-opening' : "oxc",
            'Cleavage' : 'oxc',

            # 2. Oxidation
            'Hydroxylation' : 'oxi',
            'Oxidiation' : 'oxi',
            'Oxidation' : 'oxi',
            'UnspOxidation' : 'oxi',

            # 3. Epoxidation
            'Epoxidation' : 'epo',


            # 4.Substitution
            'Dehalogenation' : 'sut',
            'Desulfuration' : 'sut',
            'Deboronation' : 'sut',
            'Denitrosation' : 'sut',
            
            # 5. Dehydrogenation
            'Dehydrogenation' : 'dhy',
            'Dehydration' : 'dhy',

            # 6. SPN-Oxidation
            'N-Oxidation': 'spn',
            'S-Oxidation': 'spn',            

            # 7. Hydrolysis
            'Hydrolysis' : 'hys',

            # 8. Reduction
            'Reduction' : 'rdc',
            'UnspReduction' : 'rdc',

            # ETC
            'Rearrangement' : 'Rearrangement', 
            'Cyclization' : 'Cyclization',

            }
    
    reactions = [i for i in list(set(reactions)) if i]           
    labels = {
                'bond_som' : [0] * len(bonds),
                'bond_oxc' : [0] * len(bonds),
                'bond_oxi' : [0] * len(bonds),
                'bond_epo' : [0] * len(bonds),
                'bond_sut' : [0] * len(bonds),
                'bond_dhy' : [0] * len(bonds),
                'bond_hys' : [0] * len(bonds),
                'bond_rdc' : [0] * len(bonds),

                'atom_som' : [0] * len(atoms),
                'atom_spn' : [0] * len(atoms),
                'atom_oxc' : [0] * len(atoms),
                'atom_oxi' : [0] * len(atoms),
                'atom_epo' : [0] * len(atoms),
                'atom_sut' : [0] * len(atoms),
                'atom_dhy' : [0] * len(atoms),
                'atom_hys' : [0] * len(atoms),
                'atom_rdc' : [0] * len(atoms),
                }

    for reaction in reactions:
        reaction = check_ns_oxidation(reaction)
        reaction_atoms, reaction_type, _ = reaction[1:-1].split(';')
        reaction_type = type_collect[reaction_type]

        if reaction_type == 'Rearrangement':
            continue

        if reaction_type == 'spn':
            r_atom_idx = int(reaction_atoms.split(',')[0]) - 1
            labels['atom_som'][r_atom_idx] = 1
            labels[f'atom_{reaction_type}'][r_atom_idx] = 1
        
        for n in range(len(bonds)):
            s_atom_idx, e_atom_idx, s_atom, e_atom = bonds_idx[n][0], bonds_idx[n][1], bonds[n][0], bonds[n][1]
            is_react = check_reaction(s_atom_idx, e_atom_idx, s_atom, e_atom, reaction)
            
            if is_react:
                labels['atom_som'][s_atom_idx], labels['atom_som'][e_atom_idx] = 1, 1
                labels['bond_som'][n] = 1
                
                if f'atom_{reaction_type}' in labels:
                    labels[f'atom_{reaction_type}'][s_atom_idx], labels[f'atom_{reaction_type}'][e_atom_idx] = 1, 1
                if f'bond_{reaction_type}' in labels:
                    labels[f'bond_{reaction_type}'][n] = 1

       
    return labels

class CustomDataset(InMemoryDataset):
    def __init__(self, root='dataset_path', transform=None, pre_transform=None, df=None, args=None, cyp_list=[], mode='train'):
        self.df = df        
        self.mode = mode        
        self.cyp_list = cyp_list        
        self.args= args                
        self.from_save_pt = True        
        super().__init__(root, transform, pre_transform, df)        

    @property
    def raw_file_names(self):        
        return [f'raw_{i+1}.pt' for i in range(self.df.shape[0])]

    @property
    def processed_file_names(self):
        return [f'data_{i+1}.pt' for i in range(self.df.shape[0])]        

    def len(self):
        return self.df.shape[0]

    def get(self, idx):

        return self.make_graph(idx)
        

    def process(self):        
        pass

    def make_graph(self, idx):
        mol = self.df.loc[idx, 'ROMol']
        mol_H = AllChem.AddHs( mol, addCoords=True)

        atoms_h = [i.GetSymbol() for i in mol_H.GetAtoms()]
        bonds_idx_h = [(i.GetBeginAtomIdx(),i.GetEndAtomIdx()) for i in mol_H.GetBonds()]
        bonds_h = [(i.GetBeginAtom().GetSymbol(),i.GetEndAtom().GetSymbol()) for i in mol_H.GetBonds()]
        
        graph_h = mol2graph(mol_H)
        graph_h.x[:, 4] = torch.LongTensor([i.GetTotalNumHs() for i in mol.GetAtoms()] + [0] * (mol_H.GetNumAtoms() - mol.GetNumAtoms()) )
        
        eq_atoms, eq_bonds = get_equivalent_bonds(mol_H)
        
        graph_h.equivalent_atoms = eq_atoms
        graph_h.equivalent_bonds = eq_bonds
        
        graph_h.smile =  Chem.MolToSmiles(mol_H)
        graph_h.spn_atom = torch.BoolTensor([i in ['S', 'N'] for i in atoms_h ])
        
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

        return graph_h 