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

    if reaction_type in ['S-Oxidation', 'N-Oxidation', 'P-Oxidation']:
        return 0

    # 둘다 숫자인 경우는 해당 atom 인덱스 일치 여부 확인
    if reaction_atom_e == 'H': # Hydroxylation or Oxidation
        if ((reaction_atom_s == bond_atom_sidx) and (bond_atom_e_symbol == 'H')) or ((reaction_atom_s == bond_atom_eidx) and (bond_atom_s_symbol ==  'H')): 
            return 1
    else:
        reaction_atom_e = int(reaction_atom_e)
        if ((bond_atom_sidx == reaction_atom_s) and (bond_atom_eidx == reaction_atom_e)) or ((bond_atom_sidx == reaction_atom_e) and (bond_atom_eidx == reaction_atom_s)):                    
            return 1        
    return 0


  
def mol2bond_label(atoms, bonds, bonds_idx, reactions, return_type=False):
    type_collect = {
                '' : '',
                'Cleavage' : 'Cleavage', 
                'Hydroxylation' : 'Hydroxylation',

                'Oxidiation' : 'Oxidation',
                'Oxidation' : 'Oxidation',
                'Epoxidation' : 'Oxidation',
                'UnspOxidation' : 'Oxidation',

                'Rearrangement' : 'Rearrangement', 
                'Reduction' : 'Reduction', 
                'Dehydrogenation' : 'Dehydrogenation',
                'Desulfuration' : 'Desulfuration',
                'Denitrosation' : 'Denitrosation',
                'UnspReduction' : 'Reduction',

                'N-Oxidation': 'SPN-Oxidation',
                'S-Oxidation': 'SPN-Oxidation'}
    
    reactions = [i for i in list(set(reactions)) if i]
   
    bond_label = [0] * len(bonds) # Reaction?    
    
    bond_cleavage = [0] * len(bonds) # Reaction?
    bond_reduction = [0] * len(bonds)
    bond_hydroxylation = [0] * len(bonds)
    bond_oxidation = [0] * len(bonds)

    bond_reaction_type = [''] * len(bonds)
        
    # spn_atoms_idx = [idx for idx, i in enumerate(atoms) if i in ['S', 'P', 'N']]
    atom_label = [0] * len(atoms)
    atom_spn = [0] * len(atoms)
    atom_hydroxylation = [0] * len(atoms)
    atom_oxidation  = [0] * len(atoms)
    atom_cleavage = [0] * len(atoms)
    atom_reduction = [0] * len(atoms)

    atom_reaction_type = [''] * len(atoms)

    for reaction in reactions:
        reaction = check_ns_oxidation(reaction)
        reaction_atoms, reaction_type, _ = reaction[1:-1].split(';')

        if reaction_type == 'S-Oxidation':
            r_atom_idx = int(reaction_atoms.split(',')[0]) - 1
            atom_label[r_atom_idx] = 1
            atom_spn[r_atom_idx] = 1     
            atom_oxidation[r_atom_idx] = 1
            atom_reaction_type[r_atom_idx] = 'S-Oxidation'

        if reaction_type == 'N-Oxidation': # 'P-Oxidation'
            r_atom_idx = int(reaction_atoms.split(',')[0]) - 1
            atom_label[r_atom_idx] = 1
            atom_spn[r_atom_idx] = 1
            atom_oxidation[r_atom_idx] = 1
            atom_reaction_type[r_atom_idx] = 'N-Oxidation'

        # make bond label
        for n in range(len(bonds)):
            s_atom_idx, e_atom_idx, s_atom, e_atom = bonds_idx[n][0], bonds_idx[n][1], bonds[n][0], bonds[n][1]
            is_react = check_reaction(s_atom_idx, e_atom_idx, s_atom, e_atom, reaction)
            
            if is_react:
                atom_label[s_atom_idx] = 1
                atom_label[e_atom_idx] = 1
                atom_reaction_type[s_atom_idx] = reaction_type
                atom_reaction_type[e_atom_idx] = reaction_type
                if type_collect[reaction_type] == 'Hydroxylation':
                    atom_hydroxylation[s_atom_idx] = 1
                    atom_hydroxylation[e_atom_idx] = 1                
                
                if (type_collect[reaction_type] == 'Oxidation'):
                    atom_oxidation[s_atom_idx] = 1
                    atom_oxidation[e_atom_idx] = 1

                if (type_collect[reaction_type] == 'Oxidation') and ('H' in reaction_atoms) : # n-H인 Oxidaiton은 Hydroxylation으로 가정
                    atom_hydroxylation[s_atom_idx] = 1
                    atom_hydroxylation[e_atom_idx] = 1                                

                if type_collect[reaction_type] == 'Cleavage':
                    atom_cleavage[s_atom_idx] = 1
                    atom_cleavage[e_atom_idx] = 1

                if type_collect[reaction_type] == 'Reduction':
                    atom_reduction[s_atom_idx] = 1
                    atom_reduction[e_atom_idx] = 1

            if is_react:
                bond_label[n] = 1                
                bond_reaction_type[n] = reaction_type
                if type_collect[reaction_type] == 'Oxidation':
                    bond_oxidation[n] = 1

                if type_collect[reaction_type] == 'Cleavage':
                    bond_cleavage[n] = 1

                if type_collect[reaction_type] == 'Hydroxylation':
                    bond_hydroxylation[n] = 1

                if (type_collect[reaction_type] == 'Oxidation') and ('H' in reaction_atoms) :
                    bond_hydroxylation[n] = 1                    

                if type_collect[reaction_type] == 'Reduction':
                    bond_reduction[n] = 1
    
    return (
            bond_label,
            bond_cleavage,
            bond_reduction,
            bond_hydroxylation,
            bond_oxidation,
            atom_label,
            atom_cleavage,
            atom_reduction,
            atom_hydroxylation,
            atom_oxidation,
            atom_spn
            )    

class CustomDataset(InMemoryDataset):
    def __init__(self, root='dataset_path', transform=None, pre_transform=None, df=None, args=None, cyp_list=[], mode='train'):
        self.df = df        
        self.mode = mode        
        self.cyp_list = cyp_list        
        self.args= args                
        self.from_save_pt = True
        self.tasks = ['subs', 'bond', 'atom',  'spn', 'H', 'clv', 'nh_oxi', 'nn_oxi']

        self.type_collect = {
                            '' : '',
                            'Cleavage' : 'Cleavage', 
                            'Hydroxylation' : 'Hydroxylation',

                            'Oxidiation' : 'Oxidation',
                            'Oxidation' : 'Oxidation',
                            'Epoxidation' : 'Oxidation',
                            'UnspOxidation' : 'Oxidation',

                            'Rearrangement' : 'Rearrangement', 
                            'Reduction' : 'Reduction', 
                            'Dehydrogenation' : 'Dehydrogenation',
                            'Desulfuration' : 'Desulfuration',
                            'Denitrosation' : 'Denitrosation',
                            'UnspReduction' : 'UnspReduction',

                            'N-Oxidation': 'SPN-Oxidation',
                            'S-Oxidation': 'SPN-Oxidation',
                            }    
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
        if self.mode == 'train' and random.random() > 0.7:
            graph_h = deepcopy(self.graph_h_list[idx])
            x = graph_h.x[:, 4]
            x = x-1
            x[x<0]=0
            graph_h.x[:,4] = x

            # graph_h = deepcopy(self.graph_h_list[idx])
            # x = graph_h.x[:, 4]
            # x = x+1
            # x[x>8]=8
            # graph_h.x[:,4] = x
            # return graph_h
        
        return self.graph_h_list[idx]
        

    def process(self):
        if self.from_save_pt:
            self.graph_h_list = []
            for mid in self.df['POS_ID'].tolist():                
                self.graph_h_list.append(torch.load(f'graph_pt/{mid}_addh.pt'))                
                
            return
