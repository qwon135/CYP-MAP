from dualgraph.mol import smiles2graphwithface
from dualgraph.dataset import DGData
import torch, random
import numpy as np
import torch_geometric.transforms as T
from rdkit.Chem import AllChem
from rdkit import Chem
from torch_geometric.data import InMemoryDataset
from dualgraph.dataset import DGData
from add_equevalent import get_equivalent_bonds
from torch_geometric.utils import to_dense_batch, dense_to_sparse
from torch_geometric.utils import shuffle_node, mask_feature, dropout_node, dropout_edge
from torch_geometric.transforms import RandomNodeSplit
from torch_geometric.utils.subgraph import subgraph


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
    bond_nn_oxidation = [0] * len(bonds) # Reaction?
    bond_cleavage = [0] * len(bonds) # Reaction?
    bond_reduction = [0] * len(bonds)

    bond_reaction_type = [''] * len(bonds)
        
    # spn_atoms_idx = [idx for idx, i in enumerate(atoms) if i in ['S', 'P', 'N']]
    atom_label = [0] * len(atoms)
    atom_spn = [0] * len(atoms)
    atom_hydroxylation = [0] * len(atoms)
    atom_nh_oxidation = [0] * len(atoms)
    atom_oxidation  = [0] * len(atoms)

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
                
                    atom_nh_oxidation[s_atom_idx] = 1
                    atom_nh_oxidation[e_atom_idx] = 1
                
            if is_react:
                bond_label[n] = 1                
                bond_reaction_type[n] = reaction_type
                if type_collect[reaction_type] == 'Oxidation':
                    bond_nn_oxidation[n] = 1
                elif type_collect[reaction_type] == 'Cleavage':
                    bond_cleavage[n] = 1
                if type_collect[reaction_type] == 'Reduction':
                    bond_reduction[n] = 1
    if return_type:
        return bond_label, bond_nn_oxidation, bond_cleavage, atom_label, atom_spn, atom_hydroxylation, atom_nh_oxidation, atom_oxidation, atom_reaction_type, bond_reaction_type
    return bond_label, bond_nn_oxidation, bond_cleavage, bond_reduction, atom_label, atom_spn, atom_hydroxylation, atom_nh_oxidation, atom_oxidation
    


def get_class_weight(train_df, class_type, n_classes, cyp):
    class_weight = list(range(n_classes))
    for i in range(train_df['ROMol'].shape[0]):
        mol = train_df.loc[i, 'ROMol']
        bonds_idx = [(i.GetBeginAtomIdx(),i.GetEndAtomIdx()) for i in mol.GetBonds()]
        bonds = [(i.GetBeginAtom().GetSymbol(),i.GetEndAtom().GetSymbol()) for i in mol.GetBonds()]    
        
        reactions = train_df.loc[i, cyp].split('\n')
        bond_label = mol2bond_label(bonds, bonds_idx, reactions, class_type)
        for label in bond_label:
            class_weight[label] += 1
    
    class_weight = sum(class_weight) / torch.Tensor(class_weight)
    return class_weight

from copy import deepcopy

def drop_node_edge(batch_org, p, cyp_list):    
    batch = deepcopy(batch_org)
    prob = torch.rand(batch.num_nodes, device=batch.edge_index.device)
    node_mask = prob > p    
    node_mask[batch.y_atom['CYP_REACTION'].bool()] = True

    edge_index, _, edge_mask = subgraph(node_mask, batch.edge_index,
                                        num_nodes=batch.num_nodes,
                                        return_edge_mask=True)
    
    node_mask_idx = torch.zeros(node_mask.shape[0]).long()
    node_mask_idx[node_mask] = torch.arange(node_mask.sum().item())

    edge_mask_idx = torch.zeros(edge_mask.shape[0]).long()
    edge_mask_idx[edge_mask] = torch.arange(edge_mask.sum().item())
    
    batch.edge_index = node_mask_idx[edge_index]

    ring_index = batch.ring_index[:, edge_mask]
    # ring_index = edge_mask_idx[ring_index]

    batch.ring_index = ring_index
    batch.edge_attr = batch.edge_attr[edge_mask]
    
    batch.n_edges = edge_mask.sum().item()
    batch.n_nodes = node_mask.sum().item()

    nf_node_mask = ~torch.isin(batch.nf_node, torch.where(~node_mask)[0])[0]

    for ridx in range(batch.num_rings):        
        ring_node_mask = nf_node_mask[batch.nf_ring[0] == ridx+1]
        if not ring_node_mask.all():
            batch.ring_mask[ridx] = False

    batch.nf_node = batch.nf_node[:, nf_node_mask]
    batch.nf_node = node_mask_idx[batch.nf_node]
    batch.nf_ring = batch.nf_ring[:, nf_node_mask]

    batch.n_nfs = batch.nf_node.size(1)
    
    edge_mask = edge_mask.view(edge_mask.shape[0] // 2, 2)
    edge_mask = edge_mask[:, 0]

    batch.x = batch.x[node_mask]
    batch.spn_atom = batch.spn_atom[node_mask]
    batch.has_H_atom = batch.has_H_atom[node_mask]
    batch.not_has_H_bond = batch.not_has_H_bond[edge_mask]
        
    for cyp in cyp_list:
        batch.y_spn[cyp] = batch.y_spn[cyp][node_mask]
        batch.y_atom[cyp] = batch.y_atom[cyp][node_mask]
        batch.y_hydroxylation[cyp] = batch.y_hydroxylation[cyp][node_mask]
        batch.y_nh_oxidation[cyp] = batch.y_nh_oxidation[cyp][node_mask]
        
        batch.y[cyp] = batch.y[cyp][edge_mask]
        batch.y_cleavage[cyp] = batch.y_cleavage[cyp][edge_mask]
        batch.y_nn_oxidation[cyp] = batch.y_nn_oxidation[cyp][edge_mask]

    return batch

def shuffle_graph(graph_org, cyp_list):    
    graph = deepcopy(graph_org)

    perm = torch.randperm(graph.x.shape[0])

    graph.x = graph.x[perm]
    graph.spn_atom = graph.spn_atom[perm]
    graph.has_H_atom = graph.has_H_atom[perm]
    graph.edge_index = perm[graph.edge_index]
    graph.nf_node = perm[graph.nf_node]

    for cyp in cyp_list:
        graph.y_spn[cyp] = graph.y_spn[cyp][perm]
        graph.y_atom[cyp] = graph.y_atom[cyp][perm]
        graph.y_hydroxylation[cyp] = graph.y_hydroxylation[cyp][perm]
        graph.y_nh_oxidation[cyp] = graph.y_nh_oxidation[cyp][perm]
    
    return graph

def node_mask(graph_org, p):
    graph = deepcopy(graph_org)
    mask_value = torch.LongTensor([[118, 4, 11, 11, 9, 5, 5, 1, 1]])
    mask_value = torch.repeat_interleave(mask_value, graph.x.shape[0], dim=0)

    _, mask = mask_feature(graph.x.float(), p=p, mode='all')
    
    mask[:, [0, 2, 4, 6, 7]] = True
    mask[graph.y_atom['CYP_REACTION'].bool(), :] = True    
    graph.x[~mask] = mask_value[~mask]
    return graph

class CustomDataset(InMemoryDataset):
    def __init__(self, root='dataset_path', transform=None, pre_transform=None, df=None, args=None, class_type=3, add_H=False, cyp_list=[], mode='train', position_dir = 'positions'):
        self.df = df        
        self.mode = mode
        self.class_type = class_type
        self.cyp_list = cyp_list
        self.pe_transform = T.AddRandomWalkPE(walk_length=20, attr_name='pe')
        self.args= args
        self.add_H = add_H
        self.position_dir = position_dir
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
        return self.graph_list[idx], self.graph_h_list[idx]
        if (self.add_H == 'random') and (self.mode=='train'): 
            if random.random() > 0.5:
                graph = self.graph_h_list[idx]
            else:    
                graph = self.graph_list[idx]
        elif self.add_H:
            graph = self.graph_h_list[idx]
        else:
            graph = self.graph_list[idx]
        
        # if self.mode=='train':
        #     if random.random() < self.args.drop_node_p:
        #         graph = drop_node_edge(graph, p=0.05, cyp_list=self.cyp_list)
                
        #     if random.random() < self.args.mask_node_p:
        #         graph = node_mask(graph, 0.05)
        #     # if random.random() < 0.3:
        #     #     graph = shuffle_graph(graph, self.cyp_list)
        return graph

    def process(self):
        if self.from_save_pt:
            self.graph_list, self.graph_h_list = [], []
            for mid in self.df['POS_ID'].tolist():                
                self.graph_h_list.append(torch.load(f'graph_pt/{mid}_addh.pt'))                
                self.graph_list.append(torch.load(f'graph_pt/{mid}.pt'))
            return 
        
        self.labels, self.label_substrate, self.atom_labels = {}, {}, {}
        self.bond_nn_oxidation, self.bond_cleavage = {}, {}
        self.atom_spn, self.atom_hydroxylation, self.atom_nh_oxidation = {}, {}, {}

        for cyp in self.cyp_list:
            self.labels[cyp] = []
            self.label_substrate[cyp] = []
            self.atom_labels[cyp] = []
            self.bond_nn_oxidation[cyp] = []
            self.bond_cleavage[cyp] = []
            self.atom_spn[cyp] = []
            self.atom_hydroxylation[cyp] = []
            self.atom_nh_oxidation[cyp] = []

        self.atoms, self.atoms_h = [], []
        self.graph_list, self.graph_h_list = [], []         

        for i in range(self.df.shape[0]):
            pos_id = self.df.loc[i, 'POS_ID']

            mol = self.df.loc[i, 'ROMol']
            atoms = [i.GetSymbol() for i in mol.GetAtoms()]
            bonds_idx = [(i.GetBeginAtomIdx(),i.GetEndAtomIdx()) for i in mol.GetBonds()]
            bonds = [(i.GetBeginAtom().GetSymbol(),i.GetEndAtom().GetSymbol()) for i in mol.GetBonds()]

            mol_H = AllChem.AddHs( mol, addCoords=True)
            atoms_h = [i.GetSymbol() for i in mol_H.GetAtoms()]
            bonds_idx_h = [(i.GetBeginAtomIdx(),i.GetEndAtomIdx()) for i in mol_H.GetBonds()]
            bonds_h = [(i.GetBeginAtom().GetSymbol(),i.GetEndAtom().GetSymbol()) for i in mol_H.GetBonds()]

            self.atoms.append(atoms)
            self.atoms_h.append(atoms_h)

            graph = mol2graph(mol)
            graph.equivalent_bonds = get_equivalent_bonds(mol)
            graph.pe = None
            graph.smile =  Chem.MolToSmiles(mol)
            graph.spn_atom = torch.BoolTensor([i in ['S', 'N'] for i in atoms ])
            graph.has_H_atom =  torch.BoolTensor([i.GetTotalNumHs() for i in mol.GetAtoms()])
            graph.not_has_H_bond = torch.BoolTensor([('H'not in i) for i in bonds])

            graph_h = mol2graph(mol_H)
            graph_h.equivalent_bonds = get_equivalent_bonds(mol_H)
            graph_h.pe = None
            graph_h.smile =  Chem.MolToSmiles(mol_H)
            graph_h.spn_atom = torch.BoolTensor([i in ['S', 'N'] for i in atoms_h ])
            graph_h.has_H_atom = torch.BoolTensor(graph.has_H_atom.tolist() + [False] * (len(atoms_h) - len(atoms)))
            graph_h.not_has_H_bond = torch.BoolTensor([('H' not in i) for i in bonds_h])

            for cyp in self.cyp_list:                
                reactions = self.df.loc[i, cyp].split('\n')
                reactions = [check_ns_oxidation(i) for i in reactions]
                if self.add_H:
                    bond_label, bond_nn_oxidation, bond_cleavage, atom_label, atom_spn, atom_hydroxylation, atom_nh_oxidation = mol2bond_label(atoms=atoms_h, bonds=bonds_h, bonds_idx=bonds_idx_h, reactions=reactions)
                else:
                    bond_label, bond_nn_oxidation, bond_cleavage, atom_label, atom_spn, atom_hydroxylation, atom_nh_oxidation = mol2bond_label(atoms=atoms, bonds=bonds, bonds_idx=bonds_idx, reactions=reactions)
                
                self.labels[cyp].append(bond_label)
                self.atom_labels[cyp].append(atom_label)
                self.label_substrate[cyp].append([1 if self.df.loc[i, cyp] != '' else 0])
                self.bond_nn_oxidation[cyp].append(bond_nn_oxidation)
                self.bond_cleavage[cyp].append(bond_cleavage)

                self.atom_spn[cyp].append(atom_spn)
                self.atom_hydroxylation[cyp].append(atom_hydroxylation)
                self.atom_nh_oxidation[cyp].append(atom_nh_oxidation)
            
            graph.atoms = self.atoms[i]
            graph_h.atoms = self.atoms_h[i]            

            y, y_substrate, y_atom = {}, {}, {}
            y_nn_oxidation, y_cleavage = {}, {}
            y_spn, y_hydroxylation, y_nh_oxidation = {}, {}, {}

            for cyp in self.cyp_list:
                y[cyp] = torch.FloatTensor(self.labels[cyp][i])
                y_substrate[cyp] = torch.FloatTensor(self.label_substrate[cyp][i])
                y_atom[cyp] = torch.FloatTensor(self.atom_labels[cyp][i])
                y_nn_oxidation[cyp] = torch.FloatTensor(self.bond_nn_oxidation[cyp][i])

                y_cleavage[cyp] = torch.FloatTensor(self.bond_cleavage[cyp][i])
                y_spn[cyp] = torch.FloatTensor(self.atom_spn[cyp][i])
                y_hydroxylation[cyp] = torch.FloatTensor(self.atom_hydroxylation[cyp][i])
                y_nh_oxidation[cyp] = torch.FloatTensor(self.atom_nh_oxidation[cyp][i])
            
            if self.add_H:    

                graph_h.y = y
                graph_h.y_substrate = y_substrate
                graph_h.y_atom = y_atom
                
                graph_h.y_nn_oxidation = y_nn_oxidation
                graph_h.y_cleavage = y_cleavage
                
                graph_h.y_spn = y_spn
                graph_h.y_hydroxylation = y_hydroxylation
                graph_h.y_nh_oxidation = y_nh_oxidation

                graph_h.mid = self.df.loc[i, 'POS_ID']        
            else:                           
                graph.y = y
                graph.y_substrate = y_substrate
                graph.y_atom = y_atom
                
                graph.y_nn_oxidation = y_nn_oxidation
                graph.y_cleavage = y_cleavage
                
                graph.y_spn = y_spn
                graph.y_hydroxylation = y_hydroxylation
                graph.y_nh_oxidation = y_nh_oxidation

                graph.mid = self.df.loc[i, 'POS_ID']        

            self.graph_list.append(graph)
            self.graph_h_list.append(graph_h)
