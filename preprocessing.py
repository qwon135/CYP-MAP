import os
from sklearn.model_selection import train_test_split
import pickle
import lmdb
import pandas as pd
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')  
import warnings
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
from dgllife.data.uspto import get_bond_changes, get_pair_label, load_one_reaction, process_file, default_atom_pair_featurizer, default_atom_pair_featurizer

def process_reaction(reaction):
    bond_changes = get_bond_changes(reaction)
    formatted_reaction = '{} {}\n'.format(
        reaction, ';'.join(['{}-{}-{}'.format(x[0], x[1], x[2]) for x in bond_changes]))
    return formatted_reaction

def processing(reaction):
    mol, reaction, graph_edits =  load_one_reaction(process_reaction(reaction))
    
    atom_pair_label = get_pair_label(mol, graph_edits)
    return mol, reaction, atom_pair_label, graph_edits

def to_matrix(y_true, n_classes):
    y_true = np.array(y_true)
    y_true_ = np.zeros([y_true.shape[0], n_classes])

    for n, i in enumerate(y_true):
        y_true_[n][i] = 1
    return y_true_
def smi2_2Dcoords(mol):    
    mol = AllChem.AddHs(mol)
    AllChem.Compute2DCoords(mol)
    coordinates = mol.GetConformer().GetPositions().astype(np.float32)
    len(mol.GetAtoms()) == len(coordinates), "2D coordinates shape is not align with this"
    return coordinates


def smi2_3Dcoords(mol, cnt):    
    mol = AllChem.AddHs(mol)
    coordinate_list=[]
    for seed in range(cnt):
        try:
            res = AllChem.EmbedMolecule(mol, randomSeed=seed)  # will random generate conformer with seed equal to -1. else fixed random seed.
            if res == 0:
                try:
                    AllChem.MMFFOptimizeMolecule(mol)       # some conformer can not use MMFF optimize
                    coordinates = mol.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(mol)
                    
            elif res == -1:
                mol_tmp = Chem.MolFromSmiles(mol)
                AllChem.EmbedMolecule(mol_tmp, maxAttempts=5000, randomSeed=seed)
                mol_tmp = AllChem.AddHs(mol_tmp, addCoords=True)
                try:
                    AllChem.MMFFOptimizeMolecule(mol_tmp)       # some conformer can not use MMFF optimize
                    coordinates = mol_tmp.GetConformer().GetPositions()
                except:
                    print("Failed to generate 3D, replace with 2D")
                    coordinates = smi2_2Dcoords(mol) 
        except:
            print("Failed to generate 3D, replace with 2D")
            coordinates = smi2_2Dcoords(mol) 
        coordinates = smi2_2Dcoords(mol) 
        assert len(mol.GetAtoms()) == len(coordinates), "3D coordinates shape is not align with {}".format(mol)
        coordinate_list.append(coordinates.astype(np.float32))
    return coordinate_list


def inner_smi2coords(mol, reaction_smi, target, only_2d):    
    cnt = 10 # conformer num,all==11, 10 3d + 1 2d    
    if len(mol.GetAtoms()) > 400:
        coordinate_list =  [smi2_2Dcoords(mol)] * (cnt+1)
        print("atom num >400,use 2D coords")
    else:
        if only_2d:
            coordinate_list =  [smi2_2Dcoords(mol)] * (cnt+1)
        else:
            coordinate_list = smi2_3Dcoords(mol,cnt)
            coordinate_list.append(smi2_2Dcoords(mol).astype(np.float32))
    mol = AllChem.AddHs( mol, addCoords=True)
    atoms = [atom.GetSymbol() for atom in mol.GetAtoms()]  # after add H 
    return pickle.dumps({
        'atoms': atoms, 
        'coordinates': coordinate_list, 
        'mol':mol,'smi': reaction_smi, 
        'target': target}, protocol=-1)


def smi2coords(content, only_2d=False):    
    mol, reaction_smi, atom_pair_label, graph_edits = processing(content)
    try:
        return inner_smi2coords(mol, reaction_smi.split('>')[0], atom_pair_label, only_2d)
    except:
        print("failed")
        return None

def reaction_type2label_type1(x):
    # 0. Non-reaction
    # 1. Cleavage
    # 2. Hydroxylation
    # 3. Oxidation
    # 4. Others
    reaction_type2label_dict = {
            'cleavage' : 1,
            'dehydrogenation' : 4,
            'denitrosation' : 4,
            'desulfation' : 4,
            'desulfuration' : 4,
            'desulphuration' : 4,
            'epoxidation' : 4,
            'hdroxylation' : 2,
            'hydrlxylation' : 2,
            'hydroxyaltion' : 2,
            'hydroxylation' : 2,
            'hydroxylaton' : 2,
            'n-oxidation' : 4,
            'oxdation' : 3,
            'oxidaiton' : 3,
            'oxidation' : 3,
            'oxidiation' : 3,
            'oxydation' : 3,
            'rearrangement' : 4,
            'redcution' : 4,
            'reduction' : 4,
            'reducttion' : 4,
            'redunction' : 4,
            'reuction' : 4,
            'ruction' : 4,
            's-oxidation' : 4,
            'unspoxidation' : 4,
            'unspreduction' : 4}
    return reaction_type2label_dict[x]

def reaction_type2label_type2(x):
    # 0. Non-reaction
    # 1. Cleavage
    # 2. Hydroxylation | Oxidation
    # 3. Others
    reaction_type2label_dict = {
            'cleavage' : 1,
            'dehydrogenation' : 3,
            'denitrosation' : 3,
            'desulfation' : 3,
            'desulfuration' : 3,
            'desulphuration' : 3,
            'epoxidation' : 3,
            'hdroxylation' : 2,
            'hydrlxylation' : 2,
            'hydroxyaltion' : 2,
            'hydroxylation' : 2,
            'hydroxylaton' : 2,
            'n-oxidation' : 3,
            'oxdation' : 2,
            'oxidaiton' : 2,
            'oxidation' : 2,
            'oxidiation' : 2,
            'oxydation' : 2,
            'rearrangement' : 3,
            'redcution' : 3,
            'reduction' : 3,
            'reducttion' : 3,
            'redunction' : 3,
            'reuction' : 3,
            'ruction' : 3,
            's-oxidation' : 3,
            'unspoxidation' : 3,
            'unspreduction' : 3}
    
    return reaction_type2label_dict[x]
def check_reaction(s_atom_idx, e_atom_idx, s_atom, e_atom, reaction, class_type = 1):
    if not reaction:
        return 0
    rea_idx, reaction_type, _ = reaction[1:-1].split(';')
    rea_sidx, rea_eidx = rea_idx.split(',')                        
                
    rea_sidx = int(rea_sidx)
    if rea_eidx.isalpha(): # 글자면            
        if (rea_sidx-1 == s_atom_idx) and (rea_eidx == e_atom):            
            if class_type  == 1:
                return reaction_type2label_type1(reaction_type.lower())
            elif class_type == 2:
                return reaction_type2label_type2(reaction_type.lower())
    else:
        rea_eidx = int(rea_eidx)
        if (rea_sidx-1 == s_atom_idx) and (rea_eidx+1 == e_atom):
            if class_type  == 1:
                return reaction_type2label_type1(reaction_type.lower())
            elif class_type == 2:
                return reaction_type2label_type2(reaction_type.lower())

    return 0

def mol2bond_label(bonds, bonds_idx, reactions, class_type):
    
    bond_label = [0] * len(bonds)
    
    for reaction in reactions:
        for n in range(len(bonds)):
            s_atom_idx, e_atom_idx, s_atom, e_atom = bonds_idx[n][0], bonds_idx[n][1], bonds[n][0], bonds[n][1]
            
            is_react = check_reaction(s_atom_idx, e_atom_idx, s_atom, e_atom, reaction, class_type)
            bond_label[n] = is_react
    return bond_label

def predict_som(model, batch, token_index):
    src_tokens = batch['net_input']['src_tokens']
    src_distance = batch['net_input']['src_distance']
    src_coord = batch['net_input']['src_coord']
    src_edge_type = batch['net_input']['src_edge_type']

    padding_mask = src_tokens.eq(0)
    if not padding_mask.any():
        padding_mask = None

    x = model.embed_tokens(src_tokens)
    
    def get_dist_features(dist, et):
            n_node = dist.size(-1)
            gbf_feature = model.gbf(dist, et)
            gbf_result = model.gbf_proj(gbf_feature)
            graph_attn_bias = gbf_result
            graph_attn_bias = graph_attn_bias.permute(0, 3, 1, 2).contiguous()
            graph_attn_bias = graph_attn_bias.view(-1, n_node, n_node)
            return graph_attn_bias

    graph_attn_bias = get_dist_features(src_distance, src_edge_type)
    
    (encoder_rep,
    encoder_pair_rep,
    delta_encoder_pair_rep,
    x_norm,
    delta_encoder_pair_rep_norm
                            ) = model.encoder(x, padding_mask=padding_mask, attn_mask=graph_attn_bias) # encoder output -> x, attn_mask, delta_pair_repr, x_norm, delta_pair_repr_norm
    encoder_pair_rep[encoder_pair_rep == float("-inf")] = 0    
    
    mol = Chem.MolFromSmiles(batch['target']['smi_name'][0])
    mol = AllChem.AddHs( mol, addCoords=True)
    bonds_idx = [(i.GetBeginAtomIdx(),i.GetEndAtomIdx()) for i in mol.GetBonds()]

    delta_encoder_pair_rep = delta_encoder_pair_rep[:, :, token_index, :][:, token_index, :, :]    
    logits = model.classification_heads['som'](delta_encoder_pair_rep[0][np.array(bonds_idx).T])
    return logits, padding_mask

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional

def label_to_one_hot_label(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
    eps: float = 1e-6,
    ignore_index=255,
) -> torch.Tensor:
    shape = labels.shape    
    one_hot = torch.zeros((shape[0], ignore_index+1) + shape[1:], device=device, dtype=dtype)
                
    one_hot = one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps
        
    ret = torch.split(one_hot, [num_classes, ignore_index+1-num_classes], dim=1)[0]
    
    return ret


def focal_loss(input, target, alpha, gamma, reduction, eps, ignore_index):
    if not isinstance(input, torch.Tensor):
        raise TypeError(f"Input type is not a torch.Tensor. Got {type(input)}")

    if not len(input.shape) >= 2:
        raise ValueError(f"Invalid input shape, we expect BxCx*. Got: {input.shape}")

    if input.size(0) != target.size(0):
        raise ValueError(f'Expected input batch_size ({input.size(0)}) to match target batch_size ({target.size(0)}).')
    
    n = input.size(0) 
        
    out_size = (n,) + input.size()[2:]
            
    if target.size()[1:] != input.size()[2:]:
        raise ValueError(f'Expected target size {out_size}, got {target.size()}')

    if not input.device == target.device:
        raise ValueError(f"input and target must be in the same device. Got: {input.device} and {target.device}")
    
    if isinstance(alpha, float):
        pass
    elif isinstance(alpha, np.ndarray):
        alpha = torch.from_numpy(alpha)        
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)
    elif isinstance(alpha, torch.Tensor):        
        alpha = alpha.view(-1, len(alpha), 1, 1).expand_as(input)       
        
        
    input_soft = F.softmax(input, dim=1) + eps
            
    target_one_hot = label_to_one_hot_label(target.long(), num_classes=input.shape[1], device=input.device, dtype=input.dtype, ignore_index=ignore_index)
    
    weight = torch.pow(1.0 - input_soft, gamma)
            
    focal = -alpha * weight * torch.log(input_soft)
        
    loss_tmp = torch.sum(target_one_hot * focal, dim=1)

    if reduction == 'none':        
        loss = loss_tmp
    elif reduction == 'mean':        
        loss = torch.mean(loss_tmp)
    elif reduction == 'sum':        
        loss = torch.sum(loss_tmp)
    else:
        raise NotImplementedError(f"Invalid reduction mode: {reduction}")
    return loss

class FocalLoss(nn.Module):

    def __init__(self, alpha, gamma = 2.0, reduction = 'mean', eps = 1e-8, ignore_index=30):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.eps = eps
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return focal_loss(input, target, self.alpha, self.gamma, self.reduction, self.eps, self.ignore_index)

