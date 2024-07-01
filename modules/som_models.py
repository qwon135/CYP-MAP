import argparse
import os.path as osp
from typing import Any, Dict, Optional
import numpy as np

import torch
from torch.nn import Sequential, Dropout, Linear
# from torch_geometric.nn import Linear
from typing import Any, Dict, Optional
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.nn.pool import TopKPooling
from .dualgraph.gnn import GNN2, GNN
from torch_geometric.utils import softmax
# from torch_scatter import scatter
from torch_geometric.nn import NNConv
from torch_geometric.utils import scatter

class Attention(torch.nn.Module):
    def __init__(self, channels, dropout, n_classes):
        super().__init__()
        self.attn = Sequential(
                                Linear(channels, channels, bias=False),
                                torch.nn.PReLU(init=0.05),
                                Dropout(dropout),
                                # torch.nn.Sigmoid(),
                                torch.nn.Tanh()
                                )
        # self.fc = Sequential(
        #                 Dropout(dropout),
        #                 Linear(channels, n_classes),
        #                 )
        self.fc = Sequential(
            torch.nn.LayerNorm(channels),
            Linear(channels, channels),
            torch.nn.BatchNorm1d(channels),
            # torch.nn.InstanceNorm1d(channels),
            torch.nn.PReLU(init=0.05),
            Dropout(dropout),
            Linear(channels, n_classes)
        )

    def forward(self, x, return_attn=False):        
        A = self.attn(x)
        mul_x = torch.mul(x, A)
        if return_attn:
            return mul_x, self.fc(mul_x)
        return self.fc(mul_x)

class SOMPredictor(torch.nn.Module):
    def __init__(self, channels, dropout, n_classes, cyp_list):
        super().__init__()
        self.cyp_list = cyp_list
        self.proj_edge = torch.nn.Sequential(
                            Linear(channels ,channels),
                            torch.nn.ReLU(),
                            Linear(channels ,channels)
                            )
        self.attn_fc = torch.nn.ModuleDict()
        for cyp in self.cyp_list:
            self.attn_fc[cyp] = Attention(channels, dropout, n_classes)

    def forward(self, x):
        x = self.proj_edge(x)
        logits = {}
        for cyp in self.cyp_list:
            logits[cyp] = self.attn_fc[cyp](x)
        return logits    

class SOMPredictorV2(torch.nn.Module):
    def __init__(self, channels, dropout_som, dropout_type, n_classes, cyp_list):
        super().__init__()
        self.cyp_list = cyp_list
        self.proj_edge = torch.nn.Sequential(
                            Linear(channels, channels),
                            torch.nn.PReLU(init=0.05),
                            Linear(channels, channels)
                            )
        self.reaction_head = torch.nn.ModuleDict()
        self.subtype_head = torch.nn.ModuleDict()
        for cyp in self.cyp_list:
            self.reaction_head[cyp] = Attention(channels, dropout_som, 1)  # 반응 여부 예측 (0: 반응 없음, 1: 반응 있음)
            self.subtype_head[cyp] = Attention(channels + 1, dropout_type, n_classes-1)  # subtype 예측 (1, 2, 3)

    def forward(self, x):
        x = self.proj_edge(x)
        logits = {}
        for cyp in self.cyp_list:
            reaction_logits = self.reaction_head[cyp](x)
            reaction_probs = reaction_logits.sigmoid()

            subtype_input = torch.cat([x, reaction_probs], dim=-1)
            subtype_logits = self.subtype_head[cyp](subtype_input)
            logits[cyp] = torch.cat([reaction_logits, subtype_logits], dim=-1)
        return logits   

class SOMPredictorV3(torch.nn.Module):
    def __init__(self, channels, dropout_som, dropout_type, n_classes, cyp_list):
        super().__init__()
        self.cyp_list = cyp_list
        self.proj_edge = torch.nn.Sequential(
                            Linear(channels, channels),
                            torch.nn.PReLU(init=0.05),
                            Linear(channels, channels)
                            )
        self.reaction_head = torch.nn.ModuleDict()
        self.subtype_head = torch.nn.ModuleDict()
        self.n_classes = n_classes
        for cyp in self.cyp_list:
            self.reaction_head[cyp] = Attention(channels, dropout_som, 1)  # 반응 여부 예측 (0: 반응 없음, 1: 반응 있음)
            self.subtype_head[cyp] = torch.nn.ModuleDict()
            for i in range(n_classes-1):
                self.subtype_head[cyp][str(i)] = Attention(channels + 1, dropout_type, 1)  # subtype 예측 (1, 2, 3)

    def forward(self, x):
        x = self.proj_edge(x)
        logits = {}
        for cyp in self.cyp_list:            
            reaction_logits = self.reaction_head[cyp](x)
            reaction_probs = reaction_logits.sigmoid()
            subtype_input = torch.cat([x, reaction_probs], dim=-1)

            logit = [reaction_logits]
            for i in range(self.n_classes-1):            
                subtype_logits = self.subtype_head[cyp][str(i)](subtype_input)
                logit.append(subtype_logits)

            logits[cyp] = torch.cat(logit, dim=-1)
        return logits   

class GNNSOM(torch.nn.Module):
    def __init__(self, 
                 channels = 512, num_layers = 2, gnn_num_layers=8,latent_size = 128, dropout=0.1, dropout_fc=0.1, dropout_som_fc=0.1, dropout_type_fc=0.1, n_classes=1, use_face=True, node_attn=True, face_attn=True,
                 encoder_dropout= 0.0, pooling='sum', 
                cyp_list= ['BOM_1A2', 'BOM_2A6', 'BOM_2B6', 'BOM_2C8', 'BOM_2C9', 'BOM_2C19', 'BOM_2D6', 'BOM_2E1', 'BOM_3A4'],                
                ):
        super().__init__()
        self.gnn = GNN2(
                        mlp_hidden_size = channels,
                        mlp_layers = num_layers,
                        num_message_passing_steps=gnn_num_layers,
                        latent_size = latent_size,
                        use_layer_norm = True,
                        use_bn=False,
                        use_face=use_face,
                        som_mode=True,
                        ddi=True,
                        dropedge_rate = dropout,
                        dropnode_rate = dropout,
                        dropout = dropout,
                        dropnet = dropout,
                        global_reducer = pooling,
                        node_reducer = pooling,
                        face_reducer = pooling,
                        graph_pooling = pooling,
                        node_attn = node_attn,
                        face_attn = face_attn,
                        encoder_dropout=encoder_dropout                        
                        )      

        self.bond_fc = torch.nn.ModuleDict() # Predict Reaction
        self.atom_fc = torch.nn.ModuleDict()
        self.cyp_list = cyp_list
        self.pooling_u=True
        if self.pooling_u:
            self.substrate_fc = torch.nn.Sequential(
                                            torch.nn.LayerNorm(latent_size * 4),
                                            Linear(latent_size * 4 , latent_size * 2),
                                            torch.nn.BatchNorm1d(latent_size *2),
                                            torch.nn.PReLU(init=0.05),
                                            torch.nn.Dropout(dropout_fc),

                                            torch.nn.LayerNorm(latent_size * 2),
                                            Linear(latent_size * 2 , latent_size),
                                            torch.nn.BatchNorm1d(latent_size),
                                            torch.nn.PReLU(init=0.05),
                                            torch.nn.Dropout(dropout_fc),

                                            Linear(latent_size , len(cyp_list))
                                            )
        else:
            self.substrate_fc = torch.nn.Sequential(
                                            # torch.nn.LayerNorm(latent_size),
                                            # Linear(latent_size , latent_size),
                                            # torch.nn.BatchNorm1d(latent_size),
                                            # torch.nn.PReLU(init=0.05),
                                            torch.nn.Dropout(dropout_fc),
                                            Linear(latent_size , len(cyp_list))
                                            )
        self.substrate_fc[-1].weight.data.normal_(mean=0.0, std=0.01)
        
        self.atom_tasks = ['atom_som', 'atom_spn', 'atom_dea', 'atom_epo', 'atom_oxi', 'atom_dha', 'atom_dhy', 'atom_rdc']
        self.bond_tasks = ['bond_som', 'bond_dea', 'bond_epo', 'bond_oxi', 'bond_dha', 'bond_dhy', 'bond_rdc']
        self.tasks = ['subs'] + self.atom_tasks + self.bond_tasks
        
        self.atom_fc = SOMPredictorV2(latent_size, dropout_som_fc, dropout_type_fc, len(self.atom_tasks), cyp_list)
        self.bond_fc = SOMPredictorV2(latent_size, dropout_som_fc, dropout_type_fc, len(self.bond_tasks), cyp_list)
                
    def forward(self, batch):
        mol_feat_atom, mol_feat_bond, mol_feat_ring, x, edge_attr, u = self.gnn(batch)

        # if self.pooling_edge:
        if self.pooling_u:
            mol_feature = torch.cat([mol_feat_atom, mol_feat_bond, mol_feat_ring, u], -1)
        else:
            mol_feature = u        

        edge_attr = edge_attr.view(edge_attr.shape[0] // 2, 2, edge_attr.shape[-1])
        edge_attr = edge_attr.sum(1)
        
        logits = {task : {} for task in self.tasks}

        substrate_logit = self.substrate_fc(mol_feature.detach())
        bond_logits = self.bond_fc(edge_attr)
        atom_logits = self.atom_fc(x)

        for cyp_idx, cyp in enumerate(self.cyp_list):
            logits['subs'][cyp] = substrate_logit[:, cyp_idx]

            for tsk_idx, tsk in enumerate(self.bond_tasks):
                logits[tsk][cyp] = bond_logits[cyp][:, tsk_idx]

            for tsk_idx, tsk in enumerate(self.atom_tasks):
                logits[tsk][cyp] = atom_logits[cyp][:, tsk_idx]
                
        return logits

    def forward_with_loss(self, batch, loss_fn_ce, loss_fn_bce, device, args):

        logits = self.forward(batch)

        loss_dict = {'total_loss' : 0, 'valid_loss' : 0}

        pred_dict = {}

        spn_atom = batch.spn_atom.to(device)
        has_H_atom = batch.has_H_atom.to(device)
        not_has_H_bond = batch.not_has_H_bond.to(device)
        bond_all = torch.ones_like(not_has_H_bond).bool().to(device)
        atom_all = torch.ones_like(spn_atom).bool().to(device)
        first_H_bond_idx = batch.first_H_bond_idx.to(device)
        bond_with_first_H = not_has_H_bond + first_H_bond_idx
        has_H_bond = ~(not_has_H_bond.to(device))
        pred_dict['spn_atom'] = spn_atom
        pred_dict['has_H_atom'] = has_H_atom
        pred_dict['not_has_H_bond'] = not_has_H_bond
        pred_dict['has_H_bond'] = has_H_bond

        for cyp in self.cyp_list:
            loss_dict[f'{cyp}_subs_loss'] = loss_fn_bce(logits['subs'][cyp], batch.y_substrate[cyp]) / batch.y_substrate[cyp].shape[0]
            loss_dict[f'{cyp}_subs_loss'] = loss_dict[f'{cyp}_subs_loss'] * args.substrate_loss_weight
            pred_dict[f'{cyp}_subs_logits'] = logits['subs'][cyp]
            pred_dict[f'{cyp}_subs_label'] = batch.y_substrate[cyp]

            loss_dict['total_loss'] += loss_dict[f'{cyp}_subs_loss']
            
            for tsk in self.tasks[1:]:
                if 'atom' in tsk:
                    loss_dict[f'{cyp}_{tsk}_loss'] = self.get_loss(loss_fn_bce, logits[tsk][cyp], batch.y[cyp][tsk], atom_all, args.atom_loss_weight, args.reduction)
                elif 'bond' in tsk:
                    loss_dict[f'{cyp}_{tsk}_loss'] = self.get_loss(loss_fn_bce, logits[tsk][cyp], batch.y[cyp][tsk], bond_all, args.bond_loss_weight, args.reduction)

                loss_dict['total_loss'] += loss_dict[f'{cyp}_{tsk}_loss']

                pred_dict[f'{cyp}_{tsk}_logits'] = logits[tsk][cyp]
                pred_dict[f'{cyp}_{tsk}_label'] = batch.y[cyp][tsk]

        loss_dict['total_loss'] =loss_dict['total_loss'] / len(self.cyp_list)
        return logits, loss_dict, pred_dict
    
    def get_loss(self, loss_fn, logits, labels, select_index, task_weight, reduction):
        if not select_index.cpu().sum():
            return 0
                
        loss = loss_fn(logits[select_index], labels[select_index])
        if reduction == 'sum':
            loss = loss / select_index.sum()
                
        return loss * task_weight