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
        self.fc = Sequential(                        
                        Dropout(dropout),
                        Linear(channels, n_classes),
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
    def __init__(self, channels, dropout, n_classes, cyp_list):
        super().__init__()
        self.cyp_list = cyp_list
        self.proj_edge = torch.nn.Sequential(
                            Linear(channels, channels),
                            torch.nn.ReLU(),
                            Linear(channels, channels)
                            )
        self.reaction_head = torch.nn.ModuleDict()
        self.subtype_head = torch.nn.ModuleDict()
        for cyp in self.cyp_list:
            self.reaction_head[cyp] = Attention(channels, dropout, 1)  # 반응 여부 예측 (0: 반응 없음, 1: 반응 있음)
            self.subtype_head[cyp] = Attention(channels + 1, dropout, n_classes-1)  # subtype 예측 (1, 2, 3)

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

class SOMPredictorV5(torch.nn.Module):
    def __init__(self, channels, dropout, n_classes, cyp_list):
        super().__init__()
        self.cyp_list = cyp_list

        self.proj_edge =  torch.nn.ModuleDict()        
        self.reaction_predictor = torch.nn.ModuleDict()

        for cyp in cyp_list:
            self.proj_edge[cyp] = torch.nn.Sequential(
                            Linear(channels, channels),
                            torch.nn.ReLU(),
                            Linear(channels, channels)
                            )
            self.reaction_predictor[cyp] = Attention(channels, dropout, 1)

        self.subtype_predictor = Attention(channels+1, dropout, n_classes-1)

    def forward(self, x):        
        logits = {}
        for cyp in self.cyp_list:
            x_cyp = self.proj_edge[cyp](x)
            reaction_logits = self.reaction_predictor[cyp](x_cyp)
            x_for_subtype = torch.cat([x, reaction_logits.sigmoid()], -1)
            subtype_logits = self.subtype_predictor(x_for_subtype)            
            logits[cyp] = torch.cat([reaction_logits, subtype_logits], dim=-1)
        return logits  
    
class GNNSOM(torch.nn.Module):
    def __init__(self, 
                 channels = 512, num_layers = 2, gnn_num_layers=8,latent_size = 128, dropout=0.1, dropout_fc=0.1, dropout_som_fc=0.1, dropout_type_fc=0.1, n_classes=1, use_mamba=False, use_face=True, node_attn=True, face_attn=True,
                 encoder_dropout= 0.0, pooling='sum', 
                cyp_list= ['BOM_1A2', 'BOM_2A6', 'BOM_2B6', 'BOM_2C8', 'BOM_2C9', 'BOM_2C19', 'BOM_2D6', 'BOM_2E1', 'BOM_3A4'],
                use_som_v2=False
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
                        use_mamba=use_mamba,
                        node_attn = node_attn,
                        face_attn = face_attn,
                        encoder_dropout=encoder_dropout,
                        use_pe=False
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
        if use_som_v2:
            self.atom_fc = SOMPredictorV2(latent_size, dropout_som_fc, 4, cyp_list) # Any Reaction, spn-oxidation, hydroxylation, n-h Oxidation
            self.bond_fc = SOMPredictorV2(latent_size, dropout_som_fc, 4, cyp_list) # Any reaction, Cleavage, n-n Oxidation, Reduction
        else:
            self.atom_fc = SOMPredictor(latent_size, dropout_som_fc, 4, cyp_list) # Any Reaction, spn-oxidation, hydroxylation, n-h Oxidation
            self.bond_fc = SOMPredictor(latent_size, dropout_som_fc, 4, cyp_list) # Any reaction, Cleavage, n-n Oxidation        
        self.tasks = ['subs', 'bond', 'atom',  'spn', 'H', 'clv', 'nh_oxi', 'nn_oxi', 'rdc']
        self.atom_tasks = ['atom',  'spn', 'H',  'nh_oxi']
        self.bond_tasks = ['bond', 'clv','nn_oxi', 'rdc']

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

            logits['bond'][cyp] = bond_logits[cyp][:, 0]
            logits['clv'][cyp] = bond_logits[cyp][:, 1]
            logits['nn_oxi'][cyp] = bond_logits[cyp][:, 2]
            logits['rdc'][cyp] = bond_logits[cyp][:, 3]

            logits['atom'][cyp] = atom_logits[cyp][:, 0]
            logits['spn'][cyp] = atom_logits[cyp][:, 1]
            logits['H'][cyp] = atom_logits[cyp][:, 2]
            logits['nh_oxi'][cyp] = atom_logits[cyp][:, 3]

        return logits

    def forward_with_loss(self, batch, loss_fn_ce, loss_fn_bce, device, args):

        logits = self.forward(batch)

        loss_dict = {'total_loss' : 0, 'valid_loss' : 0}

        pred_dict = {}

        spn_atom = batch.spn_atom.to(device)
        has_H_atom = batch.has_H_atom.to(device)
        not_has_H_bond = batch.not_has_H_bond.to(device)
        
        pred_dict['spn_atom'] = spn_atom
        pred_dict['has_H_atom'] = has_H_atom
        pred_dict['not_has_H_bond'] = not_has_H_bond

        for cyp_idx, cyp in enumerate(self.cyp_list):
            labels = {'subs' : batch.y_substrate[cyp],
                    'bond' : batch.y[cyp],
                    'atom' : batch.y_atom[cyp],
                    'spn' : batch.y_spn[cyp],
                    'H' : batch.y_hydroxylation[cyp],
                    'clv' : batch.y_cleavage[cyp],
                    'nh_oxi' : batch.y_nh_oxidation[cyp],
                    'nn_oxi' : batch.y_nn_oxidation[cyp],
                    'rdc' :batch.y_bond_reduction[cyp] ,
                    'atom_oxi': batch.y_atom_oxidation[cyp],
                    }
            
            if args.reduction == 'sum':
                loss_dict[f'{cyp}_subs_loss'] = loss_fn_bce(logits['subs'][cyp], labels['subs']) / labels['subs'].shape[0]
                loss_dict[f'{cyp}_atom_loss'] = loss_fn_bce(logits['atom'][cyp], labels['atom']) / labels['atom'].shape[0]
                loss_dict[f'{cyp}_bond_loss'] = loss_fn_bce(logits['bond'][cyp], labels['bond']) / labels['bond'].shape[0]

                loss_dict[f'{cyp}_clv_loss'] = loss_fn_bce(logits['clv'][cyp], labels['clv']) / labels['clv'].shape[0]
                loss_dict[f'{cyp}_rdc_loss'] = loss_fn_bce(logits['rdc'][cyp], labels['rdc']) / labels['rdc'].shape[0]
                
                # loss_dict[f'{cyp}_H_loss'] = loss_fn_bce(logits['H'][cyp], labels['H']) / labels['H'].shape[0]
                # loss_dict[f'{cyp}_nh_oxi_loss'] = loss_fn_bce(logits['nh_oxi'][cyp], labels['nh_oxi']) / labels['nh_oxi'].shape[0]
                # loss_dict[f'{cyp}_spn_loss'] = loss_fn_bce(logits['spn'][cyp], labels['spn']) / labels['spn'].shape[0]
                # loss_dict[f'{cyp}_nn_oxi_loss'] = loss_fn_bce(logits['nn_oxi'][cyp], labels['nn_oxi']) / labels['nn_oxi'].shape[0]

                loss_dict[f'{cyp}_H_loss'] = loss_fn_bce(logits['H'][cyp][has_H_atom], labels['H'][has_H_atom]) / labels['H'][has_H_atom].shape[0] if has_H_atom.sum() else 0
                loss_dict[f'{cyp}_nh_oxi_loss'] = loss_fn_bce(logits['nh_oxi'][cyp][has_H_atom], labels['nh_oxi'][has_H_atom]) /  labels['nh_oxi'][has_H_atom].shape[0] if has_H_atom.sum() else 0
                loss_dict[f'{cyp}_spn_loss'] = (loss_fn_bce(logits['spn'][cyp][spn_atom], labels['spn'][spn_atom])) /  labels['spn'][spn_atom].shape[0] if spn_atom.sum() else 0
                loss_dict[f'{cyp}_nn_oxi_loss'] = loss_fn_bce(logits['nn_oxi'][cyp][not_has_H_bond], labels['nn_oxi'][not_has_H_bond]) / labels['nn_oxi'].shape[0] if not_has_H_bond.sum() else 0
            else:                
                loss_dict[f'{cyp}_subs_loss'] = loss_fn_bce(logits['subs'][cyp], labels['subs'])
                loss_dict[f'{cyp}_bond_loss'] = loss_fn_bce(logits['bond'][cyp], labels['bond'])

                loss_dict[f'{cyp}_clv_loss'] = loss_fn_bce(logits['clv'][cyp], labels['clv'])
                loss_dict[f'{cyp}_rdc_loss'] = loss_fn_bce(logits['rdc'][cyp], labels['rdc'])

                loss_dict[f'{cyp}_nn_oxi_loss'] = loss_fn_bce(logits['nn_oxi'][cyp], labels['nn_oxi'])
                loss_dict[f'{cyp}_atom_loss'] = loss_fn_bce(logits['atom'][cyp], labels['atom'])

                loss_dict[f'{cyp}_H_loss'] = loss_fn_bce(logits['H'][cyp], labels['H'])
                loss_dict[f'{cyp}_nh_oxi_loss'] = loss_fn_bce(logits['nh_oxi'][cyp], labels['nh_oxi'])
                loss_dict[f'{cyp}_spn_loss'] = loss_fn_bce(logits['spn'][cyp], labels['spn'])
                
            loss_dict[f'{cyp}_bond_loss'] = loss_dict[f'{cyp}_bond_loss'] * args.bond_loss_weight
            loss_dict[f'{cyp}_clv_loss'] = loss_dict[f'{cyp}_clv_loss'] * args.bond_loss_weight
            loss_dict[f'{cyp}_nn_oxi_loss'] = loss_dict[f'{cyp}_nn_oxi_loss'] * args.bond_loss_weight
            loss_dict[f'{cyp}_rdc_loss'] = loss_dict[f'{cyp}_rdc_loss'] * args.bond_loss_weight

            loss_dict[f'{cyp}_atom_loss'] = loss_dict[f'{cyp}_atom_loss'] * args.atom_loss_weight
            loss_dict[f'{cyp}_H_loss'] = loss_dict[f'{cyp}_H_loss'] * args.atom_loss_weight
            loss_dict[f'{cyp}_nh_oxi_loss'] = loss_dict[f'{cyp}_nh_oxi_loss'] * args.atom_loss_weight
            loss_dict[f'{cyp}_spn_loss'] = loss_dict[f'{cyp}_spn_loss'] * args.atom_loss_weight
            loss_dict[f'{cyp}_subs_loss'] = loss_dict[f'{cyp}_subs_loss'] * args.substrate_loss_weight

            loss_dict['total_loss'] += loss_dict[f'{cyp}_bond_loss']
            loss_dict['total_loss'] += loss_dict[f'{cyp}_clv_loss']
            loss_dict['total_loss'] += loss_dict[f'{cyp}_nn_oxi_loss']
            loss_dict['total_loss'] += loss_dict[f'{cyp}_rdc_loss']
            loss_dict['total_loss'] += loss_dict[f'{cyp}_atom_loss']
            loss_dict['total_loss'] += loss_dict[f'{cyp}_H_loss']
            loss_dict['total_loss'] += loss_dict[f'{cyp}_nh_oxi_loss']
            loss_dict['total_loss'] += loss_dict[f'{cyp}_spn_loss']
            loss_dict['total_loss'] += loss_dict[f'{cyp}_subs_loss'] 

            pred_dict[f'{cyp}_subs_logits'] = logits['subs'][cyp]
            pred_dict[f'{cyp}_subs_label'] = labels['subs']

            pred_dict[f'{cyp}_bond_logits'] = logits['bond'][cyp]
            pred_dict[f'{cyp}_bond_label'] = labels['bond']

            pred_dict[f'{cyp}_clv_logits'] = logits['clv'][cyp]
            pred_dict[f'{cyp}_clv_label'] = labels['clv']

            pred_dict[f'{cyp}_nn_oxi_logits'] = logits['nn_oxi'][cyp]
            pred_dict[f'{cyp}_nn_oxi_label'] = labels['nn_oxi']

            pred_dict[f'{cyp}_rdc_logits'] = logits['rdc'][cyp]
            pred_dict[f'{cyp}_rdc_label'] = labels['rdc']

            pred_dict[f'{cyp}_atom_logits'] = logits['atom'][cyp]
            pred_dict[f'{cyp}_atom_label'] = labels['atom']
            
            pred_dict[f'{cyp}_spn_logits'] = logits['spn'][cyp]
            pred_dict[f'{cyp}_spn_label'] = labels['spn']

            pred_dict[f'{cyp}_H_logits'] = logits['H'][cyp]
            pred_dict[f'{cyp}_H_label'] = labels['H']

            pred_dict[f'{cyp}_nh_oxi_logits'] = logits['nh_oxi'][cyp]
            pred_dict[f'{cyp}_nh_oxi_label'] = labels['nh_oxi']
            
        loss_dict['total_loss'] =loss_dict['total_loss'] / len(self.cyp_list)
        return logits, loss_dict, pred_dict
