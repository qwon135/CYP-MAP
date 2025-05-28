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
from .dualgraph.gnn import GNN2, GNN, one_hot_bonds, one_hot_atoms
from torch_geometric.utils import softmax

# from torch_scatter import scatter
from torch_geometric.nn.models import GCN, GIN, GAT
from torch_geometric.nn import MessagePassing
# from torch_geometric.nn import MPNN
from torch_geometric.utils import scatter, add_self_loops
from torch_geometric.nn.pool import global_add_pool
from modules.ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

class MPNNConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(MPNNConv, self).__init__(aggr='add')  # "Add" aggregation
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.edge_lin = torch.nn.Linear(13, out_channels)

    def forward(self, x, edge_index, edge_attr):
        # Add self-loops to the adjacency matrix
        edge_index, edge_attr = add_self_loops(edge_index, edge_attr, num_nodes=x.size(0))
        
        # Linearly transform node feature matrix
        x = self.lin(x)

        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_j, edge_attr):
        # x_j has shape [E, out_channels]
        # edge_attr has shape [E, 1]

        return x_j + self.edge_lin(edge_attr)

    def update(self, aggr_out):
        return aggr_out

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


class CYPMAP_GNN(torch.nn.Module):
    def __init__(self, 
                 channels = 512, num_layers = 2, gnn_num_layers=8,latent_size = 128, dropout=0.1, dropout_fc=0.1, dropout_som_fc=0.1, dropout_type_fc=0.1, n_classes=1, use_face=True, node_attn=True, face_attn=True,
                 encoder_dropout= 0.0, pooling='sum', gnn_type = 'gnn',
                cyp_list= ['BOM_1A2', 'BOM_2A6', 'BOM_2B6', 'BOM_2C8', 'BOM_2C9', 'BOM_2C19', 'BOM_2D6', 'BOM_2E1', 'BOM_3A4'],                
                ):
        super().__init__()
        self.gnn_type = gnn_type
        if self.gnn_type.lower() != 'gnn':
            if self.gnn_type.lower() == 'gcn':
                self.gnn = GCN(in_channels=sum(get_atom_feature_dims()), hidden_channels=latent_size, num_layers=gnn_num_layers)
            elif self.gnn_type.lower() == 'gin':
                self.gnn = GIN(in_channels=sum(get_atom_feature_dims()), hidden_channels=latent_size, num_layers=gnn_num_layers)
            elif self.gnn_type.lower() == 'gat':
                self.gnn = GAT(in_channels=sum(get_atom_feature_dims()), hidden_channels=latent_size, num_layers=gnn_num_layers)
            elif self.gnn_type.lower() == 'mpnn':
                self.gnn = MPNNConv(in_channels=sum(get_atom_feature_dims()), out_channels=latent_size)
            self.node_proj = torch.nn.Sequential(
                                        torch.nn.Linear(latent_size, latent_size),
                                        torch.nn.PReLU(init=0.05),
                                        torch.nn.Linear(latent_size, latent_size)
                                        )
            self.bond_proj = torch.nn.Sequential(
                                        torch.nn.Linear(latent_size, latent_size),
                                        torch.nn.PReLU(init=0.05),
                                        torch.nn.Linear(latent_size, latent_size)
                                        )
            self.pooling_u=False

        else:
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
            self.pooling_u=True

        self.bond_fc = torch.nn.ModuleDict() # Predict Reaction
        self.atom_fc = torch.nn.ModuleDict()
        self.cyp_list = cyp_list
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
        
        self.atom_tasks = ['atom_som', 'atom_spn', 'atom_oxc', 'atom_oxi', 'atom_epo', 'atom_sut', 'atom_dhy', 'atom_hys', 'atom_rdc']
        self.bond_tasks = ['bond_som', 'bond_oxc', 'bond_oxi', 'bond_epo', 'bond_sut', 'bond_dhy', 'bond_hys', 'bond_rdc']
        self.tasks = ['subs'] + self.atom_tasks + self.bond_tasks
        
        self.atom_fc = SOMPredictorV2(latent_size, dropout_som_fc, dropout_type_fc, len(self.atom_tasks), cyp_list)
        self.bond_fc = SOMPredictorV2(latent_size, dropout_som_fc, dropout_type_fc, len(self.bond_tasks), cyp_list)
                
    def forward(self, batch):
        if self.gnn_type == 'gnn':
            mol_feat_atom, mol_feat_bond, mol_feat_ring, x, edge_attr, u, x_encoder_node, edge_attr_encoder_edge = self.gnn(batch)

            x = x + x_encoder_node
            edge_attr = edge_attr + edge_attr_encoder_edge

            if self.pooling_u:
                mol_feature = torch.cat([mol_feat_atom, mol_feat_bond, mol_feat_ring, u], -1)
            else:
                mol_feature = u        

            edge_attr = edge_attr.view(edge_attr.shape[0] // 2, 2, edge_attr.shape[-1])
            edge_attr = edge_attr.sum(1)
        else:
            x = one_hot_atoms(batch.x)
            edge_attr = one_hot_bonds(batch.edge_attr)

            if self.gnn_type in ['gcn', 'gin', 'gat']:
                x = self.gnn(x = x, edge_index =batch.edge_index)
            elif self.gnn_type in ['mpnn']:
                x = self.gnn(x = x, edge_index =batch.edge_index, edge_attr = edge_attr)

            edge_attr = x[batch.edge_index].mean(0)            
            edge_attr = edge_attr.view(edge_attr.shape[0] // 2, 2, edge_attr.shape[-1]).sum(1)

            x, edge_attr = self.node_proj(x), self.bond_proj(edge_attr)            
            mol_feature = global_add_pool(x, batch.batch)
        
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

    def forward_pred(self, batch):

        logits = self.forward(batch)

        loss_dict = {'total_loss' : 0, 'valid_loss' : 0}

        pred_dict = {}        

        for cyp_idx, cyp in enumerate(self.cyp_list):
            pred_dict[f'{cyp}_subs_logits'] = logits['subs'][cyp]

            for tsk in self.tasks[1:]:                
                pred_dict[f'{cyp}_{tsk}_logits'] = logits[tsk][cyp]
        
        return pred_dict
