import argparse
import os.path as osp
from typing import Any, Dict, Optional

import torch
from torch.nn import (
    BatchNorm1d,
    Embedding,
    Linear,
    ModuleList,
    ReLU,
    Sequential,
)

import torch_geometric.transforms as T
from torch_geometric.nn import GINEConv, GATConv, global_add_pool
import inspect, random, os
from typing import Any, Dict, Optional

import torch.nn.functional as F
from torch import Tensor
from torch.nn import Dropout, Linear, Sequential

from torch_geometric.nn.conv import MessagePassing
from torch_geometric.nn.inits import reset
from torch_geometric.nn.resolver import (
    activation_resolver,
    normalization_resolver,
)
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_batch, unbatch_edge_index

from mamba_ssm import Mamba
from torch_geometric.utils import degree, sort_edge_index

from dualgraph.gnn import one_hot_atoms, one_hot_bonds
import math
from torch_geometric.utils import softmax
from torch_scatter import scatter
from torch_geometric.nn.aggr import AttentionalAggregation
from dualgraph.gnn import GNN, GNN2

class GNNSubstrate(torch.nn.Module):

    def __init__(self):
        super(GNNSubstrate, self).__init__()
        self.gnn = GNN2(
                        mlp_hidden_size = 512,
                        mlp_layers = 2,
                        latent_size = 128,
                        use_layer_norm = True,
                        use_face=True, 
                        ddi=True, 
                        dropedge_rate = 0.1, 
                        dropnode_rate = 0.1,
                        dropout = 0.1,
                        dropnet = 0.1, 
                        global_reducer = "sum", 
                        node_reducer = "sum", 
                        face_reducer = "sum", 
                        graph_pooling = "sum",                        
                        node_attn = True, 
                        face_attn = False)

        
        self.substrate_fc = torch.nn.ModuleDict()
        self.cyp_list= ['BOM_1A2', 'BOM_2A6', 'BOM_2B6', 'BOM_2C8', 'BOM_2C9', 'BOM_2C19', 'BOM_2D6', 'BOM_2E1', 'BOM_3A4']
        for cyp in self.cyp_list:
            self.substrate_fc[cyp] = torch.nn.Sequential(
                                        torch.nn.LayerNorm(128),
                                        torch.nn.Linear(128, 128,),
                                        torch.nn.BatchNorm1d(128),
                                        torch.nn.Dropout(0.1),
                                        torch.nn.ReLU(),
                                        torch.nn.Linear(128, 1),
                                        )
            
            self.substrate_fc[cyp][-1].weight.data.normal_(mean=0.0, std=0.01)        

    def forward(self, batch):
        mol_feature = self.gnn(batch)
        substrate = {}
        for cyp in self.cyp_list:
            substrate[cyp] = self.substrate_fc[cyp](mol_feature).squeeze(-1)
        
        return substrate

    def get_loss(self, substrate, batch, loss_fn, device):        
        loss_dict = {'total_loss' : 0}
        for cyp in self.cyp_list:                        
            substrate_label = batch.y_substrate[cyp].float().to(device)            
            loss_dict[f'substrate_{cyp}_loss'] = loss_fn(substrate[cyp], substrate_label)
            loss_dict['total_loss'] += loss_dict[f'substrate_{cyp}_loss']
        loss_dict['total_loss'] =loss_dict['total_loss'] / len(self.cyp_list)
        return loss_dict
    
    def infer(self, batch):
        substrate = self.forward( batch)                
                        
        preds = {}
        for cyp in self.cyp_list:            
            y_prob_substrate = substrate[cyp].sigmoid().cpu()
            y_pred_substrate = torch.ge(substrate[cyp].sigmoid().cpu(), 0.5).long()            
            
            preds['y_prob_substrate'] = y_prob_substrate
            preds['y_pred_substrate'] = y_pred_substrate

        return preds

class MLPwoLastAct(torch.nn.Module):
    def __init__(
        self,
        input_size,
        output_sizes,
        use_layer_norm=False,
        activation=torch.nn.ReLU,
        dropout=0.0,
        layernorm_before=False,
        use_bn=False,
    ):
        super().__init__()
        module_list = []
        if not use_bn:
            if layernorm_before:
                module_list.append(torch.nn.LayerNorm(input_size))

            for i, size in enumerate(output_sizes):
                module_list.append(torch.nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(activation())
                    if dropout > 0:
                        module_list.append(torch.nn.Dropout(dropout))
                input_size = size

            if not layernorm_before and use_layer_norm:
                module_list.append(torch.nn.LayerNorm(input_size))
        else:
            for i, size in enumerate(output_sizes):
                module_list.append(torch.nn.Linear(input_size, size))
                if i < len(output_sizes) - 1:
                    module_list.append(torch.nn.BatchNorm1d(size))
                    module_list.append(activation())
                    if dropout > 0:
                        module_list.append(torch.nn.Dropout(p=dropout))
                input_size = size

        self.module_list = torch.nn.ModuleList(module_list)
        self.reset_parameters()

    def reset_parameters(self):
        for item in self.module_list:
            if hasattr(item, "reset_parameters"):
                item.reset_parameters()

    def forward(self, x):
        for item in self.module_list:
            x = item(x)
        return x


class GPSConv(torch.nn.Module):
    def __init__(
        self,
        channels: int,
        conv: Optional[MessagePassing],
        dropout: float = 0.0,
        act: str = 'relu',
        order_by_degree: bool = False,
        shuffle_ind: int = 0,
        d_state: int = 16,
        d_conv: int = 4,
        act_kwargs: Optional[Dict[str, Any]] = None,
        norm: Optional[str] = 'batch_norm',
        norm_kwargs: Optional[Dict[str, Any]] = None,
    ):
        super().__init__()

        self.channels = channels
        self.conv = conv
        self.dropout = dropout
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        
        assert (self.order_by_degree==True and self.shuffle_ind==0) or (self.order_by_degree==False), f'order_by_degree={self.order_by_degree} and shuffle_ind={self.shuffle_ind}'
        

        self.self_attn = Mamba(
            d_model=channels,
            d_state=d_state,
            d_conv=d_conv,
            expand=1
        )
            
        self.mlp = Sequential(
            Linear(channels, channels * 2),
            activation_resolver(act, **(act_kwargs or {})),
            Dropout(dropout),
            Linear(channels * 2, channels),
            Dropout(dropout),
        )

        norm_kwargs = norm_kwargs or {}
        self.norm1 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm2 = normalization_resolver(norm, channels, **norm_kwargs)
        self.norm3 = normalization_resolver(norm, channels, **norm_kwargs)

        self.norm_with_batch = False
        if self.norm1 is not None:
            signature = inspect.signature(self.norm1.forward)
            self.norm_with_batch = 'batch' in signature.parameters

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if self.conv is not None:
            self.conv.reset_parameters()
        self.attn._reset_parameters()
        reset(self.mlp)
        if self.norm1 is not None:
            self.norm1.reset_parameters()
        if self.norm2 is not None:
            self.norm2.reset_parameters()
        if self.norm3 is not None:
            self.norm3.reset_parameters()

    def permute_within_batch(self, x, batch):
        # Enumerate over unique batch indices
        unique_batches = torch.unique(batch)
        
        # Initialize list to store permuted indices
        permuted_indices = []

        for batch_index in unique_batches:
            # Extract indices for the current batch
            indices_in_batch = (batch == batch_index).nonzero().squeeze()
            
            # Permute indices within the current batch
            permuted_indices_in_batch = indices_in_batch[torch.randperm(len(indices_in_batch))]
            
            # Append permuted indices to the list
            permuted_indices.append(permuted_indices_in_batch)
        
        # Concatenate permuted indices into a single tensor
        permuted_indices = torch.cat(permuted_indices)

        return permuted_indices

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        batch: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Runs the forward pass of the module."""
        hs = []
        if self.conv is not None:  # Local MPNN.
            h = self.conv(x, edge_index, **kwargs)
            h = F.dropout(h, p=self.dropout, training=self.training)
            h = h + x
            if self.norm1 is not None:
                if self.norm_with_batch:
                    h = self.norm1(h, batch=batch)
                else:
                    h = self.norm1(h)
            hs.append(h)
            
            
        if self.order_by_degree:
            deg = degree(edge_index[0], x.shape[0]).to(torch.long)
            order_tensor = torch.stack([batch, deg], 1).T
            _, x = sort_edge_index(order_tensor, edge_attr=x)
            
        if self.shuffle_ind == 0:
            h, mask = to_dense_batch(x, batch)
            h = self.self_attn(h)[mask]
        else:
            mamba_arr = []
            for _ in range(self.shuffle_ind):
                h_ind_perm = self.permute_within_batch(x, batch)
                h_i, mask = to_dense_batch(x[h_ind_perm], batch)
                h_i = self.self_attn(h_i)[mask][h_ind_perm]
                mamba_arr.append(h_i)
            h = sum(mamba_arr) / self.shuffle_ind        
        
        h = F.dropout(h, p=self.dropout, training=self.training)
        h = h + x  # Residual connection.
        if self.norm2 is not None:
            if self.norm_with_batch:
                h = self.norm2(h, batch=batch)
            else:
                h = self.norm2(h)
        hs.append(h)

        out = sum(hs)  # Combine local and global outputs.

        out = out + self.mlp(out)
        if self.norm3 is not None:
            if self.norm_with_batch:
                out = self.norm3(out, batch=batch)
            else:
                out = self.norm3(out)

        return out, h

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}({self.channels}, '
                f'conv={self.conv}, heads={self.heads})')


class GraphModelSubstrate(torch.nn.Module):
    def __init__(self, channels = 256, pe_dim = 3, num_layers = 2, model_type = 'mamba', shuffle_ind = 0, d_state = 16, d_conv = 4, n_classes=1, dropout=0.1, order_by_degree = False, 
                cyp_list= ['BOM_1A2', 'BOM_2A6', 'BOM_2B6', 'BOM_2C8', 'BOM_2C9', 'BOM_2C19', 'BOM_2D6', 'BOM_2E1', 'BOM_3A4']):
        super().__init__()

        self.node_emb = torch.nn.Sequential(
                        Linear(173, channels - pe_dim) 
                        )
        
        self.pe_emb = torch.nn.Sequential(
                         BatchNorm1d(pe_dim),
                         Linear(3, pe_dim)
                        )
        
        self.edge_emb = torch.nn.Sequential(
                        Linear(13, channels)
                        )

        self.model_type = model_type
        self.shuffle_ind = shuffle_ind
        self.order_by_degree = order_by_degree
        
        self.convs = ModuleList()
        for _ in range(num_layers):
            nn = Sequential(
                Linear(channels, channels),                
                ReLU(),
                Linear(channels, channels),
            )
            conv = GPSConv(channels, GINEConv(nn),
                        #    dropout=0.1,
                            shuffle_ind=self.shuffle_ind,
                            order_by_degree=self.order_by_degree,
                            d_state=d_state, d_conv=d_conv)
            self.convs.append(conv)
      
        self.substrate_fc = torch.nn.ModuleDict()
        self.cyp_list = cyp_list

        for cyp in cyp_list:
            self.substrate_fc[cyp] = torch.nn.Sequential(
                                            torch.nn.LayerNorm(channels),
                                            Linear(channels , channels),
                                            torch.nn.BatchNorm1d(channels),                                   
                                            torch.nn.ReLU(),
                                            torch.nn.Dropout(dropout),
                                            Linear(channels , 1)
                                            )
            self.substrate_fc[cyp][-1].weight.data.normal_(mean=0.0, std=0.01)

    def forward(self, batch_data):
        x, pe, edge_index, edge_attr, batch = batch_data.x, batch_data.pe, batch_data.edge_index, batch_data.edge_attr, batch_data.batch
        x = self.node_emb(one_hot_atoms(x))
        edge_attr = self.edge_emb(one_hot_bonds(edge_attr))
        
        pe = self.pe_emb(pe)
        x = torch.cat((x, pe), 1)
        # x = x+pe
        for conv in self.convs:            
            x, h = conv(x, edge_index, batch, edge_attr=edge_attr)

        moleculer_features = global_add_pool(x, batch)

        substrate = {}
        for cyp in self.cyp_list:
            substrate[cyp] = self.substrate_fc[cyp](moleculer_features).squeeze(-1)

        return substrate

    def get_loss(self, substrate, batch, loss_fn, device):        
        loss_dict = {'total_loss' : 0}        
        for cyp in self.cyp_list:                        
            substrate_label = batch.y_substrate[cyp].float().to(device)            
            loss_dict[f'substrate_{cyp}_loss'] = loss_fn(substrate[cyp], substrate_label)
            loss_dict['total_loss'] += loss_dict[f'substrate_{cyp}_loss']
        loss_dict['total_loss'] =loss_dict['total_loss'] / len(self.cyp_list)
        return loss_dict
    
    def infer(self, batch):
        substrate = self.forward( batch)                
                        
        preds = {}
        for cyp in self.cyp_list:            
            y_prob_substrate = substrate[cyp].sigmoid().cpu()
            y_pred_substrate = torch.ge(substrate[cyp].sigmoid().cpu(), 0.5).long()            
            
            preds['y_prob_substrate'] = y_prob_substrate
            preds['y_pred_substrate'] = y_pred_substrate

        return preds            