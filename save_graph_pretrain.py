import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, RobustScaler, normalize
import torch, os, random, copy
import numpy as np
import gc
from torch.nn.utils import clip_grad_norm_
from glob import glob
from torch import nn
from torch.nn import functional as F
# from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot as plt
from torch_ema import ExponentialMovingAverage
from modules.ogb.utils import smiles2graph
from modules.dualgraph.mol import smiles2graphwithface, simles2graphwithface_with_mask
from modules.dualgraph.gnn import one_hot_atoms, one_hot_bonds, GNN2
from rdkit.Chem import AllChem
from rdkit import Chem
from torch_geometric.data import Dataset, InMemoryDataset
from modules.dualgraph.dataset import DGData
from torch_geometric.loader import DataLoader
# from rdkit.Chem import PandasTools
from glob import glob
import argparse
import os
from itertools import repeat
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import subgraph, to_networkx

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

def main(args):
    data = []

    for csv_path in glob('pretrain_data/*/*.csv'):
        df = pd.read_csv(csv_path, index_col=None)
        dset_name = csv_path.split('/')[1]
        df['dataset_name'] = dset_name
        df['MOL_ID'] = f'{dset_name}_' + df.index.astype(str).str.zfill(7)
        data.append(df[['MOL_ID', 'dataset_name', 'smiles']])
    data =pd.concat(data).reset_index(drop=True)
    data = data.drop_duplicates(['smiles']).reset_index(drop=True)
    data = data[~data['MOL_ID'].isin(['chembl34_2311326'])] .reset_index(drop=True) # Error moleculer
    data.to_parquet('pretrain_data.parquet')

    for mid, smile in tqdm(data[['MOL_ID', 'smiles']].values):
        graph = mol2graph(smile)
        torch.save(graph, f'pretrain_data/graph_pt/{mid}.pt')
    
def parse_args():
    parser = argparse.ArgumentParser()    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()    
    main(args)    