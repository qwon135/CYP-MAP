import sys
import pandas as pd
import torch, os, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))
from rdkit import Chem
from rdkit.Chem import AllChem
from glob import glob
from tqdm import tqdm
from modules.dualgraph.mol import smiles2graphwithface
from modules.dualgraph.dataset import DGData

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
    os.makedirs(f'pretrain_data/graph_pt', exist_ok=True)
    if not os.path.exists(f'pretrain_data.parquet'):
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
    else:
        data = pd.read_parquet('pretrain_data.parquet').loc[:5000]

    for mid, smile in tqdm(data[['MOL_ID', 'smiles']].values):

        mol = Chem.MolFromSmiles(smile)
        mol_h = AllChem.AddHs( mol, addCoords=True)                        
        graph_h = mol2graph(mol_h)

        torch.save(graph_h, f'pretrain_data/graph_pt/{mid}_addH_Bond.pt')

        graph = mol2graph(smile)
        torch.save(graph, f'pretrain_data/graph_pt/{mid}.pt')
        
    
def parse_args():
    parser = argparse.ArgumentParser()    
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()    
    main(args)    