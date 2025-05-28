import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', )))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'modules')))

import pandas as pd
import torch, random, time, gc, argparse
import numpy as np
from torch import nn
from tqdm import tqdm
from torch_ema import ExponentialMovingAverage
from modules.dualgraph.gnn import  GNN2
from torch_geometric.data import  InMemoryDataset
from torch_geometric.loader import DataLoader

from torch_geometric.utils import subgraph, to_networkx

def get_time(start_time, step, total_steps):
    current_time = time.time()
    elapsed_time = current_time - start_time
    
    # 예상 남은 시간 계산
    steps_per_sec = (step + 1) / elapsed_time
    remaining_steps = total_steps - (step + 1)
    estimated_time_remaining = remaining_steps / steps_per_sec
    
    # 시간 형식 지정
    elapse_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
    remain_time = time.strftime("%H:%M:%S", time.gmtime(estimated_time_remaining))
    return remain_time, elapse_time

class MoleculeDataset_graphcl(InMemoryDataset):
    def __init__(self,
                 root='dataset_path',
                 transform=None,
                 pre_transform=None, 
                 df=None):
        self.df = df
        self.aug_prob = None
        self.aug_mode = 'sample'
        self.aug_strength = 0.2
        self.augmentations = [self.node_drop, self.subgraph,
                              self.edge_pert, self.attr_mask, lambda x: x]
        super().__init__(root, transform, pre_transform, df)
        
    @property
    def raw_file_names(self):
        return [f'raw_{i+1}.pt' for i in range(self.df.shape[0])]

    @property
    def processed_file_names(self):
        return [f'data_{i+1}.pt' for i in range(self.df.shape[0])]
    
    def set_augMode(self, aug_mode):
        self.aug_mode = aug_mode

    def set_augStrength(self, aug_strength):
        self.aug_strength = aug_strength

    def set_augProb(self, aug_prob):
        self.aug_prob = aug_prob

    def node_drop(self, data):
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        drop_num = int(node_num * self.aug_strength)

        idx_perm = np.random.permutation(node_num)
        idx_nodrop = idx_perm[drop_num:].tolist()
        idx_nodrop.sort()

        node_mask = torch.zeros(node_num).bool()
        node_mask[~torch.tensor(idx_nodrop)]=True

        edge_idx, edge_attr, edge_mask = subgraph(subset=idx_nodrop,
                                        edge_index=data.edge_index,
                                        edge_attr=data.edge_attr,
                                        relabel_nodes=True,
                                        num_nodes=node_num,
                                        return_edge_mask=True)
        data = self.ring_drop(data, node_mask, edge_mask)
        
        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nodrop]        

        data.__num_nodes__, _ = data.x.shape
        return data

    def edge_pert(self, data):        
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        pert_num = int(edge_num * self.aug_strength)

        # delete edges
        idx_drop = np.random.choice(edge_num, (edge_num - pert_num),
                                    replace=False)
        edge_index = data.edge_index[:, idx_drop]
        edge_attr = data.edge_attr[idx_drop]

        # add edges
        adj = torch.ones((node_num, node_num))
        adj[edge_index[0], edge_index[1]] = 0
        # edge_index_nonexist = adj.nonzero(as_tuple=False).t()
        edge_index_nonexist = torch.nonzero(adj, as_tuple=False).t()
        idx_add = np.random.choice(edge_index_nonexist.shape[1],
                                    pert_num, replace=False)
        edge_index_add = edge_index_nonexist[:, idx_add]
        # random 4-class & 3-class edge_attr for 1st & 2nd dimension
        edge_attr_add_1 = torch.tensor(np.random.randint(
            4, size=(edge_index_add.shape[1], 1)))
        edge_attr_add_2 = torch.tensor(np.random.randint(
            3, size=(edge_index_add.shape[1], 1)))
        edge_attr_add_3 = torch.tensor(np.random.randint(
            2, size=(edge_index_add.shape[1], 1)))
        edge_attr_add = torch.cat((edge_attr_add_1, edge_attr_add_2, edge_attr_add_3), dim=1)
        edge_index = torch.cat((edge_index, edge_index_add), dim=1)
        

        edge_attr = torch.cat((edge_attr, edge_attr_add), dim=0)

        data.edge_index = edge_index
        data.edge_attr = edge_attr
        return data

    def attr_mask(self, data):

        _x = data.x.clone()
        node_num, _ = data.x.size()
        mask_num = int(node_num * self.aug_strength)

        token = data.x.float().mean(dim=0).long()
        idx_mask = np.random.choice(
            node_num, mask_num, replace=False)

        _x[idx_mask] = token
        data.x = _x
        return data    
    
    def ring_drop(self, graph, node_mask, edge_mask):
        node_mask_idx = (node_mask.cumsum(0) - 1) * node_mask
        edge_mask_idx = (edge_mask.cumsum(0) - 1) * edge_mask

        graph.ring_index = graph.ring_index[:, edge_mask]

        graph.n_edges = edge_mask.sum().item()
        graph.n_nodes = node_mask.sum().item()

        nf_node_mask = ~torch.isin(graph.nf_node, torch.where(~node_mask)[0])[0]

        graph.nf_node = node_mask_idx[graph.nf_node[:, nf_node_mask]]
        graph.nf_ring = graph.nf_ring[:, nf_node_mask]
        graph.n_nfs = graph.nf_node.size(1)
        return graph
    
    def subgraph(self, data):

        G = to_networkx(data)
        node_num, _ = data.x.size()
        _, edge_num = data.edge_index.size()
        sub_num = int(node_num * (1 - self.aug_strength))

        idx_sub = [np.random.randint(node_num, size=1)[0]]
        idx_neigh = set([n for n in G.neighbors(idx_sub[-1])])

        while len(idx_sub) <= sub_num:
            if len(idx_neigh) == 0:
                idx_unsub = list(set([n for n in range(node_num)]).difference(set(idx_sub)))
                idx_neigh = set([np.random.choice(idx_unsub)])
            sample_node = np.random.choice(list(idx_neigh))

            idx_sub.append(sample_node)
            idx_neigh = idx_neigh.union(
                set([n for n in G.neighbors(idx_sub[-1])])).difference(set(idx_sub))

        idx_nondrop = idx_sub
        idx_nondrop.sort()
        
        node_mask = torch.zeros(node_num).bool()
        node_mask[~torch.tensor(idx_nondrop)]=True

        edge_idx, edge_attr, edge_mask = subgraph(subset=idx_nondrop,
                                       edge_index=data.edge_index,
                                       edge_attr=data.edge_attr,
                                       relabel_nodes=True,
                                       num_nodes=node_num,
                                       return_edge_mask=True)
        data = self.ring_drop(data, node_mask, edge_mask)

        data.edge_index = edge_idx
        data.edge_attr = edge_attr
        data.x = data.x[idx_nondrop]
        data.__num_nodes__, _ = data.x.shape
        return data

    def get(self, idx):
        sid = self.sid_list[idx]        

        h_choice = random.random()        
        if h_choice < 0.33:  # addH/noH            
            data1 = torch.load(f'pretrain_data/graph_pt/{sid}_addH_Bond.pt')
            data2 = torch.load(f'pretrain_data/graph_pt/{sid}.pt')                            
            
        elif h_choice < 0.66:  # addH/addH
            data1 = torch.load(f'pretrain_data/graph_pt/{sid}_addH_Bond.pt')
            data2 = torch.load(f'pretrain_data/graph_pt/{sid}_addH_Bond.pt')
        else:  # noH/noH
            data1 = torch.load(f'pretrain_data/graph_pt/{sid}.pt')
            data2 = torch.load(f'pretrain_data/graph_pt/{sid}.pt')

        if self.aug_mode == 'no_aug':
            n_aug1, n_aug2 = 4, 4
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'uniform':
            n_aug = np.random.choice(25, 1)[0]
            n_aug1, n_aug2 = n_aug // 5, n_aug % 5
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        elif self.aug_mode == 'sample':
            n_aug = np.random.choice(25, 1, p=self.aug_prob)[0]
            n_aug1, n_aug2 = n_aug // 5, n_aug % 5
            data1 = self.augmentations[n_aug1](data1)
            data2 = self.augmentations[n_aug2](data2)
        else:
            raise ValueError
        
        return data1, data2
    
    def process(self):
        self.sid_list = []        
        
        for i in range(self.df.shape[0]):
            sid = self.df.loc[i, 'MOL_ID']
            self.sid_list.append(sid)
    def len(self):
        return self.df.shape[0]

class GraphContrastiveLearning(torch.nn.Module):

    def __init__(self):
        super(GraphContrastiveLearning, self).__init__()
        self.ddi = True
        self.gnn = GNN2(
                        mlp_hidden_size = 512,
                        mlp_layers = 2,
                        num_message_passing_steps=8,
                        latent_size = 128,
                        use_layer_norm = True,
                        use_bn=False,
                        use_face=True,
                        som_mode=False,
                        ddi=True,
                        dropedge_rate = 0.0,
                        dropnode_rate = 0.0,
                        dropout = 0.0,
                        dropnet = 0.0,
                        global_reducer = 'sum',
                        node_reducer = 'sum',
                        face_reducer = 'sum',
                        graph_pooling = 'sum',                        
                        node_attn = True,
                        face_attn = True,
                        encoder_dropout=0.0,                        
                        )
                        
        self.proj = nn.Sequential(
                        nn.Linear(128, 128),
                        nn.ReLU(),
                        nn.Linear(128, 128),
                        )
    def forward(self, batch):
        mol = self.gnn(batch).squeeze(1)
        return self.proj(mol)

def loss_cl(x1, x2):
    T = 0.1
    batch_size, _ = x1.size()
    x1_abs = x1.norm(dim=1)
    x2_abs = x2.norm(dim=1)

    sim_matrix = torch.einsum('ik,jk->ij', x1, x2) / torch.einsum('i,j->ij', x1_abs, x2_abs)
    sim_matrix = torch.exp(sim_matrix / T)
    pos_sim = sim_matrix[range(batch_size), range(batch_size)]
    loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim)
    loss = - torch.log(loss).mean()
    return loss

def main(args):
    data = pd.read_parquet('pretrain_data.parquet').loc[:5000]
    train_dataset = MoleculeDataset_graphcl(df = data)
    train_dataset.set_augMode('sample')
    train_dataset.set_augProb(np.ones(25) / 25)
    train_dataset.set_augStrength(0.2)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers = args.num_workers)

    model = GraphContrastiveLearning().to(args.device)

    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay = args.weight_decay)
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, verbose=False)    
    total_steps = len(train_loader)    

    device = args.device

    best_val_loss = 1e6    
    for epoch in range(1, args.epochs+1):

        model.train()
        train_loss, step_loss = 0, 0
        start_time = time.time()        

        for step, (batch1, batch2) in enumerate(train_loader):
            x1, x2 = model(batch1.to(device)), model(batch2.to(device))            
            loss = loss_cl(x1, x2)

            optim.zero_grad()
            loss.backward()
            optim.step()
            ema.update()
            
            train_loss += loss
            step_loss += loss.cpu().item()

            if (step % 50) == 0 and (step != 0):
                step_loss /= 50
                remain_time, elapse_time = get_time(start_time, step, total_steps)
                print(f'EPOCH : {epoch} | step : {step} / {total_steps} | step_loss : {step_loss:.4f} | elapse : {elapse_time} | remain : {remain_time}')
                step_loss = 0

        if train_loss < best_val_loss:
            best_val_loss = train_loss
            torch.save(model.gnn.state_dict(), f'ckpt_pretrain/gnn_pretrain.pt')            

        scheduler.step()
        torch.save(
                {
                'optimizer_state_dict': optim.state_dict(),
                'model_state_dict': model.state_dict(),
                'gnn_state_dict' : model.gnn.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'ema_state_dict' : ema.state_dict()
                },
                f'ckpt_pretrain/gnn_pretrain_epoch{epoch}.pt')

        print(f'EPOCH : {epoch} | train_loss : {train_loss/len(train_loader):.4f}')


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--lr", type=int, default=5e-5)
    parser.add_argument("--weight_decay", type=float, default=5e-5)
    parser.add_argument("--ema_decay", type=float, default=0.999)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--num_workers", type=int, default=12)

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()    
    main(args)        