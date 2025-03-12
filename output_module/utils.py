import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score, average_precision_score
from torch_geometric.utils import unbatch
import warnings
warnings.filterwarnings('ignore', '')
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

def som_unbatch(x, num_count, num_mol, mid_list):
    # print(torch.arange(num_mol).shape, torch.LongTensor(num_count).shape)
    batch = torch.arange(num_mol).repeat_interleave(torch.LongTensor(num_count))
    
    new_dict = {}
    for k,v in x.items():
        # print(torch.tensor(v).shape, batch.shape)
        x_unbatch = unbatch(torch.tensor(v), batch)
        new_dict[k] = {mid : [round(j,3) for j in i.tolist()] for i, mid in zip(x_unbatch, mid_list)}
            
    return new_dict

def to_matrix(y_true, n_classes):
    y_true = np.array(y_true)
    y_true_ = np.zeros([y_true.shape[0], n_classes])

    for n, i in enumerate(y_true):
        y_true_[n][i] = 1
    return y_true_

def custom_jaccard_score(test_bond_label, test_bond_preds):
    try:
        [[TN, FP], [FN, TP]] = confusion_matrix(test_bond_label, test_bond_preds)    
        return TP / (TP + FP + FN)
    except:
        return 0

def calculate_roc_auc(y_true, y_score):
    if set(y_true) == set([0]):
        return 0
    if set(y_true) == set([1]):
        return 1    
    y_true, y_score = np.array(y_true), np.array(y_score)
    # print(y_true.shape, y_score.shape)    
    return roc_auc_score(y_true, y_score)

class Validator:
    def __init__(self, cyp_list, metric_mode):
        self.cyp_list = cyp_list
        self.valid_loss_dict = {'total_loss' : 0, 'valid_loss' : 0}    
        self.y_true = {}
        self.y_prob = {}

        self.mid_list = []        
        self.node_batch, self.n_nodes = [], []
        self.edge_batch, self.n_edges = [], []
        self.equivalent_bonds = []
        self.equivalent_atoms = []

        self.spn_atom = []
        self.has_H_atom = []
        self.not_H_bond = []
        self.bonds_idx_h = []
        self.first_H_bond_idx = []
        self.metric_mode = metric_mode
        self.tasks = ['subs', 'bond_som', 'atom_som',  'atom_spn', 'oxc', 'oxi', 'epo', 'sut', 'dhy', 'hys', 'rdc']
        for task in self.tasks:
            self.y_true[task] = {}
            self.y_prob[task] = {}

            for cyp in cyp_list:
                self.y_true[task][cyp] = []
                self.y_prob[task][cyp] = []            
                self.valid_loss_dict[f'{cyp}_{task}_loss'] = 0

    def add_probs(self, batch, prediction):
        for cyp in self.cyp_list:
            for task in self.tasks:
                if (self.metric_mode == 'atom') and (task in ['oxc', 'oxi', 'epo', 'sut', 'dhy', 'hys', 'rdc']):
                    probs = prediction[f'{cyp}_atom_{task}_atom_logits'].sigmoid()#.cpu().tolist()
                    probs = probs[batch.edge_index]                    
                    probs = probs.mean(0)                    
                    probs = probs.view(probs.shape[0] // 2, 2)
                    probs = probs.mean(-1)
                    self.y_prob[task][cyp] += probs.cpu().tolist()

                elif self.metric_mode == 'atom_bond' and (task in ['oxc', 'oxi', 'epo', 'sut', 'dhy', 'hys', 'rdc']):
                    atom_probs = prediction[f'{cyp}_bond_{task}_atom_logits'].sigmoid()#.cpu().tolist()
                    atom_probs = atom_probs[batch.edge_index]                    
                    atom_probs = atom_probs.mean(0)                    
                    atom_probs = atom_probs.view(atom_probs.shape[0] // 2, 2)
                    atom_probs = atom_probs.mean(-1)

                    bond_probs = prediction[f'{cyp}_bond_{task}_logits'].sigmoid()

                    self.y_prob[task][cyp] += (atom_probs/2 + bond_probs/2).cpu().tolist()
                else:                    
                    if task in ['oxc', 'oxi', 'epo', 'sut', 'dhy', 'hys', 'rdc']:
                        self.y_prob[task][cyp] += prediction[f'{cyp}_bond_{task}_logits'].sigmoid().cpu().tolist()
                    else:
                        self.y_prob[task][cyp] += prediction[f'{cyp}_{task}_logits'].sigmoid().cpu().tolist()


    def get_probs(self, task, cyp):
        if task == 'som':                
            y_true = np.array(self.y_true['bond_som'][cyp] + self.y_true['atom_spn'][cyp])
            y_prob = np.array(self.y_prob['bond_som'][cyp] + self.y_prob['atom_spn'][cyp])
            bonds_with_firstH = (np.array(self.not_H_bond) + np.array( self.first_H_bond_idx)).tolist()
            pos = bonds_with_firstH + self.spn_atom

            return y_true[pos], y_prob[pos]
        
        y_prob = np.array(self.y_prob[task][cyp])
        y_true = np.array(self.y_true[task][cyp])

        if task in ['bond_som', 'oxc', 'oxi', 'epo', 'sut', 'dhy', 'hys', 'rdc']:
            bonds_with_firstH = (np.array(self.not_H_bond) + np.array( self.first_H_bond_idx)).tolist()
            y_prob = y_prob[bonds_with_firstH]
            y_true = y_true[bonds_with_firstH]
        elif task in ['atom_spn']:
            y_prob = y_prob[self.spn_atom]
            y_true = y_true[self.spn_atom]
        elif task in ['dhy']:
            y_prob = y_prob[self.first_H_bond_idx]
            y_true = y_true[self.first_H_bond_idx]
        
        return y_true, y_prob

    def add_graph_info(self, batch):

        if self.node_batch:
            self.node_batch += (batch.batch + (self.node_batch[-1] + 1)).tolist()
        else:
            self.node_batch += batch.batch.tolist()
        
        edge_batch = torch.repeat_interleave(torch.arange(batch.num_graphs).to(batch.x.device), batch.n_edges, dim=0)
        edge_batch = edge_batch.view(edge_batch.shape[0]//2, 2)[:, 0].cpu()
        if self.edge_batch:
            self.edge_batch += (edge_batch + self.edge_batch[-1] + 1).tolist()
        else:
            self.edge_batch += edge_batch.tolist()

        self.equivalent_bonds += batch.equivalent_bonds
        self.equivalent_atoms += batch.equivalent_atoms

        self.n_nodes += batch.n_nodes.tolist()
        self.n_edges += (batch.n_edges//2).tolist()

        self.has_H_atom += batch.has_H_atom.tolist()
        self.not_H_bond += batch.not_has_H_bond.tolist()        
        self.first_H_bond_idx += batch.first_H_bond_idx.tolist()  
        self.spn_atom += batch.spn_atom.tolist()
        self.bonds_idx_h += batch.bonds_idx_h

    # def adjust_substrate(self, sub_th):
    #     for cyp in self.cyp_list:
    #         y_prob_sub = torch.FloatTensor(self.y_prob['subs'][cyp])
    #         n_nodes = torch.LongTensor(self.n_nodes)
    #         n_edges = torch.LongTensor(self.n_edges)

    #         sub_node = (torch.repeat_interleave(y_prob_sub, n_nodes) > sub_th).long().numpy()
    #         sub_edge = (torch.repeat_interleave(y_prob_sub, n_edges) > sub_th).long().numpy()
    
    #         for task in self.tasks:
    #             if task == 'subs':
    #                 continue
    #             if task in ['bond', 'clv', 'hdx', 'oxi', 'rdc']:
    #                 self.y_prob[task][cyp] = (np.array(self.y_prob[task][cyp]) * sub_edge).tolist()
    #             else:
    #                 self.y_prob[task][cyp] = (np.array(self.y_prob[task][cyp]) * sub_node).tolist()

    # def adjust_som(self, th):
    #     for cyp in self.cyp_list:            
    #         for task in self.tasks:
    #             if task == 'subs':
    #                 continue
    #             if task in ['spn']:
    #                 self.y_prob[task][cyp] = (np.array(self.y_prob[task][cyp]) * (np.array(self.y_prob['atom'][cyp]) >th).astype(int)).tolist()
    #             elif task in ['clv', 'hdx', 'oxi', 'rdc',]:
    #                 self.y_prob[task][cyp] = (np.array(self.y_prob[task][cyp]) * (np.array(self.y_prob['bond'][cyp]) >th).astype(int) ).tolist()

    def add_loss(self, loss_dict):
        self.valid_loss_dict['total_loss'] += loss_dict['total_loss']
        for cyp in self.cyp_list:
            self.valid_loss_dict['valid_loss'] += loss_dict[f'{cyp}_bond_som_loss']            
            self.valid_loss_dict['valid_loss'] += loss_dict[f'{cyp}_atom_spn_loss']

            for task in self.tasks:
                if task in [ 'oxc', 'oxi', 'epo', 'sut', 'dhy', 'hys', 'rdc']:
                    self.valid_loss_dict[f'{cyp}_{task}_loss'] += loss_dict[f'{cyp}_bond_{task}_loss']
                else:
                    self.valid_loss_dict[f'{cyp}_{task}_loss'] += loss_dict[f'{cyp}_{task}_loss']
                
    
    def get_probs_by_mid(self, mid):
        pass


    def unbatch(self):
        node_batch = np.repeat(np.arange(len(self.n_nodes)), self.n_nodes)
        edge_batch = np.repeat(np.arange(len(self.n_edges)), self.n_edges)

        self.y_prob_unbatch = {}        
        self.has_H_atom_unbatch = self.som_unbatch(self.has_H_atom, node_batch)
        self.has_H_bond_unbatch = self.som_unbatch(~np.array(self.not_H_bond), edge_batch)
        for tsk in [ 'bond_som', 'atom_som',  'atom_spn', 'oxc', 'oxi', 'epo', 'sut', 'dhy', 'hys', 'rdc']:
            self.y_prob_unbatch[tsk] = {}            

        for tsk in ['bond_som', 'oxc', 'oxi', 'epo', 'sut', 'dhy', 'hys', 'rdc']:
            for cyp in self.cyp_list:
                self.y_prob_unbatch[tsk][cyp] = self.som_unbatch(self.y_prob[tsk][cyp], edge_batch)                
        
        for tsk in ['atom_som', 'atom_spn']:
            for cyp in self.cyp_list:
                self.y_prob_unbatch[tsk][cyp] = self.som_unbatch(self.y_prob[tsk][cyp], node_batch)                             

    def som_unbatch(self, x, batch):
        batch = torch.from_numpy(batch)
        x = np.array(x)
        x = torch.from_numpy(x)
        
        x_unbatch = unbatch(x, batch)
        x_unbatch = [i.numpy() for i in x_unbatch]
                
        return x_unbatch

def validation(model, valid_loader, loss_fn_ce, loss_fn_bce, args):
    device = args.device
    model.eval()

    validator = Validator(cyp_list=model.cyp_list, metric_mode = args.metric_mode)

    for batch in valid_loader:        
        batch = batch.to(device)
        validator.add_graph_info(batch)
        with torch.no_grad():
            prediction = model.forward_pred(batch)        
        validator.add_probs(batch, prediction)

    scores = {k:v/ len(valid_loader) for k,v in validator.valid_loss_dict.items()}

    if args.adjust_substrate:
        validator.adjust_substrate(args.substrate_th)
    
    scores['validator'] = validator

    return scores