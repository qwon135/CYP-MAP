import torch
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score
import numpy as np
from rdkit import Chem
from tqdm import tqdm
from rdkit.Chem import AllChem
from rdkit import RDLogger
from sklearn.metrics import roc_auc_score
RDLogger.DisableLog('rdApp.*')  
import warnings
warnings.filterwarnings(action='ignore')
from multiprocessing import Pool
from unimol.models import UniMolModel
from unicore import tasks
from unicore.data import Dictionary
from torch_ema import ExponentialMovingAverage
from unicore import checkpoint_utils, distributed_utils, options, utils
from unicore import models
from unicore.data.sort_dataset import  EpochShuffleDataset
from unicore.data import LMDBDataset, EpochBatchIterator, CountingIterator, UnicoreDataset
from unicore.data.data_utils import collate_tokens, collate_tokens_2d, batch_by_size
from unicore.data.sort_dataset import  EpochShuffleDataset
from torch import nn
from rdkit.Chem import Draw, PandasTools
import argparse, random, os
from som_utils import validation, get_loaders
import pandas as pd
import re
from pp_unimol import mol2bond_label, predict_som, FocalLoss, to_matrix

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)    
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)    
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def remove_alpha_dot(text):
    pattern = r'[A-Za-z]\.'  # 알파벳 + 마침표에 대응하는 패턴
    return re.sub(pattern, '', text)

def main(args):
    if args.class_type == 1:
        n_classes = 5
    elif args.class_type == 2:
        n_classes = 4
    df = PandasTools.LoadSDF('/home/pjh/workspace/SOM/data/EBoMD_train_0213.sdf')
    df = df.reset_index(drop=True)
    test_df = PandasTools.LoadSDF('/home/pjh/workspace/SOM/data/EBoMD_test_0213.sdf')
    test_df = test_df.reset_index(drop=True)
    test_df = test_df.fillna('')
    test_df = test_df[test_df[args.cyp] != ''].reset_index(drop=True)
    test_df[args.cyp] = test_df[args.cyp].apply(remove_alpha_dot)

    smile2bond_target = {}
    smile2bond_idx = {}
    cyp = args.cyp
    class_type = args.class_type

    df = pd.concat([df, test_df]).reset_index(drop=True)
    df = df.fillna('')
    for idx in range( df.shape[0]):
        # try:
        reactions = df.loc[idx, cyp].split('\n')
        
        mol = df['ROMol'][idx]        
        smile =  Chem.MolToSmiles(mol)
                
        bonds_idx = [(i.GetBeginAtomIdx(),i.GetEndAtomIdx()) for i in AllChem.AddHs( mol, addCoords=True).GetBonds()]
        bonds = [(i.GetBeginAtom().GetSymbol(),i.GetEndAtom().GetSymbol()) for i in AllChem.AddHs( mol, addCoords=True).GetBonds()]    
        bond_label = mol2bond_label(bonds, bonds_idx, reactions, class_type)
        smile2bond_target[smile] = bond_label
        smile2bond_idx[smile] = bonds_idx
        # except:
        #     continue
    train_args = options.get_parser(None, default_task='unimol')
    unimol_args, _ = train_args.parse_known_args()

    unimol_args.data = './som'
    unimol_args.task_name = 'som'
    unimol_args.mask_prob=0.1
    unimol_args.leave_unmasked_prob = 0.1
    unimol_args.random_token_prob = 0.1
    unimol_args.dict_name = 'dict.txt'
    unimol_args.only_polar = -1
    unimol_args.conf_size = 11
    unimol_args.no_shuffle = True
    unimol_args.mode = 'train'
    unimol_args.remove_hydrogen = False
    unimol_args.remove_polar_hydrogen = False
    unimol_args.max_atoms = 512
    unimol_args.noise_type = None
    unimol_args.noise = False
    unimol_args.arch = 'unimol'
    unimol_args.finetune_mol_model = None
    unimol_args.num_recycles = 1
    unimol_args.coord_loss = 0
    unimol_args.distance_loss=False
    unimol_args.classification_head_name = 'som'
    unimol_args.num_classes = 1
    unimol_args.masked_token_loss = 0.1
    unimol_args.masked_coord_loss = 0.1

    task = tasks.setup_task(unimol_args)

    model = task.build_model(unimol_args)
    missing_key = model.load_state_dict(torch.load('mol_pre_no_h_220816.pt', 'cpu',)['model'], strict=False)

    model.classification_heads.som = nn.Sequential(
                                            # nn.Dropout(0.2),
                                            nn.Linear(64, n_classes)
                                            )

    optimizer = torch.optim.AdamW([
                        {'params': model.embed_tokens.parameters(),'lr' : 1e-6},
                        {'params': model.encoder.parameters(),'lr' : 1e-6},
                        {'params': model.classification_heads.som.parameters()}], lr=1e-4, weight_decay=1e-3)
    ema = ExponentialMovingAverage(model.parameters(), decay=0.995)

    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=1, verbose=False)
    loss_fn = FocalLoss(alpha=0.1)
    dictionary = Dictionary.load('som/dict.txt')

    for epoch in range(args.epochs):
        train_loader, valid_loader, test_loader = get_loaders(task, args.cyp, args.class_type)
        train_loss = 0    
        model.train()
        for batch in train_loader:
            smile = batch['target']['smi_name'][0]
            mol = Chem.MolFromSmiles(smile)
            som_target = smile2bond_target[ batch['target']['smi_name'][0]]
            som_target = torch.from_numpy(np.array(som_target))

            token_index = torch.BoolTensor( [i not in dictionary.special_index() for i in  batch['net_input']['src_tokens'][0].tolist()])
            logits, padding_mask = predict_som(model, batch, token_index)    
            optimizer.zero_grad()
            loss = loss_fn(logits, som_target)
            loss.backward()
            optimizer.step()       
            ema.update()
    
            train_loss += loss.cpu().item()
        train_loss /= len(train_loader)
        lr_scheduler.step()

        valid_loss, mol_f1s, mol_prc, mol_rec, bond_f1s, bond_prc, bond_rec, bond_auc, jac_score = validation(model, valid_loader, smile2bond_target, dictionary, loss_fn, n_classes)
        print(f'EPOCH : {epoch} | train_loss : {train_loss:.4f} | valid_loss : {valid_loss:.4f} | mol_f1s : {mol_f1s:.4f} | mol_prc : {mol_prc:.4f} | mol_rec : {mol_rec:.4f} | bond_f1s : {bond_f1s:.4f} | bond_prc : {bond_prc:.4f} | bond_rec : {bond_rec:.4f} | bond_auc : {bond_auc:.4f} | jac_score : {jac_score:.4f}')        
        test_loss, mol_f1s, mol_prc, mol_rec, bond_f1s, bond_prc, bond_rec, bond_auc, jac_score = validation(model, test_loader, smile2bond_target, dictionary, loss_fn, n_classes)
        print(f'EPOCH : {epoch} | train_loss : {train_loss:.4f} | test_loss : {test_loss:.4f} | mol_f1s : {mol_f1s:.4f} | mol_prc : {mol_prc:.4f} | mol_rec : {mol_rec:.4f} | bond_f1s : {bond_f1s:.4f} | bond_prc : {bond_prc:.4f} | bond_rec : {bond_rec:.4f} | bond_auc : {bond_auc:.4f} | jac_score : {jac_score:.4f}')        
        print()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cyp", type=str, default='BOM_1A2')
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--class_type", type=int, default=1)
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)    
    main(args)