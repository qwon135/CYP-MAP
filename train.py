import torch, argparse
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
import random, os
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from torch_geometric.loader import DataLoader
from modules.som_dataset import CustomDataset
from modules.som_models import GNNSOM
from sklearn.model_selection import train_test_split, StratifiedKFold
import warnings
from torch import nn
from utils import validation, to_matrix
# from segmentation_models_pytorch.losses import FocalLoss, DiceLoss
from torch_ema import ExponentialMovingAverage
from tabulate import tabulate
from timm.loss import BinaryCrossEntropy
warnings.filterwarnings('ignore', '')
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
torch.multiprocessing.set_sharing_strategy('file_system')

def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = False  # type: ignore

def CYP_REACTION(x):
    cyp_col = ['BOM_1A2', 'BOM_2A6', 'BOM_2B6', 'BOM_2C8', 'BOM_2C9', 'BOM_2C19', 'BOM_2D6', 'BOM_2E1', 'BOM_3A4',]
    cyp_reactions = x[cyp_col].tolist()
    cyp_reactions = [i for i in cyp_reactions if i] 
    return '\n'.join( cyp_reactions )

def get_logs(epoch, train_loss, scores, cyp_list, args, mode='Valid'):
    logs = f'Epoch : {epoch} | Train Loss : {train_loss:.4f} | {mode} Loss : {scores["total_loss"]:.4f} {scores["valid_loss"]:.4f}\n'
    
    table_data1 = []
    table_data2 = []
    table_data3 = []

    for cyp in cyp_list:        
        headers1 = ['CYP', 
                #    'auc_subs', 'apc_subs', 'f1s_subs', 'n_subs',
                   'subs_loss', 'bond_som_loss', 'atom_spn_loss',
                   'dea_loss', 'epo_loss', 'oxi_loss', 'dha_loss', 'dhy_loss', 'rdc_loss'
                   ]
        
        headers2 = ['CYP',
                    'jac_bond_som', 'f1s_bond_som', 'apc_bond_som', #'n_bond_som',
                    'jac_atom_spn', 'f1s_atom_spn', 'apc_atom_spn', #'n_atom_spn',                    
                    'jac_som', 'f1s_som', 'apc_som', #'n_som',
                    'auc_subs', 'apc_subs', 'f1s_subs',
                    ]   
        headers3 = ['CYP', 
                    'jac_dea', 'f1s_dea', 'apc_dea',# 'n_dea',
                    'jac_epo', 'f1s_epo', 'apc_epo',#  'n_epo',
                    'jac_oxi', 'f1s_oxi', 'apc_oxi',#  'n_oxi',
                    'jac_dha', 'f1s_dha', 'apc_dha',#  'n_dha',
                    'jac_dhy', 'f1s_dhy', 'apc_dhy',#  'n_dhy',
                    'jac_rdc', 'f1s_rdc', 'apc_rdc',#  'n_rdc',

                    ]                            
        row1, row2, row3 = [cyp], [cyp], [cyp]
        for header in headers1[1:]:
            if 'loss' in header or header[:2] == 'n_':
                row1.append(scores[cyp][header])
            else:
                row1.append(scores[cyp][args.th][header])
                
                
        for header in headers2[1:]:
            if 'loss' in header or header[:2] == 'n_':
                row2.append(scores[cyp][header])
            else:
                row2.append(scores[cyp][args.th][header])
        
        for header in headers3[1:]:
            if 'loss' in header or header[:2] == 'n_':
                row3.append(scores[cyp][header])
            else:
                row3.append(scores[cyp][args.th][header])


        # table_data1.append(row1)
        table_data2.append(row2)
        table_data3.append(row3)        

    # logs += (tabulate(table_data1, headers1, tablefmt="grid", floatfmt=".4f") + '\n')
    logs += (tabulate(table_data2, headers2, tablefmt="grid", floatfmt=".4f") + '\n')
    logs += tabulate(table_data3, headers3, tablefmt="grid", floatfmt=".4f")
    
    return logs
def cosine_scheduler(base_value, final_value, epochs):

    iters = np.arange(epochs)
    schedule = final_value + 0.5 * (base_value - final_value) * (1 + np.cos(np.pi * iters / len(iters)))
    
    return schedule  

def upscaling(df, cyp_list):
    n_data = df.shape[0]
    up_sample_df = []
    for cyp in cyp_list:
        if cyp == 'CYP_REACTION':continue
        df_reaction = df[df[cyp] != ''].reset_index(drop=True)
        df_non_reaction = df[df[cyp] == ''].reset_index(drop=True)

        df_reaction = df_reaction.sample((n_data//2) // len(cyp_list), replace=True).reset_index(drop=True)
        df_non_reaction = df_non_reaction.sample((n_data//2) // len(cyp_list)).reset_index(drop=True)

        up_sample_df.append(df_reaction)
        up_sample_df.append(df_non_reaction)

    up_sample_df = pd.concat(up_sample_df).reset_index(drop=True)

    pos_id_counts = up_sample_df['POS_ID'].value_counts()
    pos_id_counts = pos_id_counts[pos_id_counts <= 3]
    up_sample_df = up_sample_df[up_sample_df['POS_ID'].isin(pos_id_counts.index)]

    print('Upsample ratio!')
    for cyp in cyp_list:
        print(f'{cyp} :({df[df[cyp] != ""].shape[0]} -> {up_sample_df[up_sample_df[cyp] != ""].shape[0]}) / {up_sample_df.shape[0]}')
    print()

    return up_sample_df

def upscaling_v2(df, cyp_list):
    n_data = df.shape[0]
    non_reaction_df = df[df['CYP_REACTION'] == ''].reset_index(drop=True)
    up_sample_df = [non_reaction_df.sample((non_reaction_df.shape[0]//2), replace=True).reset_index(drop=True)]
    for cyp in cyp_list:
        if cyp == 'CYP_REACTION':continue
        df_reaction = df[df[cyp] != ''].reset_index(drop=True)        
        df_reaction = df_reaction.sample((n_data//2) // len(cyp_list), replace=True).reset_index(drop=True)
        up_sample_df.append(df_reaction)
        

    up_sample_df = pd.concat(up_sample_df).reset_index(drop=True)

    pos_id_counts = up_sample_df['POS_ID'].value_counts()
    pos_id_counts = pos_id_counts[pos_id_counts <= 3]
    up_sample_df = up_sample_df[up_sample_df['POS_ID'].isin(pos_id_counts.index)].reset_index(drop=True)

    print('Upsample ratio!')
    for cyp in cyp_list:
        print(f'{cyp} :({df[df[cyp] != ""].shape[0]} -> {up_sample_df[up_sample_df[cyp] != ""].shape[0]}) / ({df.shape[0]} -> {up_sample_df.shape[0]})')
    print()

    return up_sample_df

def upscaling_v3(df, cyp_list, ratio):
    n_data = df.shape[0]
    reaction_df = df[df['CYP_REACTION'] != ''].reset_index(drop=True)
    non_reaction_df = df[df['CYP_REACTION'] == ''].reset_index(drop=True)
    non_reaction_df = non_reaction_df.sample(int(non_reaction_df.shape[0] * ratio), replace=True).reset_index(drop=True)
    
    up_sample_df = pd.concat([reaction_df, non_reaction_df]).reset_index(drop=True)

    print('Upsample ratio!')
    for cyp in cyp_list:
        print(f'{cyp} :({df[df[cyp] != ""].shape[0]} -> {up_sample_df[up_sample_df[cyp] != ""].shape[0]}) / ({df.shape[0]} -> {up_sample_df.shape[0]})')
    print()

    return up_sample_df


def main(args):
    if not os.path.exists('log_text'):os.mkdir('log_text')
    if not os.path.exists('ckpt'):os.mkdir('ckpt')
    seed_everything(args.seed)
    device = args.device    

    cyp_list = [f'BOM_{i}'.replace(f'BOM_CYP_REACTION', 'CYP_REACTION') for i in args.cyp_list.split()]

    if args.train_with_non_reaction:
        print(f'load train_nonreact_0628.sdf!')
        df = PandasTools.LoadSDF('data/train_nonreact_0628.sdf')
    else:
        print(f'load train_0628.sdf!')
        df = PandasTools.LoadSDF('data/train_0628.sdf')

    test_df = PandasTools.LoadSDF('data/test_0628.sdf')
    
    df['CYP_REACTION'], test_df['CYP_REACTION'] = df.apply(CYP_REACTION, axis=1), test_df.apply(CYP_REACTION, axis=1)    

    df['POS_ID'], test_df['POS_ID'] = 'TRAIN' + df.index.astype(str).str.zfill(4), 'TEST' + test_df.index.astype(str).str.zfill(4)
    df['is_react'] = (df['CYP_REACTION'] == '').astype(int).astype(str)

    train_df, valid_df = train_test_split(df, stratify=df['is_react'], random_state=args.seed, test_size=0.2)
    train_df, valid_df = train_df.reset_index(drop=True), valid_df.reset_index(drop=True)    

    print('Reaction Ratio data : ', df[df['CYP_REACTION'] != ''].shape[0], '/', df.shape[0])
    print('Reaction Ratio train data : ', train_df[train_df['CYP_REACTION'] != ''].shape[0], '/', train_df.shape[0])
    
    if args.filt_decoy:
        remove_decoy_ids_60 = pd.read_csv('data/remove_decoy_ids_85.csv', index_col=None)
        remove_ids = remove_decoy_ids_60['remove_ids'].tolist()
        n_train, n_valid = train_df.shape[0], valid_df.shape[0]
        train_df = train_df[~train_df['ID'].isin(remove_ids)].reset_index(drop=True)
        valid_df = valid_df[~valid_df['ID'].isin(remove_ids)].reset_index(drop=True)

        print(f'Filt Decoy ! Train {n_train} -> {train_df.shape[0]}, Valid {n_valid} -> {valid_df.shape[0]}')

    if not args.upscaling:        
        train_dataset = CustomDataset(df=train_df,  args=args, cyp_list=cyp_list, mode='train')
        train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)

    valid_dataset = CustomDataset(df=valid_df, args=args, cyp_list=cyp_list, mode='test')
    valid_loader = DataLoader(valid_dataset, num_workers=8, batch_size=args.batch_size, shuffle=False)

    test_dataset = CustomDataset(df=test_df, args=args, cyp_list=cyp_list, mode='test')
    test_loader = DataLoader(test_dataset, num_workers=8, batch_size=args.batch_size, shuffle=False)

    epochs = args.epochs
    model = GNNSOM(
                num_layers=args.num_layers, 
                gnn_num_layers = args.gnn_num_layers,
                pooling=args.pooling,
                dropout=args.dropout, 
                dropout_fc = args.dropout_fc,
                dropout_som_fc=args.dropout_som_fc,
                dropout_type_fc=args.dropout_type_fc,                
                cyp_list=cyp_list, 
                use_face = True if args.use_face else False, 
                node_attn = True if args.node_attn else False,
                face_attn = True if args.face_attn else False,
                encoder_dropout = args.encoder_dropout,                                
                    ).to(device)     
    if args.pretrain:
        state_dict=  torch.load(args.pretrain, map_location='cpu')
        if 'gnn_state_dict' in state_dict.keys():
            state_dict = state_dict['gnn_state_dict']
        e = model.gnn.load_state_dict(state_dict, strict=False)
        
        print(e)
    loss_fn_ce = nn.CrossEntropyLoss(reduction=args.reduction)
    # loss_fn_bce = nn.BCEWithLogitsLoss(reduction=args.reduction, pos_weight=torch.FloatTensor([1.5]).to(args.device))    
    loss_fn_bce = nn.BCEWithLogitsLoss(reduction=args.reduction)    

    param_groups = [
        {"params": [], "lr": args.gnn_lr}, # model.convs의 매개변수들, 학습률 args.gnn_lr
        {"params": [], "lr": args.clf_lr},  # 나머지 매개변수들, 학습률 args.clf_lr
        {"params": [], "lr": args.clf_lr}  # 나머지 매개변수들, 학습률 args.clf_lr
    ]    
    for name, param in model.named_parameters():
        if  'gnn' in name:
            param_groups[0]["params"].append(param) # convs가 이름에 포함되면 첫번째 그룹에 추가
        elif 'substrate_fc' in name:
            param_groups[1]["params"].append(param) # convs가 이름에 포함되면 첫번째 그룹에 추가
        else:
            param_groups[2]["params"].append(param) # 아니면 두번째 그룹에 추가

    if args.optim == 'adamw':
        optim = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    elif args.optim == 'adam':
        optim = torch.optim.Adam(param_groups, weight_decay=args.weight_decay)

    if args.warmup:
        print('do warmup!')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optim, T_0=12, T_mult=1, verbose=False)
    else:
        print('not warmup!')
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=args.epochs, verbose=False)
    
    ema = ExponentialMovingAverage(model.parameters(), decay=args.ema_decay)

    best_loss = 1e6
    patience = args.patience
    best_validloss_testscores=None
    loss_df = []
    
    log_path = f'log_text/seed{str(args.seed).zfill(2)}_{args.save_name}.txt'
    if os.path.exists(log_path):os.remove(log_path)
    with open(log_path, 'a') as f:
        f.write(f"CONFIG : \n")    
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
        with open(log_path, 'a') as f:
            f.write(f"{arg}: {value}\n")
    for epoch in range(epochs):
        if args.upscaling:            
            up_train_df = upscaling_v2(train_df, cyp_list)            

            # train_dataset = CustomDataset(df=up_train_df,  class_type=class_type, args=args, cyp_list=cyp_list, mode='train')
            train_dataset = CustomDataset(df=up_train_df,  args=args, cyp_list=cyp_list, mode='train')
            train_loader = DataLoader(train_dataset, num_workers=8, batch_size=args.batch_size, shuffle=True, drop_last=True)
        
        train_loss = 0
        model.train()
        for step, batch in enumerate(train_loader):         
            batch = batch.to(device)

            optim.zero_grad()
            
            _, loss_dict, _ = model.forward_with_loss(batch, loss_fn_ce, loss_fn_bce, device, args)            
            
            loss = loss_dict['total_loss']
            loss.backward()
            if args.grad_norm:
                grad_=torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_norm)
            
            optim.step()   
            ema.update()
    
            train_loss += loss.cpu().item()
        train_loss /= len(train_loader)
        val_scores = validation(model, valid_loader, loss_fn_ce, loss_fn_bce, args)        
        val_loss_dict = {'epoch' : epoch}
        for cyp in cyp_list:
            for task in ['subs', 'bond_som', 'atom_som',  'atom_spn', 'dea', 'epo','oxi', 'dha', 'dhy', 'rdc']:
                val_loss_dict[f'{cyp}_{task}_loss'] = val_scores[f'{cyp}_{task}_loss'].cpu().item()

        loss_df.append(val_loss_dict)        
        if args.save_name:
            pd.DataFrame(loss_df).to_csv(f'log_text/loss_{args.save_name}.csv', index=None)
        else:
            pd.DataFrame(loss_df).to_csv(f'log_text/loss_seed{args.seed}.csv', index=None)
            
        val_logs = get_logs(epoch, train_loss, val_scores, cyp_list, args, mode='Valid', )        

        with open(log_path, 'a') as f:
            f.write(val_logs + '\n\n')
        
        print(val_logs)

        print()
        if val_scores['valid_loss'] < best_loss:
            best_loss = val_scores['valid_loss']
            if args.save_name:
                torch.save(model.state_dict(), f'ckpt/{args.seed}_{args.save_name}.pt')
            else:
                torch.save(model.state_dict(), f'ckpt/{args.seed}.pt')

            test_scores = validation(model, test_loader, loss_fn_ce, loss_fn_bce, args)
            best_validloss_testscores = get_logs(epoch, train_loss, test_scores, cyp_list, args, mode='Test')

            patience = args.patience
            if args.print_test_every:
                print(best_validloss_testscores)
                with open(log_path, 'a') as f:
                    print('!!!!Test!!!!\n\n')
                    f.write('!!!!Test!!!!\n\n')
                    f.write(best_validloss_testscores + '\n\n')            
        else:
            patience -= 1
        if not patience:
            print('Early stop')
            break
        
        scheduler.step()
    print('best_validloss_testscores !')
    print(best_validloss_testscores)
    with open(log_path, 'a') as f:
        f.write(best_validloss_testscores)

    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gnn_num_layers", type=int, default=8)
    parser.add_argument("--gnn_lr", type=float, default=4e-5)
    parser.add_argument("--clf_lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-3)
    parser.add_argument("--ema_decay", type=float, default=0.995)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--dropout_fc", type=float, default=0.0)
    parser.add_argument("--dropout_som_fc", type=float, default=0.0)
    parser.add_argument("--dropout_type_fc", type=float, default=0.0)
    parser.add_argument("--encoder_dropout", type=float, default=0.0)
    parser.add_argument("--class_type", type=int, default=2)
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--upscaling", type=int, default=1)
    parser.add_argument("--use_focal", type=int, default=0)
    parser.add_argument("--train_with_non_reaction", type=int, default=1)
    parser.add_argument("--test_only_reaction_mol", type=int, default=0)
    parser.add_argument("--add_equivalent", type=bool, default=True)
    parser.add_argument("--substrate_th", type=float, default=0.5)
    parser.add_argument("--save_name", type=str, default='')
    parser.add_argument("--th", type=float, default=0.15)
    parser.add_argument("--adjust_substrate", type=int, default=0)
    parser.add_argument("--filt_som", type=int, default=0)
    parser.add_argument("--gnn_type", type=str, default='gnn')
    parser.add_argument("--pretrain", type=str, default='pretrain/ckpt_pretrain/gnn_pretrain.pt')
    parser.add_argument("--optim", type=str, default='adamw')
    parser.add_argument("--use_face", type=int, default=1)
    parser.add_argument("--node_attn", type=int, default=1)
    parser.add_argument("--face_attn", type=int, default=1)
    parser.add_argument("--grad_norm", type=int, default=50)
    parser.add_argument("--substrate_loss_weight", type=float, default=0.05)
    parser.add_argument("--bond_loss_weight", type=float, default=1.0)
    parser.add_argument("--atom_loss_weight", type=float, default=1.0)
    parser.add_argument("--som_type_loss_weight", type=float, default=1.0)
    parser.add_argument("--equivalent_mean", type=int, default=0)
    parser.add_argument("--average", type=str, default='binary')
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--cyp_list", type=str, default='1A2 2A6 2B6 2C8 2C9 2C19 2D6 2E1 3A4 CYP_REACTION')
    parser.add_argument("--pooling", type=str, default='sum')    
    parser.add_argument("--train_add_H_random", type=int, default=0)
    parser.add_argument("--add_H", type=int, default=1)
    parser.add_argument("--drop_node_p", type=float, default=0.0)
    parser.add_argument("--mask_node_p", type=float, default=0.0)
    parser.add_argument("--metric_mode", type=str, default='bond')
    parser.add_argument("--train_only_spn_H_atom", type=int, default=0)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--print_test_every", type=int, default=1)
    parser.add_argument("--filt_decoy", type=int, default=0)
    parser.add_argument("--reduction", type=str, default='mean')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)
    main(args)