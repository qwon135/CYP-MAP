import torch, argparse
import torch_geometric.transforms as T
from torch_geometric.loader import DataLoader
from matplotlib import pyplot as plt
import inspect, random, os
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem
from rdkit import Chem
from multiprocessing import Pool
from rdkit.Chem import Draw, PandasTools
from torch_geometric.loader import DataLoader
from sklearn.metrics import accuracy_score, f1_score, jaccard_score, precision_score, recall_score, roc_auc_score, average_precision_score
from modules.som_dataset import CustomDataset, get_class_weight
from modules.som_models import  GNNSOM
from sklearn.model_selection import train_test_split
import warnings, json
from torch import nn
from utils import validation, to_matrix
from segmentation_models_pytorch.losses import FocalLoss, DiceLoss
from torch_ema import ExponentialMovingAverage
from tabulate import tabulate

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')

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

def to_csv(path, table_data, headers):
    score_logs = tabulate(table_data, headers, tablefmt="tsv")
    text_file=open(path,"w")
    text_file.write(score_logs)
    text_file.close()
    df = pd.read_csv(path, sep='\t')
    df.to_csv(path, index=None)

def get_logs(scores, cyp_list, args):        
    logs = ''
    table_data1 = []
    table_data2 = []
    table_data3 = []
    table_data4 = []
    for cyp in cyp_list:        
        headers1 = ['CYP', 
                   'auc_subs', 'apc_subs', 'f1s_subs', 'n_subs',
                   'subs_loss', 'bond_loss', 'atom_loss',
                    'clv_loss', 'oxi_loss', 'hdx_loss', 'rdc_loss',  'spn_loss'
                   ]
        
        headers2 = ['CYP',
                    'jac_bond', 'f1s_bond', 'apc_bond', 'n_bond',
                    'jac_spn', 'f1s_spn', 'apc_spn', 'n_spn',                    
                    'jac_som', 'f1s_som', 'apc_som', 'n_som',
                    ]   
        headers3 = ['CYP',                     
                    'jac_hdx', 'f1s_hdx', 'apc_hdx', 'n_hdx',
                    'jac_oxi', 'f1s_oxi', 'apc_oxi',  'n_oxi', 
                    'jac_clv', 'f1s_clv', 'apc_clv',  'n_clv',                    
                    'jac_rdc', 'f1s_rdc', 'apc_rdc',  'n_rdc', 
                    
                    ]       
        headers4 = ['CYP',
                    'n_subs', 'n_bond', 'n_spn', 'n_som', 'n_hdx', 'n_oxi', 'n_clv', 'n_rdc'
                    ]          
        row1, row2, row3, row4 = [cyp], [cyp], [cyp], [cyp]
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

        for header in headers4[1:]:
            if 'loss' in header or header[:2] == 'n_':
                row4.append(scores[cyp][header])
            else:
                row4.append(scores[cyp][args.th][header])

        table_data1.append(row1)
        table_data2.append(row2)
        table_data3.append(row3)        
        table_data4.append(row4)        

    logs += (tabulate(table_data1, headers1, tablefmt="grid", floatfmt=".4f") + '\n')
    logs += (tabulate(table_data2, headers2, tablefmt="grid", floatfmt=".4f") + '\n')
    logs += (tabulate(table_data3, headers3, tablefmt="grid", floatfmt=".4f") + '\n')
    logs += tabulate(table_data4, headers4, tablefmt="grid", floatfmt=".4f")

    return logs


def main(args):
    seed_everything(args.seed)
    device = args.device    
    class_type = args.class_type
    for arg, value in vars(args).items():
        print(f"{arg}: {value}")
    if args.class_type == 3:
        args.n_classes = 1
        n_classes = 1
    elif args.class_type == 2:
        args.n_classes = 4
        n_classes = 4
    elif args.class_type == 1:
        n_classes = 5
        args.n_classes = 5
    cyp = args.cyp
    cyp_list = [f'BOM_{i}'.replace(f'BOM_CYP_REACTION', 'CYP_REACTION') for i in args.cyp_list.split()]
    
    test_df = PandasTools.LoadSDF('../data/test_0611.sdf')
    test_df['CYP_REACTION'] = test_df.apply(CYP_REACTION, axis=1)
    test_df['POS_ID'] = 'TEST' + test_df.index.astype(str).str.zfill(4)

    if args.wo_no_id_ebomd:
        test_df = test_df[test_df['InChIKey'] != ''].reset_index(drop=True)

    df = PandasTools.LoadSDF('../data/train_nonreact_0611.sdf')
    df['CYP_REACTION'] = df.apply(CYP_REACTION, axis=1)
    df['POS_ID'] = 'TRAIN' + df.index.astype(str).str.zfill(4)

    test_dataset = CustomDataset(df=test_df, class_type=class_type, args=args, cyp_list=cyp_list, add_H=True if args.add_H else False, mode='test')
    test_loader = DataLoader(test_dataset, num_workers=2, batch_size=16, shuffle=False)
        
    model = GNNSOM(
                num_layers=args.num_layers, 
                gnn_num_layers = args.gnn_num_layers,
                pooling=args.pooling,
                dropout=args.dropout, 
                use_mamba= True if args.use_mamba else False, 
                cyp_list=cyp_list, 
                use_face = True if args.use_face else False, 
                node_attn = True if args.node_attn else False,
                face_attn = True if args.face_attn else False,
                encoder_dropout = args.encoder_dropout,
                n_classes=n_classes,
                use_som_v2=True,

                    ).to(device)
    model.load_state_dict(torch.load(args.ckpt, 'cpu'))
        
    loss_fn_bce = nn.BCEWithLogitsLoss()
    loss_fn_ce = nn.CrossEntropyLoss()

    print(args.adjust_substrate)
    test_scores = validation(model, test_loader, loss_fn_ce, loss_fn_bce, args)
    mname = args.ckpt.split('/')[-1].split('.')[0]
    best_validloss_testscores = get_logs(test_scores, cyp_list=cyp_list, args=args)
    print(best_validloss_testscores)

    metrics = [
        'auc_subs', 'apc_subs', 'f1s_subs',
        'jac_bond', 'f1s_bond', 'prc_bond', 'rec_bond', 'auc_bond', 'apc_bond',
        'jac_spn', 'f1s_spn', 'prc_spn', 'rec_spn', 'auc_spn', 'apc_spn',
        'jac_som', 'f1s_som', 'prc_som', 'rec_som', 'auc_som', 'apc_som',

        'jac_hdx', 'f1s_hdx', 'prc_hdx', 'rec_hdx', 'auc_hdx', 'apc_hdx',
        'jac_oxi', 'f1s_oxi', 'prc_oxi', 'rec_oxi', 'auc_oxi', 'apc_oxi',
        'jac_clv', 'f1s_clv', 'prc_clv', 'rec_clv', 'auc_clv', 'apc_clv',        
        'jac_rdc', 'f1s_rdc', 'apc_rdc', 'rec_rdc', 'auc_rdc', 'apc_rdc',
        
        ]
    score_df = []    
    test_scores = validation(model, test_loader, loss_fn_ce, loss_fn_bce, args)
    
    for cyp in cyp_list:    
        for metric in metrics:
            score_df.append({'cyp' : cyp, 'seed' : args.seed, 'metric' : metric, 'score' : test_scores[cyp][args.th][metric], 'threshold' : args.th})

    score_df = pd.DataFrame(score_df)
    score_df.to_csv(f'infer/{mname}.csv', index=None)    

    # with open(f'infer/{mname}.json', 'w') as file:
    #     json.dump(test_scores['mid2pred'], file)
    
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cyp", type=str, default='BOM_1A2')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n_classes", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=16)    
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gnn_num_layers", type=int, default=8)
    parser.add_argument("--gnn_lr", type=float, default=5e-5)
    parser.add_argument("--clf_lr", type=float, default=2e-4)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--encoder_dropout", type=float, default=0.0)
    parser.add_argument("--class_type", type=int, default=2)
    parser.add_argument("--wo_no_id_ebomd", type=int, default=1)
    parser.add_argument("--use_focal", type=bool, default=False)
    parser.add_argument("--use_non_reaction", type=bool, default=False)
    parser.add_argument("--substrate_th", type=float, default=0.5)
    parser.add_argument("--th", type=float, default=0.1)
    parser.add_argument("--average", type=str, default='binary')
    parser.add_argument("--ckpt", type=str, default='')
    parser.add_argument("--gnn_type", type=str, default='gnn')
    parser.add_argument("--use_mamba", type=int, default=0)
    parser.add_argument("--use_face", type=int, default=1)
    parser.add_argument("--node_attn", type=int, default=1)
    parser.add_argument("--face_attn", type=int, default=1)
    parser.add_argument("--adjust_substrate", type=int, default=0)
    parser.add_argument("--filt_som", type=int, default=0)
    parser.add_argument("--test_only_reaction_mol", type=int, default=0)
    parser.add_argument("--equivalent_mean", type=int, default=0)
    parser.add_argument("--substrate_loss_weight", type=float, default=0.33)
    parser.add_argument("--bond_loss_weight", type=float, default=0.33)
    parser.add_argument("--atom_loss_weight", type=float, default=0.33)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--add_H", type=int, default=1)
    parser.add_argument("--train_only_spn_H_atom", type=int, default=0)
    parser.add_argument("--pooling", type=str, default='sum')
    parser.add_argument("--reduction", type=str, default='sum')
    parser.add_argument("--cyp_list", type=str, default='1A2 2A6 2B6 2C8 2C9 2C19 2D6 2E1 3A4 CYP_REACTION')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)    
    main(args)    