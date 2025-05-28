import torch, argparse
from torch_geometric.loader import DataLoader
import  random, os, time
import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from torch_geometric.loader import DataLoader
from modules.som_models import CYPMAP_GNN
import warnings
from torch import nn
from utils import validation
from tabulate import tabulate
from modules.som_dataset import CustomDataset
warnings.filterwarnings('ignore', '')
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
                   'subs_loss', 'bond_som_loss', 'atom_spn_loss',
                   'oxc_loss', 'oxi_loss', 'epo_loss', 'sut_loss', 'dhy_loss', 'hys_loss', 'rdc_loss'
                   ]
        
        headers2 = ['CYP',
                    'jac_bond_som', 'f1s_bond_som', 'apc_bond_som', 'n_bond_som',
                    'jac_atom_spn', 'f1s_atom_spn', 'apc_atom_spn', 'n_atom_spn',             
                    'jac_som', 'f1s_som', 'apc_som', 'n_som',
                    ]   
        headers3 = ['CYP', 
                    'jac_oxc', 'f1s_oxc', 'apc_oxc', 'n_oxc',
                    'jac_oxi', 'f1s_oxi', 'apc_oxi', 'n_oxi',
                    'jac_epo', 'f1s_epo', 'apc_epo', 'n_epo',
                    'jac_sut', 'f1s_sut', 'apc_sut', 'n_sut',
                    'jac_dhy', 'f1s_dhy', 'apc_dhy', 'n_dhy',
                    'jac_hys', 'f1s_hys', 'apc_hys', 'n_hys',
                    'jac_rdc', 'f1s_rdc', 'apc_rdc', 'n_rdc',

                    ]       
        headers4 = ['CYP',
                    'n_subs', 'n_bond_som', 'n_atom_spn', 'n_som', 'n_oxc', 'n_oxi', 'n_epo', 'n_sut', 'n_dhy', 'n_hys', 'n_rdc',
                    ]          
        row1, row2, row3, row4 = [cyp], [cyp], [cyp], [cyp]
        for header in headers1[1:]:
            if 'loss' in header or header[:2] == 'n_':row1.append(scores[cyp][header])
            else:row1.append(scores[cyp][args.th][header])
                                
        for header in headers2[1:]:
            if 'loss' in header or header[:2] == 'n_':row2.append(scores[cyp][header])
            else:row2.append(scores[cyp][args.th][header])
        
        for header in headers3[1:]:
            if 'loss' in header or header[:2] == 'n_':row3.append(scores[cyp][header])
            else:row3.append(scores[cyp][args.th][header])

        for header in headers4[1:]:
            if 'loss' in header or header[:2] == 'n_':row4.append(scores[cyp][header])
            else:row4.append(scores[cyp][args.th][header])

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
    if not os.path.exists('infer'):os.mkdir('infer')
    seed_everything(args.seed)
    device = args.device    

    cyp = args.cyp
    cyp_list = [f'BOM_{i}'.replace(f'BOM_CYP_REACTION', 'CYP_REACTION') for i in args.cyp_list.split()]
    
    
    test_df = PandasTools.LoadSDF('data/cyp_map_test.sdf')
    test_df['CYP_REACTION'] = test_df.apply(CYP_REACTION, axis=1)
    test_df['POS_ID'] = 'TEST' + test_df.index.astype(str).str.zfill(4)
    if args.demo:
        test_df = test_df.sample(1000, replace=True).reset_index(drop=True)
    
    args.pos_weight = torch.tensor([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])

    test_dataset = CustomDataset(df=test_df, args=args, cyp_list=cyp_list, mode='test')
    test_loader = DataLoader(test_dataset, num_workers=2, batch_size=16, shuffle=False)
    
    model = CYPMAP_GNN(
                num_layers=args.num_layers,
                gnn_num_layers = args.gnn_num_layers,
                pooling=args.pooling,
                dropout=args.dropout,
                cyp_list=cyp_list, 
                use_face = True if args.use_face else False, 
                node_attn = True if args.node_attn else False,
                face_attn = True if args.face_attn else False,
                encoder_dropout = args.encoder_dropout,
                    ).to(device)
    
    if not args.demo and args.ckpt:
        model.load_state_dict(torch.load(args.ckpt, 'cpu'))
        
    loss_fn_bce = nn.BCEWithLogitsLoss()
    loss_fn_ce = nn.CrossEntropyLoss()
    
    start_time = time.time()
    test_scores = validation(model, test_loader, loss_fn_ce, loss_fn_bce, args)

    end_time = time.time()
    elapsed_time = end_time - start_time

    mname = args.ckpt.split('/')[-1].split('.')[0]
    best_validloss_testscores = get_logs(test_scores, cyp_list=cyp_list, args=args)
    if not args.demo:
        print(best_validloss_testscores)

    metrics = [
        'auc_subs', 'apc_subs', 'f1s_subs', 'rec_subs', 'prc_subs',
        'jac_bond_som', 'f1s_bond_som', 'prc_bond_som', 'rec_bond_som', 'auc_bond_som', 'apc_bond_som',
        'jac_atom_spn', 'f1s_atom_spn', 'prc_atom_spn', 'rec_atom_spn', 'auc_atom_spn', 'apc_atom_spn',
        
        'jac_oxc', 'f1s_oxc', 'prc_oxc', 'rec_oxc', 'auc_oxc', 'apc_oxc',
        'jac_oxi', 'f1s_oxi', 'prc_oxi', 'rec_oxi', 'auc_oxi', 'apc_oxi',
        'jac_epo', 'f1s_epo', 'prc_epo', 'rec_epo', 'auc_epo', 'apc_epo',
        'jac_sut', 'f1s_sut', 'prc_sut', 'rec_sut', 'auc_sut', 'apc_sut',
        'jac_dhy', 'f1s_dhy', 'prc_dhy', 'rec_dhy', 'auc_dhy', 'apc_dhy',
        'jac_hys', 'f1s_hys', 'prc_hys', 'rec_hys', 'auc_hys', 'apc_hys',
        'jac_rdc', 'f1s_rdc', 'prc_rdc', 'rec_rdc', 'auc_rdc', 'apc_rdc',

        'jac_som', 'f1s_som', 'prc_som', 'rec_som', 'auc_som', 'apc_som',
        ]

    score_df = []    
    
    for cyp in cyp_list:    
        for metric in metrics:
            score_df.append({'cyp' : cyp, 'seed' : args.seed, 'metric' : metric, 'score' : test_scores[cyp][args.th][metric], 'threshold' : args.th})

    score_df = pd.DataFrame(score_df)
    score_df.to_csv(f'infer/{mname}.csv', index=None)        

    with open(f'infer/{mname}.txt', 'w') as f:
        f.write(best_validloss_testscores)
    
    print(f"Inference on {len(test_loader.dataset)} molecules took {elapsed_time:.2f} seconds.")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cyp", type=str, default='BOM_1A2')
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=1)    
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--gnn_num_layers", type=int, default=8)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--encoder_dropout", type=float, default=0.0)
    parser.add_argument("--class_type", type=int, default=2)
    parser.add_argument("--use_focal", type=bool, default=False)
    parser.add_argument("--use_non_reaction", type=bool, default=False)
    parser.add_argument("--substrate_th", type=float, default=0.5)
    parser.add_argument("--th", type=float, default=0.1)
    parser.add_argument("--average", type=str, default='binary')
    parser.add_argument("--ckpt", type=str, default='')
    parser.add_argument("--gnn_type", type=str, default='gnn')    
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
    parser.add_argument("--som_type_loss_weight", type=float, default=1.0)
    parser.add_argument("--device", type=str, default='cuda:0')
    parser.add_argument("--metric_mode", type=str, default='bond')
    parser.add_argument("--add_H", type=int, default=1)
    parser.add_argument("--train_only_spn_H_atom", type=int, default=0)
    parser.add_argument("--pooling", type=str, default='sum')
    parser.add_argument("--reduction", type=str, default='mean')
    parser.add_argument("--cyp_list", type=str, default='1A2 2A6 2B6 2C8 2C9 2C19 2D6 2E1 3A4 CYP_REACTION')
    parser.add_argument('--demo', action='store_true', help='run in demo mode')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    seed_everything(args.seed)    
    main(args)    