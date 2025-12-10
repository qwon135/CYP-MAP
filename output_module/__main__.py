import torch
from torch_geometric.loader import DataLoader
import numpy as np
from multiprocessing import Pool
from rdkit.Chem import Draw, PandasTools
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(os.path.dirname(current_dir))
sys.path.append(parent_dir)
from modules.som_dataset import CustomDataset
from modules.som_models import CYPMAP_GNN
from utils import validation
import datetime
import warnings
import argparse
from image_output import *
from process_metabolite import *
import json

warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')
warnings.filterwarnings('ignore', '')


def sort_atoms_by_canonical_rank(mol):
    ranks = Chem.CanonicalRankAtoms(mol)
    atom_order = sorted(range(len(ranks)), key=lambda x: ranks[x])
    sorted_mol = Chem.RenumberAtoms(mol, atom_order)
    return sorted_mol


def sort_atoms_by_atomic_number(mol):
    mol = sort_atoms_by_canonical_rank(mol)
    atom_order = sorted(range(mol.GetNumAtoms()), key=lambda i: mol.GetAtomWithIdx(i).GetAtomicNum(), reverse=True)
    sorted_mol = Chem.RenumberAtoms(mol, atom_order)
    return sorted_mol


def load_sdf_file(file_path):
    ro_mols = []
    sanitized_mols = []
    suppl = Chem.SDMolSupplier(file_path)
    for x in suppl:
        if x is not None:
            x = sort_atoms_by_atomic_number(x)
            ro_mols.append(x)
            try:
                x_with_h = Chem.AddHs(x, addCoords=True)
                Chem.SanitizeMol(x_with_h)
                for idx, atom in enumerate(x_with_h.GetAtoms()):
                    atom.SetAtomMapNum(idx + 1)
                sanitized_mols.append(x_with_h)
            except:
                sanitized_mols.append('')

    df = PandasTools.LoadSDF(file_path)
    df['ROMol'] = ro_mols
    df['Molecules'] = sanitized_mols
    return df


def process_single_smiles(smiles):
    sanitized_mol = None
    mol = sort_atoms_by_atomic_number(Chem.MolFromSmiles(smiles))
    if mol:
        try:
            mol_with_h = Chem.AddHs(mol, addCoords=True)
            Chem.SanitizeMol(mol_with_h)
            for idx, atom in enumerate(mol_with_h.GetAtoms()):
                atom.SetAtomMapNum(idx + 1)
            sanitized_mol = mol_with_h
        except:
            sanitized_mol = None
    return pd.DataFrame({'SMILES': [smiles], 'Molecules': [sanitized_mol], 'ROMol': [mol]})


def CYP_REACTION(x):
    cyp_col = ['BOM_1A2', 'BOM_2A6', 'BOM_2B6', 'BOM_2C8', 'BOM_2C9', 'BOM_2C19', 'BOM_2D6', 'BOM_2E1', 'BOM_3A4']
    cyp_reactions = x[cyp_col].tolist()
    cyp_reactions = [i for i in cyp_reactions if i]
    return '\n'.join(cyp_reactions)


cyp_list = [f'BOM_{i}' for i in '1A2 2A6 2B6 2C8 2C9 2C19 2D6 2E1 3A4'.split()] + ['CYP_REACTION']


class CONFIG:
    substrate_loss_weight = 0.33
    bond_loss_weight = 0.33
    atom_loss_weight = 0.33
    pos_weight = torch.ones(10)
    som_type_loss_weight = 0.33
    class_type = 2
    th = 0.1
    substrate_th = 0.5
    adjust_substrate = False
    test_only_reaction_mol = False
    equivalent_bonds_mean = False
    average = 'binary'
    device = 'cpu'
    reduction = 'sum'
    metric_mode = 'bond'
    n_classes = 5


config = CONFIG()


def run_model_validation(model_path, test_df):
    model = CYPMAP_GNN(
        num_layers=2,
        gnn_num_layers=8,
        pooling='sum',
        dropout=0.1,
        cyp_list=cyp_list,
        use_face=True,
        node_attn=True,
        face_attn=True,
        n_classes=config.n_classes,
    ).to(config.device)

    test_df['CYP_REACTION'] = test_df.apply(CYP_REACTION, axis=1)

    test_dataset = CustomDataset(df=test_df, cyp_list=cyp_list, args=config)
    test_loader = DataLoader(test_dataset, num_workers=2, batch_size=8, shuffle=False)

    loss_fn_ce, loss_fn_bce = torch.nn.CrossEntropyLoss(), torch.nn.BCEWithLogitsLoss()

    config.ckpt = model_path  # Load the model checkpoint
    model.load_state_dict(torch.load(config.ckpt, 'cpu'))
    test_scores = validation(model, test_loader, loss_fn_ce, loss_fn_bce, config)
    validator = test_scores['validator']
    validator.unbatch()

    def collect_validator_data(validator, test_df):
        validator_data = {}

        for key in validator.y_prob.keys():
            validator_data[key] = {}
            if key not in validator.y_prob_unbatch:
                for sub_key in validator.y_prob[key].keys():
                    y_prob_data = validator.y_prob[key][sub_key]
                    data_list = []

                    for idx in range(test_df.shape[0]):
                        test_id = test_df.loc[idx, 'POS_ID']
                        y_prob_mol = y_prob_data[idx]

                        data_list.append({
                            'id': test_id,
                            'y_prob': y_prob_mol,
                        })

                    validator_data[key][sub_key] = data_list

            else:
                for sub_key in validator.y_prob[key].keys():
                    if sub_key not in validator.y_prob_unbatch[key]:
                        continue

                    y_prob_data = validator.y_prob_unbatch[key][sub_key]
                    data_list = []

                    for idx in range(test_df.shape[0]):
                        test_id = test_df.loc[idx, 'POS_ID']
                        y_prob_mol = y_prob_data[idx]

                        data_list.append({
                            'id': test_id,
                            'y_prob': y_prob_mol.tolist(),
                            'len_y_prob': len(y_prob_mol),
                        })

                    validator_data[key][sub_key] = data_list

        return validator_data

    validator_data = collect_validator_data(validator, test_df)
    return validator_data


def calculate_max_SOM_data(data):
    max_data = {}

    for key in data.keys():
        max_data[key] = {}
        for sub_key in data[key].keys():
            max_data[key]['max'] = [{'y_prob': []} for _ in range(len(data[key][sub_key]))]

    for key in data.keys():
        for sub_key in data[key].keys():
            if sub_key != 'CYP_REACTION':
                for i in range(len(data[key][sub_key])):
                    max_data[key]['max'][i]['y_prob'].append(data[key][sub_key][i]['y_prob'])

    for key in max_data.keys():
        for sub_key in max_data[key].keys():
            for i in range(len(max_data[key][sub_key])):
                y_prob_list = np.array(max_data[key][sub_key][i]['y_prob'])
                max_data[key][sub_key][i]['y_prob'] = np.max(y_prob_list, axis=0).tolist()

    return max_data


def calculate_mean_data(data_list):
    mean_data = {}

    for key in data_list[0].keys():
        mean_data[key] = {}
        for sub_key in data_list[0][key].keys():
            mean_data[key][sub_key] = [{'y_prob': []} for _ in range(len(data_list[0][key][sub_key]))]

    for data in data_list:
        for key in data.keys():
            for key in data.keys():
                for sub_key in data[key].keys():
                    for i in range(len(data[key][sub_key])):
                        mean_data[key][sub_key][i]['y_prob'].append(data[key][sub_key][i]['y_prob'])

    for key in mean_data.keys():
        for sub_key in mean_data[key].keys():
            for i in range(len(mean_data[key][sub_key])):
                y_prob_list = np.array(mean_data[key][sub_key][i]['y_prob'])

                mean_data[key][sub_key][i]['y_prob'] = np.mean(y_prob_list, axis=0).tolist()

    max_data = calculate_max_SOM_data(mean_data)
    for key in mean_data.keys():
        mean_data[key]['max'] = {}
        mean_data[key]['max'] = max_data[key]['max']

    return mean_data


def main(smiles=None, sdf=None, subtype='all', base_dir='./output', output_type='default',
         mode='default', custom_threshold=0.5):  # custom_threshold 인자 추가
    if sdf:
        test_df = load_sdf_file(sdf)
    if smiles:
        test_df = process_single_smiles(smiles)

    test_df['POS_ID'] = 'MOL' + test_df.index.astype(str).str.zfill(4)
    name_dict = {'1A2': 'BOM_1A2', '2A6': 'BOM_2A6', '2B6': 'BOM_2B6', '2C8': 'BOM_2C8', '2C9': 'BOM_2C9',
                 '2C19': 'BOM_2C19',
                 '2D6': 'BOM_2D6', '2E1': 'BOM_2E1', '3A4': 'BOM_3A4', 'all': 'CYP_REACTION', 'sub9': 'max'}

    subtype = name_dict[subtype]
    cyp_list = [f'BOM_{i}' for i in '1A2 2A6 2B6 2C8 2C9 2C19 2D6 2E1 3A4'.split()] + ['CYP_REACTION']
    for column in cyp_list:
        if column not in test_df.columns:
            test_df[column] = None

    models = sorted([os.path.join('ckpt', f) for f in os.listdir('ckpt') if f.endswith('.pt')])
    reaction_rules = pd.read_csv('./output_module/reaction_rules/cyp_map_rules.csv', encoding='cp949')
    output_data = []
    for model_path in models:
        validation_data = run_model_validation(model_path, test_df)
        output_data.append(validation_data)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    seed_mean_output = calculate_mean_data(output_data)
    image_dir_path = os.path.join(base_dir, f'image_output_{timestamp}')
    metabolite_path = os.path.join(base_dir, f'metabolite_output_{timestamp}')
    # metabolite_image_path = os.path.join(metabolite_path, 'images')
    os.makedirs(image_dir_path, exist_ok=True)

    subtypes_threshold = {
        'BOM_1A2': {'thr': 0.26, 'subthr': 0.25, 'som_thr': 0.26, 'low_thr': 0.17},
        'BOM_2A6': {'thr': 0.14, 'subthr': 0.18, 'som_thr': 0.14, 'low_thr': 0.08},
        'BOM_2B6': {'thr': 0.18, 'subthr': 0.40, 'som_thr': 0.18, 'low_thr': 0.11},
        'BOM_2C8': {'thr': 0.14, 'subthr': 0.31, 'som_thr': 0.14, 'low_thr': 0.07},
        'BOM_2C9': {'thr': 0.23, 'subthr': 0.43, 'som_thr': 0.23, 'low_thr': 0.14},
        'BOM_2C19': {'thr': 0.26, 'subthr': 0.38, 'som_thr': 0.26, 'low_thr': 0.15},
        'BOM_2D6': {'thr': 0.22, 'subthr': 0.48, 'som_thr': 0.22, 'low_thr': 0.14},
        'BOM_2E1': {'thr': 0.19, 'subthr': 0.20, 'som_thr': 0.19, 'low_thr': 0.10},
        'BOM_3A4': {'thr': 0.21, 'subthr': 0.71, 'som_thr': 0.21, 'low_thr': 0.12},
        'CYP_REACTION': {'thr': 0.23, 'subthr': 0.55, 'som_thr': 0.23, 'low_thr': 0.12}
    }

    if mode == 'custom':
        print(f"Running in Custom Mode with threshold: {custom_threshold}")
        for key in subtypes_threshold:
            subtypes_threshold[key]['thr'] = custom_threshold
            subtypes_threshold[key]['som_thr'] = custom_threshold
            subtypes_threshold[key]['low_thr'] = custom_threshold

    if output_type == 'only-som':
        if mode != 'broad':
            if subtype == 'max':
                SoM_to_image(test_df, seed_mean_output, image_dir_path, 'max', 0.26, 0.53)
                for key in subtypes_threshold.keys():
                    if key != 'CYP_REACTION':
                        threshold = subtypes_threshold[key]['som_thr']
                        sub_threshold = subtypes_threshold[key]['subthr']
                        SoM_to_image(test_df, seed_mean_output, image_dir_path, key, threshold, sub_threshold)
            else:
                threshold = subtypes_threshold[subtype]['som_thr']
                sub_threshold = subtypes_threshold[subtype]['subthr']
                SoM_to_image(test_df, seed_mean_output, image_dir_path, subtype, threshold, sub_threshold)

        else:
            if subtype == 'max':
                SoM_to_image(test_df, seed_mean_output, image_dir_path, 'max', 0.15, 0.53)
                for key in subtypes_threshold.keys():
                    if key != 'CYP_REACTION':
                        threshold = subtypes_threshold[key]['low_thr']
                        sub_threshold = subtypes_threshold[key]['subthr']
                        SoM_to_image(test_df, seed_mean_output, image_dir_path, key, threshold, sub_threshold)
            else:
                threshold = subtypes_threshold[subtype]['low_thr']
                sub_threshold = subtypes_threshold[subtype]['subthr']
                SoM_to_image(test_df, seed_mean_output, image_dir_path, subtype, threshold, sub_threshold)

    if output_type == 'raw-score':
        score_json_path = os.path.join(base_dir, f'score_dict_{timestamp}.json')
        with open(score_json_path, 'w') as json_file:
            json.dump(seed_mean_output, json_file, indent=4)
        aligned_sdf_path = os.path.join(base_dir, f'aligned_molecules_{timestamp}.sdf')
        writer = Chem.SDWriter(aligned_sdf_path)
        for i, row in test_df.iterrows():
            if row['ROMol'] and isinstance(row['ROMol'], Chem.Mol):
                mol = row['ROMol']
                writer.write(mol)
        writer.close()

    if output_type == 'default':
        os.makedirs(metabolite_path, exist_ok=True)
        # os.makedirs(metabolite_image_path, exist_ok=True)
        if mode in ['default', 'broad', 'custom']:
            metabolite_output = make_reaction_output_reaction(test_df, seed_mean_output, subtypes_threshold, mode,
                                                              prediction_type=subtype,
                                                              reaction_rules=reaction_rules)
            SoM_to_image_metabolite(test_df, seed_mean_output, image_dir_path, subtype, 0, 0, metabolite_output)
            if subtype == 'max':
                for key in subtypes_threshold.keys():
                    if key != 'CYP_REACTION':
                        threshold = subtypes_threshold[key]['low_thr'] if mode == 'broad' else subtypes_threshold[key]['som_thr']
                        sub_threshold = subtypes_threshold[key]['subthr']
                        SoM_to_image_metabolite(test_df, seed_mean_output, image_dir_path, key, threshold,sub_threshold, metabolite_output)
            save_metabolites_to_sdf_total(metabolite_output, metabolite_path, subtypes_threshold)

        # sdf_files = [f for f in os.listdir(metabolite_path) if f.endswith('.sdf')]
        # for sdf_file in sdf_files:
        # pos_id = sdf_file.split('_')[0]
        # template = test_df.loc[test_df['POS_ID'] == pos_id, 'ROMol'].values[0]
        # AllChem.Compute2DCoords(template)
        # save_metabolite_image_web(metabolite_path, sdf_file, template, metabolite_image_path)
        
    print('Done')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Process a molecule (SMILES or SDF) and generate images and metabolites.")
    parser.add_argument('--smiles', type=str, help='SMILES string of the molecule.')
    parser.add_argument('--sdf', type=str, help='Path to the SDF file.')
    parser.add_argument('--subtype', type=str, default='all',
                        help="Subtype to predict. Options: 'sub9', 'all', '1A2', '2A6', '2B6', '2C8', '2C9', '2C19', '2D6', '2E1', '3A4'")
    parser.add_argument('--base_dir', type=str, default='./output', help='Base directory to save outputs.')
    parser.add_argument('--output_type', type=str, default='default',
                        help='Type of output to generate (e.g., default, only-som, raw-score).')

    parser.add_argument('--mode', type=str, default='default',
                        help="Prediction mode. Options: 'default', 'broad', 'custom'.")
    parser.add_argument('--custom_threshold', type=float, default=0.5,
                        help='Threshold value for custom mode (default: 0.5).')

    args = parser.parse_args()

    main(smiles=args.smiles, sdf=args.sdf, subtype=args.subtype, base_dir=args.base_dir,
         output_type=args.output_type, mode=args.mode, custom_threshold=args.custom_threshold)