import re
import json
import time
from rdkit.Chem import PandasTools
import numpy as np
import pandas as pd
from rdkit.Chem import AllChem, rdChemReactions
import rdkit.Chem as Chem
from rdkit.Chem.rdChemReactions import SanitizeRxn, PreprocessReaction, ReactionFromSmarts, SanitizeFlags
from rdkit.Chem import rdFMCS
from rdkit.Chem import Draw
import random
import os
from rdkit.Chem import SDWriter, Descriptors
import traceback
from output_utils import *
from rdkit import RDLogger

def extract_pairs(mol, features, threshold, mode):
    pairs = []
    pairs_score = {}

    if mode == 'bond':
        for idx, bond in enumerate(mol.GetBonds()):
            begin_atom_idx = str(bond.GetBeginAtom().GetAtomMapNum())
            if bond.GetEndAtom().GetAtomicNum() == 1:
                end_atom_idx = 'nan'
            else:
                end_atom_idx = str(bond.GetEndAtom().GetAtomMapNum())
            char_list = sorted([begin_atom_idx, end_atom_idx])
            position1 = char_list[0]
            position2 = char_list[1]
            pair = f'<{position1},{position2}>'
            pairs_score[pair] = features[idx]
            if features[idx] >= threshold:
                pairs.append(pair)

    elif mode == 'spn':
        SPN_count = 0
        for idx, atom in enumerate(mol.GetAtoms()):
            if atom.GetAtomicNum() in [7, 16]:
                label = features[SPN_count]
                pair = f'<{atom.GetAtomMapNum()},nan>'
                pairs_score[pair] = label
                if label >= threshold:
                    pairs.append(pair)
                SPN_count += 1

    elif mode == 'atom':
        atom_count = 0
        for idx, atom in enumerate(mol.GetAtoms()):
            label = features[atom_count]
            pair = f'<{atom.GetAtomMapNum()},nan>'
            pairs_score[pair] = label
            if label >= threshold:
                pairs.append(pair)
            atom_count += 1

    pairs = list(set(pairs))

    return pairs, pairs_score

def combine_dictionaries(*dicts):
    combined_dict = {}
    current_key = 0

    for d in dicts:
        for key in d:
            combined_dict[current_key] = d[key]
            current_key += 1

    return combined_dict

def remove_bond_changes_from_som_list(som_list, bond_changes):
    """
    Removes bond changes from som_list and returns the residual list.
    """
    residual_list = som_list.copy()
    for bond_change in bond_changes:
        if bond_change in residual_list:
            residual_list.remove(bond_change)
    return residual_list


def is_radical(mol):
    """
    Check if the molecule is a radical
    """
    for atom in mol.GetAtoms():
        if atom.GetNumRadicalElectrons() > 0:
            return True

    return False
def is_geminal_diol(mol):
    """
    Check if the molecule is a geminal diol.
    """
    geminal_diol_pattern = Chem.MolFromSmarts('[C]([OH])([OH])')
    if mol.HasSubstructMatch(geminal_diol_pattern):
        return True
    return False

def Get_Bond_Change(substrate_smiles, metabolite_smiles, tag):
    smirks, additional_oxygen_map = convert_to_smirks(substrate_smiles, metabolite_smiles)
    smirks = fix_missing_map_numbers(smirks)
    bond_changes = get_bond_changes(smirks)
    BOM_rxn_list = []
    Oxidation_list = []
    Ox = 0

    for bond_change in bond_changes:
        atom_number_0, atom_number_1 = bond_change[0], bond_change[1]
        if atom_number_0 in additional_oxygen_map:
            atom_number_0 = 'nan'
            Ox = 1
        if atom_number_1 in additional_oxygen_map:
            atom_number_1 = 'nan'
            Ox = 1
        char_list = sorted([str(atom_number_0), str(atom_number_1)])
        position1 = char_list[0]
        position2 = char_list[1]
        bond_string = f'<{position1},{position2}>'
        BOM_rxn_list.append(bond_string)
        if Ox == 1:
            Oxidation_list.append(bond_string)

    if tag == 'oxi' or tag == 'spn':
      if any('nan' not in element for element in BOM_rxn_list):
          pass
      else:
         BOM_rxn_list = Oxidation_list

    elif tag == 'oxc':
      BOM_rxn_list = [element for element in BOM_rxn_list if 'nan' not in element]
      pass #  bond

    else:
      BOM_rxn_list = [element for element in BOM_rxn_list if 'nan' not in element] # only bond

    return BOM_rxn_list, smirks

def run_reactions(reactant_dict, target_reaction_rules, target_BoM, mode, tag):
    RDLogger.DisableLog('rdApp.*')
    product_dict = reactant_dict.copy()
    if 0 in product_dict:
        del product_dict[0]

    product_idx = 0

    for key, reactant_data in reactant_dict.items():
        reactant = Chem.MolFromSmiles(reactant_data['smiles'])

        try:
            reactant_smiles = Chem.MolToSmiles(AllChem.RemoveHs(reactant), isomericSmiles=False)
            reactant_with_h = AllChem.AddHs(reactant)
        except Exception as e:
            continue

        for i, rxn_string in enumerate(target_reaction_rules['SMIRKS Strings']):
            reaction_name = target_reaction_rules['Name'][i]
            rxn = AllChem.ReactionFromSmarts(rxn_string)
            products = rxn.RunReactants([reactant_with_h])

            for product in products:
                try:
                    [Chem.SanitizeMol(metabolite) for metabolite in product]
                    product_smiles = []
                    radical = 0
                    for metabolite in product:
                        numbering_metabolite(metabolite, reactant)
                        if is_radical(metabolite):
                            radical += 1

                        smiles_metabolite = Chem.MolToSmiles(Chem.rdmolops.RemoveHs(metabolite), isomericSmiles=False)
                        product_smiles.append(smiles_metabolite)

                    if radical < 1:
                        product_smiles_str = '.'.join(product_smiles)
                        try:
                            bond_changes, smirks = Get_Bond_Change(reactant_smiles, product_smiles_str, tag)
                            if all(item in target_BoM for item in bond_changes):
                                product_dict[product_idx] = {
                                    'smiles': product_smiles_str,
                                    'reaction_name': reaction_name,
                                    'mol': product,
                                    'bond_changes': bond_changes,
                                    'tag': [tag]
                                }
                                product_idx += 1
                        except Exception as e:
                            pass
                except Exception as e:
                    pass

    return product_dict

def check_target_metabolite(m, target_BoM, reaction_rules, n, mode, tag):
    RDLogger.DisableLog('rdApp.*')
    for idx, atom in enumerate(m.GetAtoms()):
        atom.SetAtomMapNum(idx + 1)

    substrate_smiles = Chem.MolToSmiles(m, isomericSmiles=True)
    reactant_info = {'smiles': substrate_smiles,
                  'reaction_name': "",
                  'mol': [],
                  'bond_changes': [],
                  # 'score': [],
                  'tag': [tag]}
    reactant_dict = { 0 : reactant_info }

    for i in range(n):
        if i == 0:
            product_dict = run_reactions(reactant_dict, reaction_rules, target_BoM, mode, tag)
        else:
            product_dict = run_reactions(product_dict, reaction_rules, target_BoM, mode, tag)
    return product_dict

def calculate_scores(bond_changes, atom_score, bond_score, reaction_score):
    """Calculate final scores based on bond changes, atom scores, bond scores, and reaction scores."""
    scores = []
    scores_reaction = []
    scores_bond = []

    for bond_change in bond_changes:
        if bond_change in atom_score and bond_change in reaction_score:
            score = np.sqrt(atom_score[bond_change] * reaction_score[bond_change])
            scores_bond.append(atom_score[bond_change])
            scores_reaction.append(reaction_score[bond_change])
            scores.append(score)

        elif bond_change in bond_score and bond_change in reaction_score:
            score = np.sqrt(bond_score[bond_change] * reaction_score[bond_change])
            scores_bond.append(bond_score[bond_change])
            scores_reaction.append(reaction_score[bond_change])
            scores.append(score)

    if scores:
        final_score = np.mean(scores)
        final_score_reaction = np.mean(scores_reaction)
        final_score_bond = np.mean(scores_bond)
    else:
        final_score = 0
        final_score_reaction = 0
        final_score_bond = 0

    return final_score, final_score_reaction, final_score_bond

def make_reaction_output_reaction(df, SoM_dict, subtypes_threshold, mode, prediction_type, reaction_rules):
    n = 1
    RDLogger.DisableLog('rdApp.*')
    sub_keys_list = [sub_key for key in SoM_dict.keys() for sub_key in SoM_dict[key].keys()]
    output_dict = {}

    if prediction_type != 'max':
        subtypes_threshold = {key: value for key, value in subtypes_threshold.items() if key == prediction_type}

    threshold_dict = {
        'oxi': {'thr': 0.177, 'low_thr': 0.085},
        'oxc': {'thr': 0.160, 'low_thr': 0.018},
        'epo': {'thr': 0.026, 'low_thr': 0.004},
        'sut': {'thr': 0.075, 'low_thr': 0.075},
        'dhy': {'thr': 0.008, 'low_thr': 0.004},
        'rdc': {'thr': 0.009, 'low_thr': 0.008},
        'hys': {'thr': 0.024, 'low_thr': 0.005},
        'spn': {'thr': 0.040, 'low_thr': 0.004}
    }

    if mode == 'broad':
        key_1 = 'low_thr'
    else:
        key_1 = 'thr'

    for i in range(len(df['Molecules'])):
        index = 'MOL' + '0'*(4 - len(str(i))) + f'{i}'
        try:
            print(f"Processing index: {index}")
            mol = df['Molecules'][i]
            molblock = Chem.MolToMolBlock(mol)
            name = molblock.split('\n')[0].strip()

            for idx, atom in enumerate(mol.GetAtoms()):
                atom.SetAtomMapNum(idx + 1)
            mol2 = df['ROMol'][i]
            smiles = clear_mapping(Chem.MolToSmiles(mol2))
            output_dict[index] = {'Name': name, 'substrate': smiles}
            subtype_dict = {}
            atom_score_dict = {}
            bond_score_dict = {}
            subs_dict ={}

            for subtype in sub_keys_list:
                bond_features = SoM_dict['bond_som'][subtype][i]['y_prob']
                atom_features = SoM_dict['atom_som'][subtype][i]['y_prob']
                subs_prob = SoM_dict['subs'][subtype][i]['y_prob']
                _, bond_score = extract_pairs(mol, bond_features, 0, 'bond')
                _, atom_score = extract_pairs(mol, atom_features, 0, 'atom')
                atom_score_dict[subtype] = atom_score
                bond_score_dict[subtype] = bond_score
                subs_dict[subtype] = subs_prob

            try:
                bond_features = SoM_dict['bond_som'][prediction_type][i]['y_prob']
                atom_features = SoM_dict['atom_som'][prediction_type][i]['y_prob']
                oxc_features = SoM_dict['oxc']['CYP_REACTION'][i]['y_prob']
                epo_features = SoM_dict['epo']['CYP_REACTION'][i]['y_prob']
                oxi_features = SoM_dict['oxi']['CYP_REACTION'][i]['y_prob']
                sut_features = SoM_dict['sut']['CYP_REACTION'][i]['y_prob']
                dhy_features = SoM_dict['dhy']['CYP_REACTION'][i]['y_prob']
                rdc_features = SoM_dict['rdc']['CYP_REACTION'][i]['y_prob']
                hys_features = SoM_dict['hys']['CYP_REACTION'][i]['y_prob']
                spn_features = SoM_dict['atom_spn']['CYP_REACTION'][i]['y_prob']

                for idx, atom in enumerate(mol.GetAtoms()):
                    atom.SetAtomMapNum(idx + 1)

                bond_list, bond_score = extract_pairs(mol, bond_features, 0, 'bond')
                atom_list, atom_score = extract_pairs(mol, atom_features, 0, 'atom')
                oxi_list, oxi_score = extract_pairs(mol, oxi_features, threshold_dict['oxi'][key_1], 'bond')
                oxc_list, oxc_score = extract_pairs(mol, oxc_features, threshold_dict['oxc'][key_1], 'bond')
                epo_list, epo_score = extract_pairs(mol, epo_features, threshold_dict['epo'][key_1],'bond')
                sut_list, sut_score = extract_pairs(mol, sut_features, threshold_dict['sut'][key_1], 'bond')
                dhy_list, dhy_score = extract_pairs(mol, dhy_features, threshold_dict['dhy'][key_1], 'bond')
                rdc_list, rdc_score = extract_pairs(mol, rdc_features, threshold_dict['rdc'][key_1], 'bond')
                hys_list, hys_score = extract_pairs(mol, hys_features, threshold_dict['hys'][key_1], 'bond')
                spn_list, spn_score = extract_pairs(mol, spn_features, threshold_dict['spn'][key_1], 'spn')

                som_list = bond_list + atom_list

                if not som_list:
                    subtype_dict[prediction_type] = {'metabolite': {}, 'total_SoM': {}}
                    continue

                reaction_rules_subset = reaction_rules[
                    (reaction_rules['Common Type'] == 'Oxidative Cleavage') |
                    (reaction_rules['Common Type'] == 'Cleavage')
                ].reset_index(drop=True)

                oxc_dict = check_target_metabolite(mol, oxc_list, reaction_rules_subset, n, 'bond', 'oxc')
                epo_dict = check_target_metabolite(mol, epo_list, reaction_rules[reaction_rules['Common Type'] == 'Epoxidation'].reset_index(drop=True), n, 'bond', 'epo')
                oxi_dict_atom = check_target_metabolite(mol, oxi_list, reaction_rules[reaction_rules['Common Type'] == 'Oxidation'].reset_index(drop=True), n, 'atom', 'oxi')
                oxi_dict_bond = check_target_metabolite(mol, oxi_list, reaction_rules[reaction_rules['Common Type'] == 'Oxidation'].reset_index(drop=True), n, 'bond', 'oxi')

                reaction_rules_subset = reaction_rules[
                    (reaction_rules['Common Type'] == 'Substitution') |
                    (reaction_rules['Common Type'] == 'Desulfuration')
                ].reset_index(drop=True)

                sut_dict = check_target_metabolite(mol, sut_list, reaction_rules_subset, n, 'bond', 'sut')
                dhy_dict_atom = check_target_metabolite(mol, dhy_list, reaction_rules[reaction_rules['Common Type'] == 'Dehydrogenation'].reset_index(drop=True), n, 'atom', 'dhy')
                dhy_dict_bond = check_target_metabolite(mol, dhy_list, reaction_rules[reaction_rules['Common Type'] == 'Dehydrogenation'].reset_index(drop=True), n, 'bond', 'dhy')
                rdc_dict = check_target_metabolite(mol, rdc_list, reaction_rules[reaction_rules['Common Type'] == 'Reduction'].reset_index(drop=True), n, 'bond', 'rdc')
                hys_dict = check_target_metabolite(mol, hys_list, reaction_rules[reaction_rules['Common Type'] == 'Hydrolysis'].reset_index(drop=True), n, 'bond', 'hys')
                spn_dict = check_target_metabolite(mol, spn_list, reaction_rules[reaction_rules['Common Type'] == 'SPN-Oxidation'].reset_index(drop=True), n, 'atom', 'spn')

                product_dict = combine_dictionaries(oxc_dict, epo_dict, oxi_dict_atom, oxi_dict_bond, sut_dict, dhy_dict_atom, dhy_dict_bond, rdc_dict, spn_dict, hys_dict)
                score_dict = {
                    'oxc': oxc_score, 'oxi': oxi_score, 'epo': epo_score,
                    'sut': sut_score, 'dhy': dhy_score, 'rdc': rdc_score,
                    'spn': spn_score, 'hys': hys_score
                }

                metabolite_dict = {}
                CYP_all_som = {}
                for subtype in subtypes_threshold.keys():
                    CYP_all_som[subtype] = []
                all_som = []
                for key, product in product_dict.items():
                    metabolite_smiles = clear_mapping(product['smiles'])
                    reaction_name = product['reaction_name']
                    bond_changes = product['bond_changes']
                    reaction_score = score_dict[product['tag'][0]]

                    final_score, final_score_reaction, final_score_bond = calculate_scores(bond_changes, atom_score, bond_score, reaction_score)

                    subtype_score = {}
                    subtype_bond_score = {}
                    subtype_reaction_score = {}

                    for subtype in sub_keys_list:
                        subtype_score[subtype], subtype_reaction_score[subtype], subtype_bond_score[subtype] = calculate_scores(bond_changes, atom_score_dict[subtype], bond_score_dict[subtype], reaction_score)

                    CYP_types = []
                    for subtype in subtypes_threshold.keys():
                        cyp_score = subtype_bond_score[subtype]
                        cyp_subs_score = subs_dict[subtype]
                        check_score = subtypes_threshold[subtype][key_1]
                        check_sub_score = subtypes_threshold[subtype]['subthr']
                        if cyp_score >= check_score and cyp_subs_score >= check_sub_score:
                            if prediction_type == 'max' and subtype in ['max', 'CYP_REACTION']:
                                pass
                            else:
                                CYP_types.append(subtype)
                                if subtype not in CYP_all_som:
                                    CYP_all_som[subtype].append(bond_changes)
                                else:
                                    CYP_all_som[subtype].append(bond_changes)

                    if len(CYP_types) == 0:
                        continue

                    all_som.append(bond_changes)

                    if metabolite_smiles in metabolite_dict:
                        if reaction_name not in metabolite_dict[metabolite_smiles]['reaction']:
                            metabolite_dict[metabolite_smiles]['reaction'].append(reaction_name)
                        if bond_changes not in metabolite_dict[metabolite_smiles]['SoM']:
                            metabolite_dict[metabolite_smiles]['SoM'].append(bond_changes)
                        if final_score not in metabolite_dict[metabolite_smiles]['Score']:
                            if final_score is not None and metabolite_dict[metabolite_smiles]['Score'][0] < final_score:
                                metabolite_dict[metabolite_smiles]['Score'] = [final_score]
                                metabolite_dict[metabolite_smiles]['Reaction_Score'] = [final_score_reaction]
                                metabolite_dict[metabolite_smiles]['Bond_Score'] = [final_score_bond]
                    else:
                        metabolite_dict[metabolite_smiles] = {
                            'mol': product['mol'],
                            'reaction': [reaction_name],
                            'SoM': [bond_changes],
                            'Score': [final_score],
                            'Reaction_Score': [final_score_reaction],
                            'Bond_Score': [final_score_bond],
                            'CYP_types': CYP_types,
                            'Subtype_Bond_Score': subtype_bond_score,
                            'Subtype_Reaction_Score': subtype_reaction_score,
                            'Subtype_Score': subtype_score,
                            'Subtype_Subs_Score': subs_dict
                        }
                total_som = list(set(map(tuple, all_som)))
                subtype_dict[prediction_type] = {'metabolite': metabolite_dict, 'total_SoM': total_som, 'total_SoM_CYP': CYP_all_som}
            except Exception as e:
                print(f"Error processing CYP subtype {prediction_type}: {e}")
                print(traceback.format_exc())
            output_dict[index]['outputs'] = subtype_dict
        except Exception as e:
            print(f"Error processing index {index}: {e}")
            subtype_dict[prediction_type] = {'metabolite': '', 'total_SoM': f'{e}'}
            output_dict[index]['outputs'] = subtype_dict
            print(traceback.format_exc())
    return output_dict

def save_metabolites_to_sdf_total(output_dict, base_path, subtypes_threshold):
    """
    Saves metabolite information to SDF files based on the given output reaction dictionary.

    """
    for test_key, test_value in output_dict.items():
        try:
            for Cyp_type, output_value in test_value['outputs'].items():
                if Cyp_type == 'max':
                  Cyp_type = 'SUB9'
                elif Cyp_type == 'CYP_REACTION':
                  Cyp_type = 'CYP_ALL'
                sdf_filename = os.path.join(base_path, f"{test_key}_{Cyp_type}.sdf")
                with SDWriter(sdf_filename) as writer:
                    metabolites = output_value['metabolite']
                    if metabolites:
                        sorted_metabolites = sorted(
                                                    metabolites.items(),
                                                    key=lambda item: float(item[1]['Score'][0]) if item[1]['Score'][0] is not None else 0,
                                                    reverse=True)
                        rank = 1
                        previous_score = None
                        processed_smiles = set()
                        for smiles, mol_data in sorted_metabolites:
                            current_score = float(mol_data['Score'][0])
                            if previous_score is not None and current_score != previous_score:
                                rank += 1
                            previous_score = current_score
                            mols = mol_data['mol']
                            reaction = mol_data['reaction']
                            reaction_score = mol_data['Reaction_Score'][0]
                            bond_score= mol_data['Bond_Score'][0]
                            CYP_types = mol_data['CYP_types']
                            cyp_scores=''
                            for subtype in CYP_types:
                              cyp_som_score = "{:.4f}".format(mol_data['Subtype_Bond_Score'][subtype])
                              cyp_reaction_score = "{:.4f}".format(mol_data['Subtype_Reaction_Score'][subtype])
                              cyp_score = "{:.4f}".format(mol_data['Subtype_Score'][subtype])
                              cyp_subs_score = "{:.4f}".format(mol_data['Subtype_Subs_Score'][subtype])
                              subtype_label = "CYP_ALL" if subtype == "CYP_REACTION" else subtype
                              row = f'[{subtype_label.split("_")[1]}] SOM_score: {cyp_som_score}, Reaction_score: {cyp_reaction_score}, Score: {cyp_score}\n'
                              cyp_scores += row
                              reaction_str = ','.join(reaction)
                            for mol_total in mols:
                                if mol_total is None:
                                    print(test_key, reaction, smiles)
                                    continue
                                if is_geminal_diol(mol_total):
                                    mol_total = Chem.AddHs(mol_total, addCoords=True)
                                    diol_to_carbonyl = '[CX4:1]([OH:2])([OH:3])[#6:4]>>[CX3:1](=[O])[#6:4]'
                                    rxn = rdChemReactions.ReactionFromSmarts(diol_to_carbonyl)
                                    products = rxn.RunReactants([mol_total])
                                    mol_total = products[0][0]
                                    reaction_str = 'oxidation_(geminal_diol_intermediate)'
                                frags = Chem.GetMolFrags(mol_total, asMols=True)
                                for mol in frags:
                                    mol_smiles = Chem.MolToSmiles(mol)
                                    try:
                                        unmapped_smiles = clear_mapping(mol_smiles)
                                    except Exception as e:
                                        unmapped_smiles = smiles
                                    if unmapped_smiles in processed_smiles:
                                        continue
                                    processed_smiles.add(unmapped_smiles)
                                    mol = Chem.AddHs(mol, addCoords=True)

                                    MW = str(round(Descriptors.MolWt(mol), 1))
                                    score = str(current_score)

                                    mol.SetProp('SMILES', unmapped_smiles)
                                    mol.SetProp('Molecular Weight', MW)
                                    mol.SetProp('Substrate_ID', str(test_key))
                                    mol.SetProp('Substrate', str(test_value['Name']))
                                    mol.SetProp('Substrate_SMILES', str(test_value['substrate']))
                                    mol.SetProp('Reaction', reaction_str)
                                    mol.SetProp('Score', str(round(float(score), 4)))
                                    mol.SetProp('Reaction_Score', str(round(float(reaction_score), 4)))
                                    mol.SetProp('SoM_Score', str(round(float(bond_score), 4)))
                                    mol.SetProp('Rank', str(rank))
                                    mol.SetProp('Subtype', ', '.join([("CYP_ALL" if cyp == "CYP_REACTION" else cyp).split('_')[1] for cyp in CYP_types]))
                                    mol.SetProp('Subtype_Scores', cyp_scores)
                                    # mol.SetProp('SoM', SoM)
                                    if float(MW) > 44.009 and float(MW)*5 > Descriptors.MolWt(Chem.MolFromSmiles(test_value['substrate'])): #CO2
                                        try:
                                          writer.write(mol)
                                        except:
                                          mol = Chem.MolFromSmiles(unmapped_smiles)
                                          writer.write(mol)
        except Exception as e:
            print(f"Error processing {test_key}: {e}")
            sdf_filename = os.path.join(base_path, f"{test_key}_{Cyp_type}.sdf")
            with open(sdf_filename, 'w') as sdf_file:
                pass
