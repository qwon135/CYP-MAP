import numpy as np
import pandas as pd
from rdkit.Chem import PandasTools
from rdkit.Chem.Draw import IPythonConsole
from IPython.display import SVG
from rdkit.Chem import AllChem
import cairosvg
import tempfile
from rdkit.Chem import Draw
import os
from rdkit.Chem import rdFMCS
from rdkit import Chem, Geometry
from rdkit.Chem import AllChem, Draw, rdMolAlign, rdDepictor
from pathlib import Path
import re
from output_utils import *
import svgutils.transform as sg

def hsvToRgb(h, s, v):
    c = v * s
    h /= 30.0
    x = c * (1 - abs(h % 2 - 1))
    if 0 <= h < 1:
        r, g, b = (c, x, 0)
    elif 1 <= h < 2:
        r, g, b = (x, c, 0)
    elif 2 <= h < 3:
        r, g, b = (0, c, x)
    elif 3 <= h < 4:
        r, g, b = (0, x, c)
    elif 4 <= h < 5:
        r, g, b = (x, 0, c)
    else:
        r, g, b = (c, 0, x)
    m = v - c
    r, g, b = r + m, g + m, b + m

    r, g, b = max(0, min(r, 1)), max(0, min(g, 1)), max(0, min(b, 1))
    return (r, g, b)

def clean_svg(svg_text):
    svg_text = svg_text.strip()
    if svg_text.startswith('\ufeff'):
        svg_text = svg_text[1:]
    return svg_text

def SoM_to_image(df, som_dict, dir_path, subtype, input_threshold, sub_threshold):
  for i in df.index:
    try:
      database_id = f'MOL' + '0'*(4-len(str(i))) + str(i)
      mol = df['ROMol'][i]
      bond_features = som_dict['bond_som'][subtype][i]['y_prob']
      atom_features = som_dict['atom_som'][subtype][i]['y_prob']
      subs_prob = som_dict['subs'][subtype][i]['y_prob']
      if subs_prob < sub_threshold:
          threshold = 1
      else:
          threshold = input_threshold
      highlights_bond = {}
      highlights_atom = {}
      highlight_line_width = {}
      highlightRadii = {}
      minY = 0
      maxY = 0.8

      dwg = Draw.MolDraw2DSVG(600, 400)
      opt = dwg.drawOptions()
      opt.continuousHighlight = True
      opt.baseFontSize = 0.7
      opt.bondLineWidth = 2.6
      opt.fixedBondLength = 30

      for idx, bond in enumerate(mol.GetBonds()):
          label = bond_features[idx]
          bond.ClearProp('bondNote')
          value_threshold = label
          if value_threshold < threshold:
            value_threshold = 0
          else:
            value_threshold = value_threshold/0.8
          highlights_bond[bond.GetIdx()] = [hsvToRgb(200, value_threshold, 0.94)]
          highlight_line_width[bond.GetIdx()] = int(5*(label - minY) / (maxY - minY))

      for idx, atom in enumerate(mol.GetAtoms()):
          label = atom_features[idx]
          atom.ClearProp('AtomMapNumber')
          atom.ClearProp('atomNote')
          value_threshold = label
          if value_threshold < threshold:
            value_threshold = 0
          else:
            value_threshold = value_threshold/0.8
          highlights_atom[atom.GetIdx()] = [hsvToRgb(200, value_threshold, 0.94)]
          highlightRadii[atom.GetIdx()] = 0.4

      AllChem.Compute2DCoords(mol)
      dwg.DrawMoleculeWithHighlights(mol, '', highlights_atom, highlights_bond, highlightRadii, highlight_line_width)
      dwg.FinishDrawing()
      svg_text = dwg.GetDrawingText()
      SVG(dwg.GetDrawingText().replace('svg:', ''))

      with tempfile.NamedTemporaryFile(delete=True) as tmp:
          tmp.write(svg_text.encode())
          tmp.flush()
          image_subtype = "SUB9" if subtype == "max" else "CYP_ALL" if subtype == "CYP_REACTION" else subtype
          cairosvg.svg2png(url=tmp.name, write_to=f"{dir_path}/{database_id}_{image_subtype}.png")

    except Exception as e:
      print(e)

def SoM_to_image_metabolite(df, som_dict, dir_path, subtype, input_threshold, sub_threshold, output_dict):
  for i in df.index:
    try:
        database_id = f'MOL' + '0'*(4-len(str(i))) + str(i)
        mol = df['ROMol'][i]
        if subtype in output_dict[database_id]['outputs']:
            total_som = extract_numbers(output_dict[database_id]['outputs'][subtype]['total_SoM'])
        else:
            total_som = output_dict[database_id]['outputs']['max']['total_SoM_CYP'][subtype]
            total_som = extract_numbers(list(set(map(tuple, total_som))))
        bond_features = som_dict['bond_som'][subtype][i]['y_prob']
        atom_features = som_dict['atom_som'][subtype][i]['y_prob']
        subs_prob = som_dict['subs'][subtype][i]['y_prob']
        if subs_prob < sub_threshold:
            threshold = 1
        else:
            threshold = input_threshold
        highlights_bond = {}
        highlights_atom = {}
        highlight_line_width = {}
        highlightRadii = {}
        minY = 0
        maxY = 0.8
        dwg = Draw.MolDraw2DSVG(600, 400)
        opt = dwg.drawOptions()
        opt.continuousHighlight = True
        opt.baseFontSize = 0.7
        opt.bondLineWidth = 2.6
        opt.fixedBondLength = 30

        for idx, atom in enumerate(mol.GetAtoms()):
          atom.SetAtomMapNum(idx + 1)

        for idx, bond in enumerate(mol.GetBonds()):
            label = bond_features[idx]
            value_threshold = label
            if value_threshold < threshold:
              value_threshold = 0
            else:
                value_threshold = value_threshold / 0.8
            begin_atom_idx = str(bond.GetBeginAtom().GetAtomMapNum())
            end_atom_idx = str(bond.GetEndAtom().GetAtomMapNum())
            if begin_atom_idx in total_som and end_atom_idx in total_som and label > threshold:
              highlights_bond[bond.GetIdx()] = [hsvToRgb(200, value_threshold, 0.94)]
              highlight_line_width[bond.GetIdx()] = int(5*(label - minY) / (maxY - minY))

        for idx, atom in enumerate(mol.GetAtoms()):
            label = atom_features[idx]
            value_threshold = label
            if value_threshold < threshold:
              value_threshold = 0
            else:
                value_threshold = value_threshold/0.8
            if str(atom.GetAtomMapNum()) in total_som:
              highlights_atom[atom.GetIdx()] = [hsvToRgb(200, value_threshold, 0.94)]
              highlightRadii[atom.GetIdx()] = 0.4

        for idx, bond in enumerate(mol.GetBonds()):
            bond.ClearProp('bondNote')

        for idx, atom in enumerate(mol.GetAtoms()):
            atom.ClearProp('molAtomMapNumber')

        bond_updates = {}
        line_width_updates = {}

        for key, value in highlights_bond.items():
            bond = mol.GetBondWithIdx(key)
            equivalent_bonds = find_equivalent_bonds(bond, mol)
            for eq_bond in equivalent_bonds:
                bond_updates[eq_bond.GetIdx()] = value
                line_width_updates[eq_bond.GetIdx()] = highlight_line_width[key]

        highlights_bond.update(bond_updates)
        highlight_line_width.update(line_width_updates)

        atom_updates = {}
        radii_updates = {}

        for key, value in highlights_atom.items():
            atom = mol.GetAtomWithIdx(key)
            equivalent_atoms = find_equivalent_atoms(atom, mol)
            for eq_atom in equivalent_atoms:
                atom_updates[eq_atom.GetIdx()] = value
                radii_updates[atom.GetIdx()] = 0.4

        highlights_atom.update(atom_updates)
        highlightRadii.update(radii_updates)

        for idx, atom in enumerate(mol.GetAtoms()):
            if atom.GetIdx() not in highlights_atom:
                highlights_atom[atom.GetIdx()] = [hsvToRgb(200, 0, 0.94)]
                highlightRadii[atom.GetIdx()] = 0.4

        for idx, bond in enumerate(mol.GetBonds()):
            if bond.GetIdx() not in highlights_bond:
                highlights_bond[bond.GetIdx()] = [hsvToRgb(200, 0, 0.94)]
                highlight_line_width[bond.GetIdx()] = 2

        AllChem.Compute2DCoords(mol)
        dwg.DrawMoleculeWithHighlights(mol, '', highlights_atom, highlights_bond, highlightRadii, highlight_line_width)
        dwg.FinishDrawing()
        svg_text = dwg.GetDrawingText()
        SVG(dwg.GetDrawingText().replace('svg:', ''))

        with tempfile.NamedTemporaryFile(delete=True) as tmp:
            tmp.write(svg_text.encode())
            tmp.flush()
            image_subtype = "SUB9" if subtype == "max" else "CYP_ALL" if subtype == "CYP_REACTION" else subtype
            cairosvg.svg2png(url=tmp.name, write_to=f"{dir_path}/{database_id}_{image_subtype}.png")
            
    except Exception as e:
      SoM_to_image(df, som_dict, dir_path, subtype, 1, 1)
      print(e)

def save_metabolite_image_web(sdf_dir, sdf_file, ref_mol, output_dir):
    try:
        sdf_path = os.path.join(sdf_dir, sdf_file)
        suppl = Chem.SDMolSupplier(sdf_path)
        base_name = os.path.splitext(sdf_file)[0].replace('_', '-')
        base_name = base_name.split('-')[0] + '_' + '-'.join(base_name.split('-')[1:])

        molecules = [mol for mol in suppl if mol is not None]

        for i, probe_mol in enumerate(molecules):

            for atom in probe_mol.GetAtoms():
                atom.ClearProp('molAtomMapNumber')
            mols = [ref_mol, probe_mol]

            score = probe_mol.GetProp('Score') if probe_mol.HasProp('Score') else 'N/A'
            score = str(round(float(score), 4))
            Rank = str(probe_mol.GetProp('Rank') if probe_mol.HasProp('Rank') else 'N/A')
            bond_score = probe_mol.GetProp('Bond_Score') if probe_mol.HasProp('Bond_Score') else 'N/A'
            reaction_score = probe_mol.GetProp('Reaction_Score') if probe_mol.HasProp('Reaction_Score') else 'N/A'
            reaction = probe_mol.GetProp('Reaction') if probe_mol.HasProp('Reaction') else 'N/A'
            reaction = remove_last_hyphen(reaction.split(',')[0].split('(')[0])
            Subtype = probe_mol.GetProp('Subtype') if probe_mol.HasProp('Subtype') else base_name.split('-')[-1]
            Subtype = Subtype.replace(" ", "")
            Subtype_Scores = probe_mol.GetProp('Subtype_Scores') if probe_mol.HasProp('Subtype_Scores') else 'N/A'
            Subtype_Scores_df = convert_string_to_csv_format(Subtype_Scores, reaction)

            mcs = rdFMCS.FindMCS(mols, threshold=0.8, completeRingsOnly=True, ringMatchesRingOnly=True)
            patt = Chem.MolFromSmarts(mcs.smartsString)
            refMatch = ref_mol.GetSubstructMatch(patt)
            probeMatch = probe_mol.GetSubstructMatch(patt)

            if len(refMatch) == len(probeMatch):
                coord_dict = {probeMatch[j]: Geometry.Point2D(ref_mol.GetConformer().GetAtomPosition(refMatch[j]).x, ref_mol.GetConformer().GetAtomPosition(refMatch[j]).y) for j in range(len(refMatch))}
                AllChem.Compute2DCoords(probe_mol, coordMap=coord_dict)
            else:
                AllChem.Compute2DCoords(probe_mol)

            rms = AllChem.AlignMol(probe_mol, ref_mol, atomMap=list(zip(probeMatch, refMatch)))

            try:
                Chem.Kekulize(probe_mol)
            except:
                print(f"Kekulization failed for molecule {i} in file {sdf_file}")

            drawer_probe = Draw.MolDraw2DSVG(600, 400)

            opt = drawer_probe.drawOptions()
            opt.baseFontSize = 0.7
            opt.bondLineWidth = 2.6
            opt.fixedBondLength = 30

            drawer_probe.DrawMolecule(probe_mol)
            drawer_probe.FinishDrawing()
            svg_text = drawer_probe.GetDrawingText().replace('svg:', '')

            with tempfile.NamedTemporaryFile(delete=True) as tmp:
                tmp.write(svg_text.encode())
                tmp.flush()

                fig = sg.fromstring(svg_text)
                label = sg.TextElement(30, 30, "Rank: " + Rank, size=24, font='sans-serif', anchor='start', color='black')
                fig.append(label)

                output_file = f"{output_dir}/{base_name}_{i:03d}_{score}_{bond_score}_{reaction_score}_{Subtype}_{reaction}.png"
                fig.save(tmp.name)
                cairosvg.svg2png(url=tmp.name, write_to=output_file)

                ###### For CYP-MAP web ######
                subtype_list = Subtype.split(',')
                if len(subtype_list) >= 1:
                    for subtype in subtype_list:
                        type_path = os.path.join(output_dir, subtype)
                        os.makedirs(type_path, exist_ok=True)

                        target_row = Subtype_Scores_df[Subtype_Scores_df['Subtype'] == subtype]
                        if not target_row.empty:
                            SOM_score = target_row['SOM_score'].iloc[0]
                            Reaction_score = target_row['Reaction_score'].iloc[0]
                            Score = target_row['Score'].iloc[0]
                            Reaction = target_row['Reaction'].iloc[0]

                            subtype_output_file = f"{type_path}/{base_name}_{i:03d}_{Score}_{SOM_score}_{Reaction_score}_{subtype}_{Reaction}.png"
                            cairosvg.svg2png(url=tmp.name, write_to=subtype_output_file)

    except Exception as e:
        print(e)