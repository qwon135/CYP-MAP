import argparse
from collections import defaultdict
import csv
from functools import cache
import gzip
from pathlib import Path

import matplotlib
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score, roc_curve

matplotlib.use('Agg')
import matplotlib.pyplot as plt


RESULTS = Path(__file__).resolve().parent
MODELS = ('FAME2', 'D-CyPre', 'CyProduct', 'DeepMetab', 'CYP-MAP (All)', 'CYP-MAP (9 Subtypes)')
SUBTYPE_MODELS = ('D-CyPre', 'CyProduct', 'DeepMetab', 'CYP-MAP')
SUBTYPES = ('1A2', '2A6', '2B6', '2C8', '2C9', '2C19', '2D6', '2E1', '3A4')
METRICS = ('F1', 'Jaccard', 'Precision', 'Recall')
COLORS = dict(zip(MODELS, ('#f57ac0', '#7096e8', '#82d5a2', '#ffb255', '#9582c4', '#644ca2')))
COLORS['CYP-MAP'] = COLORS['CYP-MAP (All)']


def read_csv(path):
    with gzip.open(path, 'rt', encoding='utf-8', newline='') as stream:
        yield from csv.DictReader(stream)


def write_csv(path, header, rows):
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.writer(stream, lineterminator='\n')
        writer.writerow(header)
        writer.writerows(rows)


def oracle_threshold(observed, score):
    order = np.argsort(-score, kind='stable')
    ordered_scores = score[order]
    ends = np.r_[np.flatnonzero(ordered_scores[:-1] != ordered_scores[1:]), len(score) - 1]
    true_positives = np.cumsum(observed[order])[ends]
    f1 = 2 * true_positives / (ends + 1 + observed.sum())
    best = ends[np.flatnonzero(np.isclose(f1, f1.max(), rtol=0, atol=1e-12))[-1]]
    if best + 1 < len(score):
        return float(ordered_scores[best + 1])
    return float(np.nextafter(ordered_scores[-1], -np.inf))


def bootstrap_scores(observed, predicted):
    rng = np.random.RandomState(42)
    confusion_codes = 2 * observed + predicted
    counts = np.array([
        np.bincount(confusion_codes[rng.randint(0, len(observed), size=len(observed))], minlength=4)
        for _ in range(1000)
    ])
    tn, fp, fn, tp = counts.T
    keep = ((tp + fn) > 0) & ((tn + fp) > 0)
    tp, fp, fn = tp[keep], fp[keep], fn[keep]
    numerators = np.array([2 * tp, tp, tp, tp])
    denominators = np.array([2 * tp + fp + fn, tp + fp + fn, tp + fp, tp + fn])
    samples = np.divide(numerators, denominators, out=np.zeros_like(numerators, dtype=float),
                        where=denominators != 0)
    samples.sort(axis=1)
    count = samples.shape[1]
    return samples.mean(axis=1), samples[:, int(0.05 * count)], samples[:, int(0.95 * count)]


def plot_bars(axis, bars, models, categories, subtype, oracle):
    positions = np.arange(len(categories))
    width = 0.78 / len(models)
    for index, model in enumerate(models):
        metrics = [bars[model, cyp][:, 0] for cyp in categories] if subtype else bars[model, 'All'].T
        means, lows, highs = np.asarray(metrics).T
        errors = np.maximum(0, np.array([means - lows, highs - means]))
        centers = positions + (index - (len(models) - 1) / 2) * width
        axis.bar(centers, means, width, yerr=errors, label=model, color=COLORS[model],
                 hatch='///' if oracle else '', edgecolor='#444444' if oracle else 'white',
                 linewidth=0.4, capsize=2, error_kw={'elinewidth': 0.7})
    axis.set_xticks(positions, categories)
    axis.set_ylim(0, 0.8 if subtype else 1.0)
    axis.set_ylabel('F1' if subtype else 'Score')
    axis.set_axisbelow(True)
    axis.grid(axis='y', color='0.88', linewidth=0.7)
    axis.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8)


def reproduce_figure(output):
    grouped = defaultdict(list)
    for row in read_csv(RESULTS / 'SoM' / 'figure3_predictions.csv.gz'):
        grouped[row['model'], row['cyp']].append((int(row['observed']), float(row['score']),
                                                  int(row['default_call'])))
    models_by_cyp = {'All': MODELS, **dict.fromkeys(SUBTYPES, SUBTYPE_MODELS)}

    fig = plt.figure(figsize=(12, 12), layout='constrained')
    grid = fig.add_gridspec(5, 2, height_ratios=(1.35, 1, 1, 1, 1))
    roc_axis = fig.add_subplot(grid[0, 0])
    pr_axis = fig.add_subplot(grid[0, 1])
    roc_axis.set_title('a   ROC curves', loc='left', fontweight='bold')
    pr_axis.set_title('Precision–recall curves', loc='left', fontweight='bold')
    roc_axis.set(xlabel='False positive rate', ylabel='True positive rate', xlim=(0, 1), ylim=(0, 1.02))
    pr_axis.set(xlabel='Recall', ylabel='Precision', xlim=(0, 1), ylim=(0, 1.02))
    roc_axis.plot([0, 1], [0, 1], '--', color='0.55', linewidth=1)

    summary = []
    bars = {'Default': {}, 'Oracle': {}}
    for cyp, models in models_by_cyp.items():
        for model in models:
            labels, scores, calls = zip(*grouped[model, cyp])
            observed, score, default_call = np.asarray(labels), np.asarray(scores), np.asarray(calls)
            if cyp == 'All':
                roc_auc = roc_auc_score(observed, score)
                ap = average_precision_score(observed, score)
                fpr, tpr, _ = roc_curve(observed, score)
                precision, recall, _ = precision_recall_curve(observed, score)
                roc_axis.plot(fpr, tpr, color=COLORS[model], label=f'{model}: {roc_auc:.3f}')
                pr_axis.plot(recall, precision, color=COLORS[model], label=f'{model}: {ap:.3f}')
                summary.extend(('Curves', model, cyp, metric, value, '', '', '')
                               for metric, value in (('AUROC', roc_auc), ('AUPRC', ap)))
            threshold = oracle_threshold(observed, score)
            for mode, predicted, cutoff in (('Default', default_call, ''),
                                             ('Oracle', score > threshold, threshold)):
                stats = bootstrap_scores(observed, predicted)
                bars[mode][model, cyp] = np.asarray(stats)
                n_metrics = len(METRICS) if cyp == 'All' else 1
                for index in range(n_metrics):
                    value, low, high = (series[index] for series in stats)
                    summary.append((mode, model, cyp, METRICS[index], value, low, high, cutoff))

    labels = [row[0] for row in grouped[MODELS[0], 'All']]
    pr_axis.axhline(np.mean(labels), linestyle='--', color='0.55', linewidth=1)
    roc_axis.legend(loc='lower right', fontsize=8, title='AUROC')
    pr_axis.legend(loc='upper left', bbox_to_anchor=(1.01, 1), fontsize=8, title='AUPRC')
    for position, mode, title in ((1, 'Default', 'b   Released operating points'),
                                  (3, 'Oracle', 'c   Oracle sensitivity analysis')):
        aggregate = fig.add_subplot(grid[position, :])
        aggregate.set_title(title, loc='left', fontweight='bold')
        plot_bars(aggregate, bars[mode], MODELS, METRICS, subtype=False, oracle=mode == 'Oracle')
        subtypes = fig.add_subplot(grid[position + 1, :])
        plot_bars(subtypes, bars[mode], SUBTYPE_MODELS, SUBTYPES, subtype=True, oracle=mode == 'Oracle')

    fig.savefig(output / 'figure3.png', dpi=200)
    plt.close(fig)
    write_csv(output / 'figure3_metrics.csv',
              ('panel', 'model', 'cyp', 'metric', 'value', 'ci_low', 'ci_high', 'threshold'), summary)


def reproduce_table(output):
    groups = defaultdict(lambda: defaultdict(list))
    for row in read_csv(RESULTS / 'metabolites' / 'table3_predictions.csv.gz'):
        parent = groups[row['model']][row['compound_id']]
        if row['smiles']:
            parent.append(row['smiles'])

    generator = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    @cache
    def fingerprint(smiles):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f'Invalid SMILES: {smiles}')
        return generator.GetFingerprint(mol).ToBitString()

    reference = groups.pop('Reference')
    truth = {parent: [fingerprint(smiles) for smiles in entries] for parent, entries in reference.items()}
    n_true = sum(map(len, truth.values()))
    n_true_substrates = sum(bool(entries) for entries in truth.values())
    summary = []
    for model, by_parent in groups.items():
        matched = n_pred = matched_substrates = n_pred_substrates = 0
        for parent, observed in truth.items():
            predicted = by_parent.get(parent, [])
            fragments = {fingerprint(fragment) for cell in predicted
                         for fragment in cell.replace(',', '.').split('.') if fragment}
            matched += sum(value in fragments for value in observed)
            n_pred += len(predicted)
            n_pred_substrates += bool(predicted)
            matched_substrates += bool(predicted) and bool(observed)
        union = n_true + n_pred - matched
        f1 = 2 * matched / (n_true + n_pred)
        jaccard = matched / union
        precision = matched / n_pred if n_pred else 0
        recall = matched / n_true
        substrate_f1 = 2 * matched_substrates / (n_true_substrates + n_pred_substrates)
        summary.append((model, *(f'{value:.4f}' for value in (f1, jaccard, precision, recall, substrate_f1)),
                        matched, n_true - matched, n_pred))

    write_csv(output / 'table3.csv', ('Model', 'F1', 'Jaccard Score', 'Precision', 'Recall',
                                      'Substrate F1-score', 'Matched Metabolite', 'Missed Metabolite',
                                      'Output Size'), summary)


def main():
    parser = argparse.ArgumentParser(description='Reproduce the CYP-MAP benchmark figure and table.')
    parser.add_argument('--output-dir', type=Path, default=Path('benchmark_output'))
    output = parser.parse_args().output_dir
    output.mkdir(parents=True, exist_ok=True)
    reproduce_figure(output)
    reproduce_table(output)


if __name__ == '__main__':
    main()
