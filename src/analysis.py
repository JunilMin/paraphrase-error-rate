"""Build example-level forgetting outcomes, compare predictors, and plot saved results."""

from pathlib import Path
import pandas as pd
from .data import read_json, read_jsonl, write_table

def transition(correct_A, correct_B, correct_C):
    return {'forgotten_after_B': not correct_B if correct_A else None,
            'forgotten_after_C': not correct_C if correct_A else None}

def sample_report(frame):
    pool = frame[frame.correct_after_A.eq(True)]
    row = {'seed': int(frame.seed.iloc[0]), 'n_evaluated_A': len(frame),
           'n_correct_after_A': len(pool), 'n_with_inconsistency': int(pool.inconsistency.notna().sum())}
    notes = []
    for stage in 'BC':
        events = int(pool[f'forgotten_after_{stage}'].eq(True).sum())
        row[f'n_forgotten_after_{stage}'] = events
        row[f'forgetting_rate_{stage}'] = events / len(pool) if len(pool) else None
        if min(events, len(pool) - events) == 0:
            notes.append(f'{stage}: one outcome class or no A-correct examples; AUROC/AP unavailable')
        elif min(events, len(pool) - events) < 30:
            notes.append(f'{stage}: few minority events; estimates may be unstable')
    row['note'] = '; '.join(notes)
    return pd.DataFrame([row])

def build_table(run):
    run = Path(run); state = read_json(run / 'run.json')
    rows = {stage: read_jsonl(run / f'after_{stage}.jsonl') for stage in 'ABC'}
    maps = {stage: {r['example_id']: r for r in records} for stage, records in rows.items()}
    if any(len(maps[s]) != len(rows[s]) or set(maps[s]) != set(maps['A']) for s in 'ABC'):
        raise ValueError('Checkpoint evaluation IDs do not match')
    path = run / 'inconsistency.jsonl'
    sem = {r['example_id']: r for r in read_jsonl(path)} if path.exists() else {}
    result = []
    for identifier, a in maps['A'].items():
        row = {k: a[k] for k in ['example_id', 'original_text', 'gold_label', 'text_group']}
        row.update(seed=state['seed'], confidence=a['confidence'], entropy=a['entropy'],
                   margin=a['margin'], gold_nll=a['gold_nll'], inconsistency=None, n_paraphrases=0)
        if a['correct']: row.update(sem.get(identifier, {}))
        for stage in 'ABC':
            r = maps[stage][identifier]
            if r['original_text'] != a['original_text'] or r['gold_label'] != a['gold_label']:
                raise ValueError('Example changed across stages')
            row[f'pred_after_{stage}'] = r['pred']
            row[f'correct_after_{stage}'] = r['correct']
        row.update(transition(row['correct_after_A'], row['correct_after_B'], row['correct_after_C']))
        result.append(row)
    frame = pd.DataFrame(result)
    write_table(run / 'results.csv', frame)
    summary = sample_report(frame)
    write_table(run / 'summary.csv', summary)
    print(summary.to_string(index=False), flush=True)
    return frame

from pathlib import Path
import warnings
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold, StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from .data import read_json, write_table, write_json

BASELINE = ['confidence', 'entropy', 'margin', 'gold_nll']
SIGNALS = {'inconsistency': 1, 'confidence': -1, 'entropy': 1, 'margin': -1, 'gold_nll': 1}

def metrics(y, scores):
    y = np.asarray(y, dtype=int)
    if len(np.unique(y)) < 2:
        return {'auroc': None, 'average_precision': None}
    return {'auroc': float(roc_auc_score(y, scores)), 'average_precision': float(average_precision_score(y, scores))}

def make_folds(y, groups, requested, seed):
    y = np.asarray(y, dtype=int); groups = np.asarray(groups)
    if len(np.unique(y)) < 2: return []
    limit = min(requested, *(len(np.unique(groups[y == c])) for c in [0, 1]))
    if limit < requested: warnings.warn(f'CV folds reduced from {requested} to at most {limit}')
    for k in range(limit, 1, -1):
        splitter = (StratifiedKFold(k, shuffle=True, random_state=seed) if len(np.unique(groups)) == len(groups)
                    else StratifiedGroupKFold(k, shuffle=True, random_state=seed))
        splits = list(splitter.split(np.zeros(len(y)), y, groups)) if isinstance(splitter, StratifiedGroupKFold) else list(splitter.split(np.zeros(len(y)), y))
        if all(len(np.unique(y[a])) == 2 and len(np.unique(y[b])) == 2 for a, b in splits): return splits
    return []

def analyze_seed(frame, folds=5, predictions=None, baseline_features=None):
    baseline_features = BASELINE if baseline_features is None else list(baseline_features)
    seed = int(frame.seed.iloc[0])
    pool = frame[frame.correct_after_A.eq(True)]
    # Every predictor is compared on the same complete-case A-correct examples.
    data = pool.dropna(subset=list(SIGNALS)).copy()
    if len(data) < len(pool): warnings.warn(f'Seed {seed}: paraphrase coverage {len(data)}/{len(pool)}; inspect selection bias')
    scores = []; associations = []
    for stage in 'BC':
        target = f'forgotten_after_{stage}'
        subset = data.dropna(subset=[target]); y = subset[target].astype(int).to_numpy()
        note = ''
        minority = min(int((y == 0).sum()), int((y == 1).sum()))
        if minority == 0: note = 'One outcome class or empty pool; AUROC/AP unavailable'
        elif minority < 30: note = 'Few minority events; estimates may be unstable'
        if note: warnings.warn(f'Seed {seed}, after {stage}: {note}')
        base = {'seed': seed, 'endpoint': stage, 'n_eligible': len(pool), 'n_analyzed': len(subset), 'n_forgotten': int(y.sum())}
        for signal, sign in SIGNALS.items():
            scores.append(dict(base, predictor=signal, evaluation='fixed ranking', folds=0,
                               **metrics(y, sign * subset[signal].to_numpy()), note=note))
            association = None
            if subset[signal].nunique() > 1 and len(np.unique(y)) == 2:
                association = float(spearmanr(subset[signal], y).statistic)
            associations.append(dict(base, signal=signal, spearman_r=association))
        splits = make_folds(y, subset.text_group.to_numpy(), folds, seed)
        for name, features in [('baseline_combined', baseline_features), ('baseline_plus_inconsistency', baseline_features + ['inconsistency'])]:
            values = {'auroc': None, 'average_precision': None}
            if splits:
                predicted = np.empty(len(y))
                fold_ids = np.empty(len(y), dtype=int)
                for fold_id, (train, test) in enumerate(splits):
                    model = make_pipeline(StandardScaler(), LogisticRegression(C=1., max_iter=3000, random_state=seed))
                    model.fit(subset.iloc[train][features], y[train])
                    predicted[test] = model.predict_proba(subset.iloc[test][features])[:, 1]
                    fold_ids[test] = fold_id
                if predictions is not None:
                    predictions.append(pd.DataFrame({'seed': seed, 'endpoint': stage,
                        'example_id': subset.example_id.to_numpy(), 'text_group': subset.text_group.to_numpy(),
                        'target': y, 'fold': fold_ids, 'predictor': name, 'probability': predicted}))
                values = metrics(y, predicted)
            scores.append(dict(base, predictor=name, evaluation='out-of-fold logistic regression',
                               folds=len(splits), **values, note=note if splits else 'Insufficient independent outcome groups for CV'))
    return pd.DataFrame(scores), pd.DataFrame(associations)


def paired_differences(oof, repeats=5000):
    """Estimate paired percentile intervals conditional on fixed OOF predictions."""
    if repeats < 1:
        raise ValueError('Bootstrap repeats must be positive')
    rows = []
    for (seed, endpoint), part in oof.groupby(['seed', 'endpoint']):
        wide = part.pivot(index=['example_id', 'text_group', 'target', 'fold'],
                          columns='predictor', values='probability').reset_index()
        y = wide.target.to_numpy()
        baseline = wide.baseline_combined.to_numpy()
        augmented = wide.baseline_plus_inconsistency.to_numpy()
        groups = wide.groupby('text_group', sort=True).indices
        members = list(groups.values())
        rng_seed = int(seed) * 100 + ord(endpoint)
        rng = np.random.default_rng(rng_seed)
        observed_base = metrics(y, baseline)
        observed_aug = metrics(y, augmented)
        draws = {'auroc': [], 'average_precision': []}
        for _ in range(repeats):
            indices = np.concatenate([members[i] for i in rng.integers(len(members), size=len(members))])
            if len(np.unique(y[indices])) < 2:
                continue
            a = metrics(y[indices], augmented[indices])
            b = metrics(y[indices], baseline[indices])
            for metric in draws:
                draws[metric].append(a[metric] - b[metric])
        for metric, values in draws.items():
            if not values:
                raise ValueError('No valid two-class bootstrap samples')
            low, high = np.quantile(values, [.025, .975])
            rows.append({'seed': seed, 'endpoint': endpoint, 'metric': metric,
                         'baseline': observed_base[metric], 'augmented': observed_aug[metric],
                         'difference': observed_aug[metric] - observed_base[metric],
                         'ci_low': low, 'ci_high': high, 'valid_repeats': len(values),
                         'requested_repeats': repeats, 'bootstrap_seed': rng_seed,
                         'method': 'paired text-group cluster percentile bootstrap; fixed OOF predictions'})
        print(f'Paired intervals: seed {seed}, after {endpoint}', flush=True)
    return pd.DataFrame(rows)

def analyze(runs, output, bootstrap_repeats=5000):
    output = Path(output); comparisons = []; correlations = []; summaries = []
    protocol = None; seen = set(); predictions = []
    for run in runs:
        run = Path(run); state = read_json(run / 'run.json')
        if state['config']['mode'] != 'main' or state['status'] != 'complete':
            raise ValueError(f'Analysis requires a completed main run: {run}')
        if state['seed'] in seen: raise ValueError('Provide only one run per seed')
        if protocol is not None and protocol != state['config']: raise ValueError('Do not combine different protocols')
        protocol = state['config']; seen.add(state['seed'])
        frame = pd.read_csv(run / 'results.csv', dtype={'example_id': str})
        scores, associations = analyze_seed(frame, protocol['statistics']['folds'], predictions)
        comparisons.append(scores); correlations.append(associations); summaries.append(sample_report(frame))
    if not comparisons: raise ValueError('No completed main experiments found')
    scores = pd.concat(comparisons, ignore_index=True)
    write_table(output / 'predictor_scores.csv', scores)
    write_table(output / 'associations.csv', pd.concat(correlations, ignore_index=True))
    write_table(output / 'sample_summary.csv', pd.concat(summaries, ignore_index=True))
    long = scores.melt(id_vars=['seed', 'endpoint', 'predictor'], value_vars=['auroc', 'average_precision'], var_name='metric')
    summary = long.groupby(['endpoint', 'predictor', 'metric'], as_index=False).agg(mean=('value', 'mean'), std=('value', 'std'), n_seeds=('value', 'count'))
    write_table(output / 'seed_summary.csv', summary)
    if predictions:
        oof = pd.concat(predictions, ignore_index=True)
        write_table(output / 'oof_predictions.csv', oof)
        write_table(output / 'paired_differences.csv', paired_differences(oof, bootstrap_repeats))
    quality_path = Path(runs[0]).parents[2] / 'results/paraphrase_quality.json'
    if quality_path.exists():
        quality = read_json(quality_path)
        write_json(output / 'paraphrase_quality.json', quality)
        lines = ['Experiment outputs from the frozen paraphrase bank.', '']
        for key, value in quality.items():
            lines.append(f'{key}: {value}')
        lines.extend(['', 'Quality metadata are copied from the recorded JSON; assistant review is not independent human validation.'])
        (output / 'README.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print(f'Analyzed seeds {sorted(seen)}. Results: {output}', flush=True)
    return scores

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def paper_figures(runs, tables, output):
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    frames = [pd.read_csv(Path(run) / 'results.csv') for run in runs]
    rows = pd.concat(frames, ignore_index=True)
    rows = rows[rows.correct_after_A.eq(True)].dropna(subset=['inconsistency'])
    plt.rcParams.update({'font.size': 9, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    for stage, ax in zip('BC', axes):
        for seed, part in rows.groupby('seed'):
            rate = part.groupby('inconsistency')[f'forgotten_after_{stage}'].mean()
            ax.plot(rate.index, rate.values, 'o-', label=f'Seed {seed}')
        if len(rows): ax.legend()
        else: ax.text(.5, .5, 'No inconsistency measurements', ha='center', transform=ax.transAxes)
        ax.set(title=f'After {stage}', xlabel='Paraphrase inconsistency', ylabel='Forgetting rate', ylim=(0, 1))
    fig.tight_layout(); fig.savefig(output / 'forgetting_by_inconsistency.png', dpi=300); plt.close(fig)
    scores = pd.read_csv(Path(tables) / 'predictor_scores.csv')
    predictors = list(scores.predictor.unique()); fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for row, stage in enumerate('BC'):
        for col, metric in enumerate(['auroc', 'average_precision']):
            ax = axes[row, col]
            for offset, (seed, part) in enumerate(scores[scores.endpoint.eq(stage)].groupby('seed')):
                part = part.set_index('predictor').reindex(predictors)
                ax.scatter(np.arange(len(predictors)) + offset * .07, part[metric], label=f'Seed {seed}')
            ax.set_xticks(range(len(predictors)), predictors, rotation=35, ha='right')
            ax.set(title=f'After {stage}: {metric}', ylim=(0, 1))
            if row == 0 and col == 0: ax.legend()
    fig.tight_layout(); fig.savefig(output / 'predictor_performance.png', dpi=300); plt.close(fig)
