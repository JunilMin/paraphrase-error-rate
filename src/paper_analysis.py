"""Calculate the manuscript’s post hoc comparisons from saved example results."""
from pathlib import Path
import hashlib
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from .data import read_json, write_json, write_table
from .analysis import analyze_seed, paired_differences, metrics


def paper_analysis(runs, output, bootstrap_repeats=5000):
    output = Path(output)
    tables = output / 'tables'; figures = output / 'figures'
    tables.mkdir(parents=True, exist_ok=True); figures.mkdir(parents=True, exist_ok=True)
    distributions = []; binary_rates = []; conditioned = []; transitions = []
    redundancy = []; predictions = []; scores = []; sources = []
    protocol = None
    seen = set()
    for run in map(Path, runs):
        state = read_json(run / 'run.json')
        if state['status'] != 'complete' or state['config']['mode'] != 'main':
            raise ValueError('Paper analysis requires completed main runs')
        if protocol is not None and protocol != state['config']:
            raise ValueError('Cannot combine different experiment settings')
        protocol = state['config']
        seed = state['seed']
        if seed in seen: raise ValueError('Duplicate seed')
        seen.add(seed)
        path = run / 'results.csv'
        sources.append({'seed': seed, 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()})
        frame = pd.read_csv(path, dtype={'example_id': str, 'text_group': str})
        if frame.example_id.duplicated().any(): raise ValueError('Duplicate example ID')
        pool = frame[frame.correct_after_A.eq(True)].copy()
        if pool[['inconsistency', 'confidence', 'entropy', 'margin', 'gold_nll']].isna().any().any():
            raise ValueError('Incomplete eligible-example features; no silent exclusions')
        if not np.isclose(pool.inconsistency.to_numpy()[:, None], [0, 1/3, 2/3, 1]).any(axis=1).all():
            raise ValueError('PER is not a three-paraphrase fraction')
        if not pool.n_paraphrases.eq(3).all(): raise ValueError('Expected three paraphrases')
        redundancy.append({'seed': seed, 'n_eligible': len(pool),
            'duplicate_text_groups': len(pool) - pool.text_group.nunique(),
            'max_abs_gold_nll_plus_log_confidence': float(np.abs(pool.gold_nll + np.log(pool.confidence)).max()),
            'spearman_confidence_negative_gold_nll': float(spearmanr(pool.confidence, -pool.gold_nll).statistic)})
        for level in [0, 1/3, 2/3, 1]:
            part = pool[np.isclose(pool.inconsistency, level)]
            row = {'seed': seed, 'per': level, 'count': len(part), 'share': len(part)/len(pool)}
            for stage in 'BC':
                row[f'events_{stage}'] = int(part[f'forgotten_after_{stage}'].sum())
                row[f'rate_{stage}'] = part[f'forgotten_after_{stage}'].mean() if len(part) else np.nan
            distributions.append(row)
        for positive in [False, True]:
            part = pool[pool.inconsistency.gt(0).eq(positive)]
            row = {'seed': seed, 'per_positive': positive, 'count': len(part), 'share': len(part)/len(pool)}
            for stage in 'BC':
                row[f'events_{stage}'] = int(part[f'forgotten_after_{stage}'].sum())
                row[f'rate_{stage}'] = part[f'forgotten_after_{stage}'].mean() if len(part) else np.nan
            binary_rates.append(row)
        # Match the manuscript's rank-based quintile with original row order for ties.
        low = pool.sort_values('confidence', kind='stable').iloc[:int(np.ceil(len(pool)*.2))]
        for stage in 'BC':
            target = f'forgotten_after_{stage}'
            events = int(pool[target].sum())
            conditioned.append({'seed': seed, 'endpoint': stage, 'n_eligible': len(pool),
                'n_low_confidence': len(low), 'spearman_per_confidence': float(spearmanr(pool.inconsistency, pool.confidence).statistic),
                'total_events': events, 'low_confidence_events': int(low[target].sum()),
                'event_share_low_confidence': float(low[target].sum()/events) if events else np.nan,
                **metrics(low[target].astype(int), low.inconsistency)})
        a = frame.correct_after_A.eq(True); b = frame.correct_after_B.eq(True); c = frame.correct_after_C.eq(True)
        row = {'seed': seed, 'n_evaluated': len(frame), 'n_A_wrong': int((~a).sum()),
            'accuracy_A': a.mean(), 'accuracy_B': b.mean(), 'accuracy_C': c.mean(), 'accuracy_drop_A_C': a.mean()-c.mean(),
            'A_wrong_B_correct': int((~a & b).sum()), 'A_wrong_C_correct': int((~a & c).sum()),
            'B_transitions': int((a & ~b).sum()), 'B_recovered_C': int((a & ~b & c).sum()),
            'C_transitions': int((a & ~c).sum()), 'C_new_since_B': int((a & b & ~c).sum())}
        row.update(A_wrong_B_correct_rate=row['A_wrong_B_correct']/row['n_A_wrong'] if row['n_A_wrong'] else np.nan,
                   A_wrong_C_correct_rate=row['A_wrong_C_correct']/row['n_A_wrong'] if row['n_A_wrong'] else np.nan,
                   B_recovery_rate=row['B_recovered_C']/row['B_transitions'] if row['B_transitions'] else np.nan)
        transitions.append(row)
        result, _ = analyze_seed(frame, protocol['statistics']['folds'], predictions,
                                  baseline_features=['confidence', 'entropy', 'margin'])
        scores.append(result[result.predictor.isin(['baseline_combined', 'baseline_plus_inconsistency'])])
    tables_to_write = {'per_distribution': pd.DataFrame(distributions), 'per_binary_rates': pd.DataFrame(binary_rates),
        'confidence_conditioned': pd.DataFrame(conditioned), 'reverse_transitions': pd.DataFrame(transitions),
        'feature_redundancy': pd.DataFrame(redundancy), 'three_feature_scores': pd.concat(scores, ignore_index=True)}
    values = tables_to_write['three_feature_scores'].melt(id_vars=['seed', 'endpoint', 'predictor'],
                                                       value_vars=['auroc', 'average_precision'], var_name='metric')
    tables_to_write['three_feature_summary'] = values.groupby(['endpoint', 'predictor', 'metric'], as_index=False).agg(
        mean=('value', 'mean'), std=('value', 'std'), n_seeds=('value', 'count'))
    oof = pd.concat(predictions, ignore_index=True)
    tables_to_write['three_feature_oof'] = oof
    tables_to_write['three_feature_paired_differences'] = paired_differences(oof, bootstrap_repeats)
    for name, table in tables_to_write.items(): write_table(tables / f'{name}.csv', table)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    dist = tables_to_write['per_distribution']
    fig, axes = plt.subplots(1, len(seen), figsize=(9, 3.5), squeeze=False)
    for ax, (seed, part) in zip(axes[0], dist.groupby('seed')):
        ax.bar(range(4), part.share*100)
        ax.set(xticks=range(4), xticklabels=['0', '1/3', '2/3', '1'], title=f'Seed {seed}',
               xlabel='Paraphrase error rate', ylabel='Eligible examples (%)', ylim=(0, 100))
    fig.tight_layout(); fig.savefig(figures/'per_distribution.png', dpi=300); plt.close(fig)
    write_json(output/'analysis_metadata.json', {'sources': sources, 'bootstrap_repeats': bootstrap_repeats,
        'features': ['confidence', 'entropy', 'margin'], 'low_confidence_fraction': .2,
        'quintile_rule': 'ceil(0.2*n); stable confidence sort, original row order breaks ties',
        'exploratory': True, 'language_model_retrained': False,
        'interval_scope': 'fixed OOF predictions; paired text-group percentile bootstrap; no multiplicity correction'})
    print(f'Paper analyses saved to {output}', flush=True)
    return tables_to_write
