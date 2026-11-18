"""Load settings, prepare dataset splits, remove training overlap, and save records."""

import hashlib
import json
import time
from pathlib import Path
from uuid import uuid4

def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False).encode()).hexdigest()

def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))

def _replace_with_retry(tmp, path):
    """Allow brief Windows file locks while preserving atomic replacement."""
    for attempt in range(10):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            if attempt == 9:
                raise
            time.sleep(0.5)

def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + uuid4().hex[:8] + '.tmp')
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False), encoding='utf-8')
    _replace_with_retry(tmp, path)

def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding='utf-8-sig').splitlines() if line.strip()]

def write_jsonl(path, rows):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name('.' + uuid4().hex[:8] + '.tmp')
    tmp.write_text(''.join(json.dumps(r, ensure_ascii=False, allow_nan=False) + '\n' for r in rows), encoding='utf-8')
    _replace_with_retry(tmp, path)

def write_table(path, frame):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False, encoding='utf-8-sig')

from pathlib import Path
import yaml

def load_config(path, pilot=False):
    cfg = yaml.safe_load(Path(path).read_text(encoding='utf-8'))
    if len(cfg['tasks']) != 3 or len({t['key'] for t in cfg['tasks']}) != 3:
        raise ValueError('Specify three distinct tasks in A/B/C order')
    for task in cfg['tasks']:
        if len(task['labels']) != len(task['dataset_labels']) or len(set(task['labels'])) != len(task['labels']):
            raise ValueError('Candidate label indices must match dataset labels')
    if pilot:
        cfg['training']['examples_per_task'] = cfg['pilot']['examples_per_task']
        cfg['data']['eval_limit_A'] = cfg['pilot']['eval_examples_A']
    cfg['mode'] = 'pilot' if pilot else 'main'
    for key in ['examples_per_task', 'batch_size', 'gradient_accumulation', 'epochs']:
        if cfg['training'][key] < 1: raise ValueError(f'{key} must be positive')
    if cfg['paraphrase']['count'] < 1 or cfg['statistics']['folds'] < 2:
        raise ValueError('Invalid paraphrase count or fold count')
    return cfg

from collections import Counter, defaultdict
import random
import re
from pathlib import Path

def normalized(text):
    return re.sub(r'\s+', ' ', text).strip().casefold()

def format_records(dataset, task, split):
    names = getattr(dataset.features.get(task.get('label_field', 'label')), 'names', None)
    if names is not None and list(names) != task['dataset_labels']:
        raise ValueError(f"Unexpected label order for {task['key']}: {names}")
    result = []
    for i, row in enumerate(dataset):
        text = '\n'.join(str(row[f]) for f in task['text_fields']).strip()
        label = int(row[task.get('label_field', 'label')])
        if not text or not 0 <= label < len(task['labels']):
            raise ValueError(f'Invalid example {i}')
        result.append({'example_id': str(i), 'original_text': text, 'gold_label': label,
                       'text_group': digest(normalized(text)), 'split': split, 'task': task['key']})
    return result

def stratified_sample(rows, size, seed):
    if size is None:
        return list(rows)
    if size > len(rows) or size < len({r['gold_label'] for r in rows}):
        raise ValueError('Requested sample must fit available data and contain every class')
    groups = defaultdict(list)
    for r in rows:
        groups[r['gold_label']].append(r)
    rng = random.Random(seed)
    quotas = {k: size * len(v) / len(rows) for k, v in groups.items()}
    counts = {k: int(v) for k, v in quotas.items()}
    for k in sorted(groups, key=lambda k: (-(quotas[k] - counts[k]), k))[:size - sum(counts.values())]:
        counts[k] += 1
    if min(counts.values()) == 0:
        raise ValueError('Sample too small for stratification; increase size')
    result = []
    for k in sorted(groups):
        rng.shuffle(groups[k]); result.extend(groups[k][:counts[k]])
    rng.shuffle(result)
    return result

def clean_training(train, evaluation):
    """Drop training groups with conflicting labels.

    Evaluation overlap has precedence; audit counts are disjoint and sum to
    the total removed rows. Evaluation records themselves are never modified.
    """
    excluded = {r['text_group'] for r in evaluation}
    first_label = {}; conflicting = set()
    for r in train:
        key = r['text_group']
        if key in excluded:
            continue
        if key in first_label and first_label[key] != r['gold_label']:
            conflicting.add(key)
        else:
            first_label.setdefault(key, r['gold_label'])
    conflict_records = defaultdict(list)
    unique = {}; overlaps = duplicates = 0
    for r in train:
        key = r['text_group']
        if key in excluded:
            overlaps += 1; continue
        if key in conflicting:
            conflict_records[key].append({'example_id': r['example_id'], 'gold_label': r['gold_label']})
            continue
        if key in unique:
            duplicates += 1; continue
        unique[key] = r
    audit = [{'text_group': key, 'labels': sorted({r['gold_label'] for r in records}),
              'n_rows': len(records), 'records': records}
             for key, records in sorted(conflict_records.items())]
    return list(unique.values()), {'train_eval_overlap_removed': overlaps, 'train_duplicates_removed': duplicates,
                                'train_conflict_policy': 'exclude_entire_conflicting_group_after_eval_overlap_removal',
                                'train_conflicting_groups_removed': len(audit),
                                'train_conflicting_rows_removed': sum(r['n_rows'] for r in audit),
                                'train_conflict_audit': audit,
                                'eval_duplicates_retained': len(evaluation) - len(excluded)}

def prepare_data(cfg, root, seed, loader=None):
    from datasets import load_dataset
    loader = loader or load_dataset
    key = digest({'format_version': 4, 'tasks': cfg['tasks'], 'data': cfg['data'],
                  'train_size': cfg['training']['examples_per_task'], 'seed': seed})[:16]
    folder = Path(root) / 'results' / 'cache' / 'processed' / key
    if (folder / 'manifest.json').exists():
        return folder, read_json(folder / 'manifest.json')
    meta = {'format_version': 4, 'seed': seed, 'tasks': {}}
    loaded = []
    for task in cfg['tasks']:
        kwargs = {'revision': task['revision'], 'cache_dir': str(Path.home() / '.cache' / 'paper_datasets')}
        if task.get('config_name'): kwargs['name'] = task['config_name']
        if task.get('data_files'): kwargs['data_files'] = task['data_files']
        ds = loader(task['dataset'], **kwargs)
        train = format_records(ds[task.get('train_split', 'train')], task, 'train')
        evaluation = format_records(ds[task.get('eval_split', 'test')], task, 'test')
        loaded.append((task, train, evaluation))
    all_evaluation = [r for _, _, evaluation in loaded for r in evaluation]
    for index, (task, train, evaluation) in enumerate(loaded):
        source_train_size, source_eval_size = len(train), len(evaluation)
        train, checks = clean_training(train, all_evaluation)
        print(f"{task['key']}: removed {checks['train_eval_overlap_removed']} evaluation-overlap rows, "
              f"{checks['train_duplicates_removed']} same-label duplicate rows, "
              f"{checks['train_conflicting_rows_removed']} rows in "
              f"{checks['train_conflicting_groups_removed']} conflicting-label groups", flush=True)
        checks['eval_duplicates_retained'] = len(evaluation) - len({r['text_group'] for r in evaluation})
        checks['overlap_scope'] = 'All three official evaluation splits excluded from every training task'
        train_size = cfg['training']['examples_per_task']
        train = stratified_sample(train, train_size, seed)
        if index == 0:
            evaluation = stratified_sample(evaluation, cfg['data']['eval_limit_A'], seed)
        item = {'dataset': task['dataset'], 'revision': task['revision'], 'checks': checks,
                'source_train_size': source_train_size, 'source_eval_size': source_eval_size,
                'train_size': len(train), 'eval_size': len(evaluation) if index == 0 else 0,
                'train_class_counts': dict(Counter(r['gold_label'] for r in train)),
                'eval_class_counts': dict(Counter(r['gold_label'] for r in evaluation)) if index == 0 else {}}
        splits = [('train', train)] + ([('eval', evaluation)] if index == 0 else [])
        for split, rows in splits:
            item[split] = task['key'] + '_' + split + '.jsonl'
            write_jsonl(folder / item[split], rows)
        meta['tasks'][task['key']] = item
    write_json(folder / 'manifest.json', meta)
    return folder, meta

def load_split(folder, meta, task, split):
    rows = read_jsonl(Path(folder) / meta['tasks'][task['key']][split])
    return rows

def export_human_review(root, sample_size=200, seed=20261004):
    """Export an unscored source-level sample for two independent reviewers."""
    import csv
    root = Path(root)
    bank_path = root / 'results/cache/paraphrases/ag_news.jsonl'
    bank = read_jsonl(bank_path)
    sources = {r['example_id']: r['original_text'] for r in read_jsonl(root / 'results/main/seed_13/after_A.jsonl')}
    previous = root / 'results/cache/paraphrases/review_sample.csv'
    excluded = set()
    if previous.exists():
        with previous.open(encoding='utf-8-sig', newline='') as handle:
            excluded = {str(r['example_id']) for r in csv.DictReader(handle)}
    eligible = sorted((r for r in bank if str(r['example_id']) not in excluded),
                      key=lambda r: int(r['example_id']))
    selected = random.Random(seed).sample(eligible, sample_size)
    output = root / 'results/human_review'
    if output.exists() and any(output.iterdir()):
        raise FileExistsError('Review files already exist; preserve reviewer responses.')
    output.mkdir(parents=True, exist_ok=True)
    columns = ['review_id', 'original_text', 'paraphrase_text', 'meaning_preserved',
               'label_preserved', 'error_type', 'notes']
    records = [{'review_id': f"{r['example_id']}:{i}", 'original_text': sources[str(r['example_id'])],
                'paraphrase_text': candidate, 'meaning_preserved': '', 'label_preserved': '',
                'error_type': '', 'notes': ''}
               for r in selected for i, candidate in enumerate(r['paraphrases'])]
    for reviewer in [1, 2]:
        with (output / f'reviewer_{reviewer}.csv').open('w', encoding='utf-8-sig', newline='') as handle:
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader(); writer.writerows(records)
    write_json(output / 'sample.json', {'seed': seed, 'source_examples': sample_size,
        'pairs': len(records), 'excluded_previous_source_examples': sorted(excluded),
        'bank_sha256': __import__('hashlib').sha256(bank_path.read_bytes()).hexdigest(),
        'sampling': 'uniform source-example sample; all three candidates per source',
        'status': 'awaiting independent human review'})
    (output / 'README.md').write_text(
        '# Independent paraphrase review\n\n'
        'Two reviewers should work independently and without model predictions or forgetting outcomes. '
        'Read the complete source and candidate. Enter yes, no, or uncertain for meaning_preserved '
        'and label_preserved. Check entities, numbers, time, negation, qualifiers, and added or omitted facts. '
        'Label preservation means that the same AG News category remains appropriate: World, Sports, '
        'Business, or Sci/Tech. Record error_type and notes for no or uncertain decisions. '
        'Leave uncertain cases unresolved until discussion. After both reviews, retain the original '
        'responses and create a separate adjudicated table. Do not edit the frozen bank directly.\n\n'
        'Sampling is at source-example level; the three candidates within a source are correlated. '
        'Report disagreement and quality estimates with this clustering in mind. These blank forms '
        'are not evidence that validation has been completed.\n', encoding='utf-8')
    return output
