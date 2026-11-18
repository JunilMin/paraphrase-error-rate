# Count the two reviewers’ ratings and report AI judgments separately.
import argparse
import collections
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        rows = list(csv.DictReader(stream))
    ids = [r['review_id'] for r in rows]
    if len(ids) != len(set(ids)):
        raise ValueError(f'Duplicate review IDs in {path.name}')
    return {r['review_id']: r for r in rows}


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(folder, output):
    inputs = {name: folder / (name + '.csv') for name in
              ['reviewer_1', 'reviewer_2', 'assistant_adjudication_200']}
    first, second, assistant = [read_csv(p) for p in inputs.values()]
    if set(first) != set(second) or len(first) != 600:
        raise ValueError('The independent forms must cover the same 600 pairs.')
    expected = {k for k in first if
                first[k]['meaning_preserved'] != second[k]['meaning_preserved'] or
                first[k]['meaning_preserved'] == 'uncertain' or
                first[k]['label_preserved'] != second[k]['label_preserved'] or
                first[k]['label_preserved'] == 'uncertain'}
    if set(assistant) != expected or len(expected) != 200:
        raise ValueError('The AI file must cover exactly the 200 unresolved pairs.')
    sources = collections.Counter(k.split(':')[0] for k in first)
    if len(sources) != 200 or set(sources.values()) != {3}:
        raise ValueError('Expected 200 source examples with three paraphrases each.')
    matrix = collections.Counter()
    merged = []
    for key, one in first.items():
        two = second[key]
        for col in ['original_text', 'paraphrase_text']:
            if one[col] != two[col] or (key in assistant and one[col] != assistant[key][col]):
                raise ValueError(f'Changed input text: {key}, {col}')
        for row in [one, two]:
            if any(row[c] not in {'yes', 'no', 'uncertain'} for c in
                   ['meaning_preserved', 'label_preserved']):
                raise ValueError(f'Invalid human response: {key}')
        matrix[(one['meaning_preserved'], two['meaning_preserved'])] += 1
        row = dict(review_id=key, original_text=one['original_text'],
                   paraphrase_text=one['paraphrase_text'],
                   reviewer_1_meaning=one['meaning_preserved'],
                   reviewer_2_meaning=two['meaning_preserved'],
                   reviewer_1_label=one['label_preserved'],
                   reviewer_2_label=two['label_preserved'])
        if key in assistant:
            ai = assistant[key]
            if ai['consensus_status'] != 'pending_human_confirmation' or ai['decision_provenance'] != 'assistant_independent_adjudication_not_two_reviewer_consensus':
                raise ValueError(f'Unexpected AI provenance: {key}')
            for prefix, human in [('r1', one), ('r2', two)]:
                for dest, source in [('meaning', 'meaning_preserved'), ('error_type', 'error_type')]:
                    if ai[f'{prefix}_{dest}'] != human[source]:
                        raise ValueError(f'Changed original response: {key}')
            meaning, label = ai['final_meaning_preserved'], ai['final_label_preserved']
            if meaning not in {'yes', 'no', 'uncertain'} or label not in {'yes', 'no', 'uncertain'}:
                raise ValueError(f'Invalid AI judgment: {key}')
            error, note = ai['final_error_type'], ai['adjudication_notes']
            if not note.strip() or (meaning != 'yes' and not error.strip()):
                raise ValueError(f'Missing AI explanation: {key}')
            provenance, status = 'ai_assisted_pending_human_confirmation', 'pending_human_confirmation'
        else:
            meaning, label = one['meaning_preserved'], one['label_preserved']
            # Keep both reviewers’ descriptions when they differ.
            error = ' | '.join(dict.fromkeys(x for x in [one['error_type'], two['error_type']] if x))
            note = ' | '.join(dict.fromkeys(x for x in [one['notes'], two['notes']] if x))
            provenance, status = 'two_independent_human_ratings_agree', 'independent_agreement'
        row.update(assisted_meaning=meaning, assisted_label=label, error_type=error,
                   notes=note, decision_provenance=provenance, confirmation_status=status)
        merged.append(row)
    n = len(first)
    agreement = sum(v for (a, b), v in matrix.items() if a == b)
    counts = lambda data, col: {v: sum(r[col] == v for r in data) for v in ['yes', 'no', 'uncertain']}
    h1, h2 = counts(first.values(), 'meaning_preserved'), counts(second.values(), 'meaning_preserved')
    expected_agreement = sum(h1[v] * h2[v] for v in h1) / n**2
    sample = json.loads((folder / 'sample.json').read_text(encoding='utf-8-sig'))
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / 'meaning_confusion_matrix.csv', [dict(reviewer_1=a, reviewer_2=b, count=matrix[(a,b)])
              for a in ['yes','no','uncertain'] for b in ['yes','no','uncertain']])
    write_csv(output / 'independent_ratings_summary.csv', [dict(reviewer=name, criterion=criterion,
              judgment=value, count=count, denominator=n, proportion=count/n)
              for name, data in [('reviewer_1', first), ('reviewer_2', second)]
              for criterion in ['meaning_preserved','label_preserved']
              for value, count in counts(data.values(), criterion).items()])
    write_csv(output / 'ai_assisted_labels_600.csv', merged)
    group = collections.defaultdict(list)
    for row in merged:
        group[row['review_id'].split(':')[0]].append(row['assisted_meaning'])
    summary = dict(status='independent_human_ratings_with_separate_ai_assisted_resolution',
        imported_on='2026-10-17', human_review_provenance='User reports two different people reviewed independently.',
        ai_resolution_provenance='AI resolutions approved for supplementary reporting; no joint human confirmation.',
        pairs=n, source_examples=len(sources), paraphrases_per_source=3,
        sampling=sample, reviewer_1_meaning=h1, reviewer_2_meaning=h2,
        reviewer_1_label=counts(first.values(),'label_preserved'),
        reviewer_2_label=counts(second.values(),'label_preserved'),
        meaning_agreement_count=agreement, meaning_agreement_rate=agreement/n,
        meaning_cohens_kappa=(agreement/n-expected_agreement)/(1-expected_agreement),
        label_agreement_rate=sum(first[k]['label_preserved']==second[k]['label_preserved'] for k in first)/n,
        label_cohens_kappa=None, label_kappa_reason='Both reviewers assigned yes to every pair; kappa is undefined.',
        disagreement_pairs=sum(a!=b for a,b in ((first[k]['meaning_preserved'],second[k]['meaning_preserved']) for k in first)),
        both_uncertain_pairs=matrix[('uncertain','uncertain')], unresolved_for_ai=len(expected),
        ai_resolution_counts=counts(assistant.values(),'final_meaning_preserved'),
        ai_assisted_combined_meaning=counts(merged,'assisted_meaning'),
        ai_assisted_combined_label=counts(merged,'assisted_label'),
        combined_sources_all_three_yes=sum(all(x=='yes' for x in values) for values in group.values()),
        pending_human_confirmation=len(expected), exhaustive_semantic_validation=False,
        frozen_bank_changed=False, model_results_changed=False,
        input_sha256={name+'.csv':hashlib.sha256(path.read_bytes()).hexdigest() for name,path in inputs.items()},
        limitations=['Pairs are clustered within 200 sources; they are not 600 independent source samples.',
          'The sampling frame excludes sources from the earlier 50-candidate preparation review.',
          'Reviewer disagreement is substantial; AI resolution is not human consensus.',
          'Broad category preservation does not establish semantic equivalence.',
          'The review was added after the experiments; the bank and model outcomes were not changed.'])
    (output/'summary.json').write_text(json.dumps(summary,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
    print(json.dumps({k:summary[k] for k in ['pairs','source_examples','meaning_agreement_rate','meaning_cohens_kappa','ai_assisted_combined_meaning','combined_sources_all_three_yes']},indent=2))
    return summary


if __name__ == '__main__':
    parser=argparse.ArgumentParser(description='Summarize human ratings separately from AI-assisted resolutions; no model inference.')
    parser.add_argument('--review-dir',type=Path,default=Path(__file__).resolve().parents[1]/'results/human_review')
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    summarize(args.review_dir,args.output or args.review_dir/'analysis')
