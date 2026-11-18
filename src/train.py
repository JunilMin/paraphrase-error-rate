"""Train each task in sequence, save checkpoints, and resume completed stages."""

import random
from pathlib import Path
import torch
from .data import prepare_data, load_split
from .evaluate import evaluate, sequence_log_likelihood
from .analysis import build_table
from .data import read_json, write_json, read_jsonl, write_jsonl
from .model import load_model, seed_everything
from .model import encode_candidates, collate_sequences
from .paraphrase import prepare_paraphrases, measure_inconsistency

def train_stage(model, tokenizer, task, rows, cfg, seed):
    tr = cfg['training']; seed_everything(seed); rng = random.Random(seed)
    model.train()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                                 lr=tr['learning_rate'], weight_decay=tr['weight_decay'])
    # Exact example exposure, including the final partial accumulation window.
    stream = []
    for _ in range(tr['epochs']):
        order = list(range(len(rows))); rng.shuffle(order); stream.extend(order)
    device = next(model.parameters()).device
    window = tr['batch_size'] * tr['gradient_accumulation']
    losses = []; steps = 0
    for start in range(0, len(stream), window):
        indices = stream[start:start + window]; optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.0
        for b in range(0, len(indices), tr['batch_size']):
            chosen = [rows[i] for i in indices[b:b + tr['batch_size']]]
            sequences = [encode_candidates(tokenizer, task, r['original_text'], cfg['evaluation']['max_length'])[r['gold_label']] for r in chosen]
            batch = collate_sequences(sequences, tokenizer.pad_token_id, device)
            sums, counts = sequence_log_likelihood(model, batch)
            # Mean per-example continuation NLL per token; not test-set loss.
            objective = (-sums / counts).sum() / len(indices)
            objective.backward(); loss_sum += objective.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), tr['max_grad_norm'])
        optimizer.step(); steps += 1; losses.append(loss_sum)
        if steps % 25 == 0: print(f"{task['key']}: optimizer step {steps}, NLL {loss_sum:.4f}", flush=True)
    return {'unique_examples_used': len(set(stream)), 'examples_seen': len(stream), 'optimizer_steps': steps,
            'effective_epochs': len(stream) / len(rows), 'batch_size': tr['batch_size'],
            'gradient_accumulation': tr['gradient_accumulation'], 'learning_rate': tr['learning_rate'],
            'optimizer_reinitialized_at_stage': True, 'mean_training_token_nll': sum(losses) / len(losses),
            'training_token_nll_by_step': losses, 'stage_seed': seed}

def resolve_checkpoint(run, stage):
    """Resolve a saved adapter relative to its run directory."""
    path = Path(run) / 'checkpoints' / f'after_{stage}'
    if not (path / 'adapter_config.json').exists():
        raise FileNotFoundError(f'Checkpoint is missing: {path}. Restore the adapter files to resume training.')
    return str(path)

def run_seed(cfg, root, seed, loader=None, model_loader=None):
    root = Path(root).resolve()
    run = root / 'results' / ('pilot' if cfg['mode'] == 'pilot' else 'main') / f'seed_{seed}'
    run.mkdir(parents=True, exist_ok=True)
    state_path = run / 'run.json'
    if state_path.exists():
        state = read_json(state_path)
        if state['config'] != cfg or state['seed'] != seed:
            raise ValueError('This seed already has a different configuration. Preserve its output before starting a new experiment.')
    else:
        state = {'seed': seed, 'config': cfg, 'completed': [], 'checkpoints': {}, 'exposure': {}, 'status': 'created'}
        write_json(state_path, state)
    if state['status'] == 'complete':
        print(f'Already complete: {run}', flush=True); return run
    folder, metadata = prepare_data(cfg, root, seed, loader=loader)
    evaluation = load_split(folder, metadata, cfg['tasks'][0], 'eval')
    model = tokenizer = None
    for index, stage in enumerate('ABC'):
        if stage in state['completed']: continue
        if stage == 'B' and cfg['mode'] == 'main' and not (run / 'inconsistency.jsonl').exists():
            # Freeze measurements at A; subsequent task outcomes are not used.
            originals = read_jsonl(run / 'after_A.jsonl')
            pool = [r for r in originals if r['correct']]
            if not pool: raise ValueError('No A-correct examples; inspect the pilot before proceeding')
            candidates = prepare_paraphrases(cfg, root, run, pool, originals, seed)
            if candidates is None:
                state['status'] = 'needs_paraphrases'; write_json(state_path, state)
                print(f'A saved. Supply {cfg["paraphrase"]["file"]} and rerun the same command. Run: {run}', flush=True)
                return run
            if model is None:
                seed_everything(seed)
                model, tokenizer = (model_loader or load_model)(cfg, checkpoint=resolve_checkpoint(run, 'A'), trainable=True)
            measure_inconsistency(model, tokenizer, cfg, run, pool, candidates)
        if model is None:
            seed_everything(seed)
            previous = resolve_checkpoint(run, 'ABC'[index - 1]) if index else None
            model, tokenizer = (model_loader or load_model)(cfg, checkpoint=previous, trainable=True)
        state['status'] = 'training_' + stage; write_json(state_path, state)
        rows = load_split(folder, metadata, cfg['tasks'][index], 'train')
        print(f'Task {stage}: {cfg["tasks"][index]["key"]}, {len(rows)} training examples', flush=True)
        exposure = train_stage(model, tokenizer, cfg['tasks'][index], rows, cfg, seed + index * 100003)
        checkpoint = run / 'checkpoints' / f'after_{stage}'
        checkpoint.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(checkpoint); tokenizer.save_pretrained(checkpoint)
        predictions = evaluate(model, tokenizer, cfg['tasks'][0], evaluation, cfg)
        write_jsonl(run / f'after_{stage}.jsonl', predictions)
        state['completed'].append(stage); state['checkpoints'][stage] = str(checkpoint.relative_to(run))
        state['exposure'][stage] = exposure; state['status'] = 'completed_' + stage
        write_json(state_path, state)
    build_table(run)
    state['status'] = 'complete'; write_json(state_path, state)
    return run
