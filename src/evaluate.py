"""Score class-label continuations and compute predictions and uncertainty features."""

import math
import numpy as np
import torch
from .model import encode_candidates, collate_sequences

def sequence_log_likelihood(model, batch):
    logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], use_cache=False).logits
    target = batch['labels'][:, 1:]
    valid = target != -100
    # Select continuation positions before the vocabulary softmax; prompt logits
    # must never contribute to either the score or supervised objective.
    positions = logits[:, :-1, :][valid].float()
    token_logp = -torch.nn.functional.cross_entropy(positions, target[valid], reduction='none')
    row_ids = torch.arange(target.shape[0], device=target.device)[:, None].expand_as(target)[valid]
    sums = torch.zeros(target.shape[0], device=target.device).scatter_add(0, row_ids, token_logp)
    return sums, valid.sum(1)

def probability_features(scores, gold):
    scores = np.asarray(scores, dtype=float)
    if not np.isfinite(scores).all() or len(scores) < 2: raise ValueError('Invalid candidate scores')
    weights = np.exp(scores - scores.max()); p = weights / weights.sum()
    order = np.argsort(scores)
    return {'pred': int(p.argmax()), 'correct': bool(p.argmax() == gold),
            'confidence': float(p.max()), 'entropy': float(-(p * np.log(np.clip(p, 1e-300, 1))).sum()),
            'margin': float(scores[order[-1]] - scores[order[-2]]),
            'gold_nll': float(-math.log(max(float(p[gold]), 1e-300))),
            'probabilities': p.tolist(), 'candidate_scores': scores.tolist()}

def score_text(model, tokenizer, task, text, cfg):
    settings = cfg['evaluation']; device = next(model.parameters()).device
    candidates = encode_candidates(tokenizer, task, text, settings['max_length'])
    result = []
    for start in range(0, len(candidates), settings['candidate_batch_size']):
        batch = collate_sequences(candidates[start:start + settings['candidate_batch_size']], tokenizer.pad_token_id, device)
        scores, counts = sequence_log_likelihood(model, batch)
        if settings['length_normalization'] == 'mean': scores = scores / counts
        result.extend(scores.detach().cpu().tolist())
    return result

def evaluate(model, tokenizer, task, rows, cfg):
    model.eval(); result = []
    with torch.inference_mode():
        for i, row in enumerate(rows):
            features = probability_features(score_text(model, tokenizer, task, row['original_text'], cfg), row['gold_label'])
            result.append(dict(row, **features))
            if (i + 1) % 250 == 0: print(f'Evaluated {i + 1}/{len(rows)}', flush=True)
    return result
