"""Load the Qwen model and LoRA adapter, and tokenize prompts and label candidates."""

def prompt_text(task, text):
    return f"{task['instruction']}\nAllowed labels: {' | '.join(task['labels'])}\nText: {text}\nAnswer:"

def encode_candidates(tokenizer, task, text, max_length):
    # Training and evaluation tokenize the prompt and answer separately,
    # with EOS included in the answer.
    endings = [tokenizer.encode(label, add_special_tokens=False) + [tokenizer.eos_token_id]
               for label in task['labels']]
    if tokenizer.eos_token_id is None or any(len(e) < 2 for e in endings):
        raise ValueError('Labels require nonempty tokens and an EOS token')
    def encode(body):
        msg = [{'role': 'user', 'content': prompt_text(task, body)}]
        if getattr(tokenizer, 'chat_template', None):
            return tokenizer.apply_chat_template(msg, tokenize=True, add_generation_prompt=True)
        return tokenizer.encode(prompt_text(task, body) + '\n', add_special_tokens=False)
    budget = max_length - max(map(len, endings))
    if len(encode('')) > budget:
        raise ValueError('max_length cannot fit instructions, labels and continuation')
    body_tokens = tokenizer.encode(text, add_special_tokens=False)
    prompt = encode(text)
    # Preserve instructions and the answer marker; truncate only the text.
    while len(prompt) > budget:
        body_tokens = body_tokens[:max(0, len(body_tokens) - max(1, len(prompt) - budget))]
        prompt = encode(tokenizer.decode(body_tokens, skip_special_tokens=True))
    return [(prompt + ending, [-100] * len(prompt) + ending) for ending in endings]

def collate_sequences(sequences, pad_id, device):
    import torch
    width = max(len(ids) for ids, _ in sequences)
    ids = []; targets = []; masks = []
    for tokens, labels in sequences:
        pad = width - len(tokens)
        ids.append(tokens + [pad_id] * pad)
        targets.append(labels + [-100] * pad)
        masks.append([1] * len(tokens) + [0] * pad)
    return {k: torch.tensor(v, dtype=torch.long, device=device)
            for k, v in [('input_ids', ids), ('labels', targets), ('attention_mask', masks)]}

import os
import random
import numpy as np
import torch

def seed_everything(seed):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False

def load_tokenizer(cfg):
    from transformers import AutoTokenizer
    m = cfg['model']
    tok = AutoTokenizer.from_pretrained(m['name'], revision=m['tokenizer_revision'],
                                      local_files_only=m['local_files_only'], trust_remote_code=False)
    if tok.pad_token_id is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    return tok

def load_model(cfg, checkpoint=None, trainable=True):
    from transformers import AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model, PeftModel
    m = cfg['model']; tr = cfg['training']
    device = m['device']
    if device == 'auto': device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dtype = m['dtype']
    if dtype == 'auto': dtype = 'bfloat16' if device.startswith('cuda') and torch.cuda.is_bf16_supported() else 'float32'
    if device == 'cpu' and dtype != 'float32': raise ValueError('Use float32 on CPU')
    source = str(checkpoint) if checkpoint and tr['method'] == 'full' else m['name']
    model = AutoModelForCausalLM.from_pretrained(source,
        revision=None if source != m['name'] else m['revision'], torch_dtype=getattr(torch, dtype),
        local_files_only=m['local_files_only'], trust_remote_code=False, attn_implementation='eager')
    if tr['method'] == 'lora':
        if checkpoint:
            model = PeftModel.from_pretrained(model, str(checkpoint), is_trainable=trainable)
        else:
            model = get_peft_model(model, LoraConfig(task_type='CAUSAL_LM', r=tr['lora_rank'],
                lora_alpha=tr['lora_alpha'], lora_dropout=tr['lora_dropout'], target_modules=tr['lora_targets']))
    model.to(device)
    if trainable and tr['gradient_checkpointing']:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={'use_reentrant': False})
        if hasattr(model, 'enable_input_require_grads'): model.enable_input_require_grads()
    model.config.use_cache = False
    return model, load_tokenizer(cfg)
