# Paraphrase Error Rate

Code and saved evidence for a study of whether errors on paraphrased inputs measured after an initial task identify examples that later change from correct to incorrect during sequential fine-tuning.

## Study

The original experiment fine-tunes Qwen2.5-1.5B-Instruct with LoRA on AG News, then Yelp Review Full, then DBpedia-14. The same adapter carries forward within a seed. Each task uses 10,000 training examples once. The full 7,600-example AG News test set is evaluated after every task, using seeds 13, 42, and 77.

PER is the fraction of three stored variants that the Task-A model misclassifies against the source label, measured for sources initially classified correctly. Confidence is the maximum normalized probability of the allowed label continuations, including EOS, calculated from that same model's token scores. These are measured signals, not two competing language models. Logistic regression tests whether adding PER improves prediction beyond uncertainty features.

The original results show higher later-error rates in the small PER-positive subgroup. Confidence provides stronger standalone rankings. Adding PER produces small, seed-dependent gains rather than consistent improvement. The study does not establish causal effects or irreversible forgetting. See the manuscript and saved tables for the scope of each analysis.

## Snapshot status

- Original experiment: three seeds complete.
- Additional seeds 101, 202, 303, 404, and 505: complete.
- Reversed order, AG News → DBpedia-14 → Yelp Review Full, seeds 13, 42, and 77: complete.
- Second model, SmolLM2-1.7B-Instruct, three planned seeds: ongoing when this snapshot was prepared. Partial-cohort outputs are not presented as final evidence here.
- K=5: extra variants and actual quality review are still pending. No K=5 result is claimed.
- The author supplied additional context-preservation judgments for the original 200 unresolved pairs. These are separate from the original independent review, and do not establish strict semantic equivalence.

The JSON in `docs/SNAPSHOT.json` records the capture time and included runs. This repository is a snapshot, not a live monitor of the background jobs.

## Layout

| Path | Contents |
|---|---|
| `run.py`, `src/` | Byte-preserved released experiment and analysis code |
| `config.yaml` | Original pinned model, dataset, training and scoring settings |
| `evidence/` | Gzip-compressed saved outputs, bank, provenance and SHA-256 manifest |
| `analysis/` | Read-only saved-output audit |
| `followup/prepare.py` | Prepare separate follow-up workspaces without training |
| `followup/configs/` | Follow-up configurations |
| `followup/runtime_snapshots/` | As-executed Windows controller sources, retained for provenance |
| `review/` | Human review forms, validator, submitted judgments and restoration trace |
| `paper/` | Manuscript v14, reference verification and subsequent human-review addendum |
| `notebooks/` | Original notebooks with outputs and execution counts cleared |
| `tests/` | Offline checks for leakage controls, scoring and evidence restoration |

## Reproduce from saved outputs

Restore only into a new or identical destination. Differing existing files cause an error; they are not overwritten.

```sh
python tools/restore_evidence.py --cohort original --destination results
python analysis/audit_saved_results.py --project . --out ../PER_saved_output_audit
```

The audit writes outside the source project to preserve its read-only scope. It uses the standard library and checks saved values without fitting models. To reproduce statistical analyses, use an environment with `requirements-analysis.txt` installed:

```sh
python run.py --analyze-only
python run.py --paper-analysis
python src/review_analysis.py
```

The original four-feature baseline uses confidence, entropy, margin and gold_nll. The labeled post hoc sensitivity analysis removes gold_nll because it is a deterministic transform of confidence on the initially correct pool. Five-fold training-only scaling is retained. Bootstrap intervals condition on fixed out-of-fold predictions; they do not measure new-training uncertainty or justify treating seeds as independent datasets.

## New training and follow-up reproduction

Model weights, LoRA checkpoints, virtual environments and processed training caches are excluded. Fresh LLM runs require compatible training dependencies, sufficient GPU resources, and access to the pinned Hugging Face resources. `requirements-lock.txt` records an observed environment rather than a platform-independent installation guarantee.

After restoring the original bank and metadata above, prepare a **new sibling workspace**:

```sh
python followup/prepare.py --project . --scenario more_seeds --destination ../PER_runs/more_seeds
python followup/prepare.py --project . --scenario reverse_order --destination ../PER_runs/reverse_order
python followup/prepare.py --project . --scenario second_model --destination ../PER_runs/second_model
```

In a prepared workspace, use its recorded commands:

```sh
python run.py --config followup_config.json
python followup_analysis.py
```

To reanalyze a completed follow-up cohort instead of retraining it, restore its evidence into that prepared workspace's `results` directory, then use `--analyze-only` and `followup_analysis.py`. For a fresh original three-seed experiment, use a separate code checkout and `restore_evidence.py --bank-only`; do not import completed run states into that fresh run.

The runtime snapshots contain the original machine paths and controller identities. They are historical sources, not portable launch commands. Use the preparation CLI above for a new environment. The v2 controller changes status publication and completed-run reuse, not the frozen model, data, exposure or outcome definitions.

## Human review

Open `review/human_review/PER_200_pairs_review.html` for the original blank review interface. Existing submitted responses are preserved separately in `review/completed/`; reopening the blank form does not load those responses automatically.

The author retained the main event and overall context as the criterion, allowing some factual differences: 163 of the 200 selected unresolved pairs were accepted, 37 rejected, and all retained the news category. This is an author-reported additional assessment, not consensus between the original reviewers. Do not pool it with the original 600-pair independent agreement of 68.67% or reinterpret it as strict fact-level equivalence. Excel-converted identifiers were restored through unique exact text-pair matching; the answers were not changed.

## Verification and data handling

```sh
python -m unittest discover -s tests -v
```

The tests need the training dependencies for CPU scoring imports; they do not download models or run GPU training. Gzip contents and restored bytes are checked against the manifest. The source manifest records hashes of the preserved experiment files. No token, credential, model weight or active log is needed to reproduce the saved-output audit.

Data revisions and label orders are in `config.yaml`. The bank and outputs remain tied to their original sources. Numerical screens are not semantic validation. New variants must be reviewed separately; favorable outputs must not replace original variants or determine which sources enter the analysis.

The paper file predates the additional experiments. Read it with the human-review addendum and snapshot status; it is not a claim that the ongoing follow-up has finished.
