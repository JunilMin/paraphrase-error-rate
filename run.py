"""Prepare paraphrases, run sequential training, or analyze saved results."""

import argparse
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from filelock import FileLock, Timeout
from src.data import load_config, read_json
from src.analysis import analyze, paper_figures


def run_workflow(cfg, root, seeds, prepare_only=False):
    from src.train import run_seed
    from src.data import prepare_data, load_split
    from src.paraphrase import ensure_shared_paraphrases
    root = Path(root)
    runs = [root / 'results/main' / f'seed_{seed}' for seed in seeds]
    incomplete = []
    for seed, run in zip(seeds, runs):
        if (run / 'run.json').exists():
            state = read_json(run / 'run.json')
            if state['config'] != cfg or state['seed'] != seed:
                raise ValueError(f'Existing run has different settings: {run}')
            if state['status'] == 'complete': continue
        incomplete.append(seed)
    if incomplete or prepare_only:
        if cfg['data']['eval_limit_A'] is not None:
            raise ValueError('Shared main workflow expects the full official AG News test split')
        print('[1/3] Preparing shared paraphrases (completed examples are reused).', flush=True)
        folder, metadata = prepare_data(cfg, root, seeds[0])
        originals = load_split(folder, metadata, cfg['tasks'][0], 'eval')
        ensure_shared_paraphrases(cfg, root, originals)
    if prepare_only:
        print('Preparation complete. Sample: results/cache/paraphrases/review_sample.csv', flush=True)
        return
    for index, seed in enumerate(seeds, 1):
        print(f'[2/3] Main experiment seed {seed} ({index}/{len(seeds)}).', flush=True)
        run = run_seed(cfg, root, seed)
        if read_json(run / 'run.json')['status'] != 'complete':
            raise RuntimeError(f'Main experiment paused: {run}. Analysis has not run.')
    print('[3/3] Statistical tables and figures.', flush=True)
    output = root / 'results/tables'
    analyze(runs, output); paper_figures(runs, output, root / 'results/figures')
    print(f'FINISHED. Tables: {output}; figures: {root / "results/figures"}', flush=True)
    print('Paraphrase quality notes: results/paraphrase_quality.json', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Shared paraphrases -> all main seeds -> tables and figures')
    parser.add_argument('--seed', type=int, help='Optional single seed; default: all configured seeds')
    parser.add_argument('--config', type=Path, default=ROOT / 'config.yaml')
    parser.add_argument('--prepare-only', action='store_true', help='Generate shared paraphrases and review sample, without training')
    parser.add_argument('--pilot', action='store_true', help='Run pilot training only')
    parser.add_argument('--analyze-only', action='store_true', help='Read completed main runs without training')
    parser.add_argument('--export-review', action='store_true', help='Create blank independent-review forms; no training')
    parser.add_argument('--reevaluate-bank', type=Path, help='Reevaluate a revised full bank using saved A checkpoints')
    parser.add_argument('--output', type=Path, help='New directory for revised-bank results')
    parser.add_argument('--bootstrap-repeats', type=int, default=5000, help='Paired interval resamples for analysis-only and paper-analysis')
    parser.add_argument('--paper-analysis', action='store_true', help='Reproduce exploratory analyses in the working paper from saved results')
    args = parser.parse_args()
    if args.bootstrap_repeats < 1: parser.error('Bootstrap repeats must be positive')
    if args.reevaluate_bank and not args.output: parser.error('--reevaluate-bank requires --output')
    if args.output and not args.reevaluate_bank: parser.error('--output requires --reevaluate-bank')
    if sum([args.pilot, args.analyze_only, args.prepare_only, args.export_review, bool(args.reevaluate_bank), args.paper_analysis]) > 1:
        parser.error('Choose at most one optional mode')
    cfg = load_config(args.config, pilot=args.pilot)
    seeds = [args.seed] if args.seed is not None else cfg['seeds']
    if not seeds or len(set(seeds)) != len(seeds): parser.error('Seeds must be nonempty and unique')
    (ROOT / 'results').mkdir(exist_ok=True)
    try:
        with FileLock(str(ROOT / 'results/.experiment.lock'), timeout=0):
            if args.paper_analysis:
                from src.paper_analysis import paper_analysis
                runs = [ROOT / 'results/main' / f'seed_{seed}' for seed in seeds]
                paper_analysis(runs, ROOT / 'results/paper_analysis', args.bootstrap_repeats)
            elif args.export_review:
                from src.data import export_human_review
                print(export_human_review(ROOT))
            elif args.reevaluate_bank:
                from src.paraphrase import reevaluate_bank
                print(reevaluate_bank(cfg, ROOT, seeds, args.reevaluate_bank, args.output))
            elif args.pilot:
                from src.train import run_seed
                for seed in seeds:
                    run_seed(cfg, ROOT, seed)
            elif args.analyze_only:
                runs = [ROOT / 'results/main' / f'seed_{seed}' for seed in seeds]
                output = ROOT / 'results/tables'
                analyze(runs, output, bootstrap_repeats=args.bootstrap_repeats)
                paper_figures(runs, output, ROOT / 'results/figures')
            else:
                run_workflow(cfg, ROOT, seeds, args.prepare_only)
    except Timeout:
        parser.exit(1, 'Another main workflow is already running. Do not start it twice.\n')
    except KeyboardInterrupt:
        parser.exit(130, '\nStopped. Rerun the same command to resume; unfinished training stages restart.\n')
