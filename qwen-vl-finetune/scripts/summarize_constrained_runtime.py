#!/usr/bin/env python3
"""Summarize existing rank-max step logs; no torch imports or GPU allocation."""
import argparse
import json
from pathlib import Path
import statistics


def read_measurement(experiment, kind, method, skip_steps):
    directory = experiment / kind / method
    summary_path = directory / "run_summary.json"
    log_path = directory / "step_metrics.jsonl"
    if not summary_path.exists() or not log_path.exists():
        return None
    summary = json.loads(summary_path.read_text())
    recipe = json.loads((directory / "launch_recipe.json").read_text())
    config = recipe["comparison"]["config"]
    rows = [json.loads(line) for line in log_path.read_text().splitlines() if line.strip()]
    # A resumed segment has a separate clock and its own startup overhead.
    groups = {}
    for row in rows:
        groups.setdefault(row.get("segment_start_step", 0), []).append(row)
    measured = [row for segment in groups.values() for row in segment[skip_steps:]]
    if not measured:
        raise ValueError(f"No measured steps after skipping {skip_steps} per segment: {directory}")
    if summary["profile"] != (kind == "profiles"):
        raise ValueError(f"Profile/run label mismatch: {directory}")
    seconds = statistics.median(row["seconds"] for row in measured)
    cycles = statistics.median(row["cycle_seconds"] for row in measured)
    return dict(
        method=method, complete=summary["complete"] and (directory / "EXPORT_COMPLETE").exists(),
        stop_reason=summary.get("stop_reason"), steps=summary["global_step"],
        measured_steps=len(measured), global_batch=config["global_batch"],
        max_pixels=config["max_pixels"], model_max_length=config["model_max_length"],
        median_update_seconds=seconds, median_cycle_seconds=cycles,
        examples_per_second=config["global_batch"] / seconds,
        peak_allocated_gib=max(row["peak_allocated_gib"] for row in rows),
        peak_reserved_gib=max(row["peak_reserved_gib"] for row in rows),
        # A rough training-only projection; checkpoint/export/data/startup are additional.
        projected_update_hours=seconds * config["max_steps"] / 3600,
        latest_segment_training_hours=summary["metrics"]["train_runtime"] / 3600,
    )


def summarize_matrix(spec, kind):
    from constrained_experiment import read
    experiment = spec.directory
    rows = []
    for bits in spec.bits:
        for method in ('qat', 'grace'):
            group = experiment / f'w{bits}g128'
            output = group / kind / method
            data = read_measurement(group, kind, method, 1)
            if data is None or not data['complete']:
                raise ValueError(f'Missing or incomplete output: {output}')
            if kind == 'runs':
                spec.validate_export(f'{method}_w{bits}')
            invocations = [json.loads(line) for line in (output / 'invocations.jsonl').read_text().splitlines()]
            wall = sum(row['wall_seconds'] for row in invocations)
            training = sum(row.get('train_runtime_seconds', 0) for row in invocations)
            row = dict(data, result_id=f'{method}_w{bits}',
                       trainable_parameters=sum(p['parameters'] for p in read(output / 'trainable_parameters.json').values()),
                       process_wall_hours=wall / 3600, process_gpu_hours=2 * wall / 3600,
                       recorded_training_hours=training / 3600,
                       outside_recorded_training_hours=(wall - training) / 3600)
            rows.append(row)
    lines = ['# Constrained-stage runtime and memory', '',
             f'Kind: {kind}. Costs cover only these {len(rows)} configured runs, including logged resumed/failed invocations.', '',
             '| Result | Updates | Trainable parameters | Process hours | GPU-hours (2 GPUs) | Recorded training hours | Other hours | Peak allocated GiB | Peak reserved GiB |',
             '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in rows:
        values = [row['result_id'], row['steps'], row['trainable_parameters']]
        values += [f'{row[key]:.3f}' for key in ('process_wall_hours', 'process_gpu_hours', 'recorded_training_hours',
                                                'outside_recorded_training_hours', 'peak_allocated_gib', 'peak_reserved_gib')]
        lines.append('| ' + ' | '.join(map(str, values)) + ' |')
    lines += ['', 'Process time is measured around torchrun and includes loading, training, checkpoints and export. '
              'Other hours are process time outside recorded Trainer intervals, including loading/export and failures without a Trainer summary; those components are not separately instrumented. '
              'CPU preparation/provenance and the author’s prior training are excluded. A hard kill of the launcher can prevent recording its invocation.', '',
              'Memory is the maximum PyTorch allocated/reserved value across ranks at logged training boundaries; loading/export peaks and total device occupancy are not measured. '
              'Profiles repeat the longest examples and do not measure representative training throughput.']
    report_dir = experiment / 'reports'
    report_dir.mkdir(parents=True, exist_ok=True)
    name = 'runtime_memory' if kind == 'runs' else 'profile_runtime_memory'
    (report_dir / f'{name}.md').write_text('\n'.join(lines) + '\n')
    (report_dir / f'{name}.json').write_text(json.dumps(rows, indent=2) + '\n')
    print('\n'.join(lines))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path)
    parser.add_argument("--config", type=Path, help="Combine all bit-width/method runs from a model matrix")
    parser.add_argument("--kind", choices=("profiles", "runs"), default="profiles")
    parser.add_argument("--methods", nargs="+", choices=("sft", "qat", "grace"), default=["sft", "qat", "grace"])
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--skip-steps", type=int, default=1)
    parser.add_argument("--require-complete", action="store_true")
    args = parser.parse_args()
    if args.config:
        if args.experiment:
            parser.error('Choose --config or --experiment')
        from constrained_experiment import Experiment
        summarize_matrix(Experiment(args.config), args.kind)
        return
    if args.experiment is None:
        parser.error('--experiment is required unless --config is supplied')
    if args.skip_steps < 0:
        parser.error("--skip-steps must be nonnegative")
    results = []
    lines = ["# Runtime and memory measurements", "",
             f"Experiment: `{args.experiment.resolve()}`", "",
             ("**Stress profile:** repeated longest retained examples; not representative training throughput."
              if args.kind == "profiles" else "**Training run:** actual retained-data sequence."), "",
             f"Medians exclude the first {args.skip_steps} update(s) of each invocation. Memory peaks include all logged updates.", "",
             "| Method | Complete | Updates / measured | Batch | Max pixels | Tokens | Median s/update | Median s/cycle | Examples/s | Peak allocated GiB | Peak reserved GiB | Projected update-only hours |",
             "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for method in args.methods:
        result = read_measurement(args.experiment, args.kind, method, args.skip_steps)
        if result is None:
            raise ValueError(f"Missing completed/logged output for {method}; run it before summarizing.")
        results.append(result)
        lines.append(f"| {method} | {result['complete']} | {result['steps']} / {result['measured_steps']} | "
                     f"{result['global_batch']} | {result['max_pixels']} | {result['model_max_length']} | "
                     f"{result['median_update_seconds']:.2f} | {result['median_cycle_seconds']:.2f} | "
                     f"{result['examples_per_second']:.2f} | {result['peak_allocated_gib']:.2f} | "
                     f"{result['peak_reserved_gib']:.2f} | {result['projected_update_hours']:.2f} |")
    lines += ["", "Cycle timing includes intervening checkpoint/data work. The hour projection excludes that work, loading and export; it is not an end-to-end guarantee.",
              "PyTorch memory peaks are maxima across ranks at logged training boundaries, not total device occupancy or model-loading/export peaks."]
    if args.baseline:
        lines += ["", f"Baseline: `{args.baseline.resolve()}` ({args.kind}; same timing calculation).", "",
                  "| Method | Baseline s/update | New s/update | Baseline examples/s | New examples/s | Throughput change | Reserved-memory change GiB |",
                  "| --- | ---: | ---: | ---: | ---: | ---: | ---: |"]
        for result in results:
            baseline = read_measurement(args.baseline, args.kind, result["method"], args.skip_steps)
            if baseline is None:
                lines.append(f"| {result['method']} | unavailable | — | — | — | — | — |")
                continue
            result["baseline"] = baseline
            change = (result["examples_per_second"] / baseline["examples_per_second"] - 1) * 100
            lines.append(f"| {result['method']} | {baseline['median_update_seconds']:.2f} | "
                         f"{result['median_update_seconds']:.2f} | {baseline['examples_per_second']:.2f} | "
                         f"{result['examples_per_second']:.2f} | {change:+.1f}% | "
                         f"{result['peak_reserved_gib'] - baseline['peak_reserved_gib']:+.2f} |")
        lines += ["", "Batch sizes and retained inputs differ. Compare examples/s as well as seconds/update. Stress profiles are input-envelope comparisons, not identical-data ablations."]
    output = args.experiment / args.kind
    output.mkdir(parents=True, exist_ok=True)
    suffix = "_".join(args.methods)
    report = output / f"runtime_memory_{suffix}.md"
    report.write_text("\n".join(lines) + "\n")
    report.with_suffix(".json").write_text(json.dumps(results, indent=2) + "\n")
    print("\n".join(lines))
    print(f"\nSaved: {report}")
    if args.require_complete and any(not result["complete"] for result in results):
        raise SystemExit("Run paused/incomplete: inspect stop_reason and checkpoints before continuing.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError) as error:
        raise SystemExit(f"ERROR: {error}")
