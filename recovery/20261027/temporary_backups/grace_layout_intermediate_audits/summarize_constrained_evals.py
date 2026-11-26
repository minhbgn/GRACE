#!/usr/bin/env python3
"""Combine the seven local evaluation suites using only the standard library."""

import argparse
import csv
import json
import math
import os
from pathlib import Path


METHODS = ("sft", "qat", "grace")
LABELS = {"baseline": "Vanilla 2B", "sft": "SFT", "qat": "QAT", "grace": "GRACE"}
SUITES = {
    "hallusionbench": ("HallusionBench aAcc", "aAcc"),
    "mmbench": ("MMBench EN dev (per-row)", "overall_acc"),
    "scienceqa": ("ScienceQA test (all)", "overall_acc"),
    "ai2d": ("AI2D test", "overall_acc"),
    "mmmu": ("MMMU validation", "overall_acc"),
    "seedbench_img": ("SEED-Bench-IMG", "overall_acc"),
    "mmstar": ("MMStar", "overall_acc"),
}
ID_FIELDS = {
    "hallusionbench": ("category", "subcategory", "set_id", "figure_id",
                      "question_id", "visual_input"),
    "mmbench": ("index",), "scienceqa": ("pid",), "ai2d": ("idx",),
    "mmmu": ("id",),
    # SEED's source assigns one question to two different question types.
    "seedbench_img": ("question_id", "data_id", "question_type_id"),
    "mmstar": ("index",),
}


def load_results(root, methods=METHODS):
    data = {}
    for suite, (_, metric) in SUITES.items():
        reference = None
        for method in methods:
            path = root / method / f"{suite}.json"
            payload = json.loads(path.read_text())
            summary, results = payload["summary"], payload["results"]
            n = len(results)
            correct = sum(bool(row["correct"]) for row in results)
            if not n or n != summary["total"] or correct != summary["correct"]:
                raise ValueError(f"{path}: result counts disagree with summary")
            if not math.isclose(correct / n, summary[metric], abs_tol=1e-12):
                raise ValueError(f"{path}: accuracy disagrees with predictions")
            identities = {
                tuple(row[key] for key in ID_FIELDS[suite]): row["gold"]
                for row in results
            }
            if len(identities) != n:
                raise ValueError(f"{path}: duplicate question identities")
            if reference is not None and identities != reference:
                raise ValueError(f"{path}: question identities/labels differ across methods")
            reference = identities
            # ScienceQA does not include an error count in its summary.
            summary["row_errors"] = sum(bool(row.get("error")) for row in results)
            data[method, suite] = summary
    return data


def comparison_row(label, values, counts, methods=METHODS):
    if len(set(counts)) != 1:
        raise ValueError(f"{label}: denominators differ across methods: {counts}")
    scores = dict(zip(methods, values))
    references = [m for m in ("baseline", "qat", "sft") if m in methods]
    return [label, counts[0], *(100 * value for value in values),
            *(100 * (scores["grace"] - scores[m]) for m in references)]


def markdown_table(rows, methods=METHODS):
    headers = ["Suite / metric", "N"] + [f"{LABELS[m]} (%)" for m in methods]
    headers += [f"GRACE−{LABELS[m]} (pp)" for m in ("baseline", "qat", "sft") if m in methods]
    lines = ["| " + " | ".join(headers) + " |",
             "| --- | " + " | ".join(["---:"] * (len(headers) - 1)) + " |"]
    for label, n, *values in rows:
        label = label.replace("|", "\\|").replace("\n", " ")
        scores = [f"{v:.2f}" for v in values[:len(methods)]]
        deltas = [f"{v:+.2f}" for v in values[len(methods):]]
        lines.append("| " + " | ".join([label, str(n)] + scores + deltas) + " |")
    return "\n".join(lines)


def make_report(data, root, methods=METHODS):
    def build_row(label, values, counts):
        return comparison_row(label, values, counts, methods)
    primary, extra, breakdown = [], [], []
    for suite, (label, metric) in SUITES.items():
        summaries = [data[m, suite] for m in methods]
        primary.append(build_row(label, [s[metric] for s in summaries],
                                      [s["total"] for s in summaries]))
        if suite == "hallusionbench":
            for key in ("qAcc", "fAcc"):
                extra.append(build_row(f"HallusionBench {key}",
                    [s[key] for s in summaries], [s[f"{key}_total_groups"] for s in summaries]))
        for field in sorted(k for k in summaries[0] if k.startswith("by_")):
            if any(set(s[field]) != set(summaries[0][field]) for s in summaries):
                raise ValueError(f"{suite}/{field}: categories differ across methods")
            for category in summaries[0][field]:
                groups = [s[field][category] for s in summaries]
                row = build_row(f"{suite} / {field} / {category}",
                    [g["acc"] for g in groups], [g["total"] for g in groups])
                breakdown.append(row)
                if suite == "scienceqa" and field == "by_modality":
                    extra.append(build_row(f"ScienceQA test ({category})",
                        [g["acc"] for g in groups], [g["total"] for g in groups]))
    means = [sum(data[m, s][metric] for s, (_, metric) in SUITES.items()) / len(SUITES)
             for m in methods]
    primary.append(build_row("Unweighted mean (7 suites)", means, ["—"] * len(methods)))
    diagnostics = ["| Suite | " + " | ".join(f"{LABELS[m]} errors / parse fails" for m in methods) + " |",
                   "| --- | " + " | ".join(["---:"] * len(methods)) + " |"]
    for suite in SUITES:
        cells = [f"{data[m, suite]['row_errors']} / {data[m, suite].get('parse_fail', 'not reported')}"
                 for m in methods]
        diagnostics.append("| " + " | ".join([suite] + cells) + " |")
    report = "\n\n".join([
        "# " + " / ".join(LABELS[m] for m in methods) + " evaluation comparison",
        f"Source: `{root.resolve()}`. Read {len(methods) * len(SUITES)} merged JSON files (not rank shards). "
        "Validated row counts, correct counts, primary accuracies, unique question identities "
        "and matching identities/gold labels across methods.",
        "## Primary scores", markdown_table(primary, methods),
        "All scores are percentages; deltas are percentage points (pp). The mean gives each suite "
        "equal weight, uses HallusionBench aAcc and full-test ScienceQA, and is a descriptive "
        "aggregate, not an official benchmark metric.",
        "## Additional metrics", markdown_table(extra, methods),
        "HallusionBench qAcc requires every answer in a question group to be correct; fAcc "
        "requires every answer in an image/figure group to be correct. N is the number of groups. "
        "ScienceQA image is the image-containing subset of the full test split.",
        "## Execution and parsing", "\n".join(diagnostics),
        "Errors are counted from individual predictions. Parse failures remain in the accuracy "
        "denominator. MMMU does not report a separate parse-failure count.",
        "## Interpretation limits",
        "These are the repository's local evaluation protocols. MMBench uses per-row accuracy; "
        "MMMU uses the local open-answer scorer. Do not assume official leaderboard equivalence. "
        "Vanilla 2B, when present, is Qwen3-VL-2B-Instruct without additional fine-tuning, "
        "loaded in BF16. The default launcher uses the SFT processor to match its image budget; "
        "baseline result metadata records the actual processor and budget. "
        "SFT is the fine-tuned BF16 quality reference; QAT and GRACE use the matched W4G128 training/export "
        "setup described in README_constrained.md. These scores do not measure packed INT4 "
        "inference speed or deployment memory. One seed does not establish repeatability.",
        "## All category breakdowns", markdown_table(breakdown, methods),
    ]) + "\n"
    return report, primary + extra + breakdown


# Matrix-specific validation shares the same question IDs and scoring checks.
EXPECTED = dict(hallusionbench=1129, mmbench=4329, scienceqa=4241, ai2d=3088,
                mmmu=900, seedbench_img=14233, mmstar=1500)


def read(path):
    return json.loads(Path(path).read_text())


def validate_result(path, suite, model, attn):
    payload = read(path)
    summary, results = payload['summary'], payload['results']
    n = len(results)
    if n != EXPECTED[suite] or summary['total'] != n:
        raise ValueError(f'{path}: expected {EXPECTED[suite]} questions, got {n}')
    identities = {tuple(row[key] for key in ID_FIELDS[suite]): row['gold'] for row in results}
    if len(identities) != n:
        raise ValueError(f'{path}: duplicate question identities')
    correct = sum(bool(row['correct']) for row in results)
    if correct != summary['correct'] or not math.isclose(correct / n, summary[SUITES[suite][1]], abs_tol=1e-12):
        raise ValueError(f'{path}: prediction/summary mismatch')
    expected_policy = dict(version=1, min_pixels=4096, max_pixels=100352,
                           max_new_tokens=32 if suite == 'mmmu' else 8, do_sample=False, num_beams=1,
                           repetition_penalty=1.0, context='no truncation; native model context', attention=attn)
    if suite == 'mmmu':
        expected_policy['multiple_choice_max_new_tokens'] = 8
    if summary.get('evaluation_protocol') != expected_policy:
        raise ValueError(f'{path}: evaluation protocol mismatch')
    size = summary['image_processor_size']
    if size['shortest_edge'] != 4096 or size['longest_edge'] != 100352:
        raise ValueError(f'{path}: image budget mismatch')
    for key in ('model_path', 'processor_path'):
        if Path(summary[key]).resolve() != model.resolve():
            raise ValueError(f'{path}: {key} mismatch')
    return payload



def summarize_matrix(out, write=True):
    policy = read(out / 'protocol.json')
    methods = tuple(policy['models'])
    for method in methods:
        read(out / method / 'provenance.json')
        for suite in SUITES:
            validate_result(out / method / f'{suite}.json', suite,
                            Path(policy['models'][method]['model']), policy['attention'])
    data = load_results(out, methods)  # also checks matching identities and gold across rows
    labels = list(SUITES) + ['mean']
    scores = {m: {s: 100 * data[m, s][metric] for s, (_, metric) in SUITES.items()} for m in methods}
    for m in methods:
        scores[m]['mean'] = sum(scores[m].values()) / 7
    def table(headers, rows):
        return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                         ['| ' + ' | '.join(str(x).replace('|', '\\|').replace('\n', ' ') for x in row) + ' |' for row in rows])
    rows = [[m, *[f'{scores[m][s]:.2f}' for s in labels]] for m in methods]
    report = ['# ' + policy.get('report_title', 'Checkpoint comparison'),
              f'Validated all {len(methods) * len(SUITES)} merged results, expected counts, unique identities, matching gold labels, model/processor paths and evaluation policy.',
              table(['Model', *labels], rows),
              'Scores are percentages. The unweighted seven-suite mean is descriptive, not an official leaderboard metric. '
              + policy.get('report_note', '')]
    delta_rows = []
    for pair in policy['comparisons']:
        left, right = pair['left'], pair['right']
        delta_rows.append([f"{pair['label']} ({left} minus {right})",
                           *[f'{scores[left][s] - scores[right][s]:+.2f}' for s in labels]])
    report += ['## Differences (percentage points)', table(['Comparison', *labels], delta_rows)]
    diagnostic = []
    breakdown = []
    for suite in SUITES:
        for method in methods:
            summary = data[method, suite]
            diagnostic.append([method, suite, summary['row_errors'], summary.get('parse_fail', 'not reported')])
            if suite == 'hallusionbench':
                for metric in ('qAcc', 'fAcc'):
                    breakdown.append([method, suite, metric, summary[f'{metric}_total_groups'], f'{100 * summary[metric]:.2f}'])
            for field in sorted(k for k in summary if k.startswith('by_')):
                for category, group in summary[field].items():
                    breakdown.append([method, suite, f'{field}/{category}', group['total'], f'{100 * group["acc"]:.2f}'])
    report += ['## Errors and parse failures', table(['Model', 'Suite', 'Errors', 'Parse failures'], diagnostic),
               'Failed parses remain in the denominator. MMMU uses the existing local mixed-answer scorer.',
               '## Additional metrics and categories', table(['Model', 'Suite', 'Category/metric', 'N', 'Score (%)'], breakdown),
               'Acceptance thresholds remain for supervisor discussion. BF16 storage of quantized weights does not measure packed W4/W8 memory or speed.']
    if not write:
        print('\n\n'.join(report))
        return
    (out / 'comparison.md').write_text('\n\n'.join(report) + '\n')
    for name, header, values in [('comparison.csv', ['model', *labels], rows),
                                 ('deltas.csv', ['comparison', *labels], delta_rows)]:
        with (out / name).open('w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(header)
            writer.writerows(values)
    print(f'Validated comparison written to {out / "comparison.md"}')



def main():
    experiment = Path(os.environ.get("EXPERIMENT_DIR",
        Path(__file__).resolve().parents[1] / "experiments/constrained_32k_seed42"))
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-dir", type=Path,
                        default=Path(os.environ.get("OUT_DIR", experiment / "evaluation")))
    parser.add_argument("--write", action="store_true",
                        help="Save comparison.md and comparison.csv in the results directory")
    parser.add_argument("--matrix", action="store_true", help="Summarize the explicit model matrix in protocol.json")
    args = parser.parse_args()
    try:
        if args.matrix:
            summarize_matrix(args.results_dir, write=args.write)
            return
        # Preserve compatibility with earlier three-method comparisons, but never
        # silently omit an incomplete baseline run once its directory exists.
        methods = (("baseline",) + METHODS if (args.results_dir / "baseline").exists() else METHODS)
        report, rows = make_report(load_results(args.results_dir, methods), args.results_dir, methods)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(1, f"Cannot summarize results: {exc}\n")
    print(report.split("## All category breakdowns")[0].rstrip())
    if args.write:
        (args.results_dir / "comparison.md").write_text(report)
        with (args.results_dir / "comparison.csv").open("w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["suite_metric", "n"] + [f"{m}_pct" for m in methods] +
                            [f"grace_minus_{m}_pp" for m in ("baseline", "qat", "sft") if m in methods])
            writer.writerows(rows)
        print(f"\nSaved {args.results_dir / 'comparison.md'}")
        print(f"Saved {args.results_dir / 'comparison.csv'}")


if __name__ == "__main__":
    main()
