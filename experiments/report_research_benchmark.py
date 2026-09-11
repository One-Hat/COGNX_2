"""Generate a plain-language report and README fragment from saved results."""
import argparse
import csv
import json
from pathlib import Path


def pm(stat, unit="", decimals=2):
    mean = f"{stat['mean']:.{decimals}f}"
    spread = f" ± {stat['std']:.{decimals}f}" if stat["std"] is not None else "†"
    return mean + spread + unit


def main_table(summary):
    lines = ["| Method | Final ACC (%) | Forgetting (pp) | Model/adaptive (B) | Fixed scaffold (B) | Auxiliary content (B) | Auxiliary allocated (B) | Total resident arrays (B) | Native projected inference (µJ/image) |",
             "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for method, values in summary["methods"].items():
        memory = values["memory"]
        cells = [method, pm(values["final_acc"]), pm(values["forgetting"])]
        for key in ("model_adaptive_bytes", "fixed_scaffold_bytes", "auxiliary_content_bytes",
                    "auxiliary_allocated_bytes", "total_resident_array_bytes"):
            cells.append(pm(memory[key], decimals=0))
        cells.append(pm(values["inference_projected_uj_per_sample"], decimals=3))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def generate_report(directory):
    directory = Path(directory)
    config = json.loads((directory / "config.json").read_text(encoding="utf-8"))
    summary = json.loads((directory / "summaries/summary.json").read_text(encoding="utf-8"))
    runs = json.loads((directory / "raw/per_seed_results.json").read_text(encoding="utf-8"))
    if not summary["complete"]:
        raise ValueError("Refusing to report an incomplete comparison")
    methods = summary["methods"]
    debug = config["mode"] == "quick"
    banner = ("**DEBUG RESULTS ONLY — reduced data, not research evidence. The full ten-seed experiment has NOT been completed by this quick run.**"
              if debug else "**Full-data research protocol.**")
    if len(config["seeds"]) == 1:
        banner += "\n\n† One seed: standard deviation is undefined; values are single-seed observations."
    hp = config["hyperparameters"]
    table = main_table(summary)
    sizes = "\n".join(f"| {''.join(map(str,s['classes']))} | {s['train']:,} | {s['test']:,} | {s['full_train']:,} | {s['full_test']:,} |" for s in config["task_sizes"])
    best_acc = max(methods, key=lambda m: methods[m]["final_acc"]["mean"])
    least_forgetting = min(methods, key=lambda m: methods[m]["forgetting"]["mean"])
    mn = methods["MNEMA"]
    mem = mn["memory"]
    mnema_rows = []
    for key in ("configured_budget_bytes", "implementation_row_bytes", "actual_row_bytes", "active_rows",
                "implementation_occupancy_bytes", "actual_active_payload_bytes", "allocated_faststore_bytes",
                "cortex_synaptic_adaptive_bytes", "other_adaptive_array_bytes", "fixed_scaffold_bytes", "total_resident_array_bytes"):
        stat = mem[key]
        mnema_rows.append(f"| {key} | {pm(stat, decimals=0)} | {stat['max']:,.0f} |")
    peak = mn["faststore_peak_payload_bytes"]
    mnema_rows.append(f"| peak post-sample active payload | {pm(peak, decimals=0)} | {peak['max']:,.0f} |")
    comparisons = []
    for method in ("Replay-300 (Research)", "Replay-64KiB", "DER++-300", "EWC"):
        mm = methods[method]["memory"]
        keys = ("replay_items", "replay_capacity", "image_bytes", "label_bytes", "logit_bytes", "buffer_metadata_bytes", "buffer_bytes", "fisher_bytes", "parameter_snapshot_bytes", "other_cl_array_bytes")
        comparisons.append("- **" + method + "**: " + "; ".join(f"{k}={pm(mm[k], decimals=0)}" for k in keys if k in mm) + ".")
    wins, losses = [], []
    for other in methods:
        if other == "MNEMA":
            continue
        gap = mn["final_acc"]["mean"] - methods[other]["final_acc"]["mean"]
        target = wins if gap > 0 else losses
        target.append(f"Final accuracy {'above' if gap > 0 else 'at or below'} {other} by {abs(gap):.2f} percentage points.")
    if debug:
        wins_text = "No research advantage is established by this debug run. Observed debug comparisons only: " + " ".join(wins)
        losses_text = "These are debugging observations, not full-data conclusions: " + " ".join(losses)
    else:
        wins_text = " ".join(wins) or "No final-accuracy advantage in this comparison."
        losses_text = " ".join(losses) or "No lower mean final accuracy in this comparison."
    report = f"""# MNEMA Continual-Learning Benchmark

{banner}

## 1. Research Question

How does the existing MNEMA compare with standard continual-learning strategies in accuracy, forgetting and memory usage under sequential Split-MNIST Class-IL?

## 2. Experimental Setup

MNIST handwritten digits, fixed task order 01 → 23 → 45 → 67 → 89. Each model receives the same shuffled training indices once per seed. Every prediction considers all ten classes; no inference task ID, output masking or task-specific head is used. All five test tasks are evaluated after every training task, including future-task cells; progress averages only tasks seen so far. The full mode uses 60,000 training and 10,000 test images. This run uses {sum(s['train'] for s in config['task_sizes']):,} training and {sum(s['test'] for s in config['task_sizes']):,} test images.

| Digits | Used train | Used test | Full train | Full test |
|---|---:|---:|---:|---:|
{sizes}

Seeds: {config['seeds']}. Run mode: **{config['mode']}**. Created: {config['created_utc']}. Git commit: `{config['git_commit']}`. Exact source hashes, dataset checksums and settings: [config.json](config.json). Software: `{config['software']}`. NumPy/CPU, one BLAS thread, sequential methods/seeds; no GPU or neuromorphic device assumed. Images remain uint8 in host memory and are normalized to float32 /255 per use.

The unit of statistical replication is the seed. Tables use sample standard deviation (ddof=1); SD is undefined for one seed, so quick figures omit error bars. Source/environment/configuration mismatches refuse resume; valid completed seed–method pairs are skipped. No statistical significance or tuned-best-method claim is made.

## 3. Methods

### Naive MLP
Normal online SGD; 784 → 256 ReLU → 10, fp32, learning rate {hp['lr']}. Same architecture and initialization per seed for all dense baselines. No optimizer momentum or pretrained weights.

### Replay-300 (Research)
Stores 300 uint8 images and labels using true reservoir sampling. Each incoming sample produces one update on mean CE over current plus one previous sample, then enters the reservoir. Historical replay instead used random replacement, insertion before sampling and separate SGD steps; it is preserved but is not included under this research method's name.

### Replay-64KiB
Same research replay policy with a strict 65,536-byte allocated array-payload buffer budget, including image, label and seen/size counters. Capacity is derived from NumPy dtype sizes. Inputs are stored without float inflation. This is a conventional replay method under a 64 KiB episodic-buffer constraint, **not exact memory matching to MNEMA**.

### EWC
Protects parameters important to past tasks using lambda/2 times the sum of diagonal Fisher-weighted squared deviations from each task's optimum. Lambda={hp['ewc_lambda']}. Stores separate old-task Fishers and parameter snapshots. The empirical Fisher averages squared individual cross-entropy gradients using observed training labels. Fisher sample limit={hp['fisher_samples']} (0=all training samples); actual counts per run: {[r['fisher_sample_counts'] for r in runs if r['method']=='EWC']}. This additional training-data pass makes no optimizer updates; it is reported compute overhead and is not a strictly single-access stream algorithm. No Fisher is built after the last task because no future training uses it. [Original EWC paper](https://arxiv.org/abs/1612.00796).

### DER++-300
Reservoir of uint8 images, labels and ten fp32 logits produced before the original sample's optimizer update. Loss is current CE + {hp['der_alpha']} × mean squared logit error + {hp['der_beta']} × replay-label CE. The two replay terms draw independent single examples and share one combined update. Defaults are fixed illustrative coefficients, not test-tuned optimum values. [DER++ paper](https://arxiv.org/abs/2004.07211), [authors' reference implementation](https://github.com/aimagelab/mammoth/blob/master/models/derpp.py).

### MNEMA
The unmodified implementation combines a temporal spike encoder, sparse random separator with homeostasis, FastStore associations, a shallow adaptive spiking cortex with Benna–Fusi variables, arbitration, neuromodulation and internal sleep consolidation. Dimensions and thresholds remain unchanged. Internal consolidation adds replay computation; one pass refers to external stream presentation, not equal compute across algorithms.

## Implementation Caveats Discovered During Audit

### Stateful evaluation issue
The original historical benchmark evaluated the live model; held-out inputs could change adaptive state and subsequent training. **MNEMA's native inference path is stateful. The research benchmark therefore restores the same trained checkpoint state after every test inference so held-out evaluation data cannot alter subsequent predictions or training.** This is an evaluation protocol, not a modification to MNEMA. Restored attributes: encoder threshold/reference; separator thresholds/running rates; cortex membrane/adaptation; controller bus; RNG states. A full attribute-state hash is checked before/after every evaluation set. All final-checkpoint test sets are additionally predicted in reverse order and compared image by image. The checkpoint's existing membrane state is preserved, not reset to zero. No per-image full-model deepcopy is used.

### FastStore accounting issue
The frozen implementation counts 25 bytes/occupied row, whereas its int16 weights, float32 salience and uint32 timestamp require 28 bytes/row. Consequently the configured 64 KiB limit is **not a true physical 64 KiB payload ceiling**. The benchmark measures actual values and flags exceedances without changing eviction. The 64 KiB configuration also does not include cortex or separator state. Historical results are not overwritten or directly comparable to this protocol.

## 4. Main Results

{table}

All memory values are bytes (KiB = 1,024 bytes). Auxiliary content and allocated auxiliary memory are alternative views, not additive columns. Total = adaptive + fixed + allocated auxiliary. Energy is a partial native **projection**, not measured physical energy; see Section 8.

## 5. Accuracy

Final ACC is the unweighted average of the five final task accuracies. {best_acc} has the highest observed mean ({methods[best_acc]['final_acc']['mean']:.2f}%). Task sizes differ slightly, so this is a task-macro average rather than pooled sample accuracy. {'This ranking is only a pipeline smoke-test observation.' if debug else 'Mean rankings alone do not establish statistical superiority.'}

![Final accuracy](plots/final_accuracy.png)

## 6. Forgetting

For zero-based task j=0,...,3, F_j = max(R[t,j] for t=j,...,3) − R[4,j]. Average these four values and multiply by 100 to report percentage points. The final checkpoint and checkpoints before learning j are excluded from the maximum. Negative forgetting is allowed and means final performance improved beyond previous post-learning checkpoints. The fifth task has no later learning period and is excluded. {least_forgetting} has the lowest observed mean forgetting ({methods[least_forgetting]['forgetting']['mean']:.2f} pp). Low forgetting with low accuracy can mean a model never learned much; read both metrics together.

![Retention matrices](plots/retention_matrices.png)

## 7. Memory Trade-off

Measurements are unique persistent NumPy array payload, not interpreter/process RSS. Python object headers, scalar objects, PRNG internals, shared dataset, temporary gradient/Fisher workspace and evaluation snapshots are excluded for all methods. Buffer seen/size metadata are explicitly stored and counted as int64 arrays. MNEMA scalar adaptive state is restored and hashed even though scalar-object memory is outside this array-payload accounting. Fixed scaffold includes separator connectivity and synapse constants. Cortex synaptic variables, eligibility traces, neuron arrays and separator adaptive arrays are model/adaptive state, not frozen memory.

FastStore content counts rows with salience > 0, including weights, salience and timestamps. Unused dense capacity and stale timestamps are included in allocated FastStore arrays. Active content is not a claim of a deployable packed sparse representation; such a representation could need extra row-address metadata.

| MNEMA quantity | Final mean ± SD across seeds | Maximum across seeds |
|---|---:|---:|
{chr(10).join(mnema_rows)}

Post-sample over-budget observations across runs: **{mn['faststore_over_budget_observations']}**. Monitoring occurs after each external training sample (including its consolidation); the maximum is not a bound on transient within-step occupancy. Per-task memory and per-run peaks are saved in raw results. The final active payload is measured in each run, not copied from the audit probe.

{chr(10).join(comparisons)}

Compare Replay-300, Replay-64KiB and MNEMA using both auxiliary content and total resident arrays. MNEMA has a substantial adaptive/fixed allocation beyond its occupied FastStore rows. All buffers here are bounded; do not describe fixed-capacity replay as unbounded memory growth.

![Accuracy versus memory](plots/accuracy_vs_memory.png)

## 8. Energy / Compute

The original technology cards and instrument are unchanged. Native inference counters are projected using the ASIC 45 nm card. The table reports projection per image over the same evaluation workload. MNEMA native training counts/projections are retained separately in raw records. New baseline training, EWC Fisher estimation and buffer work lack comparable complete instrumentation, so **no cross-method training-energy ranking is reported**. Native counters also omit some arithmetic, use simplifying dtype/traffic assumptions and exclude checkpoint restoration. They are not exhaustive end-to-end energy or evidence of physical hardware efficiency. Wall-clock seconds are measured on this host (including integrity checks), not energy. Raw records report SGD updates, replay draws and Fisher sample counts.

## 9. Where MNEMA Wins

{wins_text} No significance is implied. Memory and projected-operation observations must be read with the caveats above.

## 10. Where MNEMA Loses

{losses_text} Its total resident array allocation is {mem['total_resident_array_bytes']['mean']:,.0f} bytes, and its configured FastStore limit is not accurately enforced against actual payload.

## 11. Main Interpretation

{'The implementation and output pipeline have been exercised, but this quick run cannot answer the full research question. Run the ten-seed full protocol on the stronger computer before making research claims.' if debug else 'These results describe the frozen implementation under this specific protocol. Accuracy, forgetting and auxiliary memory must be considered together, including MNEMA’s larger resident scaffold. They do not establish that MNEMA solves catastrophic forgetting.'}

## 12. Limitations

Only MNIST, one fixed task order, simple MLP baselines, limited architecture scale, no validation-based hyperparameter selection, empirical rather than exact model Fisher, task boundaries available to EWC during training, unequal algorithmic compute despite one external pass, simulated/projected energy, no real neuromorphic hardware, array payload rather than peak process RAM. Reproducibility is checked within an environment; exact cross-platform floating-point equivalence is not guaranteed. {'Reduced samples and one-seed defaults are additional debug limitations.' if debug else 'Ten seeds still do not replace testing on other datasets and task orders.'}

## 13. Next Experiment

After completing full Split-MNIST, use the same declared protocol on Split-FashionMNIST to test whether the trade-off extends to less easily separated images. Do not implement it as part of this benchmark.

## Reproduction and artifacts

Full command (stronger computer only): `python experiments/run_research_benchmark.py --mode full --seeds 10`.
Quick command: `python experiments/run_research_benchmark.py --quick`.
Repeat the same command to resume; `--force` archives that mode's previous output directory and reruns all pairs. Interrupted pairs restart from the beginning; completed pairs are durable. Plots can be regenerated using `python experiments/plot_research_benchmark.py results/research_benchmark/{config['mode']}` and this report using `python experiments/report_research_benchmark.py results/research_benchmark/{config['mode']}`.

Machine-readable artifacts: [full_results.json](full_results.json), [per_seed_results.json](raw/per_seed_results.json), [summary.json](summaries/summary.json), per-pair raw JSON and exact per-seed train/test indices. Despite its generic filename, full_results.json in a quick directory contains DEBUG data and carries explicit mode metadata.
"""
    (directory / "BENCHMARK_REPORT.md").write_text(report, encoding="utf-8")
    (directory / "summaries/summary_table.md").write_text(banner + "\n\n" + table + "\n", encoding="utf-8")
    with (directory / "summaries/summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "method", "final_acc_mean", "final_acc_sd", "forgetting_mean_pp", "forgetting_sd_pp",
                         "adaptive_bytes", "fixed_bytes", "auxiliary_content_bytes", "auxiliary_allocated_bytes", "resident_array_bytes"])
        for method, values in methods.items():
            mm = values["memory"]
            writer.writerow([config["mode"], method, values["final_acc"]["mean"], values["final_acc"]["std"],
                             values["forgetting"]["mean"], values["forgetting"]["std"],
                             *[mm[k]["mean"] for k in ("model_adaptive_bytes", "fixed_scaffold_bytes", "auxiliary_content_bytes", "auxiliary_allocated_bytes", "total_resident_array_bytes")]])
    fragment = f"## Research Benchmark\n\n{banner}\n\nFull Split-MNIST, five sequential digit pairs, one external training pass, ten-class inference without task IDs; default full seeds 0–9. Methods: Naive MLP, reservoir Replay-300 (Research), Replay-64KiB, EWC, DER++-300 and frozen MNEMA.\n\nRun on a stronger machine: `python experiments/run_research_benchmark.py --mode full --seeds 10`. Resume with the same command; add `--force` to archive and rerun. Development: `python experiments/run_research_benchmark.py --quick`.\n\n{table}\n\nMemory columns are array payload; total = adaptive + fixed + allocated auxiliary. Native inference energy is an incomplete projection, not measured energy. MNEMA's configured 64 KiB limit is not an actual payload ceiling (25 bytes counted versus 28 bytes per active row). Stateless checkpoint evaluation is enforced per image. These results are not comparable directly with historical small-benchmark numbers.\n\n[Detailed report](results/research_benchmark/{config['mode']}/BENCHMARK_REPORT.md) · [Generated summary](results/research_benchmark/{config['mode']}/summaries/summary.json)\n\n![Accuracy](results/research_benchmark/{config['mode']}/plots/final_accuracy.png)\n![Memory trade-off](results/research_benchmark/{config['mode']}/plots/accuracy_vs_memory.png)\n![Retention](results/research_benchmark/{config['mode']}/plots/retention_matrices.png)\n"
    (directory / "summaries/README_SECTION.md").write_text(fragment, encoding="utf-8")
    return fragment


def update_readme(fragment):
    """Explicit publication of the generated section; numbers never copied by hand."""
    path = Path(__file__).resolve().parents[1] / "README.md"
    text = path.read_text(encoding="utf-8")
    start, end = "<!-- RESEARCH_BENCHMARK_START -->", "<!-- RESEARCH_BENCHMARK_END -->"
    block = start + "\n" + fragment + "\n" + end
    if start in text:
        before, rest = text.split(start, 1)
        _, after = rest.split(end, 1)
        text = before + block + after
    else:
        title, rest = text.split("\n", 1)
        text = title + "\n\n" + block + "\n" + rest
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--update-readme", action="store_true", help="Replace generated README section with this run's section")
    args = parser.parse_args()
    fragment = generate_report(args.directory)
    if args.update_readme:
        update_readme(fragment)
