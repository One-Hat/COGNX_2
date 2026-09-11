"""Publication-style figures built exclusively from saved benchmark summaries."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_plots(directory):
    directory = Path(directory)
    summary = json.loads((directory / "summaries/summary.json").read_text(encoding="utf-8"))
    if not summary["complete"]:
        raise ValueError("Refusing to plot an incomplete comparison")
    methods = list(summary["methods"])
    results = summary["methods"]
    labels = [m.replace(" (Research)", "\n(Research)") for m in methods]
    debug = summary["mode"] == "quick"
    banner = "DEBUG ONLY — truncated data" if debug else "Full Split-MNIST Class-IL"
    n = results[methods[0]]["final_acc"]["n"]
    errors = f"n={n}; error bars: sample SD" if n > 1 else "n=1; seed SD is undefined; no error bars"
    folder = directory / "plots"
    folder.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False,
                         "axes.spines.right": False, "savefig.dpi": 220})
    colors = plt.get_cmap("tab10")(np.arange(len(methods)))

    def save(fig, filename):
        fig.text(0.5, 0.008, f"{banner} | {errors}", ha="center", fontsize=9)
        fig.tight_layout(rect=(0, 0.045, 1, 1))
        fig.savefig(folder / filename)
        fig.savefig(folder / filename.replace(".png", ".pdf"))
        plt.close(fig)

    for metric, title, ylabel, filename in (
        ("final_acc", "Final average task accuracy", "Accuracy (%)", "final_accuracy.png"),
        ("forgetting", "Average forgetting (lower is better)", "Forgetting (percentage points)", "forgetting.png")):
        fig, ax = plt.subplots(figsize=(10, 5))
        means = [results[m][metric]["mean"] for m in methods]
        stds = [results[m][metric]["std"] for m in methods]
        ax.bar(labels, means, color=colors, yerr=stds if n > 1 else None, capsize=4)
        ax.set(title=title, ylabel=ylabel)
        ax.axhline(0, color="0.3", linewidth=0.7)
        ax.grid(axis="y", alpha=0.25)
        save(fig, filename)

    for metric, ylabel, filename in (
        ("final_acc", "Final accuracy (%)", "accuracy_vs_memory.png"),
        ("forgetting", "Forgetting (percentage points)", "forgetting_vs_memory.png")):
        fig, ax = plt.subplots(figsize=(10, 6))
        points = []
        for i, method in enumerate(methods):
            memory = results[method]["memory"]["auxiliary_content_bytes"]
            score = results[method][metric]
            x = memory["mean"] / 1024
            ax.errorbar(x, score["mean"], xerr=memory["std"] / 1024 if n > 1 else None,
                        yerr=score["std"] if n > 1 else None, fmt="o", color=colors[i], capsize=3)
            points.append((method, x, score["mean"]))
        ax.set_xscale("symlog", linthresh=1)  # Naive's zero memory remains visible.
        ax.margins(x=0.25, y=0.25)
        ax.set(xlabel="Final episodic / auxiliary payload (KiB; symlog includes zero)",
               ylabel=ylabel, title="Accuracy–memory trade-off" if metric == "final_acc" else "Forgetting–memory trade-off")
        ax.text(0.02, 0.02, "MNEMA: active FastStore payload. Other adaptive/fixed state is reported separately.\nBuffers: allocated payload + counters; EWC: Fisher + snapshots. Not exact memory matching.",
                transform=ax.transAxes, fontsize=8)
        ax.grid(alpha=0.25)
        # Place labels in display coordinates, avoiding overlaps without extra dependencies.
        fig.canvas.draw()
        boxes = []
        for method, x, y in points:
            for dx, dy in ((8, 12), (8, -22), (-8, 12), (-8, -22), (8, 36), (-8, 36), (8, -42), (-8, -42)):
                annotation = ax.annotate(method, (x, y), xytext=(dx, dy), textcoords="offset points",
                                         fontsize=9, ha="left" if dx > 0 else "right")
                box = annotation.get_window_extent(fig.canvas.get_renderer()).expanded(1.04, 1.2)
                if not any(box.overlaps(other) for other in boxes):
                    boxes.append(box)
                    break
                annotation.remove()
            else:
                ax.annotate(method, (x, y), xytext=(8, 60), textcoords="offset points", fontsize=9)
        save(fig, filename)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9), layout=None)
    for ax, method in zip(axes.flat, methods):
        matrix = np.asarray(results[method]["retention_mean"]) * 100
        ax.imshow(matrix, vmin=0, vmax=100, cmap="cividis")
        ax.set(title=method, xlabel="Evaluated digit pair", ylabel="After learning task")
        ax.set_xticks(range(5), ["01", "23", "45", "67", "89"])
        ax.set_yticks(range(5), ["1", "2", "3", "4", "5"])
        for t in range(5):
            for j in range(5):
                ax.text(j, t, f"{matrix[t, j]:.1f}", ha="center", va="center",
                        color="white" if matrix[t, j] < 50 else "black", fontsize=9)
    fig.suptitle("Retention matrices — mean accuracy (%); every cell evaluated", fontsize=14)
    save(fig, "retention_matrices.png")

    fig, ax = plt.subplots(figsize=(10, 6))
    for i, method in enumerate(methods):
        means = np.array(results[method]["seen_accuracy_mean"])
        ax.plot(range(1, 6), means, marker="o", label=method, color=colors[i])
        if n > 1:
            std = np.array(results[method]["seen_accuracy_std"])
            ax.fill_between(range(1, 6), means - std, means + std, color=colors[i], alpha=0.12)
    ax.set(xticks=range(1, 6), xlabel="Tasks learned", ylabel="Mean accuracy on tasks seen so far (%)",
           title="Continual-learning progress")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=9)
    save(fig, "continual_learning_progress.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    generate_plots(parser.parse_args().directory)
