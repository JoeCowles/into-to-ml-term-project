"""Export loss curves and an optional before/after QA comparison as PNGs."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--pretrain", type=Path, required=True, help="Pretraining metrics.jsonl")
    p.add_argument("--finetune", type=Path, required=True, help="Fine-tuning metrics.jsonl")
    p.add_argument("--comparison", type=Path, help="Evaluation comparison.json")
    p.add_argument("--out", type=Path, default=Path("runs/figures"))
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, path, title in zip(axes, [args.pretrain, args.finetune], ["Pretraining", "QA fine-tuning"]):
        rows = [json.loads(line) for line in path.read_text().splitlines()]
        trained = [r for r in rows if "train_loss" in r]
        ax.plot([r["epoch"] for r in trained], [r["train_loss"] for r in trained], label="Training", color="#2166ac")
        ax.plot([r["epoch"] for r in rows], [r["validation_loss"] for r in rows], label="Validation", color="#d6604d")
        ax.set(title=title, xlabel="Epoch", ylabel="Cross-entropy (nats / target token)")
        ax.grid(alpha=0.2)
        ax.legend()
    fig.savefig(args.out / "loss_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    if args.comparison:
        results = json.loads(args.comparison.read_text())
        attrs = sorted(set().union(*(r["by_attribute"] for r in results)))
        fig, ax = plt.subplots(figsize=(11, 5), layout="constrained")
        width = 0.8 / len(results)
        for i, result in enumerate(results):
            xs = [j - 0.4 + width/2 + i*width for j in range(len(attrs))]
            ys = [100 * result["by_attribute"].get(attr, {}).get("accuracy", 0) for attr in attrs]
            label = f"{result['stage']} (overall {result['accuracy']:.1%})"
            ax.bar(xs, ys, width, label=label)
        ax.set_xticks(range(len(attrs)), [a.replace("_", " ") for a in attrs], rotation=25, ha="right")
        ax.set(ylabel="Strict rule-based QA accuracy (%)", ylim=(0, 105), title="Reserved QA evaluation")
        ax.legend()
        ax.grid(axis="y", alpha=0.2)
        ax.set_axisbelow(True)
        fig.savefig(args.out / "qa_comparison.png", dpi=180, bbox_inches="tight")
        plt.close(fig)
    print(f"Saved figures to {args.out}")


if __name__ == "__main__":
    main()
