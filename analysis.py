"""
analysis.py — Visualize experiment progress from results.tsv.
Run after accumulating experiments to see progress over time.

Usage:
    python analysis.py                  # display plot
    python analysis.py --save progress.png  # save to file
"""

import sys
import argparse
from pathlib import Path

import numpy as np

TSV_PATH = Path("results.tsv")


def load_results(path: Path) -> list[dict]:
    """Load results.tsv into a list of dicts."""
    if not path.exists():
        print(f"No {path} found. Run some experiments first.")
        sys.exit(1)

    rows = []
    with open(path) as f:
        header = f.readline().strip().split("\t")
        for line in f:
            vals = line.strip().split("\t")
            if len(vals) != len(header):
                continue
            row = dict(zip(header, vals))
            # Convert numeric fields
            for k in ("macro_f1", "binary_f1", "multi_acc", "time_s"):
                if k in row:
                    try:
                        row[k] = float(row[k])
                    except ValueError:
                        row[k] = 0.0
            rows.append(row)
    return rows


def plot_progress(results: list[dict], save_path: str | None = None):
    """Plot macro_f1 and binary_f1 over experiments."""
    import matplotlib.pyplot as plt

    n = len(results)
    x = list(range(1, n + 1))
    macro_f1 = [r["macro_f1"] for r in results]
    binary_f1 = [r["binary_f1"] for r in results]

    # Track the running best
    best_macro = []
    current_best = 0
    for v in macro_f1:
        current_best = max(current_best, v)
        best_macro.append(current_best)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    fig.suptitle("Woahamark Detection — Experiment Progress", fontsize=14, fontweight="bold")

    # Macro F1
    ax1.scatter(x, macro_f1, c=["#2ecc71" if m >= b else "#e74c3c" for m, b in zip(macro_f1, [0] + best_macro[:-1])],
                alpha=0.7, s=40, zorder=3)
    ax1.plot(x, best_macro, color="#217AB7", linewidth=2, label="Best so far", zorder=2)
    ax1.set_ylabel("Macro F1 (primary)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.05, 1.05)

    # Binary F1
    ax2.scatter(x, binary_f1, c="#015583", alpha=0.6, s=30, zorder=3)
    ax2.set_ylabel("Binary F1 (secondary)")
    ax2.set_xlabel("Experiment #")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(-0.05, 1.05)

    # Annotate best
    if macro_f1:
        best_idx = np.argmax(macro_f1)
        ax1.annotate(
            f"Best: {macro_f1[best_idx]:.3f}\n({results[best_idx].get('tag', '?')})",
            xy=(best_idx + 1, macro_f1[best_idx]),
            xytext=(10, 10), textcoords="offset points",
            fontsize=9, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color="#217AB7"),
        )

    # Add notes as hover-like annotations for top results
    tags = [r.get("tag", "") for r in results]
    notes = [r.get("notes", "") for r in results]

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Saved to {save_path}")
    else:
        plt.show()


def print_summary(results: list[dict]):
    """Print text summary of experiment progress."""
    if not results:
        print("No results yet.")
        return

    macro_scores = [r["macro_f1"] for r in results]
    best_idx = np.argmax(macro_scores)
    best = results[best_idx]

    print(f"\n{'='*50}")
    print(f"  Experiments run: {len(results)}")
    print(f"  Best macro_f1:   {best['macro_f1']:.4f} (experiment #{best_idx+1}, tag={best.get('tag', '?')})")
    print(f"  Best binary_f1:  {max(r['binary_f1'] for r in results):.4f}")
    print(f"  Improvements:    {sum(1 for i, m in enumerate(macro_scores) if i == 0 or m > max(macro_scores[:i]))}")
    print(f"{'='*50}")

    # Show top 5
    ranked = sorted(enumerate(results), key=lambda x: x[1]["macro_f1"], reverse=True)[:5]
    print("\nTop 5 experiments:")
    for rank, (idx, r) in enumerate(ranked, 1):
        print(f"  {rank}. #{idx+1} macro_f1={r['macro_f1']:.4f} | {r.get('notes', '')[:60]}")


def main():
    parser = argparse.ArgumentParser(description="Analyze experiment progress")
    parser.add_argument("--save", type=str, help="Save plot to file instead of displaying")
    parser.add_argument("--text-only", action="store_true", help="Print text summary only (no plot)")
    args = parser.parse_args()

    results = load_results(TSV_PATH)
    print_summary(results)

    if not args.text_only:
        plot_progress(results, save_path=args.save)


if __name__ == "__main__":
    main()
