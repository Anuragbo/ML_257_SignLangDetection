from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "part1_letter_classifier" / "results"
OUT.mkdir(parents=True, exist_ok=True)


def bar_chart(title: str, labels: list[str], values: list[float], colors: list[str], out_name: str) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    bars = ax.bar(labels, values, color=colors, edgecolor="#0f172a", linewidth=1.0)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.tick_params(axis="x", labelrotation=10)

    for b, v in zip(bars, values):
        ax.text(
            b.get_x() + b.get_width() / 2,
            min(v + 0.015, 0.995),
            f"{v:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
            color="#0f172a",
            fontweight="bold",
        )

    legend_items = [
        plt.Line2D([0], [0], marker="s", color="w", markerfacecolor=c, markersize=10, label=l)
        for l, c in zip(labels, colors)
    ]
    ax.legend(handles=legend_items, title="Key", loc="lower right", frameon=True)

    fig.tight_layout()
    fig.savefig(OUT / out_name, bbox_inches="tight")
    plt.close(fig)


def grouped_bar_chart(
    title: str,
    categories: list[str],
    series_names: list[str],
    series_values: list[list[float]],
    series_colors: list[str],
    out_name: str,
) -> None:
    fig, ax = plt.subplots(figsize=(9, 4.8), dpi=150)
    x = np.arange(len(categories))
    width = 0.34

    for idx, (name, vals, color) in enumerate(zip(series_names, series_values, series_colors)):
        offset = (idx - (len(series_names) - 1) / 2) * width
        bars = ax.bar(x + offset, vals, width=width, label=name, color=color, edgecolor="#0f172a", linewidth=1.0)
        for b, v in zip(bars, vals):
            ax.text(
                b.get_x() + b.get_width() / 2,
                min(v + 0.015, 0.995),
                f"{v:.3f}",
                ha="center",
                va="bottom",
                fontsize=9,
                color="#0f172a",
                fontweight="bold",
            )

    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_ylabel("Score")
    ax.set_ylim(0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.grid(axis="y", alpha=0.25, linestyle="--")
    ax.set_axisbelow(True)
    ax.legend(title="Metric", loc="upper left", frameon=True)
    fig.tight_layout()
    fig.savefig(OUT / out_name, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    bar_chart(
        "Part 1 MediaPipe Accuracy",
        ["Random Forest", "SVM", "MLP"],
        [0.9066, 0.9011, 0.8242],
        ["#3b82f6", "#22c55e", "#f59e0b"],
        "perf_bar_mediapipe_accuracy.png",
    )
    bar_chart(
        "Part 1 Image Model Accuracy",
        ["ResNet-18", "MobileNetV2", "VGG-11-BN", "CNN"],
        [0.9894, 0.9868, 0.9815, 0.9392],
        ["#3b82f6", "#22c55e", "#f59e0b", "#ef4444"],
        "perf_bar_image_accuracy.png",
    )
    bar_chart(
        "Top Part 1 Accuracy Comparison",
        ["YOLO", "ResNet-18"],
        [0.9622, 0.9894],
        ["#14b8a6", "#3b82f6"],
        "perf_bar_yolo_vs_resnet.png",
    )
    bar_chart(
        "Part 2 WLASL Top-1",
        ["Transformer", "BiLSTM"],
        [0.344, 0.246],
        ["#3b82f6", "#22c55e"],
        "perf_bar_wlasl_top1.png",
    )
    bar_chart(
        "Part 2 WLASL Top-5",
        ["Transformer", "BiLSTM"],
        [0.705, 0.590],
        ["#3b82f6", "#22c55e"],
        "perf_bar_wlasl_top5.png",
    )
    grouped_bar_chart(
        "Part 2 WLASL Top-1 vs Top-5",
        ["Transformer", "BiLSTM"],
        ["Top-1", "Top-5"],
        [[0.344, 0.246], [0.705, 0.590]],
        ["#f59e0b", "#14b8a6"],
        "perf_bar_wlasl_top1_top5_combined.png",
    )


if __name__ == "__main__":
    main()
