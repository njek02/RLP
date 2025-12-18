import matplotlib.pyplot as plt
import numpy as np


def _apply_paper_style(figsize=(6.5, 4), dpi: int = 300) -> None:
    """
        Style taken from data science course paper plotting guidelines.
    """
    plt.rcParams.update({ 
        "figure.figsize": figsize,
        "figure.dpi": dpi,
        "font.size": 10,
        "axes.titlesize": 10,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "lines.linewidth": 1.8,
        "lines.markersize": 4,
        "axes.grid": True,
        "grid.alpha": 0.12,
        "grid.linestyle": "-",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.spines.left": True,
        "axes.spines.bottom": True,
    })
    plt.rcParams["axes.prop_cycle"] = plt.cycler(color=[
        "#1f77b4",
        "#ff7f0e",
        "#2ca02c",
        "#9467bd"
    ])


def plot_success_rate_se(success_rates, paper_style=False):
    if paper_style:
        _apply_paper_style()

    success_rates = np.array(success_rates)
    mean_sr = np.mean(success_rates)
    se_sr = np.std(success_rates, ddof=1) / np.sqrt(len(success_rates))

    plt.bar([0], [mean_sr], yerr=[se_sr], capsize=6)
    plt.xticks([0], ["Success Rate"])
    plt.ylim(0, 1)
    plt.ylabel("Mean Success Rate")
    plt.title("Mean Success Rate Across Runs\n(with Standard Error)")
    plt.tight_layout()
    plt.show()
