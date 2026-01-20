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

def plot_success_rate_se_multiple(success_rates_dict, paper_style=False, save_path=None):
    """
    success_rates_dict: dict
        Keys   -> labels for each bar
        Values -> list or array of success rates for that condition
    """
    if paper_style:
        _apply_paper_style()

    labels = list(success_rates_dict.keys())
    means = []
    ses = []
    variances = []

    for rates in success_rates_dict.values():
        rates = np.asarray(rates)
        means.append(np.mean(rates))
        ses.append(np.std(rates, ddof=1) / np.sqrt(len(rates)))
        variances.append(np.var(rates, ddof=1))
    
    print(means)
    print(ses)
    print(variances)

    x = np.arange(len(labels))

    plt.bar(x, means, yerr=ses, capsize=6)
    plt.xticks(x, labels)
    plt.ylim(0, 1)
    plt.ylabel("Mean Success Rate")
    plt.title("Success rate Comparison: Dense vs. Pruned Model")
    plt.tight_layout()

    if save_path is not None:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")

    plt.show()
