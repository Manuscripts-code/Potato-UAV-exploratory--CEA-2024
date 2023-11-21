import matplotlib.pyplot as plt
import numpy as np

from configs import configs
from utils.plot_utils import save_plot_figure

"""
This script is used to plot accuracy plot.
"""


f1_points = np.array([0.971, 0.800, 0.709, 0.653, 0.586, 0.549, 0.474])
num_varieties = [i + 1 for i in range(1, len(f1_points) + 1)]

with save_plot_figure(save_path=configs.SAVE_RESULTS_DIR / "f1_clf_results.pdf", figsize=(8, 4)) as (
    fig,
    ax,
):
    ax.spines["right"].set_linewidth(0)
    ax.spines["top"].set_linewidth(0)
    ax.set_ylabel("F1", fontsize=24)
    ax.set_xlabel("Number of varieties classified", fontsize=24)
    plt.tick_params(labelsize=22)
    plt.plot(num_varieties, f1_points, color="r", alpha=0.7, linewidth=3, label="x", marker="o")

    # ax.legend(loc="upper right", fontsize=22, framealpha=1)
    plt.xticks(range(2, 9, 1))
    plt.yticks(np.arange(0.5, 1.05, 0.1))
