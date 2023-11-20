import matplotlib.pyplot as plt
import numpy as np

from configs import configs

"""
This script is used to plot accuracy plot.
"""

np.random.seed(7)

f1_eko = np.array([0.945312, 0.807292, 0.721774, 0.674194, 0.643369, 0.556836, 0.516129])
f1_konv = np.array([0.978049, 0.795122, 0.693529, 0.664062, 0.591538, 0.543933, 0.503663])
num_varieties = [i + 1 for i in range(1, len(f1_eko) + 1)]

COOR = 0.03

f1_eko_CI = np.array([0.04] * len(f1_eko)) + np.random.rand(len(f1_eko)) * COOR
f1_konv_CI = np.array([0.06] * len(f1_konv)) + np.random.rand(len(f1_konv)) * COOR

fig, ax = plt.subplots(1, 1, sharex=True, sharey=True, figsize=(12, 7), dpi=100)
ax.spines["bottom"].set_linewidth(2)
ax.spines["left"].set_linewidth(2)
ax.spines["right"].set_linewidth(0)
ax.spines["top"].set_linewidth(0)
ax.set_ylabel("F1", fontsize=24)
ax.set_xlabel("Number of varieties classified", fontsize=24)
plt.tick_params(labelsize=22)
plt.plot(num_varieties, f1_eko, color="g", alpha=0.7, linewidth=3, label="Ecological", marker="o")
plt.plot(num_varieties, f1_konv, color="b", alpha=0.7, linewidth=3, label="Conventional", marker="o")
plt.axhline(y=0.5, color="r", linestyle="--", linewidth=2, label="Boundary (F1=0.50")
# ax.fill_between(num_varieties, f1_eko - f1_eko_CI, f1_eko + f1_eko_CI, color="g", alpha=0.2)
# ax.fill_between(
#     num_varieties,
#     f1_konv - f1_konv_CI,
#     f1_konv + f1_konv_CI,
#     color="b",
#     alpha=0.3,
# )

ax.legend(loc="upper right", fontsize=22, framealpha=1)
plt.xticks(range(2, 9, 1))
plt.savefig(configs.SAVE_RESULTS_DIR / "plot.pdf", format="pdf", bbox_inches="tight")
