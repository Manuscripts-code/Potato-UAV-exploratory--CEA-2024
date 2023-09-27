import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay, PredictionErrorDisplay, confusion_matrix


def save_features_plot(
    features: pd.DataFrame,
    features_names: list[str],
    labels: np.ndarray,
    *,
    save_path: str = "features_plot.pdf",
    x_label: str = "Spectral bands",
    y_label: str = "",
):
    assert len(features) == len(labels), "Features and labels must have the same length!"

    fig, ax = plt.subplots(figsize=(8, 7), dpi=300)
    ax.set_ylabel(y_label, fontsize=24)
    ax.set_xlabel(x_label, fontsize=24)
    ax.tick_params(axis="both", which="major", labelsize=22)
    ax.tick_params(axis="both", which="minor", labelsize=22)
    # ax.set_ylim([0, 1])
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines["right"].set_linewidth(0)
    ax.spines["top"].set_linewidth(0)

    cmap = plt.get_cmap("viridis")
    unique_labels = np.unique(labels)
    no_colors = len(unique_labels)
    colors = cmap(np.linspace(0, 1, no_colors))
    labels_color_map = {label: idx for idx, label in enumerate(np.unique(labels))}

    x_values = list(range(len(features_names)))

    for label, features_row in zip(labels, features.iterrows()):
        ax.plot(x_values, features_row[1].tolist(), color=colors[labels_color_map[label]], alpha=0.6)

    ax.set_xticks(x_values)
    ax.set_xticklabels(features_names, rotation=0)

    custom_lines = []
    for idx in range(no_colors):
        custom_lines.append(Line2D([0], [0], color=colors[idx], lw=2))

    ax.legend(custom_lines, [str(num) for num in unique_labels], fontsize=22)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def save_confusion_matrix_display(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    save_path: str = "confusion_matrix.pdf",
):
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=target_names, normalize="true"
    )
    cm_display.plot()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()


def save_prediction_errors_display(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "prediction_errors.pdf",
    kind: str = "residual_vs_predicted",
):
    pe_display = PredictionErrorDisplay.from_predictions(y_true, y_pred, kind=kind)
    pe_display.plot(kind=kind)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close()
