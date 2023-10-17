from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import yellowbrick.features as yb
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay, PredictionErrorDisplay

from configs import configs


def save_features_plot(
    features: pd.DataFrame,
    features_names: list[str],
    labels: np.ndarray,
    *,
    save_path: str | Path = "features_plot.pdf",
    x_label: str = "Spectral bands",
    y_label: str = "",
):
    assert len(features) == len(labels), "Features and labels must have the same length!"

    mpl.rcParams.update(mpl.rcParamsDefault)
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
    plt.close("all")


def save_confusion_matrix_display(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    save_path: str | Path = "confusion_matrix.pdf",
):
    mpl.rcParams.update(mpl.rcParamsDefault)
    cm_display = ConfusionMatrixDisplay.from_predictions(
        y_true, y_pred, display_labels=target_names, normalize="true"
    )
    cm_display.plot()
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close("all")


def save_prediction_errors_display(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path = "prediction_errors.pdf",
    kind: str = "residual_vs_predicted",
):
    mpl.rcParams.update(mpl.rcParamsDefault)
    pe_display = PredictionErrorDisplay.from_predictions(y_true, y_pred, kind=kind)
    pe_display.plot(kind=kind)
    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close("all")


def save_data_visualization(
    data: pd.DataFrame,
    y_data_encoded: np.ndarray,
    classes: list = None,
    save_path: str | Path = "visualization_data.pdf",
):
    mpl.rcParams.update(mpl.rcParamsDefault)

    # RadViz
    _save_path = save_path.with_name(save_path.stem + "_radviz.pdf")
    plt.subplots(figsize=(8, 7), dpi=300)
    visualizer = yb.RadViz(
        classes=classes, features=data.columns.tolist(), alpha=0.7, colormap="viridis"
    )
    visualizer.fit(data, y_data_encoded)
    visualizer.transform(data)
    visualizer.show()
    plt.savefig(_save_path, format="pdf", bbox_inches="tight")
    plt.close("all")

    # 2d PCA
    _save_path = save_path.with_name(save_path.stem + "_pca.pdf")
    plt.subplots(figsize=(8, 7), dpi=300)
    visualizer = yb.PCA(scale=True, classes=classes, alpha=0.7, colormap="viridis")
    visualizer.fit_transform(data, y_data_encoded)
    visualizer.show()
    plt.savefig(_save_path, format="pdf", bbox_inches="tight")
    plt.close("all")

    # 2d manifold
    # method used can be "tsne", "lle", "isomap", "mds" etc.
    # _method = "isomap"
    # _save_path = save_path.with_name(save_path.stem + "_manifold.pdf")
    # plt.subplots(figsize=(8, 7), dpi=300)
    # visualizer = yb.Manifold(
    #     manifold=_method, classes=classes, alpha=0.7, colormap="viridis", n_neighbors=3
    # )
    # visualizer.fit_transform(data, y_data_encoded)
    # visualizer.show()
    # plt.savefig(_save_path, format="pdf", bbox_inches="tight")
    # plt.close("all")


def save_meta_visualization(
    meta: pd.DataFrame,
    save_path: str | Path = "visualization_meta.pdf",
):
    mpl.rcParams.update(mpl.rcParamsDefault)
    for treatment in pd.unique(meta[configs.TREATMENT_ENG]):
        plt.subplots(figsize=(8, 7), dpi=300)
        ax = sns.countplot(
            data=meta,
            x=configs.DATE_ENG,
            hue=configs.VARIETY_ENG,
            alpha=0.8,
        )
        ax.set_ylabel("Count", fontsize=16)
        ax.set_xlabel(configs.DATE_ENG, fontsize=16)
        ax.tick_params(axis="both", which="major", labelsize=14)
        ax.tick_params(axis="both", which="minor", labelsize=14)
        ax.tick_params(axis="x", labelrotation=0)
        ax.grid(False)
        ax.spines["bottom"].set_linewidth(2)
        ax.spines["left"].set_linewidth(2)
        ax.spines[["right", "top"]].set_visible(False)
        ax.legend(loc="lower right", fontsize=12, framealpha=1)
        for container in ax.containers:
            ax.bar_label(container, fontsize=12, padding=3)
        plt.savefig(
            save_path.with_name("".join([save_path.stem, f"_{treatment}", save_path.suffix])),
            format="pdf",
            bbox_inches="tight",
        )
        plt.close("all")


def save_target_visualization(
    target_values: np.ndarray,
    target_labels: np.ndarray = None,
    target_type: str = "classification",
    save_path: str | Path = "visualization_target.pdf",
):
    mpl.rcParams.update(mpl.rcParamsDefault)
    _TARGET_VAL = "Target values"
    _TARGET_LAB = "Target"
    data = pd.DataFrame.from_dict({_TARGET_VAL: target_values, _TARGET_LAB: target_labels})

    plt.subplots(figsize=(8, 7), dpi=300)
    if target_type == "classification":
        ax = sns.countplot(
            data=data,
            x=_TARGET_LAB,
            alpha=0.8,
        )
    elif target_type == "regression":
        ax = sns.histplot(data=data, x=_TARGET_VAL, hue=_TARGET_LAB, alpha=0.8)
        # ax.legend()
    else:
        raise ValueError(f"Unknown target type: {target_type}")

    ax.set_xlabel(_TARGET_LAB, fontsize=16)
    ax.set_ylabel("Count", fontsize=16)
    ax.tick_params(axis="both", which="major", labelsize=14)
    ax.tick_params(axis="both", which="minor", labelsize=14)
    ax.tick_params(axis="x", labelrotation=0)
    ax.grid(False)
    ax.spines["bottom"].set_linewidth(2)
    ax.spines["left"].set_linewidth(2)
    ax.spines[["right", "top"]].set_visible(False)

    plt.savefig(save_path, format="pdf", bbox_inches="tight")
    plt.close("all")
