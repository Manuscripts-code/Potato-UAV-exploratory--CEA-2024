from contextlib import contextmanager
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scienceplots  # noqa
import seaborn as sns
import umap
import yellowbrick.features as yb
from matplotlib.colors import ListedColormap
from matplotlib.lines import Line2D
from sklearn.metrics import ConfusionMatrixDisplay, PredictionErrorDisplay, accuracy_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC

from configs import configs


@contextmanager
def save_plot_figure(
    save_path: str | Path = "plot.pdf",
    use_science_style: bool = False,
    figsize: tuple[int, int] = (8, 7),
    dpi: int = 300,
):
    mpl.rcParams.update(mpl.rcParamsDefault)
    if use_science_style:
        plt.style.use(["science", "ieee", "no-latex"])
    else:
        plt.style.use("default")
    try:
        fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
        yield fig, ax
        if save_path:
            plt.savefig(save_path, format="pdf", bbox_inches="tight")
        else:
            plt.show()
    finally:
        plt.close("all")


def save_features_plot(
    data: pd.DataFrame,
    y_data_encoded: pd.DataFrame,
    classes: list[str],
    save_path: str | Path = "features_plot.pdf",
    colormap: str = "viridis",
    x_label: str = "Spectral bands",
    y_label: str = "",
):
    with save_plot_figure(save_path) as (fig, ax):
        cmap = plt.get_cmap(colormap)
        unique_labels = np.unique(y_data_encoded)
        no_colors = len(unique_labels)

        if no_colors > 2:
            colors = cmap(np.linspace(0, 1, no_colors))
        else:
            colors = ["darkgoldenrod", "forestgreen"]

        x_values = list(range(len(data.columns)))

        grouped_data = data.groupby(y_data_encoded.to_numpy())
        mean_values = grouped_data.mean()
        std_values = grouped_data.std()

        for idx in unique_labels:
            mean = mean_values.loc[idx].tolist()
            std = std_values.loc[idx].tolist()
            ax.plot(x_values, mean, color=colors[idx], label=classes[idx], alpha=0.6)
            ax.fill_between(
                x_values,
                [m - s for m, s in zip(mean, std)],
                [m + s for m, s in zip(mean, std)],
                color=colors[idx],
                alpha=0.2,
            )

        custom_lines = []
        for idx in unique_labels:
            custom_lines.append(Line2D([0], [0], color=colors[idx], lw=2))

        ax.set_ylabel(y_label, fontsize=14)
        ax.set_xlabel(x_label, fontsize=14)
        ax.tick_params(axis="both", which="major", labelsize=12)
        ax.tick_params(axis="both", which="minor", labelsize=12)
        # ax.set_ylim([0, 1])
        # ax.spines["bottom"].set_linewidth(2)
        # ax.spines["left"].set_linewidth(2)
        ax.spines["right"].set_linewidth(0)
        ax.spines["top"].set_linewidth(0)
        ax.set_xticks(x_values)
        ax.set_xticklabels(data.columns, rotation=0)
        ax.legend(loc="upper left", fontsize=12, framealpha=1, frameon=False)


def save_confusion_matrix_display(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    target_names: list[str],
    save_path: str | Path = "confusion_matrix.pdf",
):
    with save_plot_figure(save_path):
        cm_display = ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=target_names, normalize="true"
        )
        cm_display.plot()


def save_prediction_errors_display(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str | Path = "prediction_errors.pdf",
    kind: str = "residual_vs_predicted",
):
    with save_plot_figure(save_path):
        pe_display = PredictionErrorDisplay.from_predictions(y_true, y_pred, kind=kind)
        pe_display.plot(kind=kind)


def show_parallel_coordinates(
    data: pd.DataFrame,
    y_data_encoded: np.ndarray,
    classes: list = None,
    save_path: str | Path = "visualization_parallel_coordinates.pdf",
    colormap: str = "viridis",
):
    with save_plot_figure(save_path):
        visualizer = yb.ParallelCoordinates(
            classes=classes, features=data.columns.tolist(), shuffle=True, alpha=0.7, colormap=colormap
        )
        visualizer.fit_transform(data, y_data_encoded)
        visualizer.show()


def show_radial_visualization(
    data: pd.DataFrame,
    y_data_encoded: np.ndarray,
    classes: list = None,
    save_path: str | Path = "visualization_radial.pdf",
    colormap: str = "viridis",
):
    with save_plot_figure(save_path):
        visualizer = yb.RadViz(
            classes=classes, features=data.columns.tolist(), alpha=0.7, colormap=colormap
        )
        visualizer.fit_transform(data, y_data_encoded)
        visualizer.show()


def show_2d_pca(
    data: pd.DataFrame,
    y_data_encoded: np.ndarray,
    classes: list = None,
    save_path: str | Path = "visualization_2d_pca.pdf",
    colormap: str = "viridis",
):
    with save_plot_figure(save_path):
        visualizer = yb.PCA(scale=True, classes=classes, alpha=0.7, colormap=colormap)
        visualizer.fit_transform(data, y_data_encoded)
        visualizer.show()


def show_manifold(
    data: pd.DataFrame,
    y_data_encoded: np.ndarray,
    classes: list = None,
    save_path: str | Path = "visualization_manifold.pdf",
    colormap: str = "viridis",
    method: str = "isomap",
):
    # method used can be "tsne", "lle", "isomap", "mds" etc.
    with save_plot_figure(save_path):
        visualizer = yb.Manifold(
            manifold=method, classes=classes, alpha=0.7, colormap=colormap, n_neighbors=3
        )
        visualizer.fit_transform(data, y_data_encoded)
        visualizer.show()


def show_umap(
    data: pd.DataFrame,
    y_data_encoded: np.ndarray,
    classes: list[str] = None,
    groups: list[int] = None,
    save_path: str | Path = "visualization_umap.pdf",
    colormap: str = "Spectral",
    figsize: tuple[int, int] = (8, 7),
    supervised: bool = False,
    use_internal_legend: bool = False,
    random_state: int = configs.RANDOM_SEED,
    **umap_kwargs,
):
    reducer = umap.UMAP(random_state=random_state, **umap_kwargs)
    if supervised:
        embeddings = reducer.fit_transform(data, y_data_encoded)
    else:
        embeddings = reducer.fit_transform(data)

    if groups is None:
        groups = np.zeros_like(y_data_encoded)

    cmap = plt.get_cmap(colormap)
    if int(len(classes) / len(np.unique(groups))) > 2:
        _colors = cmap(np.linspace(0, 1, int(len(classes) / len(np.unique(groups)))))
    else:
        _colors = ["red", "blue"]
    _markers = ["o", "v", "^", "<", ">", "s", "p", "*", "h", "H", "+", "x", "D", "d", "|", "_"]

    with save_plot_figure(save_path=save_path, figsize=figsize) as (fig, ax):
        # display the UMAP embedding
        y_data_encoded_ = []
        embeddings_ = []

        for group_idx in np.unique(groups):
            embedding_masked = embeddings[groups == group_idx]
            y_data_encoded_masked = y_data_encoded[groups == group_idx]

            for class_idx in np.unique(y_data_encoded_masked):
                embedding_2masked = embedding_masked[y_data_encoded_masked == class_idx]

                classes_selected = np.array(classes)[np.unique(y_data_encoded_masked)]
                classes_selected_sorted = np.sort(classes_selected)
                classes_idx_map = {
                    idx: np.where(classes_selected_sorted == item)[0][0]
                    for idx, item in enumerate(classes_selected)
                }

                ax.scatter(
                    *embedding_2masked.T,
                    s=50,
                    color=_colors[classes_idx_map[class_idx % len(np.unique(y_data_encoded_masked))]],
                    alpha=0.6,
                    marker=_markers[group_idx],
                    label=classes[class_idx],
                )

                y_data_encoded_.append(
                    [classes_idx_map[class_idx % len(np.unique(y_data_encoded_masked))]]
                    * len(y_data_encoded_masked[y_data_encoded_masked == class_idx])
                )
                embeddings_.append(embedding_2masked)

        y_data_encoded_ = np.concatenate(y_data_encoded_)
        embeddings = np.concatenate(embeddings_)

        # clf = KNeighborsClassifier(n_neighbors=3, n_jobs=-1)
        clf = SVC()
        clf.fit(embeddings, y_data_encoded_)
        y_pred = clf.predict(embeddings)
        accuracy = accuracy_score(y_data_encoded_, y_pred)

        h = 0.02
        x_min, x_max = embeddings[:, 0].min() - 1, embeddings[:, 0].max() + 1
        y_min, y_max = embeddings[:, 1].min() - 1, embeddings[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        ax.contourf(
            xx,
            yy,
            Z,
            cmap=ListedColormap(_colors),
            alpha=0.2,
        )

        handles, labels = ax.get_legend_handles_labels()
        sorted_handles_labels = sorted(zip(labels, handles), key=lambda t: t[0])  # Sort by labels
        labels, handles = zip(*sorted_handles_labels)

        accuracy_handle = Line2D(
            [0], [0], marker="o", color="w", label=f"Score:  {accuracy:.2f}", markersize=15
        )
        handles = [accuracy_handle] + list(handles)
        labels = [accuracy_handle.get_label()] + list(labels)

        if use_internal_legend:
            ax.legend(handles, labels, fontsize=18, framealpha=1)
        else:
            ax.legend(
                handles, labels, loc="center left", bbox_to_anchor=(1, 0.5), fontsize=18, framealpha=1
            )
        plt.setp(ax, xticks=[], yticks=[])


def save_data_visualization(
    data: pd.DataFrame,
    y_data_encoded: np.ndarray,
    classes: list = None,
    save_dir: Path = Path("visualization"),
    colormap="viridis",
):
    if classes is None:
        classes = np.unique(y_data_encoded)

    data = data.copy()
    data = (data - data.mean()) / data.std()

    _save_path = save_dir / "visualization_parallel_coordinates.pdf"
    show_parallel_coordinates(data, y_data_encoded, classes, _save_path, colormap)
    _save_path = save_dir / "visualization_radial.pdf"
    show_radial_visualization(data, y_data_encoded, classes, _save_path, colormap)
    _save_path = save_dir / "visualization_2d_pca.pdf"
    show_2d_pca(data, y_data_encoded, classes, _save_path, colormap)
    _save_path = save_dir / "visualization_manifold.pdf"
    show_manifold(data, y_data_encoded, classes, _save_path, colormap)
    _save_path = save_dir / "visualization_umap.pdf"
    show_umap(
        data=data,
        y_data_encoded=y_data_encoded,
        classes=classes,
        save_path=_save_path,
        colormap=colormap,
        **dict(n_neighbors=15, min_dist=1),
    )


def save_meta_visualization(
    meta: pd.DataFrame,
    save_dir: Path = Path("visualization"),
):
    for treatment in pd.unique(meta[configs.TREATMENT_ENG]):
        save_path = save_dir / f"visualization_meta_{treatment}.pdf"
        with save_plot_figure(save_path):
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


def save_target_visualization(
    target_values: np.ndarray,
    target_labels: np.ndarray = None,
    target_type: str = "classification",
    save_path: str | Path = "visualization_target.pdf",
):
    _TARGET_VAL = "Target values"
    _TARGET_LAB = "Target"
    data = pd.DataFrame.from_dict({_TARGET_VAL: target_values, _TARGET_LAB: target_labels})

    with save_plot_figure(save_path):
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
