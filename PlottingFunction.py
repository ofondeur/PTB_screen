import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import re


def plot_variable(fsom, marker, color_palette="viridis", lim=None):
    fsom = update_flowsom(fsom)
    cluster_data = fsom.get_cluster_data()
    layout = parse_layout(fsom, "MST")

    mfis = get_cluster_mfis(fsom)

    if marker not in mfis.columns:
        raise ValueError(f"Le marqueur {marker} n'existe pas dans les données.")

    fig, ax = plt.subplots()
    scatter = ax.scatter(
        layout["x"], layout["y"], c=mfis[marker], cmap=color_palette, s=100, alpha=0.8
    )

    if lim:
        scatter.set_clim(lim)

    plt.colorbar(scatter, ax=ax, label=marker)

    return ax


def plot_flowsom(
    fsom,
    view="MST",
    node_sizes=None,
    max_node_size=1,
    ref_node_size=None,
    equal_node_size=False,
    background_values=None,
    background_colors=None,
    background_size=1.5,
    equal_background_size=False,
    background_lim=None,
    alpha=0.4,
    title=None,
):
    fsom = update_flowsom(fsom)
    layout = parse_layout(fsom, view)

    cluster_data = fsom.get_cluster_data()
    node_sizes = cluster_data.obs["percentages"]

    fig, ax = plt.subplots()

    if background_values is not None:
        add_background(ax, background_values)

    ax.scatter(layout["x"], layout["y"], s=node_sizes * 100, c="blue", alpha=0.5)

    if title:
        ax.set_title(title)

    return ax


def plot_marker4(
    fsom, marker, ref_markers=None, title=None, color_palette=None, lim=None, **kwargs
):
    fsom = update_flowsom(fsom)

    mfis = get_cluster_mfis(fsom)
    channels = get_channels(fsom, marker)

    if ref_markers is None:
        ref_markers = fsom["map"]["colsUsed"]

    ref_channels = get_channels(fsom, ref_markers)
    if lim is None:
        lim = [min(mfis[ref_channels]), max(mfis[ref_channels])]
    plot_list = []
    for channel in channels:
        p = plot_variable(
            fsom,
            variable=mfis[channels.index(channel)],
            variable_name="MFI",
            color_palette=color_palette,
            lim=lim,
            **kwargs,
        )

        if title and isinstance(title, list) and len(title) > channels.index(channel):
            p.set_title(title[channels.index(channel)])
        else:
            p.set_title("")

        plot_list.append(p)

    fig, axes = plt.subplots(1, len(plot_list), figsize=(15, 5))
    for ax, plot in zip(axes, plot_list):
        ax.imshow(plot)

    plt.legend(loc="right")

    return fig


def parse_node_size(node_sizes, max_node_size, ref_node_size=None):
    node_sizes = np.array(node_sizes, dtype=float)
    node_sizes[np.isnan(node_sizes)] = 0
    if ref_node_size is None:
        ref_node_size = max(node_sizes)

    if len(np.unique(node_sizes)) == 1:
        return np.full_like(node_sizes, max_node_size, dtype=float)

    scaled = (np.log1p(node_sizes) / np.log1p(ref_node_size)) * max_node_size
    return scaled


def parse_layout(fsom, layout):
    cluster_data = fsom.get_cluster_data()

    if isinstance(layout, (pd.DataFrame, np.ndarray)):
        if layout.shape[0] == cluster_data.n_obs and layout.shape[1] == 2:
            return pd.DataFrame(layout, columns=["x", "y"])
        else:
            raise ValueError(
                f"Layout doit avoir {cluster_data.n_obs} lignes et 2 colonnes."
            )

    elif layout == "grid":
        return pd.DataFrame(cluster_data.obsm["grid"], columns=["x", "y"])

    elif layout == "MST":
        return pd.DataFrame(cluster_data.uns["graph"], columns=["x", "y"])

    else:
        raise ValueError("Le layout doit être 'MST', 'grid' ou une matrice 2D.")


def get_cluster_mfis(fsom, cols_used=False, pretty_colnames=False):
    fsom = update_flowsom(fsom)
    cluster_data = fsom.get_cluster_data()

    if cluster_data.X is None:
        raise ValueError("Aucune donnée MFI disponible dans cluster_data.")

    mfis = pd.DataFrame(cluster_data.X, columns=cluster_data.var.index)

    if cols_used:
        cols_used_indices = cluster_data.var["cols_used"]
        mfis = mfis.iloc[:, cols_used_indices]

    if pretty_colnames:
        mfis.columns = cluster_data.var["pretty_colnames"]

    return mfis


def update_flowsom(fsom):
    cluster_data = fsom.get_cluster_data()

    if "n_nodes" not in cluster_data.uns:
        raise ValueError(
            "L'information sur le nombre de clusters est manquante dans cluster_data."
        )

    # Assigner les valeurs importantes
    fsom.n_nodes = cluster_data.uns["n_nodes"]
    fsom.n_metaclusters = cluster_data.uns.get("n_metaclusters", None)

    return fsom


def get_channels(obj, markers, exact=True):
    cluster_data = obj.get_cluster_data()

    if isinstance(markers, str):
        markers = [markers]

    if not isinstance(markers, (list, np.ndarray)):
        raise ValueError("markers doit être une liste ou un tableau numpy.")

    available_markers = cluster_data.var.index  # Liste des marqueurs disponibles
    matched_channels = [m for m in markers if m in available_markers]

    if not matched_channels:
        raise ValueError(f"Aucun des marqueurs {markers} n'a été trouvé dans fsom.")

    return matched_channels


def add_scale(
    ax,
    values=None,
    colors=None,
    limits=None,
    show_legend=True,
    label_legend="",
    type="fill",
):
    if isinstance(values, (str, bool)):
        values = np.array(values, dtype="category")

    # Vérifier les couleurs
    if colors is None:
        colors = ["#FFFFFF"] * len(values)

    # Appliquer l'échelle
    if isinstance(values, np.ndarray) and np.issubdtype(values.dtype, np.number):
        # Utiliser un gradient pour les valeurs continues
        cmap = plt.cm.get_cmap("viridis")
        norm = plt.Normalize(vmin=limits[0], vmax=limits[1])
        sc = ax.scatter(
            values["x"], values["y"], c=values, cmap=cmap, norm=norm, s=values["size"]
        )
        if show_legend:
            plt.colorbar(sc, ax=ax, label=label_legend)
    else:
        # Utiliser une échelle manuelle pour les valeurs discrètes
        cmap = plt.cm.get_cmap("tab10")
        sc = ax.scatter(values["x"], values["y"], c=values, cmap=cmap, s=values["size"])
        if show_legend:
            plt.legend(loc="best")

    return ax


def add_nodes(
    ax,
    node_info=None,
    values=None,
    lim=None,
    color_palette=None,
    fill_color="white",
    show_legend=True,
    label="",
    **kwargs,
):
    if isinstance(values, (str, bool)):
        values = np.array(values, dtype="category")

    add_scale(
        ax,
        values=values,
        colors=color_palette,
        lim=lim,
        label_legend=label,
        show_legend=show_legend,
    )

    ax.scatter(node_info["x"], node_info["y"], s=node_info["size"], c=values, **kwargs)

    return ax


def add_background(
    ax, background_values, background_colors=None, background_lim=None, alpha=0.4
):
    if isinstance(background_values, (list, np.ndarray)) and isinstance(
        background_values[0], str
    ):
        background_values = np.array(background_values, dtype="category")

    add_scale(
        ax,
        values=background_values,
        colors=background_colors,
        limits=background_lim,
        label_legend="background",
    )

    scatter = ax.scatter(
        x=ax.get_children()[0].get_offsets()[:, 0],
        y=ax.get_children()[0].get_offsets()[:, 1],
        s=500,
        c=background_values,
        cmap=background_colors,
        alpha=alpha,
    )

    return ax
