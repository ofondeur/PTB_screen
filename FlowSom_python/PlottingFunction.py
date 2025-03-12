import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from getFunctions import (
    update_flowsom,
    parse_layout,
    parse_node_size,
    get_channels,
    get_cluster_mfis,
    add_nodes,
)


def plot_variable(
    fsom, variable, variable_name="", color_palette=None, lim=None, **kwargs
):
    # Mettre à jour le FlowSOM
    fsom = update_flowsom(fsom)

    # Vérifier si la longueur de la variable est correcte
    if len(variable) != len(fsom["map"]["nNodes"]):
        raise ValueError(
            f"Length of 'variable' should be equal to number of clusters in FlowSOM object "
            f"({len(fsom['map']['nNodes'])} clusters and {len(variable)} variables)."
        )

    # Plot de base avec PlotFlowSOM
    ax = plot_flowsom(fsom, **kwargs)

    # Ajouter les nœuds avec la couleur
    add_nodes(
        ax, variable=variable, color_palette=color_palette, lim=lim, label=variable_name
    )

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

    # Nombre de clusters
    n_nodes = len(fsom["map"]["pctgs"])
    is_empty = np.array(fsom["map"]["pctgs"]) == 0

    # Vérifications des tailles de vecteurs
    if node_sizes is None:
        node_sizes = fsom["map"]["pctgs"]
    if len(node_sizes) != n_nodes:
        raise ValueError(
            f"Length of 'node_sizes' should match FlowSOM clusters ({n_nodes})."
        )

    if background_values is not None and len(background_values) != n_nodes:
        raise ValueError(
            f"Length of 'background_values' should match FlowSOM clusters ({n_nodes})."
        )

    # Calcul de la disposition des nœuds
    layout = parse_layout(fsom, view)

    # Calcul de la taille des nœuds
    if ref_node_size is None:
        ref_node_size = max(node_sizes)

    if equal_node_size:
        scaled_node_size = np.full(n_nodes, max_node_size)
    else:
        scaled_node_size = parse_node_size(node_sizes, max_node_size, ref_node_size)

    scaled_node_size[is_empty] = min(max_node_size, 0.05)

    # Création du DataFrame pour le plot
    plot_df = pd.DataFrame(
        {"x": layout["x"], "y": layout["y"], "size": scaled_node_size}
    )

    # Création du plot avec seaborn
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.scatterplot(
        data=plot_df,
        x="x",
        y="y",
        size="size",
        sizes=(10, 300),
        hue=is_empty,
        palette={True: "gray", False: "white"},
        legend=False,
        ax=ax,
    )

    # Ajout du titre
    if title:
        ax.set_title(title, fontsize=14)

    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_frame_on(False)
    plt.show()

    return fig


def plot_marker2(
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
