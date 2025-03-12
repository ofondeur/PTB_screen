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
)


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
    """
    Python equivalent of PlotFlowSOM in R.
    Visualizes the FlowSOM clustering structure.
    """

    # Mise à jour de l'objet FlowSOM
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


def plot_variable(fsom, variable, variable_name="", color_palette="viridis", lim=None):
    """
    Plots a variable on a FlowSOM map using PlotFlowSOM as base.
    """

    fsom = update_flowsom(fsom)

    if len(variable) != len(fsom["map"]["pctgs"]):
        raise ValueError(
            f"Length of 'variable' should match FlowSOM clusters ({len(fsom['map']['pctgs'])})."
        )

    fig = plot_flowsom(fsom)

    # Ajouter la coloration des nœuds avec les valeurs de la variable
    sns.scatterplot(
        x=fsom["map"]["layout"]["x"],
        y=fsom["map"]["layout"]["y"],
        size=variable,
        sizes=(10, 300),
        hue=variable,
        palette=color_palette,
        legend=True,
    )

    plt.title(variable_name)
    plt.show()

    return fig


def plot_marker(
    fsom, marker, ref_markers=None, title=None, color_palette="viridis", lim=None
):
    # Mise à jour de l'objet FlowSOM (si nécessaire)
    fsom = update_flowsom(fsom)

    # Récupérer les valeurs MFI (Median Fluorescence Intensity)
    mfis = get_cluster_mfis(fsom)

    # Récupérer les colonnes associées au marqueur
    channels = get_channels(fsom, marker)

    # Calculer les limites (si non spécifiées)
    if ref_markers is None:
        ref_markers = fsom["map"]["colsUsed"]
    ref_channels = get_channels(fsom, ref_markers)

    if lim is None:
        lim = (np.min(mfis[ref_channels].values), np.max(mfis[ref_channels].values))

    # Création des sous-plots
    num_channels = len(channels)
    fig, axes = plt.subplots(1, num_channels, figsize=(5 * num_channels, 5))

    if num_channels == 1:
        axes = [axes]  # Assurer que axes est une liste même avec un seul subplot

    for i, channel in enumerate(channels):
        ax = axes[i]

        # Utiliser les valeurs MFI comme variable à représenter
        sns.heatmap(
            mfis[[channel]], ax=ax, cmap=color_palette, vmin=lim[0], vmax=lim[1]
        )

        # Ajouter un titre
        ax.set_title(title if title else channel)

    plt.tight_layout()
    return fig
