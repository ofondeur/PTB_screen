import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import collections as mc

from AddFunctions import (
    FlowSOM_colors,
    add_nodes,
    add_text,
    add_legend,
    add_MST,
    parse_node_sizes,
    gg_color_hue,
)
from getFunctions import get_channels


def plot_marker_pv_hv(
    fsompv, fsomhv, marker, lim=None, cmap=FlowSOM_colors(), **kwargs
):
    mfispv = fsompv.get_cluster_data().X
    mfishv = fsomhv.get_cluster_data().X

    if lim is None:
        marker_values_pv = fsompv.mudata["cell_data"][:, marker].X.flatten()
        marker_values_hv = fsomhv.mudata["cell_data"][:, marker].X.flatten()
        lim = (
            min(np.percentile(marker_values_pv, 5), np.percentile(marker_values_hv, 5)),
            max(
                np.percentile(marker_values_pv, 95), np.percentile(marker_values_hv, 95)
            ),
        )

    markerpv = list(get_channels(fsompv, marker).keys())[0]
    marker_indexpv = np.where(fsompv.get_cell_data().var_names == markerpv)[0][0]
    figpv = plot_variable(
        fsompv,
        variable=mfispv[:, marker_indexpv],
        cmap=cmap,
        lim=lim,
        categorical=False,
        **kwargs,
    )
    markerhv = list(get_channels(fsomhv, marker).keys())[0]
    marker_indexhv = np.where(fsomhv.get_cell_data().var_names == markerhv)[0][0]
    fighv = plot_variable(
        fsomhv,
        variable=mfishv[:, marker_indexhv],
        cmap=cmap,
        lim=lim,
        categorical=False,
        **kwargs,
    )
    return figpv, fighv


def plot_marker_ph(
    fsom, marker, ref_markers=None, lim=None, cmap=FlowSOM_colors(), **kwargs
):
    mfis = fsom.get_cluster_data().X
    if lim is None:
        marker_values_ = fsom.mudata["cell_data"][:, marker].X.flatten()
        np.percentile(marker_values_, 5)
        lim = (np.percentile(marker_values_, 5), np.percentile(marker_values_, 95))
    marker = list(get_channels(fsom, marker).keys())[0]
    marker_index = np.where(fsom.get_cell_data().var_names == marker)[0][0]
    fig = plot_variable(
        fsom,
        variable=mfis[:, marker_index],
        cmap=cmap,
        lim=lim,
        categorical=False,
        **kwargs,
    )
    return fig


def plot_variable(
    fsom,
    variable,
    cmap=FlowSOM_colors(),
    labels=None,
    text_size=5,
    text_color="black",
    lim=None,
    title=None,
    categorical=True,
    **kwargs,
):
    if not isinstance(variable, np.ndarray):
        variable = np.asarray(variable)
    assert (
        variable.shape[0] == fsom.get_cell_data().uns["n_nodes"]
    ), "Length of variable should be the same as the number of nodes in your FlowSOM object"
    if variable.dtype == "object":
        string_to_number = {
            string: index for index, string in enumerate(np.unique(variable))
        }
        variable = np.asarray([string_to_number[string] for string in variable])
    fig, ax, layout, scaled_node_size = plot_FlowSOM(fsom, **kwargs)
    nodes = add_nodes(layout, scaled_node_size)
    n = mc.PatchCollection(nodes, cmap=cmap)
    n.set_array(variable)
    if lim is not None:
        n.set_clim(lim)
    n.set_edgecolor("black")
    n.set_linewidth(0.5)
    n.set_zorder(2)
    ax.add_collection(n)
    if labels is not None:
        ax = add_text(
            ax,
            layout,
            labels,
            text_size=text_size,
            text_color=text_color,
            ha=["center"],
            va=["center"],
        )
    ax, fig = add_legend(
        fig=fig,
        ax=ax,
        data=variable,
        title="Marker",
        cmap=cmap,
        location="upper left",
        bbox_to_anchor=(1.04, 1),
        categorical=categorical,
        lim=lim,
    )
    ax.axis("equal")
    if title is not None:
        plt.title(title)
    plt.axis("off")
    return fig


def plot_FlowSOM(
    fsom,
    view="MST",
    background_values=None,
    background_cmap=gg_color_hue(),
    background_size=1.5,
    equal_background_size=False,
    node_sizes=None,
    max_node_size=1,
    ref_node_size=None,
    equal_node_size=False,
):

    # Initialization
    nNodes = fsom.get_cell_data().uns["n_nodes"]
    isEmpty = fsom.get_cluster_data().obs["percentages"] == 0

    # Warnings
    if node_sizes is not None:
        assert nNodes == len(
            node_sizes
        ), 'Length of "node_sizes" should be equal to number of clusters in FlowSOM object'

    if background_values is not None:
        assert (
            background_values.shape[0] == fsom.mudata["cell_data"].uns["n_nodes"]
        ), "Length of background_values should be equal to number of clusters in FlowSOM object"

    # Node sizes
    node_sizes = parse_node_sizes(
        fsom,
        view=view,
        node_sizes=node_sizes,
        max_node_size=max_node_size,
        ref_node_size=ref_node_size,
        equal_node_size=equal_node_size,
    )
    node_sizes[isEmpty] = min([0.05, node_sizes.max()])

    # Layout
    layout = (
        fsom.get_cluster_data().obsm["layout"]
        if view == "MST"
        else fsom.get_cluster_data().obsm["grid"]
    )

    # Start plot
    fig, ax = plt.subplots()

    # Add background
    if background_values is not None:
        if equal_background_size:
            background_size = np.repeat(
                np.max(node_sizes) * background_size, len(background_values)
            )
        else:
            background_size = (
                parse_node_sizes(
                    fsom,
                    view=view,
                    node_sizes=None,
                    max_node_size=max_node_size,
                    ref_node_size=ref_node_size,
                    equal_node_size=False,
                )
                * background_size
            )
        background = add_nodes(layout, background_size)
        b = mc.PatchCollection(background, cmap=background_cmap)
        if background_values.dtype == np.float64 or background_values.dtype == np.int64:
            b.set_array(background_values)
        else:
            b.set_array(pd.get_dummies(background_values).values.argmax(1))
        b.set_alpha(0.5)
        b.set_zorder(1)
        ax.add_collection(b)
        ax, fig = add_legend(
            fig=fig,
            ax=ax,
            data=background_values,
            title="Background",
            cmap=background_cmap,
            location="lower left",
            bbox_to_anchor=(1.04, 0),
        )

    # Add MST
    if view == "MST":
        e = add_MST(fsom)
        MST = mc.LineCollection(e)
        MST.set_edgecolor("black")
        MST.set_linewidth(0.2)
        MST.set_zorder(0)
        ax.add_collection(MST)

    # Add nodes
    nodes = add_nodes(layout, node_sizes)
    n = mc.PatchCollection(nodes)
    n.set_facecolor(["#C7C7C7" if tf else "#FFFFFF" for tf in isEmpty])  # "white")
    n.set_edgecolor("black")
    n.set_linewidth(0.1)
    n.set_zorder(2)
    ax.add_collection(n)

    return fig, ax, layout, node_sizes
