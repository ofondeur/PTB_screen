import matplotlib.colors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import collections as mc
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Wedge
from scipy.spatial.distance import pdist


def FlowSOM_colors():
    """Colormap of default FlowSOM colors."""
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
        "FlowSOM_colors",
        [
            "#00007F",
            "#0000E1",
            "#007FFF",
            "#00E1E1",
            "#7FFF7F",
            "#E1E100",
            "#FF7F00",
            "#E10000",
            "#7F0000",
        ],
    )
    return cmap


def add_nodes(layout, heights):
    if isinstance(heights, pd.Series):
        heights = heights.to_numpy()
    patches = [Circle((row[0], row[1]), heights[i]) for i, row in enumerate(layout)]
    return patches


def add_text(ax, layout, text, text_size=25, text_color="black", ha=None, va=None):
    if isinstance(text, pd.Series):
        text = text.to_numpy()
    if va is None:
        va = ["center"]
    if ha is None:
        ha = ["right"]
    if len(ha) == 1:
        ha = np.repeat(ha, len(text))
    if len(va) == 1:
        va = np.repeat(va, len(text))
    for i, row in enumerate(layout):
        ax.text(
            row[0],
            row[1],
            text[i],
            size=text_size,
            ha=ha[i],
            va=va[i],
            c=text_color,
            clip_on=False,
        )
    return ax


def add_legend(
    fig,
    ax,
    data,
    title,
    cmap,
    location="best",
    orientation="horizontal",
    bbox_to_anchor=None,
    categorical=True,
):
    if categorical:
        unique_data = sorted(np.unique(data))
        colors = cmap(np.linspace(0, 1, len(unique_data)))
        legend_elements = [
            Line2D(
                [0],
                [0],
                marker="o",
                color="w",
                label=unique_data[i],
                markerfacecolor=colors[i],
                markersize=5,
            )
            for i in range(len(unique_data))
        ]
        box = ax.get_position()
        ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
        legend = plt.legend(
            handles=legend_elements,
            loc=location,
            frameon=False,
            title=title,
            bbox_to_anchor=bbox_to_anchor,  # (1, 0.5),
            fontsize=5,
            title_fontsize=6,
        )
        plt.gca().add_artist(legend)
    else:
        norm = matplotlib.colors.Normalize(vmin=min(data), vmax=max(data))
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array(data)
        fig.colorbar(sm, ax=ax, orientation=orientation, shrink=0.4, label=title)
    return ax, fig


def parse_edges(fsom):
    edge_list = fsom.get_cluster_data().uns["graph"].get_edgelist()
    coords = fsom.get_cluster_data().obsm["layout"]
    segment_plot = [
        (
            coords[nodeID[0], 0],
            coords[nodeID[0], 1],
            coords[nodeID[1], 0],
            coords[nodeID[1], 1],
        )
        for nodeID in edge_list
    ]
    return np.asarray(segment_plot, dtype=np.float32)


def add_MST(fsom):
    edges = parse_edges(fsom)
    lines = [[(row[0], row[1]), (row[2], row[3])] for row in edges]
    return lines


def gg_color_hue():
    """Colormap of default ggplot colors."""
    cmap = matplotlib.colors.ListedColormap(
        [
            "#F8766D",
            "#D89000",
            "#A3A500",
            "#39B600",
            "#00BF7D",
            "#00BFC4",
            "#00B0F6",
            "#9590FF",
            "#E76BF3",
            "#FF62BC",
        ]
    )
    return cmap


def auto_max_node_size(layout, overlap):
    overlap = 1 + overlap
    min_distance = min(pdist(layout))
    return min_distance / 2 * overlap


def parse_node_sizes(
    fsom,
    view="MST",
    node_sizes=None,
    max_node_size=1,
    ref_node_size=None,
    equal_node_size=False,
):
    node_sizes = (
        fsom.get_cluster_data().obs["percentages"] if node_sizes is None else node_sizes
    )
    ref_node_size = max(node_sizes) if ref_node_size is None else ref_node_size
    layout = (
        fsom.get_cluster_data().obsm["layout"]
        if view == "MST"
        else fsom.get_cluster_data().obsm["grid"]
    )
    auto_node_size = auto_max_node_size(layout, 1 if view == "MST" else -0.3)  # overlap
    max_node_size = auto_node_size * max_node_size

    if equal_node_size:
        node_sizes = np.repeat(max_node_size, len(node_sizes))
    n_nodes = len(node_sizes)
    if len(np.unique(node_sizes)) == 1:
        return np.repeat(max_node_size, n_nodes)
    scaled_node_size = max_node_size * np.exp(node_sizes / ref_node_size) / np.exp(1)

    # scaled_node_size = max_node_size * (node_sizes / ref_node_size) ** 1.5

    # scaled_node_size = np.sqrt(np.multiply((np.divide(node_sizes, ref_node_size)), np.square(max_node_size)))
    return scaled_node_size
