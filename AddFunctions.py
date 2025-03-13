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


def add_MST(fsom):
    edges = parse_edges(fsom)
    lines = [[(row[0], row[1]), (row[2], row[3])] for row in edges]
    return lines
