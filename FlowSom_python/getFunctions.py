import re
import pandas as pd
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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
    n_clusters = len(fsom["map"]["pctgs"])  # Nombre de clusters

    # Si `layout` est une matrice ou un DataFrame
    if isinstance(layout, (np.ndarray, pd.DataFrame)):
        if layout.shape[0] == n_clusters and layout.shape[1] == 2:
            layout_df = pd.DataFrame(layout, columns=["x", "y"])
        else:
            raise ValueError(f"Layout must have {n_clusters} rows and 2 columns.")

    # Si `layout` est une chaîne de caractères
    elif layout == "grid":
        layout_df = pd.DataFrame(fsom["map"]["grid"], columns=["x", "y"])

    elif layout == "MST":
        layout_df = pd.DataFrame(fsom["MST"]["l"], columns=["x", "y"])

    else:
        raise ValueError(
            "Layout should be 'MST', 'grid', or a 2-column matrix/DataFrame."
        )

    return layout_df


def get_cluster_mfis(fsom, cols_used=False, pretty_colnames=False):
    fsom = update_flowsom(fsom)
    mfis = fsom["map"]["medianValues"]
    mfis.index = range(1, len(mfis) + 1)

    if "colsUsed" not in fsom["map"]:
        cols_used = False
    if "prettyColnames" not in fsom:
        pretty_colnames = False

    if cols_used and not pretty_colnames:
        mfis = mfis.loc[:, fsom["map"]["colsUsed"]]
    elif not cols_used and pretty_colnames:
        mfis.columns = fsom["prettyColnames"]
    elif cols_used and pretty_colnames:
        mfis = mfis.loc[:, fsom["map"]["colsUsed"]]
        mfis.columns = [fsom["prettyColnames"][i] for i in fsom["map"]["colsUsed"]]

    return mfis


def update_flowsom(fsom):
    if isinstance(fsom, dict) and "FlowSOM" in fsom:
        fsom["FlowSOM"]["metaclustering"] = fsom["metaclustering"]
        fsom = fsom["FlowSOM"]

    if not isinstance(fsom, dict) or "map" not in fsom:
        raise ValueError("fsom should be a FlowSOM object.")

    fsom["prettyColnames"] = [
        col.replace("(", "<").replace(")", ">")
        for col in fsom.get("prettyColnames", [])
    ]

    if "pctgs" not in fsom["map"]:
        pctgs = {str(i + 1): 0 for i in range(fsom["map"]["nNodes"])}
        pctgs_tmp = pd.Series(fsom["map"]["mapping"][:, 0]).value_counts(normalize=True)
        for k, v in pctgs_tmp.items():
            pctgs[str(k)] = v
        fsom["map"]["pctgs"] = pctgs

    if "nMetaclusters" not in fsom["map"]:
        fsom["map"]["nMetaclusters"] = len(set(fsom["metaclustering"]))

    if "metaclusterMFIs" not in fsom["map"] and "metaclustering" in fsom:
        metacluster_df = pd.DataFrame(fsom["data"])
        metacluster_df["mcl"] = [
            fsom["metaclustering"][i] for i in fsom["map"]["mapping"][:, 0]
        ]
        fsom["map"]["metaclusterMFIs"] = (
            metacluster_df.groupby("mcl").median().drop(columns=["mcl"])
        )

    return fsom


def get_channels(obj, markers, exact=True):
    if "flowFrame" in obj:
        object_channels = list(obj["flowFrame"]["parameters"]["name"])
        object_markers = list(obj["flowFrame"]["parameters"]["desc"])
    elif "FlowSOM" in obj:
        object_channels = list(obj["prettyColnames"].keys())
        object_markers = [re.sub(r" <.*", "", col) for col in obj["prettyColnames"]]
    else:
        raise ValueError("Object should be of class flowFrame or FlowSOM")

    channel_names = []
    for marker in markers:
        if isinstance(marker, int):
            channel_names.append(object_channels[marker])
        else:
            pattern = rf"^{re.escape(marker)}$" if exact else marker
            matched_idx = [
                i for i, name in enumerate(object_markers) if re.match(pattern, name)
            ]
            if matched_idx:
                channel_names.extend([object_channels[i] for i in matched_idx])
            else:
                raise ValueError(f"Marker {marker} could not be found")

    return channel_names


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
