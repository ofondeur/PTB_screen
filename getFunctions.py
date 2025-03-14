import re
import pandas as pd
import pandas as pd
import numpy as np
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import anndata as ad


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


def get_channels(obj, markers: np.ndarray, exact=True):
    assert obj.__class__.__name__ == "FlowSOM" or isinstance(
        obj, ad.AnnData
    ), "Please provide an FCS file or a FlowSOM object"
    if obj.__class__.__name__ == "FlowSOM":
        object_markers = np.asarray(
            [
                re.sub(" <.*", "", pretty_colname)
                for pretty_colname in obj.mudata["cell_data"].var["pretty_colnames"]
            ]
        )
        object_channels = np.asarray(
            [
                re.sub(r".*<(.*)>.*", r"\1", pretty_colname)
                for pretty_colname in obj.mudata["cell_data"].var["pretty_colnames"]
            ]
        )
    else:
        object_markers = np.asarray(obj.uns["meta"]["channels"]["$PnS"])
        object_channels = np.asarray(obj.uns["meta"]["channels"]["$PnN"])

    channelnames = {}
    for marker in markers:
        if isinstance(marker, int):
            i_channel = [marker]
        else:
            if exact:
                marker = r"^" + marker + r"$"
            i_channel = np.asarray(
                [
                    i
                    for i, m in enumerate(object_markers)
                    if re.search(marker, m) is not None
                ]
            )
        if len(i_channel) != 0:
            for i in i_channel:
                channelnames[object_channels[i]] = object_markers[i]
        else:
            i_channel = np.asarray(
                [
                    i
                    for i, c in enumerate(object_channels)
                    if re.search(marker, c) is not None
                ]
            )
            if len(i_channel) != 0:
                for i in i_channel:
                    channelnames[object_channels[i]] = object_channels[i]
            else:
                raise Exception(f"Marker {marker} could not be found!")
    return channelnames


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
