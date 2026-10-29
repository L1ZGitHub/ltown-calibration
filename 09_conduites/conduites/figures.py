"""Petites aides de dessin pour les carnets : le réseau, une valeur par conduite."""
import numpy as np
from matplotlib.collections import LineCollection


def segments(wn, conduites: list[str]) -> np.ndarray:
    """(conduites, 2, 2) : les coordonnées des deux bouts de chaque conduite."""
    s = []
    for p in conduites:
        lien = wn.get_link(p)
        s.append([wn.get_node(lien.start_node_name).coordinates, wn.get_node(lien.end_node_name).coordinates])
    return np.asarray(s, dtype=float)


def dessiner(ax, wn, conduites, valeurs=None, cmap="viridis", vmin=None, vmax=None, epais=None,
             titre="", barre=True):
    """Le réseau, chaque conduite colorée par `valeurs` (grise sans valeur)."""
    seg = segments(wn, conduites)
    fond = LineCollection(seg, colors="0.85", linewidths=0.6)
    ax.add_collection(fond)
    if valeurs is not None:
        v = np.asarray(valeurs, dtype=float)
        lc = LineCollection(seg, array=v, cmap=cmap, linewidths=1.6 if epais is None else epais)
        lc.set_clim(np.nanmin(v) if vmin is None else vmin, np.nanmax(v) if vmax is None else vmax)
        ax.add_collection(lc)
        if barre:
            ax.figure.colorbar(lc, ax=ax, fraction=0.03)
    ax.autoscale()
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(titre)
    return ax


def marquer(ax, wn, conduites, choix, couleur="red", taille=60, label=None):
    """Un point au milieu de chaque conduite de `choix`."""
    seg = segments(wn, [p for p in conduites if p in set(choix)])
    if len(seg):
        m = seg.mean(1)
        ax.scatter(m[:, 0], m[:, 1], s=taille, facecolors="none", edgecolors=couleur, linewidths=1.8, label=label)
