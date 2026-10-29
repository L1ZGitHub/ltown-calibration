"""Ce que les carnets partagent : le réseau, les fuites de 2018, les cartes, les dessins.

Rien ici ne calcule de modèle : on relit ce qui est livré dans `sorties/` et on dessine.
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import animation

from . import figures as F
from .sources import M

ICI = Path(__file__).resolve().parents[1]
SORTIES = ICI / "sorties"
HASARD = "0.6"                       # le gris du hasard dans toutes les figures
COULEURS = {"B": "C0", "C": "C1", "08c": "C2", "hasard": HASARD}


def reseau():
    """Le réseau de L-Town (rugosités calibrées) et la liste des 905 conduites."""
    fen = M.fenetre(2018, 0, 1)
    for nom, k in fen.coefficients.items():
        fen.wn.get_link(nom).roughness = k
    return fen.wn, list(fen.wn.pipe_name_list), fen.zones


def zone_c(wn, conduites, zones) -> np.ndarray:
    """Vrai pour les conduites dont les deux bouts sont dans la zone C."""
    return np.array([all(zones.get(n, "C") == "C" for n in (wn.get_link(p).start_node_name,
                                                            wn.get_link(p).end_node_name))
                     for p in conduites])


def fuites_2018(conduites):
    """(tableau des fuites, débits horaires 8760 × 905 en m³/h)."""
    from etapes import e00_donnees as D
    fuites, q = D.fuites_publiees(2018)
    heures = pd.date_range("2018-01-01", periods=365 * 24, freq="h")
    return fuites, q.reindex(index=heures, columns=conduites, fill_value=0.0)


def distances(wn, conduites) -> np.ndarray:
    from etapes import e05_signatures as S
    return S.distances(wn).loc[conduites, conduites].to_numpy()


def lisseur(dist, wn, conduites) -> np.ndarray:
    """La matrice du lissage sur 300 m (celle des notes : `zones.noyau`)."""
    from .zones import noyau
    return noyau(dist, np.array([wn.get_link(p).length for p in conduites]))


def carte_2018(fichier, cle, conduites) -> np.ndarray:
    """Une carte de 2018 livrée (8760 heures × 905 conduites), remise dans l'ordre `conduites`."""
    z = np.load(fichier)
    return pd.DataFrame(z[cle], columns=z["conduites"].tolist())[conduites].to_numpy()


def sigmoide(x):
    return 1 / (1 + np.exp(-np.asarray(x, dtype=np.float32)))


def rang_300m(carte, p, dist):
    """Pour chaque ligne de `carte` : la place de la 1re conduite à ≤ 300 m de la conduite p
    (1 = la toute première de la carte)."""
    ordre = np.argsort(-carte, axis=1)
    return (dist[p][ordre] <= 300).argmax(axis=1) + 1


def dessiner_carte(ax, wn, conduites, valeurs, fuites=(), titre="", cmap="magma_r", vmin=0, vmax=1,
                   barre=True, top=None):
    """Une valeur par conduite ; cercles rouges = vraies fuites ; `top` = k premières en bleu."""
    F.dessiner(ax, wn, conduites, valeurs, cmap=cmap, vmin=vmin, vmax=vmax, titre=titre, barre=barre)
    if top:
        F.marquer(ax, wn, conduites, [conduites[i] for i in np.argsort(-np.asarray(valeurs))[:top]],
                  couleur="C0", taille=40, label=f"les {top} premières")
    if len(fuites):
        F.marquer(ax, wn, conduites, list(fuites), label="vraie fuite")
    return ax


def lire_json(chemin):
    return json.loads(Path(chemin).read_text())


def animer(wn, conduites, cartes, fuites_par_pas, titres, intervalle=250, cmap="magma_r", vmax=1.0):
    """Une animation (JavaScript, lisible dans JupyterLab) : une carte par pas, vraies fuites en rouge."""
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    seg = F.segments(wn, conduites)
    milieux = seg.mean(1)
    F.dessiner(ax, wn, conduites, cartes[0], cmap=cmap, vmin=0, vmax=vmax)
    lc = ax.collections[-1]
    pts = ax.scatter([], [], s=90, facecolors="none", edgecolors="red", linewidths=2)

    def image(k):
        lc.set_array(np.asarray(cartes[k], dtype=float))
        f = fuites_par_pas[k]
        pts.set_offsets(milieux[f] if len(f) else np.empty((0, 2)))
        ax.set_title(titres[k])
        return lc, pts

    a = animation.FuncAnimation(fig, image, frames=len(cartes), interval=intervalle, blit=False)
    plt.close(fig)
    return a
