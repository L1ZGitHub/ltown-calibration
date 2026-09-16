"""Ce qu'on mesure pour dire qu'une calibration est meilleure qu'une autre.

Une remarque qui change les conclusions : **séparer le biais de la dispersion**.

    RMSE² = biais² + dispersion²

Le biais est la part constante de l'écart entre pression simulée et pression mesurée. Elle
disparaît dès qu'on regarde une variation — une différence entre deux instants, une dérive, un
écart à une moyenne glissante. Une calibration qui ne fait que retirer un décalage constant
améliore la RMSE sans rien changer à ce qui se voit dans la dynamique. C'est pourquoi les
tableaux de ce dépôt affichent toujours les trois colonnes, et pourquoi le gain à retenir est
celui de la **dispersion**.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import reseau as R


def par_capteur(simule: np.ndarray, mesure: np.ndarray,
                noms: list[str] | None = None) -> pd.DataFrame:
    """Biais, dispersion, RMSE et écart maximal, capteur par capteur, en mètres."""
    noms = noms or R.capteurs()["pressure"]
    n = min(len(simule), len(mesure))
    e = np.asarray(simule[:n], float) - np.asarray(mesure[:n], float)
    biais = e.mean(axis=0)
    rmse = np.sqrt((e ** 2).mean(axis=0))
    return pd.DataFrame({
        "biais_m": biais,
        "dispersion_m": np.sqrt(np.maximum(rmse ** 2 - biais ** 2, 0.0)),
        "rmse_m": rmse,
        "max_m": np.abs(e).max(axis=0),
    }, index=noms)


def resume(simule: np.ndarray, mesure: np.ndarray) -> dict:
    """Les mêmes grandeurs, agrégées sur tous les capteurs et tous les pas."""
    n = min(len(simule), len(mesure))
    e = (np.asarray(simule[:n], float) - np.asarray(mesure[:n], float)).ravel()
    biais, rmse = float(e.mean()), float(np.sqrt((e ** 2).mean()))
    return {"biais_m": biais,
            "dispersion_m": float(np.sqrt(max(rmse ** 2 - biais ** 2, 0.0))),
            "rmse_m": rmse,
            "max_m": float(np.abs(e).max()),
            "n_pas": int(n)}


def tableau(variantes: dict[str, np.ndarray], mesure: np.ndarray,
            reference: str | None = None) -> pd.DataFrame:
    """Une ligne par variante de modèle, avec le gain relatif sur la dispersion.

    `reference` nomme la variante servant de point de comparaison ; par défaut, la première.
    """
    t = pd.DataFrame({k: resume(v, mesure) for k, v in variantes.items()}).T
    ref = reference or next(iter(variantes))
    t["gain_dispersion"] = t["dispersion_m"] / t.loc[ref, "dispersion_m"] - 1.0
    t["gain_rmse"] = t["rmse_m"] / t.loc[ref, "rmse_m"] - 1.0
    return t


def erreur_de_pas(simule: np.ndarray, mesure: np.ndarray) -> dict:
    """Écart sur la **variation** d'un pas au suivant : la statistique que le biais ne touche pas.

    C'est la grandeur qui décide de ce qu'on peut détecter dans un résidu, puisque tout ce qui
    est constant s'y annule exactement.
    """
    n = min(len(simule), len(mesure))
    d = np.diff(np.asarray(simule[:n], float), axis=0) - np.diff(np.asarray(mesure[:n], float), axis=0)
    return {"rmse_pas_m": float(np.sqrt((d ** 2).mean())),
            "max_pas_m": float(np.abs(d).max())}


# ------------------------------------------------------------------------------------ figures
def tracer_capteur(ax, simule: np.ndarray, mesure: np.ndarray, capteur: str,
                   debut: int = 0, n_pas: int = R.PAS_JOUR * 3,
                   noms: list[str] | None = None, etiquette: str = "simulé"):
    """Superpose mesure et simulation sur un capteur, sur quelques jours."""
    noms = noms or R.capteurs()["pressure"]
    k = noms.index(capteur)
    t = np.arange(debut, min(debut + n_pas, len(mesure))) * R.PAS_MIN / 60.0 / 24.0
    s = slice(debut, debut + len(t))
    ax.plot(t, mesure[s, k], lw=1.0, color="0.25", label="mesure")
    ax.plot(t, simule[s, k], lw=1.0, label=etiquette)
    ax.set_xlabel("jours"); ax.set_ylabel("pression (m)"); ax.set_title(capteur)
    ax.legend(fontsize=8)
    return ax


def tracer_par_capteur(ax, tables: dict[str, pd.DataFrame], colonne: str = "dispersion_m"):
    """Compare une colonne de `par_capteur` entre plusieurs variantes, capteur par capteur."""
    largeur = 0.8 / len(tables)
    for i, (nom, t) in enumerate(tables.items()):
        x = np.arange(len(t)) + i * largeur
        ax.bar(x, t[colonne].to_numpy(), width=largeur, label=nom)
    t0 = next(iter(tables.values()))
    ax.set_xticks(np.arange(len(t0)) + 0.4 - largeur / 2)
    ax.set_xticklabels(t0.index, rotation=90, fontsize=7)
    ax.set_ylabel(colonne.replace("_m", " (m)")); ax.legend(fontsize=8)
    return ax
