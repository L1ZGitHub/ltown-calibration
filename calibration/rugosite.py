"""Calibration des rugosités par groupes de conduites, au sens de Levenberg-Marquardt.

L-Town compte 905 conduites. Les calibrer une par une n'a pas de sens : elles n'entrent dans les
équations que par leur résistance

    R = 10,674 · L / (C^1,852 · D^4,871)

et l'on ne dispose que de 33 capteurs de pression. La méthode retenue par l'équipe « Under
Pressure » est donc de **regrouper** les conduites — six groupes — et d'ajuster six facteurs
multiplicatifs sur le coefficient de Hazen-Williams, par moindres carrés non linéaires
(Levenberg-Marquardt).

Deux détails d'implémentation qui comptent :

* les facteurs sont paramétrés en **logarithme**, ce qui les maintient strictement positifs sans
  imposer de bornes — Levenberg-Marquardt, contrairement à une région de confiance, n'en accepte
  pas ;
* le vecteur de résidus est **sous-échantillonné à l'heure**. Une fenêtre de trois jours donne
  alors 72 × 33 = 2 376 résidus pour six paramètres, ce qui est déjà très largement surdéterminé,
  et chaque évaluation coûte une simulation complète.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import reseau as R


def facteurs_par_conduite(groupes: dict[str, str], valeurs: dict[str, float]) -> dict[str, float]:
    """Traduit {groupe: facteur} en {conduite: facteur}, la forme attendue par `reseau.preparer`."""
    return {p: float(valeurs[g]) for p, g in groupes.items() if g in valeurs}


def residus(simule: np.ndarray, mesure: np.ndarray, sous_ech: int = 12) -> np.ndarray:
    """Vecteur de résidus aplati, sous-échantillonné d'un facteur `sous_ech` (12 pas = 1 heure)."""
    n = min(len(simule), len(mesure))
    return (simule[:n:sous_ech] - mesure[:n:sous_ech]).ravel().astype(float)


def ajuster(simulateur: Callable[[dict[str, float]], np.ndarray],
            mesure: np.ndarray,
            noms_groupes: list[str],
            depart: float = 1.0,
            sous_ech: int = 12,
            max_nfev: int = 60,
            verbeux: bool = True) -> dict:
    """Ajuste un facteur de rugosité par groupe par Levenberg-Marquardt.

    `simulateur` prend un dictionnaire {groupe: facteur} et renvoie les pressions simulées aux
    33 capteurs, de même forme que `mesure`. Tout le reste de la configuration hydraulique —
    demandes, conditions aux limites, découpage en tranches — est enfermé dans cette fermeture,
    ce qui permet d'utiliser exactement la même routine d'ajustement avant et après les
    améliorations du second carnet.

    Renvoie {"facteurs", "rmse_depart", "rmse", "n_evaluations", "secondes", "trace"}.
    """
    journal = []
    t0 = time.time()

    def cout(theta):
        val = dict(zip(noms_groupes, np.exp(theta)))
        r = residus(simulateur(val), mesure, sous_ech)
        rmse = float(np.sqrt(np.mean(r ** 2)))
        journal.append({**val, "rmse": rmse})
        if verbeux:
            print(f"  [{len(journal):3d}] rmse {rmse:.4f} m   "
                  + "  ".join(f"{g}×{v:.3f}" for g, v in val.items()), flush=True)
        return r

    theta0 = np.full(len(noms_groupes), np.log(depart))
    r0 = cout(theta0)
    sol = least_squares(cout, theta0, method="lm", diff_step=0.02, max_nfev=max_nfev)
    facteurs = dict(zip(noms_groupes, np.exp(sol.x)))
    return {
        "facteurs": facteurs,
        "rmse_depart": float(np.sqrt(np.mean(r0 ** 2))),
        "rmse": float(np.sqrt(np.mean(sol.fun ** 2))),
        "n_evaluations": len(journal),
        "secondes": time.time() - t0,
        "trace": pd.DataFrame(journal),
    }


def balayage(simulateur: Callable[[dict[str, float]], np.ndarray],
             mesure: np.ndarray,
             noms_groupes: list[str],
             valeurs=(0.90, 0.95, 1.00, 1.05, 1.10),
             sous_ech: int = 12,
             verbeux: bool = True) -> pd.DataFrame:
    """Balaye **un seul** facteur appliqué à tous les groupes à la fois.

    Utile comme contrôle de l'ajustement à six paramètres : si la courbe à un paramètre atteint
    le même minimum, les cinq paramètres supplémentaires n'ont rien apporté.
    """
    lignes = []
    for f in valeurs:
        r = residus(simulateur({g: f for g in noms_groupes}), mesure, sous_ech)
        rmse, biais = float(np.sqrt(np.mean(r ** 2))), float(np.mean(r))
        lignes.append({"facteur": f, "rmse": rmse, "biais": biais,
                       "dispersion": float(np.sqrt(max(rmse ** 2 - biais ** 2, 0.0)))})
        if verbeux:
            print(f"  facteur {f:.3f}  rmse {rmse:.4f} m  biais {biais:+.4f} m  "
                  f"dispersion {lignes[-1]['dispersion']:.4f} m", flush=True)
    return pd.DataFrame(lignes)


# --------------------------------------------------------------------------------- diagnostic
def pertes_de_charge(D: np.ndarray, noeuds: list[str], n_pas: int = R.PAS_JOUR, **kw) -> pd.DataFrame:
    """Perte de charge de chaque conduite sur `n_pas`, en m : le diagnostic d'identifiabilité.

    La question que pose ce tableau est simple : **de combien la rugosité peut-elle bouger une
    pression ?** Si la perte de charge d'une conduite vaut quelques millimètres, changer son
    coefficient de 10 % en déplace une fraction — très en dessous de ce qu'un capteur arrondi au
    centimètre peut voir. Le nombre de paramètres identifiables ne se décide pas a priori, il se
    lit ici.

    La perte est calculée comme la **différence de charge entre les deux extrémités**, et non
    lue dans la colonne `headloss` du simulateur : celle-ci est normalisée par la longueur pour
    les conduites, et la confondre avec une perte en mètres fausse le diagnostic d'un facteur qui
    dépend de chaque conduite. Le contrôle est immédiat — la somme des pertes le long d'un chemin
    doit rendre l'écart de charge entre ses extrémités.
    """
    def extraire(res, wn, a, b):
        H = res.node["head"]
        return np.abs(np.column_stack([
            H[wn.get_link(p).start_node_name].to_numpy()[:b - a]
            - H[wn.get_link(p).end_node_name].to_numpy()[:b - a]
            for p in wn.pipe_name_list])).astype(np.float32)

    H = np.vstack(R.simuler(D, noeuds, n_pas, extraire, **kw))
    wn = R.charger_modele(1)
    t = pd.DataFrame({
        "mediane_m": np.median(H, axis=0),
        "maximum_m": H.max(axis=0),
        "longueur_m": [wn.get_link(p).length for p in wn.pipe_name_list],
        "diametre_mm": [wn.get_link(p).diameter * 1000 for p in wn.pipe_name_list],
    }, index=wn.pipe_name_list)
    return t.sort_values("mediane_m", ascending=False)


def concentration(pertes: pd.DataFrame) -> pd.Series:
    """Quelle part de la perte de charge totale est portée par les n % de conduites les plus chargées.

    C'est la lecture qui décide du nombre de groupes de rugosité utiles : si l'essentiel de la
    friction tient dans quelques dizaines de conduites, les centaines d'autres ne portent aucune
    information, et les regrouper finement ne fait qu'ajouter des paramètres non identifiables.
    """
    v = pertes["mediane_m"].sort_values(ascending=False).to_numpy()
    cum = np.cumsum(v) / v.sum()
    return pd.Series({f"{q} % des conduites": float(cum[max(int(len(v) * q / 100) - 1, 0)])
                      for q in (1, 5, 10, 25, 50)})
