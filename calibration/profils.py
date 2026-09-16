"""Modèle de demande de l'équipe « Under Pressure », reproduit à l'identique.

La méthode tient en deux équations.

**(1) Un compteur.** Chaque compteur communicant (AMR) est décomposé en un produit d'effets :

    d(t) = d̄ · T(t) · S(t) · R(t)

où `d̄` est la moyenne annuelle, `T(t)` la **tendance** (moyenne glissante centrée sur une
fenêtre d'une semaine, qui porte donc la saisonnalité annuelle), `S(t)` la **saisonnalité
hebdomadaire** (médianes périodiques du signal détendancé, renormalisées à 1) et `R(t)` un
résidu — que la méthode **jette**. C'est le point important : on ne cherche pas à prédire le
bruit de consommation, on cherche la partie reproductible.

**(2) Un nœud non mesuré.** Les 82 compteurs sont en zone C ; les 700 autres jonctions n'ont
aucune mesure. Le modèle les reconstruit comme un **mélange** de types de consommateurs :

    d̂ᵢ(t) = Σⱼ d̄ᵢⱼ · Tⱼ(t) · Sⱼ(t)

`d̄ᵢⱼ` est la demande nominale du nœud i pour le type j, lue directement dans le modèle — chaque
jonction y porte trois lignes de demande (résidentiel, commercial, industriel). Les formes
`Tⱼ·Sⱼ` par type, elles, ne sont pas mesurées directement : on les obtient en résolvant
l'équation (2) **à l'envers** sur les 82 nœuds où l'on a la mesure.

À noter pour lire les résultats : la somme sur les trois types est un **mélange**, pas un
produit. Le produit de l'équation (1) porte sur les effets temporels d'un même compteur.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.ndimage import uniform_filter1d

from . import reseau as R


# ------------------------------------------------------------------ équation (1) : un compteur
def tendance(x: np.ndarray, fenetre: int = R.PAS_SEMAINE) -> np.ndarray:
    """T(t) : moyenne glissante centrée sur `fenetre` pas, bords tenus (pas de trou aux extrémités).

    Une fenêtre d'exactement une semaine annule le cycle hebdomadaire : ce qui reste est la
    dérive lente — vacances d'été, saison de chauffe, croissance du parc.

    Limite à connaître : le filtre atténue aussi ce qu'il estime. Une moyenne glissante de largeur
    `fenetre` multiplie une composante de période `p` par sinc(fenetre/p) — négligeable pour une
    saisonnalité annuelle (0,1 %), mais déjà 2,5 % pour une variation de deux mois. La tendance
    restituée est donc systématiquement un peu plate ; c'est le prix d'un estimateur qui ne
    suppose aucune forme.
    """
    return uniform_filter1d(np.asarray(x, float), size=int(fenetre), mode="nearest")


def saisonnalite(x: np.ndarray, creneaux: np.ndarray, n: int = R.PAS_SEMAINE) -> np.ndarray:
    """S : profil hebdomadaire de `n` valeurs, par **médiane** de créneau, renormalisé à 1.

    Médiane et non moyenne : quelques journées atypiques par an suffisent à déformer une moyenne
    de 52 points, et le but de l'étape est précisément de retirer ces journées-là dans le résidu.
    """
    x = np.asarray(x, float)
    s = np.array([np.median(x[creneaux == k]) if np.any(creneaux == k) else 1.0 for k in range(n)])
    moy = s.mean()
    return s / moy if moy > 0 else np.ones(n)


def decomposer(serie: np.ndarray, creneaux: np.ndarray) -> dict:
    """Décompose une série de compteur en d̄, T(t), S (profil) et R(t), selon l'équation (1).

    Renvoie un dictionnaire ; `lisse` est la reconstruction d̄·T(t)·S(t), c'est-à-dire la série
    **privée de son résidu** — la seule partie que le modèle de demande conserve.
    """
    d = np.asarray(serie, float)
    moyenne = float(d.mean())
    if moyenne <= 0:                                    # compteur muet : rien à décomposer
        n = len(d)
        return {"moyenne": 0.0, "tendance": np.ones(n), "saison": np.ones(R.PAS_SEMAINE),
                "residu": np.ones(n), "lisse": np.zeros(n)}
    x = d / moyenne
    T = tendance(x)
    S = saisonnalite(x / np.maximum(T, 1e-9), creneaux)
    St = S[creneaux]
    lisse = moyenne * T * St
    residu = np.divide(d, np.maximum(lisse, 1e-9), out=np.ones_like(d), where=lisse > 1e-9)
    return {"moyenne": moyenne, "tendance": T, "saison": S, "residu": residu, "lisse": lisse}


def decomposer_amr(annee: int = 2018) -> dict:
    """Applique l'équation (1) aux 82 compteurs d'une année.

    Renvoie `{"mesure": DataFrame, "lisse": DataFrame, "moyennes": Series, "saisons": DataFrame,
    "tendances": DataFrame}` — toutes les séries en m³/h et indexées par l'horodatage des mesures.
    """
    amr = R.charger_amr(annee)
    cr = R.creneau(amr.index)
    parts = {c: decomposer(amr[c].to_numpy(), cr) for c in amr.columns}
    return {
        "mesure": amr,
        "lisse": pd.DataFrame({c: p["lisse"] for c, p in parts.items()}, index=amr.index),
        "moyennes": pd.Series({c: p["moyenne"] for c, p in parts.items()}),
        "tendances": pd.DataFrame({c: p["tendance"] for c, p in parts.items()}, index=amr.index),
        "saisons": pd.DataFrame({c: p["saison"] for c, p in parts.items()}),
        "creneaux": cr,
    }


# ------------------------------------------------- équation (2) : des compteurs vers les nœuds
def formes_par_categorie(wn, annee: int = 2018, decomposition: dict | None = None) -> pd.DataFrame:
    """Les formes temporelles `Tⱼ(t)·Sⱼ(t)` des trois types de consommateurs (T pas × 3 colonnes).

    L'équation (2) dit que la série d'un nœud est le mélange des formes de type, pondéré par ses
    demandes nominales. Sur les 82 nœuds où la série est mesurée, cette relation se retourne :
    avec `W` la matrice 82×3 des bases nominales par type et `A` les séries lissées par
    l'équation (1), on résout `W · m(t) = A(t)` au sens des moindres carrés, pour chaque pas.

    La pseudo-inverse est calculée une fois et appliquée par un seul produit matriciel : l'année
    entière coûte moins d'une seconde. Les formes sont écrêtées à zéro (une demande négative n'a
    pas de sens) puis renormalisées à une moyenne de 1, de sorte qu'elles se lisent comme des
    multiplicateurs de profil.
    """
    dec = decomposition or decomposer_amr(annee)
    A = dec["lisse"]
    W = R.bases_nominales(wn).loc[A.columns].to_numpy()          # 82 × 3
    formes = (np.linalg.pinv(W) @ A.to_numpy().T).T              # T × 3
    formes = np.clip(formes, 0.0, None)
    moy = formes.mean(axis=0)
    formes = np.divide(formes, np.where(moy > 0, moy, 1.0))
    return pd.DataFrame(formes, index=A.index, columns=list(R.CATEGORIES))


def demandes_calibrees(wn, annee: int = 2018, formes: pd.DataFrame | None = None,
                       n_pas: int | None = None, dtype=np.float32) -> tuple[np.ndarray, list[str]]:
    """Applique l'équation (2) aux 782 jonctions : renvoie (T × 782) en m³/h, et la liste des nœuds.

    Une année entière fait 105 120 × 782 valeurs, soit 330 Mo en float32 et le double en float64 :
    c'est la plus grosse matrice du dépôt. Le produit est donc fait directement dans le type
    demandé, sans passer par un intermédiaire en double précision, et `n_pas` permet de n'en
    calculer qu'un début — ce que font les carnets pour rester légers.
    """
    f = formes if formes is not None else formes_par_categorie(wn, annee)
    bases = R.bases_nominales(wn)
    M = f.to_numpy()[:n_pas].astype(dtype, copy=False)           # T × 3
    W = bases.to_numpy().T.astype(dtype, copy=False)             # 3 × 782
    return M @ W, list(bases.index)


def demandes_nominales(wn, index: pd.DatetimeIndex, dtype=np.float32) -> tuple[np.ndarray, list[str]]:
    """Le point de départ : bases nominales × profils hebdomadaires du modèle, sans rien calibrer.

    C'est l'état du modèle tel qu'il est livré. Les profils du fichier sont hebdomadaires et ne
    portent **aucune saisonnalité annuelle** — la documentation du jeu de données le dit
    explicitement. C'est la référence contre laquelle la calibration doit gagner.
    """
    noeuds = wn.junction_name_list
    profils = {p: np.asarray(wn.get_pattern(p).multipliers, float) for p in wn.pattern_name_list}
    pas = np.arange(len(index))
    D = np.zeros((len(index), len(noeuds)), dtype=dtype)
    for k, n in enumerate(noeuds):
        for d in wn.get_node(n).demand_timeseries_list:
            m = profils[d.pattern_name]
            D[:, k] += d.base_value * 3600.0 * m[pas % len(m)]
    return D, list(noeuds)
