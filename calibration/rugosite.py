"""Calibration des rugosités par groupes de conduites.

L-Town compte 905 conduites. Les calibrer une par une n'a pas de sens : elles n'entrent dans les
équations que par leur résistance

    R = 10,674 · L / (C^1,852 · D^4,871)

et l'on ne dispose que de 33 capteurs de pression. La méthode de référence les **regroupe** —
conduites de même matériau, âge, diamètre et conditions hydrauliques — et n'ajuste qu'un
coefficient de Hazen-Williams par groupe.

Le critère qu'elle minimise n'est pas une somme de carrés ordinaire. C'est un moindres carrés
pondéré, **sous contraintes de boîte**, avec fonction de perte de **Huber** et **régularisation
de Tikhonov** vers la valeur initiale du fichier de réseau :

    min_{x_L ≤ x ≤ x_U}  ½ Σ_j Σ_i H_κ( ([S·y(t_j, x)]_i − z_i^j) / σ_ij )  +  α ‖x − x_0‖²

`ajuster` implémente ces trois ingrédients, mais **seuls les deux premiers sont actifs par
défaut** : `tikhonov` vaut 0, et `sigma` vaut `None`. C'est un choix d'appel, pas une limite —
les carnets montrent ce que la régularisation change quand on l'active. À quoi ils servent :

* les **bornes** interdisent les coefficients qu'aucune conduite réelle ne porterait ;
* la **perte de Huber** empêche quelques pas de temps aberrants de gouverner l'ajustement ;
* la **régularisation** retient l'estimateur près du modèle livré là où les données ne
  contraignent rien — et sur ce réseau, elles ne contraignent presque rien.

Retirer les trois est instructif : l'optimiseur part alors chercher des coefficients multipliés
par une trentaine, améliore la fenêtre d'ajustement et dégrade tout le reste. C'est mesuré dans
le second carnet.

Une mise en garde qui vaut plus que les trois garde-fous réunis : **aucun d'eux ne rattrape une
mauvaise fenêtre d'ajustement**. Une fuite et un excès de friction produisent le même effet — une
baisse de pression en aval — et l'optimiseur, qui n'a que la rugosité sous la main, paie la fuite
avec de la rugosité. Voir `reseau.charge_de_fuite` : sur L-Town, les fenêtres employées vont de
zéro à un cinquième de la consommation en débit de fuite.

Deux détails d'implémentation. Le vecteur de résidus est **sous-échantillonné à l'heure** : une
fenêtre de trois jours donne déjà 72 × 33 = 2 376 résidus pour six paramètres, et chaque
évaluation coûte une simulation complète. Et le solveur employé est une **région de confiance
réfléchissante** plutôt que Levenberg-Marquardt au sens strict : ce dernier n'accepte pas de
bornes, alors que le critère ci-dessus en pose.
"""
from __future__ import annotations

import time
from collections.abc import Callable

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

from . import reseau as R


def facteurs_par_conduite(groupes: dict[str, str], valeurs: dict[str, float]) -> dict[str, float]:
    """Traduit {groupe: facteur} en {conduite: facteur} — pour le balayage à un seul paramètre."""
    return {p: float(valeurs[g]) for p, g in groupes.items() if g in valeurs}


def coefficients_par_conduite(groupes: dict[str, str], valeurs: dict[str, float]) -> dict[str, float]:
    """Traduit {groupe: coefficient} en {conduite: coefficient} — la forme de la référence.

    La calibration de référence estime des coefficients de Hazen-Williams **absolus**, bornés
    entre 60 et 160, et non des facteurs multiplicatifs. La différence n'est pas cosmétique : un
    facteur unique appliqué à un groupe qui mélange plusieurs coefficients d'origine ne peut pas
    les amener tous à la même valeur, alors que c'est exactement ce que veut dire « un groupe ».
    """
    return {p: float(valeurs[g]) for p, g in groupes.items() if g in valeurs}


def coefficients_initiaux(wn, groupes: dict[str, str]) -> dict[str, float]:
    """Coefficient de départ de chaque groupe : la moyenne de ses conduites, pondérée par longueur.

    Les groupes formés par diamètre mélangent parfois deux coefficients d'origine — sur L-Town,
    les conduites de 100 mm sont pour 104 d'entre elles à 120 et pour 601 à 140. La moyenne
    pondérée est le point de départ le moins arbitraire que le fichier permette.

    À noter : la référence part de valeurs par groupe qui ne figurent pas dans le fichier de
    réseau, donc d'une information sur le matériau et l'âge des conduites dont on ne dispose pas
    ici. C'est une limite de cette reproduction, pas un choix.
    """
    depart = {}
    for g in sorted(set(groupes.values())):
        pipes = [p for p, gg in groupes.items() if gg == g]
        L = np.array([wn.get_link(p).length for p in pipes])
        C = np.array([wn.get_link(p).roughness for p in pipes])
        depart[g] = float((L * C).sum() / L.sum())
    return depart


def bornes_par_groupe(wn, groupes: dict[str, str], marge: float = 0.10) -> dict[str, tuple]:
    """Bornes par groupe, déduites des coefficients que le groupe porte déjà dans le fichier.

    La référence encadre les coefficients entre 60 et 160, une plage qui convient à un réseau dont
    on connaît le matériau et l'âge des conduites. Sur L-Town le fichier ne contient que **deux**
    valeurs — 120 sur 119 conduites, 140 sur les 786 autres — si bien que cet encadrement autorise
    −56 % à +17 % autour du point de départ et ne contraint pratiquement rien.

    Si l'on sait par ailleurs que les paramètres du jeu de données ont été écartés de leur valeur
    vraie d'au plus `marge`, l'information s'encode bien mieux en contrainte qu'en pénalité : une
    borne ne se règle pas, alors qu'un coefficient de Tikhonov est un paramètre libre de plus.

    L'encadrement retenu va de `(1−marge)·min` à `(1+marge)·max` **du groupe**, et non de
    `(1±marge)·moyenne` : un groupe qui mélange deux coefficients d'origine doit pouvoir atteindre
    chacun d'eux. Sur le groupe D100, qui mêle 104 conduites à 120 et 601 à 140, la seconde règle
    exclurait la valeur 120 pourtant présente dans le fichier.

    ⚠️ `marge` est une **hypothèse sur la construction du jeu de données**, que rien dans ce dépôt
    ne permet de vérifier. À énoncer comme telle partout où le résultat est présenté.
    """
    C = pd.Series({p: wn.get_link(p).roughness for p in wn.pipe_name_list})
    bornes = {}
    for g in sorted(set(groupes.values())):
        v = C[[p for p, gg in groupes.items() if gg == g]]
        bornes[g] = (float((1.0 - marge) * v.min()), float((1.0 + marge) * v.max()))
    return bornes


def _pseudo_residus_huber(r: np.ndarray, f_scale: float) -> np.ndarray:
    """Résidus transformés pour que la somme de leurs carrés **soit** le critère de Huber.

    `least_squares(loss="huber")` applique la perte robuste à la totalité du vecteur qu'on lui
    rend — y compris, donc, aux lignes de régularisation qu'on y aurait ajoutées. La pénalité de
    Tikhonov s'en trouve écrasée : au-delà de quelques unités d'écart elle croît en |x−x₀| et non
    en (x−x₀)², c'est-à-dire qu'elle s'affaiblit précisément là où on la voudrait forte.

    On applique donc Huber **à la main** sur les seuls résidus de mesure, et on laisse
    `least_squares` en moindres carrés ordinaires. La transformation conserve le signe, pour que
    le jacobien numérique reste correct, et vaut l'identité tant que |r| ≤ f_scale.
    """
    z = (np.asarray(r, float) / f_scale) ** 2
    rho = np.where(z <= 1.0, z, 2.0 * np.sqrt(z) - 1.0)
    return np.sign(r) * f_scale * np.sqrt(rho)


def residus(simule: np.ndarray, mesure: np.ndarray, sous_ech: int = 12,
            sigma: np.ndarray | None = None) -> np.ndarray:
    """Vecteur de résidus aplati, sous-échantillonné d'un facteur `sous_ech` (12 pas = 1 heure).

    `sigma` est l'écart-type attribué à chaque capteur ; il pondère les résidus. Sans lui, tous
    les capteurs pèsent pareil, ce qui revient à les supposer d'égale qualité.
    """
    n = min(len(simule), len(mesure))
    e = (simule[:n:sous_ech] - mesure[:n:sous_ech]).astype(float)
    if sigma is not None:
        e = e / np.asarray(sigma, float)[None, :]
    return e.ravel()


def ajuster(simulateur: Callable[[dict[str, float]], np.ndarray],
            mesure: np.ndarray,
            depart: dict[str, float],
            bornes: tuple[float, float] | dict[str, tuple] = (60.0, 160.0),
            huber: float = 1.345,
            tikhonov: float = 0.0,
            sigma: np.ndarray | None = None,
            sous_ech: int = 12,
            max_nfev: int = 25,
            verbeux: bool = True) -> dict:
    """Ajuste un coefficient de Hazen-Williams par groupe, au critère décrit en tête de module.

    `depart` donne le coefficient initial de chaque groupe — voir `coefficients_initiaux`. C'est
    lui qui fixe l'ordre des paramètres, et c'est aussi le point x₀ vers lequel la régularisation
    de Tikhonov retient l'estimateur.

    `simulateur` prend un dictionnaire {groupe: coefficient} et renvoie les pressions simulées aux
    capteurs, de même forme que `mesure`. Toute la configuration hydraulique — demandes,
    conditions aux limites, découpage en tranches — est enfermée dans cette fermeture, ce qui
    permet d'employer la même routine avant et après un changement de conditions aux limites.

    `bornes` reprend l'encadrement de la référence, 60 à 160. `huber` est le seuil de bascule de
    la perte robuste, en résidus normalisés. `tikhonov` est le coefficient α ; à 0, désactivé.

    Attention en lisant le résultat : `rmse` porte sur les **seuls résidus de mesure**, pas sur le
    critère optimisé, sans quoi la régularisation la gonflerait artificiellement. `aux_bornes`
    nomme les groupes arrêtés sur une contrainte — s'il n'est pas vide, les données tiraient plus
    loin que ce que la physique autorise, et c'est une information en soi.
    """
    journal = []
    t0 = time.time()
    noms = list(depart)
    x0 = np.array([float(depart[g]) for g in noms])
    if isinstance(bornes, dict):                       # un encadrement propre à chaque groupe
        bas = np.array([float(bornes[g][0]) for g in noms])
        haut = np.array([float(bornes[g][1]) for g in noms])
    else:
        bas = np.full(len(noms), float(bornes[0]))
        haut = np.full(len(noms), float(bornes[1]))

    def mesurer(x):
        val = dict(zip(noms, x))
        r = residus(simulateur(val), mesure, sous_ech, sigma)
        journal.append({**val, "rmse": float(np.sqrt(np.mean(r ** 2)))})
        if verbeux:
            print(f"  [{len(journal):3d}] rmse {journal[-1]['rmse']:.4f} m   "
                  + "  ".join(f"{g}={v:.1f}" for g, v in val.items()), flush=True)
        return r

    def critere(x):
        # Huber est appliqué ici, sur les seuls résidus de mesure ; la pénalité de Tikhonov est
        # ajoutée ensuite et reste donc quadratique, comme le critère le demande.
        r = _pseudo_residus_huber(mesurer(x), huber)
        if tikhonov > 0:                       # α‖x − x₀‖² ajouté au vecteur de résidus
            r = np.concatenate([r, np.sqrt(tikhonov) * (x - x0)])
        return r

    rmse0 = float(np.sqrt(np.mean(mesurer(x0) ** 2)))
    sol = least_squares(critere, x0, method="trf", bounds=(bas, haut),
                        diff_step=0.02, max_nfev=max_nfev)
    coefficients = dict(zip(noms, sol.x))
    aux_bornes = [g for g, v, a, b in zip(noms, sol.x, bas, haut)
                  if min(abs(v - a), abs(v - b)) < 1e-6]
    return {
        "coefficients": coefficients,
        "depart": dict(depart),
        "rmse_depart": rmse0,
        "rmse": float(journal[-1]["rmse"]),
        "aux_bornes": aux_bornes,
        "n_evaluations": len(journal),
        "secondes": time.time() - t0,
        "trace": pd.DataFrame(journal),
    }


def balayage(simulateur: Callable[[dict[str, float]], np.ndarray],
             mesure: np.ndarray,
             depart: dict[str, float],
             valeurs=(0.90, 0.95, 1.00, 1.05, 1.10),
             sous_ech: int = 12,
             verbeux: bool = True) -> pd.DataFrame:
    """Balaye **un seul** facteur appliqué à tous les groupes à la fois.

    Utile comme contrôle de l'ajustement à six paramètres : si la courbe à un paramètre atteint
    le même minimum, les cinq paramètres supplémentaires n'ont rien apporté.
    """
    lignes = []
    for f in valeurs:
        r = residus(simulateur({g: c * f for g, c in depart.items()}), mesure, sous_ech)
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
