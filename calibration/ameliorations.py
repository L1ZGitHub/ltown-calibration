"""Quatre leviers que la calibration de référence laisse de côté.

Le modèle de demande de l'équipe « Under Pressure » et son ajustement de rugosité traitent la
partie « consommation » du problème. Il reste quatre choses que les capteurs disent et que le
modèle, tel qu'il est livré, n'écoute pas :

1. **La pompe suit une consigne, pas la réalité.** Le fichier de réseau pilote `PUMP_1` par deux
   contrôles de niveau (fermer au-dessus de 3,90 m, ouvrir en dessous de 2,40 m). La vraie pompe
   ne les respecte pas : elle est commandée autrement. Le cycle simulé se désynchronise donc du
   cycle réel, et comme chaque basculement de la pompe déplace toutes les pressions de la zone
   A+B, le décalage domine l'écart rapide. Or le débit de la pompe **est mesuré** : on peut lire
   son état plutôt que le deviner.
2. **Le réservoir se bloque.** Une fois la pompe commandée de l'extérieur, plus rien ne régule le
   niveau de T1 : il devient l'intégrale libre de l'écart entre l'entrée mesurée et la demande
   estimée, et il dérive jusqu'à saturer — plein à 4 m, ou vide. Le niveau étant lui aussi
   mesuré, il suffit de **réancrer** chaque tranche de simulation sur la mesure.
3. **La section du réservoir est une valeur de conception.** Le diamètre inscrit dans le modèle
   est un nombre rond (16 m), et on peut espérer le retrouver dans les mesures sans inverser le
   modèle : sur chaque demi-cycle de pompe, A·Δh = ∫(Q_pompe − Q_zoneC)·dt. Le résultat est
   instructif — mais pas dans le sens attendu, voir `section_par_demi_cycles`.
4. **Le niveau de la demande en zone A+B est mesuré, pas à modéliser.** La somme des débits
   d'entrée moins le débit de la pompe donne, à chaque pas, la consommation totale de A+B. Aucun
   modèle de demande ne battra une mesure directe ; autant recaler dessus.

Réserve à garder en tête pour le levier 4 : la mesure d'entrée contient **tout** ce qui sort du
réseau, fuites comprises. Recaler la demande sur le bilan de masse absorbe donc dans la demande
ce qui serait une fuite. C'est exactement ce qu'on veut pour calibrer un modèle hydraulique, et
exactement ce qu'on ne veut pas si l'on cherche ensuite des fuites dans le résidu : dans ce
second cas, il faut recaler sur une période saine et transporter le résultat.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from . import reseau as R


# ------------------------------------------------------------------ levier 1 : la pompe mesurée
def statut_pompe(annee: int, seuil: float = 1.0) -> np.ndarray:
    """État de `PUMP_1` pas par pas, lu au débitmètre : 1.0 en marche, 0.0 à l'arrêt.

    `seuil` est en m³/h. La pompe de L-Town débite environ 44 m³/h quand elle tourne et zéro
    sinon : la séparation est franche, et le seuil n'a pas besoin d'être fin.
    """
    return (R.charger_debits(annee)["PUMP_1"].to_numpy() > seuil).astype(float)


def transitions(statut: np.ndarray) -> np.ndarray:
    """Indices des pas où la pompe change d'état — les instants qui font sauter les pressions."""
    return np.flatnonzero(np.diff(statut) != 0) + 1


# ------------------------------------------------- leviers 3 et 4 : ce que les débitmètres disent
def demande_ab_mesuree(annee: int) -> pd.Series:
    """Consommation totale de la zone A+B, en m³/h, par bilan de masse aux débitmètres.

    Les deux conduites d'entrée `p227` et `p235` amènent toute l'eau du réseau ; la pompe `PUMP_1`
    en prélève une partie pour remplir le réservoir qui alimente la zone C. La différence sort
    donc forcément par la zone A+B — consommations et fuites confondues.
    """
    q = R.charger_debits(annee)
    return q["p227"] + q["p235"] - q["PUMP_1"]


def demande_c_estimee(wn, annee: int) -> pd.Series:
    """Consommation de la zone C, en m³/h, extrapolée des 82 compteurs aux 92 jonctions.

    Les compteurs couvrent presque toute la zone : le facteur d'extrapolation est le rapport des
    demandes nominales (jonctions de la zone C sur jonctions équipées), soit quelques pour cent.
    """
    amr = R.charger_amr(annee)
    bases = R.bases_nominales(wn).sum(axis=1)
    z = R.zones(wn)
    noeuds_c = [n for n, zz in z.items() if zz == "C"]
    facteur = float(bases[noeuds_c].sum() / bases[amr.columns].sum())
    return amr.sum(axis=1) * facteur


def demi_cycles(wn, annee: int, min_pas: int = 12, min_delta: float = 0.05) -> pd.DataFrame:
    """Découpe l'année en intervalles où la pompe ne change pas d'état, et fait le bilan de chacun.

    Sur un tel intervalle, le réservoir n'a qu'une entrée (la pompe, mesurée) et qu'une sortie
    (la zone C, estimée aux compteurs). Chaque demi-cycle fournit donc un triplet (Δh, V_pompe,
    V_zoneC), et la conservation du volume dit que A·Δh = V_pompe − V_zoneC.

    Les demi-cycles trop courts ou de trop faible amplitude sont écartés : leur Δh est dominé par
    l'arrondi du capteur de niveau, qui ne donne que deux décimales.
    """
    q = R.charger_debits(annee)["PUMP_1"].to_numpy()
    h = R.charger_niveau(annee).to_numpy()
    qc = demande_c_estimee(wn, annee).to_numpy()
    etat = (q > 1.0).astype(int)
    bornes = [0, *(np.flatnonzero(np.diff(etat) != 0) + 1).tolist(), len(etat)]
    dt = R.PAS_MIN / 60.0

    lignes = []
    for a, b in zip(bornes[:-1], bornes[1:]):
        if b - a < min_pas:
            continue
        dh = float(h[b - 1] - h[a])
        if abs(dh) < min_delta:
            continue
        lignes.append({"debut": a, "fin": b, "pas": b - a, "pompe": int(etat[a]), "dh": dh,
                       "V_pompe": float(np.sum(q[a:b - 1]) * dt),
                       "V_zoneC": float(np.sum(qc[a:b - 1]) * dt)})
    t = pd.DataFrame(lignes)
    if t.empty:
        raise RuntimeError("aucun demi-cycle exploitable : desserrer min_pas ou min_delta")
    t["V_net"] = t.V_pompe - t.V_zoneC
    return t


def section_par_demi_cycles(wn, annee: int, **kw) -> dict:
    """Estime la section du réservoir T1 par bilan volumique — et mesure à quel point c'est fragile.

    La relation A·Δh = V_net est une droite par l'origine, et une droite peut s'ajuster dans deux
    sens. Régresser V_net sur Δh suppose que l'erreur est sur les volumes ; régresser Δh sur V_net
    suppose qu'elle est sur les niveaux. Les deux hypothèses sont défendables — le capteur de
    niveau est arrondi au centimètre, et l'estimation de la consommation de la zone C n'est pas
    exacte — et elles encadrent l'estimateur sans biais. Quand l'encadrement est large, c'est que
    la grandeur n'est pas identifiable, et il vaut mieux le lire dans le résultat que le découvrir
    plus tard.

    `echelle_zoneC` va plus loin : il laisse la consommation de la zone C libre d'un facteur
    d'échelle, en résolvant Δh = α·V_pompe − β·V_zoneC. Le facteur trouvé dit de combien le bilan
    voudrait corriger les compteurs. Le nombre à regarder à côté est `correlation_remplissage` :
    sur un demi-cycle de remplissage, le volume pompé et le volume consommé sont tous deux
    proportionnels à la durée, donc presque colinéaires — et deux régresseurs colinéaires ne se
    séparent pas, quelle que soit la quantité de données.

    Renvoie un dictionnaire ; `section_m2` est l'estimateur usuel (V_net sur Δh), les autres clés
    servent à juger de sa crédibilité.
    """
    t = demi_cycles(wn, annee, **kw)
    a_volumes = float((t.V_net * t.dh).sum() / (t.dh ** 2).sum())        # erreur sur les volumes
    a_niveaux = float((t.V_net ** 2).sum() / (t.V_net * t.dh).sum())     # erreur sur les niveaux

    X = np.c_[t.V_pompe.to_numpy(), -t.V_zoneC.to_numpy()]
    beta, *_ = np.linalg.lstsq(X, t.dh.to_numpy(), rcond=None)
    remplissage = t[t.pompe == 1]
    correlation = float(np.corrcoef(remplissage.V_pompe, remplissage.V_zoneC)[0, 1])

    diam = lambda a: 2.0 * float(np.sqrt(a / np.pi))
    return {
        "section_m2": a_volumes,
        "diametre_m": diam(a_volumes),
        "encadrement_m2": (a_volumes, a_niveaux),
        "encadrement_diametre_m": (diam(a_volumes), diam(a_niveaux)),
        "section_echelle_libre_m2": 1.0 / beta[0],
        "echelle_zoneC": float(beta[1] / beta[0]),
        "correlation_remplissage": correlation,
        "n_demi_cycles": int(len(t)),
        "section_du_modele_m2": R.section_reservoir(wn),
        "diametre_du_modele_m": float(wn.get_node("T1").diameter),
        "demi_cycles": t,
    }


# ------------------------------------------- levier 4 : recalage du niveau sur le bilan de masse
def recaler_zone_ab(D: np.ndarray, noeuds: list[str], wn, annee: int,
                    periode: int = R.PAS_SEMAINE) -> np.ndarray:
    """Recale, par blocs de `periode` pas, le **niveau** de la demande A+B sur le bilan de masse.

    Seul le niveau d'ensemble est touché : à l'intérieur d'un bloc, la répartition entre nœuds et
    la forme temporelle viennent toujours du modèle de demande. On ne remplace donc pas le
    modèle, on lui impose la seule quantité que les débitmètres mesurent directement.

    Renvoie une copie ; `D` n'est pas modifié.
    """
    z = R.zones(wn)
    est_ab = np.array([z[n] == "AB" for n in noeuds])
    mesure = demande_ab_mesuree(annee).to_numpy()[:len(D)]
    E = np.array(D, copy=True)
    for a in range(0, len(D), periode):
        b = min(a + periode, len(D))
        modele = float(E[a:b][:, est_ab].sum())
        cible = float(np.nansum(mesure[a:b]))
        if modele > 0 and np.isfinite(cible) and cible > 0:
            E[a:b][:, est_ab] *= cible / modele
    return E
