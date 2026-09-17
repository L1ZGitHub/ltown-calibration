"""Cinq leviers que la calibration de référence laisse de côté.

Le modèle de demande de l'équipe « Under Pressure » et son ajustement de rugosité traitent la
partie « consommation » du problème. Il reste cinq choses que les capteurs disent et que le
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
   est un nombre rond (16 m). Le bilan volumique par demi-cycle de pompe semble le retrouver sans
   inverser le modèle — mais il mesure en fait une combinaison de la section et de la
   consommation de la zone C, voir `section_par_demi_cycles`. `ajuster_reservoir` lève
   l'ambiguïté en simulant la trajectoire du niveau au lieu de régresser des demi-cycles. Le
   résultat final reste négatif, pour une autre raison que l'identifiabilité : une fois le niveau
   réancré (levier 2), la section n'a plus d'effet.
4. **Le niveau de la demande en zone A+B est mesuré, pas à modéliser.** La somme des débits
   d'entrée moins le débit de la pompe donne, à chaque pas, la consommation totale de A+B. Aucun
   modèle de demande ne battra une mesure directe ; autant recaler dessus.
5. **Sur les 82 jonctions équipées, la demande est mesurée et le modèle l'écrase quand même.**
   L'équation (2) est appliquée aux 782 jonctions sans exception, alors que 82 d'entre elles ont
   leur série connue pas par pas. `injecter_compteurs` les remet à leur valeur lue.

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


def bilan_zone_c(wn, annee: int) -> pd.DataFrame:
    """Décompose la zone C par bilan de masse : ce qui entre, ce qui est consommé, ce qui fuit.

    La zone C est la seule partie du réseau dont **toute** la frontière est instrumentée. Tout ce
    qui y entre passe par la pompe, dont le débit est mesuré, et le seul stock est le réservoir,
    dont le niveau est mesuré :

        entrée = Q_pompe − A · dh/dt

    Ce qui y est consommé est donné par les 82 compteurs, extrapolés aux 92 jonctions
    (`demande_c_estimee`). La différence est donc ce qui sort du réseau sans être consommé —
    autrement dit la **fuite**, obtenue sans simuler et sans ouvrir le fichier de fuites.

    Sur 2018 l'estimation est juste à **+0,08 m³/h** près, avec un bruit de 0,23 m³/h sur des
    moyennes journalières (`test_le_bilan_de_la_zone_c_retrouve_les_fuites_publiees`). Le bruit au
    pas de 5 minutes est dix fois plus grand : le capteur de niveau n'a que deux décimales, et
    `A · dh` amplifie cet arrondi d'un facteur 201 m².

    Rien d'équivalent n'existe pour la zone A+B : la différence aux débitmètres d'entrée y donne
    la somme « fuites + consommation », que rien ne permet de séparer.
    """
    q = R.charger_debits(annee)
    h = R.charger_niveau(annee)
    section = R.section_reservoir(wn)
    entre = q["PUMP_1"] - section * h.diff().shift(-1) / (R.PAS_MIN / 60.0)
    consomme = demande_c_estimee(wn, annee)
    return pd.DataFrame({"entre_m3h": entre, "consomme_m3h": consomme,
                         "fuite_m3h": entre - consomme})


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


def ajuster_reservoir(D: np.ndarray, noeuds: list[str], wn, annee: int, *,
                      debut: int = 0, n_pas: int = 14 * R.PAS_JOUR,
                      diametres=(13.0, 14.0, 15.0, 16.0, 17.0, 18.0),
                      echelles=(1.0, 1.2, 1.4, 1.6, 1.8),
                      verbeux: bool = True) -> pd.DataFrame:
    """Estime la section de T1 **par simulation**, en visant le capteur de niveau lui-même.

    Le bilan volumique de `section_par_demi_cycles` bute sur un aliasing : sur un demi-cycle de
    remplissage, le volume pompé et le volume consommé sont tous deux proportionnels à la durée du
    cycle, donc indissociables. Cette fonction contourne le problème au lieu de le subir, en
    exploitant une asymétrie que le découpage en demi-cycles jetait :

    * pompe à l'arrêt, la pente du niveau vaut −Q_C / A — elle ne donne qu'un **rapport** ;
    * pompe en marche, elle vaut (Q_pompe − Q_C) / A, et Q_pompe, lui, est **mesuré**.

    Les deux régimes pris ensemble séparent donc A de la consommation de la zone C. On les fait
    travailler en simulant la trajectoire complète du niveau **sans jamais la réancrer** — c'est
    l'accumulation libre de l'écart qui porte l'information — et en balayant les deux paramètres :
    le diamètre du réservoir, et un facteur d'échelle sur la demande de la zone C.

    **Vérifier la colonne `sature` avant de croire le résultat.** Sur une fenêtre où le réservoir
    passe une part notable du temps collé au trop-plein, le critère ne s'aplatit pas — ce serait
    trop commode — il devient **monotone** et pousse vers le plus petit réservoir de la grille :
    un petit réservoir se remplit et se vide plus vite, donc sature moins, et l'optimiseur
    poursuit la saturation au lieu de la physique. Mesuré sur L-Town : en janvier (saturation sous
    les quelques pour cent) le minimum est intérieur, vers 15 à 16 m ; en juillet et en octobre
    (13 à 20 % de saturation) il est au bord de la grille, quelle que soit la grille.

    Deux vérifications, donc, et pas une : que l'optimum soit **intérieur**, et que le régime
    simulé soit celui qu'on croit. C'est pourquoi la saturation est rendue à côté de l'erreur.

    Renvoie un tableau trié par erreur croissante, une ligne par couple (diamètre, échelle).
    """
    niv = R.charger_niveau(annee).to_numpy()[debut:debut + n_pas]
    pompe = statut_pompe(annee)[debut:debut + n_pas + 1]
    z = R.zones(wn)
    est_c = np.array([z[n] == "C" for n in noeuds])
    base = D[debut:debut + n_pas + 1]

    lignes = []
    for k in echelles:
        E = np.array(base, copy=True)
        E[:, est_c] *= float(k)
        for d in diametres:
            h = np.concatenate(R.simuler(
                E, noeuds, n_pas,
                lambda res, w, a, b: (res.node["head"]["T1"].to_numpy()[:b - a]
                                      - w.get_node("T1").elevation),
                tranche_jours=n_pas / R.PAS_JOUR, niveau0=float(niv[0]),
                pompe=pompe, diametre_T1=float(d), verbeux=False))
            e = h - niv[:len(h)]
            lignes.append({"diametre_m": float(d), "echelle_zoneC": float(k),
                           "erreur_niveau_m": float(np.sqrt((e ** 2).mean())),
                           "sature": float(np.mean((h <= 0.001) | (h >= 3.999)))})
            if verbeux:
                print(f"  diamètre {d:5.2f} m   échelle ×{k:.2f}   "
                      f"erreur {lignes[-1]['erreur_niveau_m']:.3f} m   "
                      f"saturé {lignes[-1]['sature']:.0%}", flush=True)
    return pd.DataFrame(lignes).sort_values("erreur_niveau_m").reset_index(drop=True)


# ------------------------------------------- levier 4 : recalage du niveau sur le bilan de masse
def recaler_zone_ab(D: np.ndarray, noeuds: list[str], wn, annee: int,
                    periode: int = R.PAS_SEMAINE, debut: int = 0,
                    cible: pd.Series | None = None) -> np.ndarray:
    """Recale, par blocs de `periode` pas, le **niveau** de la demande A+B sur le bilan de masse.

    Seul le niveau d'ensemble est touché : à l'intérieur d'un bloc, la répartition entre nœuds et
    la forme temporelle viennent toujours du modèle de demande. On ne remplace donc pas le
    modèle, on lui impose la seule quantité que les débitmètres mesurent directement.

    `debut` dit à quel pas de l'année commence `D`, pour aligner la mesure sur une fenêtre.

    `cible` remplace le bilan de masse par une autre série de référence — par exemple le bilan
    **moins les fuites publiées**, qui donne la consommation réelle de la zone. Le bilan brut
    contient les fuites et les verse donc dans la demande ; une cible sans fuite ne le fait pas,
    mais elle n'est disponible que sur une année dont on connaît déjà les fuites.

    Renvoie une copie ; `D` n'est pas modifié.
    """
    z = R.zones(wn)
    est_ab = np.array([z[n] == "AB" for n in noeuds])
    serie = demande_ab_mesuree(annee) if cible is None else cible
    mesure = np.asarray(serie, float)[debut:debut + len(D)]
    E = np.array(D, copy=True)
    for a in range(0, len(D), periode):
        b = min(a + periode, len(D))
        modele = float(E[a:b][:, est_ab].sum())
        cible = float(np.nansum(mesure[a:b]))
        if modele > 0 and np.isfinite(cible) and cible > 0:
            E[a:b][:, est_ab] *= cible / modele
    return E

# ------------------------------------- levier 5 : la mesure elle-même là où le compteur existe
def injecter_compteurs(D: np.ndarray, noeuds: list[str], annee: int,
                       debut: int = 0) -> np.ndarray:
    """Remplace la demande **modélisée** par la demande **mesurée** sur les jonctions équipées.

    Le modèle de demande applique l'équation (2) aux 782 jonctions, y compris aux 82 qui portent
    un compteur — dont la série est pourtant connue pas par pas. C'est une perte sèche : sur ces
    nœuds, aucune forme estimée ne vaut la mesure. Cette fonction les remet à leur valeur lue.

    Les 10 jonctions de la zone C sans compteur gardent leur demande modélisée : leur nominal
    cumulé vaut 1,84 m³/h, soit 6 % de la zone.

    `debut` dit à quel pas de l'année commence `D`, pour aligner la mesure sur une fenêtre.

    Renvoie une copie ; `D` n'est pas modifié.
    """
    amr = R.charger_amr(annee).iloc[debut:debut + len(D)]
    E = np.array(D, copy=True)
    index = {n: k for k, n in enumerate(noeuds)}
    colonnes = [c for c in amr.columns if c in index]
    cibles = [index[c] for c in colonnes]
    E[:, cibles] = amr[colonnes].to_numpy()[:len(E)].astype(E.dtype, copy=False)
    return E


# --------------------------------------- reproduire l'année telle qu'elle a été, fuites comprises
def poser_fuites_publiees(D: np.ndarray, noeuds: list[str], wn, annee: int,
                          debut: int = 0) -> np.ndarray:
    """Ajoute le débit de fuite publié comme demande supplémentaire, au nœud amont de sa conduite.

    Le modèle de demande ne contient aucune fuite : il est construit sur des compteurs de
    consommation, qu'une conduite percée ne traverse jamais. Pour **reproduire** l'année plutôt
    que de la détecter, il faut donc les remettre — et à leur place, pas diluées sur la zone.

    Une fuite est posée en un point de sa conduite ; on la porte au nœud amont, faute de mieux. Le
    débit vient du fichier publié, donc cette fonction ne sert qu'à valider : elle utilise la
    réponse. Elle ne peut pas servir à détecter, ni à travailler sur une année sans fichier.

    Renvoie une copie ; `D` n'est pas modifié.
    """
    lk = R.charger_fuites(annee)
    if lk is None:
        return np.array(D, copy=True)
    E = np.array(D, copy=True)
    index = {n: k for k, n in enumerate(noeuds)}
    t = lk.to_numpy()[debut:debut + len(E)]
    for j, conduite in enumerate(lk.columns):
        lien = wn.get_link(conduite)
        noeud = next((n for n in (lien.start_node_name, lien.end_node_name) if n in index), None)
        if noeud is not None:
            E[:, index[noeud]] += t[:, j].astype(E.dtype, copy=False)
    return E
