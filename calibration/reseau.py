"""Accès au réseau L-Town et aux mesures : chargement, zones, simulation par tranches.

Trois pièges d'unités, réglés ici une fois pour toutes :

1. `L-TOWN.inp` est écrit en **CMH** (m³/h) mais wntr travaille en SI (m³/s). Toute série qui
   sort de ce module est en **m³/h** ; la division par 3 600 se fait au moment d'écrire dans
   le modèle, nulle part ailleurs.
2. Les compteurs AMR du fichier `*_SCADA_Demands.csv` sont en **L/h**, pas en m³/h.
3. Les CSV sont européens : séparateur `;`, virgule décimale.

Et un piège de mémoire : une année au pas de 5 min ne se simule pas d'un seul appel. wntr garde
tous ses résultats en RAM et une année complète coûte plusieurs gigaoctets. `simuler` découpe en
tranches de quelques jours et reporte l'état du réservoir d'une tranche à la suivante.
"""
from __future__ import annotations

import os
import tempfile
import uuid
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import wntr
import yaml
from wntr.network.base import LinkStatus

RACINE = Path(__file__).resolve().parent.parent
DONNEES = Path(os.environ.get("LTOWN_DIR", RACINE / "data" / "raw"))
INP = DONNEES / "L-TOWN.inp"
CONFIG = DONNEES / "dataset_configuration.yaml"

PAS_MIN = 5                       # pas de temps des mesures, et de tout ce dépôt
PAS_JOUR = 24 * 60 // PAS_MIN     # 288
PAS_SEMAINE = 7 * PAS_JOUR        # 2016
PAS_AN = 365 * PAS_JOUR           # 105 120 (ni 2018 ni 2019 n'est bissextile)
CATEGORIES = ("Residential", "Commercial", "Industrial")

FICHIERS = ["L-TOWN.inp", "dataset_configuration.yaml",
            "{a}_SCADA_Pressures.csv", "{a}_SCADA_Flows.csv",
            "{a}_SCADA_Levels.csv", "{a}_SCADA_Demands.csv"]


def verifier_donnees(annee: int = 2018) -> None:
    """Lève une erreur lisible si le dossier de données n'est pas complet."""
    manquants = [f.format(a=annee) for f in FICHIERS if not (DONNEES / f.format(a=annee)).exists()]
    if manquants:
        raise FileNotFoundError(
            f"fichiers absents de {DONNEES} : {', '.join(manquants)}.\n"
            "Déposez le jeu de données L-Town dans ce dossier, ou pointez la variable "
            "d'environnement LTOWN_DIR vers l'endroit où il se trouve.")


# --------------------------------------------------------------------------- lecture des mesures
def charger_scada(annee: int, genre: str) -> pd.DataFrame:
    """Un fichier de mesures (`Pressures`, `Flows`, `Levels`, `Demands`), indexé par horodatage."""
    df = pd.read_csv(DONNEES / f"{annee}_SCADA_{genre}.csv", sep=";", decimal=",",
                     index_col=0, parse_dates=True)
    df.index.name = "temps"
    return df


def charger_amr(annee: int) -> pd.DataFrame:
    """Les 82 compteurs AMR, **convertis de L/h en m³/h**. Colonnes = identifiants de jonction."""
    return charger_scada(annee, "Demands") / 1000.0


def charger_debits(annee: int) -> pd.DataFrame:
    """Les 3 débitmètres (`p227`, `p235`, `PUMP_1`), déjà en m³/h dans le fichier."""
    return charger_scada(annee, "Flows")


def charger_niveau(annee: int) -> pd.Series:
    """Le niveau du réservoir T1, en m."""
    return charger_scada(annee, "Levels")["T1"]


def charger_pressions(annee: int) -> pd.DataFrame:
    """Les 33 capteurs de pression en m, dans l'ordre du fichier de configuration."""
    return charger_scada(annee, "Pressures")[capteurs()["pressure"]]


def capteurs() -> dict[str, list[str]]:
    """Listes officielles du fichier de configuration : pression (33), débit (3), niveau (1)."""
    cfg = yaml.safe_load(CONFIG.read_text())
    return {k: [str(s) for s in cfg[f"{k}_sensors"]] for k in ("pressure", "flow", "level")}


def horodatage(annee: int, n_pas: int = PAS_AN) -> pd.DatetimeIndex:
    """Index de `n_pas` pas de 5 min à partir du 1er janvier de `annee`."""
    return pd.date_range(f"{annee}-01-01", periods=n_pas, freq=f"{PAS_MIN}min")


def creneau(index: pd.DatetimeIndex) -> np.ndarray:
    """Créneau hebdomadaire (0 à 2015) de chaque horodatage, **calculé par jour de la semaine**.

    Surtout pas `pas % 2016` : 2018 commence un lundi et 2019 un mardi. Un profil hebdomadaire
    estimé sur 2018 ne s'applique à 2019 que si le créneau est repéré par le jour réel.
    """
    return (index.dayofweek * PAS_JOUR + index.hour * 12 + index.minute // PAS_MIN).to_numpy()


# ------------------------------------------------------------------------ modèle et topologie
def charger_modele(duree_h: float = 24.0, pas_min: int = PAS_MIN, inp: str | Path | None = None):
    """Un modèle neuf, avec l'horizon et le pas de report demandés."""
    verifier_donnees()
    wn = wntr.network.WaterNetworkModel(str(inp or INP))
    wn.options.time.duration = int(duree_h * 3600)
    wn.options.time.report_timestep = pas_min * 60
    return wn


def graphe(wn) -> nx.Graph:
    """Tous les liens en graphe non orienté ; pompe et vannes de longueur nulle."""
    g = nx.Graph()
    for _, lien in wn.links():
        g.add_edge(lien.start_node_name, lien.end_node_name,
                   length=float(getattr(lien, "length", 0.0) or 0.0))
    return g


def zones(wn) -> dict[str, str]:
    """Zone de chaque jonction : `"C"` (derrière le réservoir T1) ou `"AB"` (alimentée par R1/R2).

    Déduit du graphe et non d'une liste écrite à la main : on coupe les deux arêtes qui traversent
    le réservoir — la pompe `PUMP_1` (n54→T1) et la sortie `p239` (T1→n343) — et la composante
    connexe de `n343` est la zone C. Résultat sur L-Town : 92 jonctions en C, 690 en A+B.
    """
    g = graphe(wn)
    g.remove_edge("n54", "T1")
    g.remove_edge("T1", "n343")
    zone_c = next(c for c in nx.connected_components(g) if "n343" in c)
    return {n: ("C" if n in zone_c else "AB") for n in wn.junction_name_list}


def bases_nominales(wn) -> pd.DataFrame:
    """Demande nominale de chaque jonction **par catégorie**, en m³/h : 782 lignes × 3 colonnes.

    Le modèle porte trois lignes de demande par jonction, une par type de consommateur, chacune
    attachée à son profil hebdomadaire. Attention : `Junction.base_demand` ne renvoie que la
    **première** des trois (donc 0 pour une jonction purement industrielle comme n1) ; il faut
    parcourir `demand_timeseries_list`.
    """
    lignes = {}
    for n in wn.junction_name_list:
        ligne = dict.fromkeys(CATEGORIES, 0.0)
        for d in wn.get_node(n).demand_timeseries_list:
            cat = str(d.pattern_name).replace("P-", "")
            ligne[cat] = ligne.get(cat, 0.0) + d.base_value * 3600.0
        lignes[n] = ligne
    return pd.DataFrame.from_dict(lignes, orient="index")[list(CATEGORIES)]


def groupes_de_rugosite(wn, n_groupes: int = 6) -> dict[str, str]:
    """Regroupe les 905 conduites par **classe de diamètre**, du plus fin au plus gros.

    L-Town compte sept diamètres (63, 75, 100, 150, 160, 200, 225 mm) très inégalement répartis :
    705 conduites en 100 mm, cinq seulement sous 100 mm. Pour six groupes, les deux plus petites
    classes sont fusionnées. Renvoie {conduite: nom de groupe}.
    """
    diam = pd.Series({p: round(wn.get_link(p).diameter * 1000, 1) for p in wn.pipe_name_list})
    classes = sorted(diam.unique())
    if n_groupes < len(classes):                       # fusionne les plus petites classes
        fusion = classes[:len(classes) - n_groupes + 1]
        etiquette = {c: (f"D<={fusion[-1]:.0f}" if c in fusion else f"D{c:.0f}") for c in classes}
    else:
        etiquette = {c: f"D{c:.0f}" for c in classes}
    return {p: etiquette[d] for p, d in diam.items()}


def section_reservoir(wn) -> float:
    """Section de T1 en m², telle qu'elle est écrite dans le modèle."""
    return float(np.pi * (wn.get_node("T1").diameter / 2.0) ** 2)


# ------------------------------------------------------------------------------- simulation
def executer(wn):
    """Un passage du moteur EPANET, dans un fichier temporaire privé nettoyé derrière lui."""
    prefixe = os.path.join(tempfile.gettempdir(), f"ltown_{os.getpid()}_{uuid.uuid4().hex}")
    try:
        return wntr.sim.EpanetSimulator(wn).run_sim(file_prefix=prefixe)
    finally:
        for ext in (".inp", ".rpt", ".bin", ".hyd"):
            try:
                os.remove(prefixe + ext)
            except OSError:
                pass


def preparer(D: np.ndarray | None, noeuds: list[str], heures: float, *,
             niveau0: float | None = None, statut_pompe=None,
             rugosite: dict[str, float] | None = None,
             pompe: np.ndarray | None = None,
             diametre_T1: float | None = None,
             inp: str | Path | None = None):
    """Un modèle prêt à tourner sur `heures`, avec une série de demande par jonction.

    `D` est en **m³/h**, de forme (T, len(noeuds)) avec T ≥ heures·12 + 1 — EPANET lit le profil
    au pas de report et reboucle silencieusement sur un profil trop court.

    `rugosite` multiplie le coefficient de Hazen-Williams des conduites nommées.

    `pompe` est le statut mesuré de `PUMP_1` pas par pas (1 en marche, 0 à l'arrêt). Quand il est
    fourni, les deux consignes de niveau du modèle sont retirées et remplacées par un profil de
    vitesse : une vitesse nulle ferme la pompe. C'est le levier du second carnet.
    """
    wn = charger_modele(heures, PAS_MIN, inp=inp)
    if D is not None:
        besoin = int(heures * 60 // PAS_MIN) + 1
        if D.shape[0] < besoin:
            raise ValueError(f"profil de {D.shape[0]} pas pour {besoin} attendus")
        for k, n in enumerate(noeuds):
            j = wn.get_node(n)
            j.demand_timeseries_list.clear()
            wn.add_pattern(f"D_{n}", list(np.asarray(D[:, k], float)))
            j.add_demand(base=1.0 / 3600.0, pattern_name=f"D_{n}")
    if niveau0 is not None:
        wn.get_node("T1").init_level = float(niveau0)
    if statut_pompe is not None:
        wn.get_link("PUMP_1").initial_status = statut_pompe
    if pompe is not None:
        for nom in list(wn.control_name_list):
            wn.remove_control(nom)
        wn.add_pattern("POMPE_MESUREE", list(np.asarray(pompe, float)))
        pm = wn.get_link("PUMP_1")
        pm.speed_timeseries.base_value = 1.0
        pm.speed_timeseries.pattern_name = "POMPE_MESUREE"
        pm.initial_status = LinkStatus.Open
    if rugosite:
        for p, f in rugosite.items():
            wn.get_link(p).roughness = wn.get_link(p).roughness * float(f)
    if diametre_T1 is not None:
        wn.get_node("T1").diameter = float(diametre_T1)
    return wn


def simuler(D, noeuds, n_pas: int, extraire, *, tranche_jours: float = 7,
            niveau0: float = 3.5, rugosite=None, pompe=None, niveau_mesure=None,
            diametre_T1=None, inp=None, verbeux: bool = True) -> list:
    """Simule `n_pas` pas de 5 min en tranches et renvoie la liste des extraits.

    `extraire(res, wn, debut, fin)` est appelée une fois par tranche et décide de ce qui est
    gardé : c'est **elle** qui borne la mémoire. Une année de pressions aux 782 jonctions pèse
    330 Mo en float32, une année aux 33 capteurs en pèse 14. Ne jamais renvoyer `res`.

    Le report d'état entre tranches — niveau de T1 et statut de la pompe au dernier pas — a été
    contrôlé : deux tranches de 24 h donnent les mêmes pressions que 48 h d'affilée à 1e-3 m près.

    `niveau_mesure` fait repartir chaque tranche du niveau **lu au capteur** plutôt que du niveau
    simulé. C'est indispensable dès que `pompe` est imposé : la pompe ne suivant plus de consigne
    de niveau, le stock du réservoir devient l'intégrale libre de l'écart entre l'entrée mesurée
    et la demande estimée, et il dérive jusqu'à vider ou noyer le réservoir.
    """
    sorties = []
    niveau, statut = float(niveau0), None
    pas_tranche = int(tranche_jours * PAS_JOUR)
    debut = 0
    while debut < n_pas:
        fin = min(debut + pas_tranche, n_pas)
        heures = (fin - debut) * PAS_MIN / 60.0
        Dt = None if D is None else D[debut:fin + 1]
        pt = None if pompe is None else pompe[debut:fin + 1]
        if niveau_mesure is not None:
            niveau = float(niveau_mesure[debut])
        wn = preparer(Dt, noeuds, heures, niveau0=niveau, statut_pompe=statut,
                      rugosite=rugosite, pompe=pt, diametre_T1=diametre_T1, inp=inp)
        res = executer(wn)
        sorties.append(extraire(res, wn, debut, fin))
        niveau = float(res.node["head"]["T1"].to_numpy()[-1] - wn.get_node("T1").elevation)
        statut = LinkStatus(int(res.link["status"]["PUMP_1"].to_numpy()[-1]))
        if verbeux:
            print(f"  tranche {debut:6d}-{fin:6d}   niveau final {niveau:5.3f} m   "
                  f"pompe {statut.name}", flush=True)
        debut = fin
    return sorties


def pressions_simulees(D, noeuds, n_pas: int, **kw) -> np.ndarray:
    """Raccourci : simule et ne garde que les 33 capteurs de pression, en float32 (T, 33)."""
    liste = capteurs()["pressure"]
    morceaux = simuler(
        D, noeuds, n_pas,
        lambda res, wn, a, b: res.node["pressure"][liste].to_numpy()[:b - a].astype(np.float32),
        **kw)
    return np.vstack(morceaux)
