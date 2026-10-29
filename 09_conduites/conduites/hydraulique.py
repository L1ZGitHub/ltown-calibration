"""Decoupage uniforme et verification EPANET sur une heure."""
from copy import deepcopy
from dataclasses import asdict, dataclass

import numpy as np
import wntr
from wntr.network import WaterNetworkModel

from .sources import M


@dataclass(frozen=True)
class Conduite:
    nom: str
    debut: str
    milieu: str
    fin: str
    segment_1: str
    segment_2: str


def liens_mesures(correspondance: list[Conduite]) -> dict[str, str]:
    par_nom = {c.nom: c for c in correspondance}
    resultat = {"PUMP_1": "PUMP_1"}
    for nom, reservoir in (("p227", "R1"), ("p235", "R2")):
        c = par_nom[nom]
        if c.debut == reservoir:
            resultat[nom] = c.segment_1
        elif c.fin == reservoir:
            resultat[nom] = c.segment_2
        else:
            raise ValueError(f"{nom} ne touche pas la frontiere attendue {reservoir}")
    return {nom: resultat[nom] for nom in M.DEBITMETRES}


def decouper(wn: WaterNetworkModel) -> tuple[WaterNetworkModel, list[Conduite]]:
    copie = deepcopy(wn)
    correspondance = []
    for nom in list(wn.pipe_name_list):
        p = wn.get_link(nom)
        milieu, second = f"m09_{nom}", f"s09_{nom}"
        if milieu in copie.node_name_list or second in copie.link_name_list:
            raise ValueError(f"Collision d'identifiant pour {nom}")
        if p.check_valve:
            raise ValueError(f"Clapet non pris en charge par ce premier test : {nom}")
        wntr.morph.split_pipe(copie, nom, second, milieu, return_copy=False)
        # split_pipe duplique minor_loss ; conserver la perte totale d'origine.
        copie.get_link(nom).minor_loss = p.minor_loss / 2
        copie.get_link(second).minor_loss = p.minor_loss / 2
        correspondance.append(Conduite(nom, p.start_node_name, milieu, p.end_node_name,
                                      nom, second))
    return copie, correspondance


def verifier() -> tuple[WaterNetworkModel, list[Conduite], dict]:
    fen = M.fenetre(2018, 0, 1)
    wn = M.R.preparer(fen.D[:13], fen.noeuds, 1, niveau0=float(fen.niveau[0]),
                      coefficients=fen.coefficients, pompe=fen.pompe[:13])
    wn.options.hydraulic.accuracy = 1e-6
    wn.options.hydraulic.trials = 200
    wn.options.hydraulic.unbalanced = "STOP"
    coupe, correspondance = decouper(wn)
    original = M.R.executer(wn)
    sain = M.R.executer(coupe)
    p = original.node["pressure"][fen.noeuds].to_numpy()
    dp = sain.node["pressure"][fen.noeuds].to_numpy() - p
    q = original.link["flowrate"][wn.pipe_name_list].to_numpy() * 3600
    noms_1 = [c.segment_1 for c in correspondance]
    noms_2 = [c.segment_2 for c in correspondance]
    q1 = sain.link["flowrate"][noms_1].to_numpy() * 3600
    q2 = sain.link["flowrate"][noms_2].to_numpy() * 3600
    # Choisir un milieu a pression positive, loin du cas emetteur sans pression.
    candidat = max(correspondance,
                   key=lambda c: float(sain.node["pressure"][c.milieu].min()))
    fuite = deepcopy(coupe)
    pression = float(sain.node["pressure"][candidat.milieu].median())
    fuite.get_node(candidat.milieu).emitter_coefficient = M.R.coefficient_emetteur(10, pression)
    res = M.R.executer(fuite)
    entre = res.link["flowrate"][candidat.segment_1].to_numpy() * 3600
    sort = res.link["flowrate"][candidat.segment_2].to_numpy() * 3600
    perdu = res.node["demand"][candidat.milieu].to_numpy() * 3600
    rapport = {
        "horizon_h": 1, "pas_min": 5, "conduites": len(correspondance),
        "noeuds_originaux": len(wn.node_name_list), "noeuds_decoupes": len(coupe.node_name_list),
        "pression_saine_max_abs_m": float(np.abs(dp).max()),
        "debit_sain_max_abs_m3h": float(max(np.abs(q1 - q).max(), np.abs(q2 - q).max())),
        "continuite_saine_max_abs_m3h": float(np.abs(q1 - q2).max()),
        "fuite": asdict(candidat),
        "fuite_min_m3h": float(perdu.min()), "fuite_max_m3h": float(perdu.max()),
        "bilan_fuite_max_abs_m3h": float(np.abs(entre - sort - perdu).max()),
        "tolerances": {"pression_m": 0.01, "debit_m3h": 0.1, "bilan_m3h": 0.01},
        "solveur": {"accuracy": 1e-6, "trials": 200, "unbalanced": "STOP"},
    }
    if not np.isfinite(dp).all() or not np.isfinite(q1).all() or not np.isfinite(q2).all():
        raise AssertionError("Resultats hydrauliques sains non finis")
    if not np.isfinite(perdu).all() or perdu.min() <= 0:
        raise AssertionError("La fuite testee ne coule pas a tous les pas")
    for cle, limite in (("pression_saine_max_abs_m", 0.01),
                        ("debit_sain_max_abs_m3h", 0.1),
                        ("continuite_saine_max_abs_m3h", 0.01),
                        ("bilan_fuite_max_abs_m3h", 0.01)):
        if not np.isfinite(rapport[cle]) or rapport[cle] > limite:
            raise AssertionError(f"{cle} = {rapport[cle]}, limite = {limite}")
    return coupe, correspondance, rapport
