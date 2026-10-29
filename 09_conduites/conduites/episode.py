"""Un episode de raccordement de six heures, sans jeu ni apprentissage."""
import argparse
import json
import tempfile
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from .hydraulique import decouper, liens_mesures
from .observations import observer, provenance_bruit
from .sources import Annees, M
from generation import jeu as J
from etapes import e10_dual as Dual


def prolonger(x: np.ndarray) -> np.ndarray:
    return np.concatenate((x, x[-1:]))


def preparer(fen, demande, coefficients, pompe, niveau, inp=None, heures: float = 6):
    wn = M.R.preparer(demande, fen.noeuds, heures, coefficients=coefficients,
                      pompe=pompe, niveau0=float(niveau), inp=inp)
    wn.options.time.duration = round(heures * 3600)   # 13/12 h tronqué donnerait 3899 s, un pas de moins
    wn.options.hydraulic.accuracy = 1e-6
    wn.options.hydraulic.trials = 200
    wn.options.hydraulic.unbalanced = "STOP"
    return decouper(wn)


class Morceaux:
    """Les résultats de plusieurs passages EPANET mis bout à bout, pas 0 à 72, comme un seul."""

    def __init__(self, resultats: list, longueurs: list[int]):
        def coller(famille, cle):
            return pd.concat([getattr(r, famille)[cle].iloc[:n] for r, n in zip(resultats, longueurs)],
                             ignore_index=True)
        self.node = {k: coller("node", k) for k in ("pressure", "demand")}
        self.link = {"flowrate": coller("link", "flowrate")}


def simuler_calendrier(fen, var, correspondance, coefficients: dict[str, float],
                       calendrier: dict[str, tuple[int, int]]):
    """La vérité avec des fuites qui commencent et s'arrêtent : EPANET garde un émetteur constant
    pendant un passage, donc un passage par morceau entre deux changements, le niveau de T1 de la
    fin d'un morceau servant de départ au suivant (comme les tranches de `calibration.reseau.simuler`).
    Sans changement, un seul passage de six heures, exactement comme avant."""
    par_nom = {c.nom: c for c in correspondance}
    coupures = sorted({0, 72, *(t for d, f in calendrier.values() for t in (d, f) if 0 < t < 72)})
    resultats, longueurs, niveau = [], [], float(fen.niveau[0])
    for a, b in zip(coupures[:-1], coupures[1:]):
        wn, corr = preparer(fen, var.D[a:b + 1], var.coefficients, fen.pompe[a:b + 1], niveau, var.inp,
                            heures=(b - a) / 12)
        if corr != correspondance:
            raise AssertionError("Correspondances differentes entre morceaux")
        for nom, c in coefficients.items():
            debut, fin = calendrier[nom]
            if debut <= a < fin:
                wn.get_node(par_nom[nom].milieu).emitter_coefficient = c
        r = M.R.executer(wn)
        if len(r.node["pressure"]) < b - a + 1:
            raise RuntimeError(f"EPANET arrete avant la fin du morceau {a}-{b}")
        niveau = float(r.node["pressure"]["T1"].iloc[b - a])
        resultats.append(r)
        longueurs.append(b - a + (b == 72))
    return Morceaux(resultats, longueurs)


def projeter(wn, correspondance, capteurs, compteurs, observations, demande,
             pression_reference, debit_reference, qv) -> tuple[dict, dict]:
    n = len(observations["niveau"])
    noms = list(wn.node_name_list)
    liens = list(wn.link_name_list)
    ni = {nom: i for i, nom in enumerate(noms)}
    li = {nom: i for i, nom in enumerate(liens)}
    champs = [
        ("pression_m", "m"), ("pression_presente", "bool"), ("pression_valide", "bool"),
        ("residu_pression_m", "m"), ("dual_m3h", "m3/h"),
        ("variation_dual_m3h", "m3/h"), ("variation_dual_valide", "bool"),
        ("amr_m3h", "m3/h"), ("amr_present", "bool"), ("amr_valide", "bool"),
        ("demande_estimee_m3h", "m3/h"), ("pression_reference_m", "m"),
        ("niveau_T1_m", "m"), ("niveau_present", "bool"), ("niveau_valide", "bool"),
    ]
    noeuds = np.zeros((n, len(noms), len(champs)), dtype=np.float32)
    cs, am = [ni[c] for c in capteurs], [ni[c] for c in compteurs]
    noeuds[:, cs, 0] = observations["capteurs"]
    noeuds[:, cs, 1] = 1
    noeuds[:, cs, 2] = observations["capteurs_valide"]
    noeuds[:, cs, 3] = observations["capteurs"] - pression_reference[:, cs]
    noeuds[:, cs, 4] = qv
    noeuds[1:, cs, 5] = np.diff(qv, axis=0)
    noeuds[1:, cs, 6] = observations["capteurs_valide"][1:] & observations["capteurs_valide"][:-1]
    noeuds[:, am, 7] = observations["amr"]
    noeuds[:, am, 8] = 1
    noeuds[:, am, 9] = observations["amr_valide"]
    noeuds[:, :, 10] = demande
    noeuds[:, :, 11] = pression_reference
    noeuds[:, ni["T1"], 12] = observations["niveau"]
    noeuds[:, ni["T1"], 13] = 1
    noeuds[:, ni["T1"], 14] = observations["niveau_valide"]
    champs_liens = [("debit_mesure_m3h", "m3/h"), ("debit_present", "bool"),
                    ("debit_valide", "bool"), ("debit_reference_m3h", "m3/h"),
                    ("etat_pompe", "bool"), ("pompe_presente", "bool"),
                    ("pompe_valide", "bool")]
    aretes = np.zeros((n, len(liens), len(champs_liens)), dtype=np.float32)
    aretes[:, :, 3] = debit_reference
    for k, segment in enumerate(liens_mesures(correspondance).values()):
        i = li[segment]
        aretes[:, i, 0] = observations["debits"][:, k]
        aretes[:, i, 1] = 1
        aretes[:, i, 2] = observations["debits_valide"][:, k]
    aretes[:, li["PUMP_1"], 4] = observations["pompe"]
    aretes[:, li["PUMP_1"], 5] = 1
    aretes[:, li["PUMP_1"], 6] = observations["pompe_valide"]
    ext = np.array([(ni[wn.get_link(l).start_node_name], ni[wn.get_link(l).end_node_name])
                    for l in liens])
    line = np.concatenate((noeuds[:, ext[:, 0]], noeuds[:, ext[:, 1]], aretes), axis=-1)
    return {
        "hydraulique_noeuds": noeuds, "hydraulique_liens": aretes, "line_sommets": line,
    }, {
        "noeuds": noms, "liens": liens, "extremites_liens": ext.tolist(),
        "champs_noeuds": champs, "champs_liens": champs_liens,
        "champs_line": [(f"debut_{nom}", u) for nom, u in champs]
        + [(f"fin_{nom}", u) for nom, u in champs] + champs_liens,
        "normalisation": None, "masque_dual": "pression_presente et pression_valide",
        "masque_residu": "pression_presente et pression_valide",
    }


def fabriquer(sortie: Path, graine: int = 9, conduite: str = "p31", *,
              jour: int = 0, fuites: dict[str, float] | None = None,
              graine_episode: int | None = None,
              calendrier: dict[str, tuple[int, int]] | None = None) -> dict:
    """`calendrier` : pour chaque fuite, (premier pas actif, premier pas inactif) ; par défaut toute
    la durée (0, 72), le contrat du pilote."""
    if sortie.exists():
        raise FileExistsError(f"Sortie deja presente : {sortie}")
    if not 0 <= jour < 364:
        raise ValueError("Jour attendu dans [0, 363]")
    fuites = {conduite: 10.} if fuites is None else dict(fuites)
    if any(not np.isfinite(q) or q <= 0 for q in fuites.values()):
        raise ValueError("Debits vises des fuites strictement positifs et finis requis")
    graine_episode = graine + 1 if graine_episode is None else graine_episode
    calendrier = {nom: (0, 72) for nom in fuites} | {k: tuple(v) for k, v in (calendrier or {}).items()}
    if set(calendrier) != set(fuites) or any(not 0 <= d < f <= 72 for d, f in calendrier.values()):
        raise ValueError("Calendrier attendu : une plage 0 <= debut < fin <= 72 par fuite")
    fen = M.fenetre(2018, jour, 1)
    r = J.VARIANTES_SEMAINE1
    with tempfile.TemporaryDirectory(prefix="ltown09_") as tmp:
        var = Annees.variante(fen, graine, graine_episode, Path(tmp))
        vrai, correspondance = preparer(fen, var.D[:73], var.coefficients,
                                        fen.pompe[:73], fen.niveau[0], var.inp)
        par_nom = {c.nom: c for c in correspondance}
        inconnues = set(fuites) - par_nom.keys()
        if inconnues:
            raise ValueError(f"Conduites inconnues : {sorted(inconnues)}")
        sain = M.R.executer(vrai)
        coefficients = {}
        for nom, debit in fuites.items():
            pression = float(sain.node["pressure"][par_nom[nom].milieu].median())
            if pression <= 0:
                raise ValueError(f"Pression insuffisante pour la fuite : {nom}")
            coefficients[nom] = M.R.coefficient_emetteur(debit, pression)
        res = simuler_calendrier(fen, var, correspondance, coefficients, calendrier) if fuites else sain
        if res.node["pressure"][fen.noeuds].to_numpy()[:72].min() < 0:
            raise ValueError("Episode non physique : pression negative a une jonction")
    mesures = liens_mesures(correspondance)
    reel = {
        "capteurs": res.node["pressure"][fen.capteurs].to_numpy()[:72],
        "debits": res.link["flowrate"][list(mesures.values())].to_numpy()[:72] * 3600,
        "niveau": res.node["pressure"]["T1"].to_numpy()[:72],
        "amr": var.D[:72, fen.mesure], "pompe": fen.pompe[:72],
    }
    obs = observer(reel, graine_episode)
    demande = fen.D[:72].copy()
    demande[:, fen.mesure] = obs["amr"]
    reference, corr_ref = preparer(fen, prolonger(demande), fen.coefficients,
                                   prolonger(obs["pompe"]), obs["niveau"][0])
    if correspondance != corr_ref:
        raise AssertionError("Correspondances variante/reference differentes")
    ref = M.R.executer(reference)
    # Le dual doit utiliser la demande estimee portant les AMR de cet episode.
    fen_dual = replace(fen, D=prolonger(demande), pompe=prolonger(obs["pompe"]),
                       niveau=prolonger(obs["niveau"]))
    qv = Dual.simuler(fen_dual, obs["capteurs"], coupe=True,
                      debit_pompe=prolonger(obs["debits"][:, 0]),
                      niveau=prolonger(obs["niveau"]), tranches=[0])
    noms = reference.node_name_list
    liens = reference.link_name_list
    estimation = np.zeros((72, len(noms)), dtype=np.float32)
    indices = {nom: i for i, nom in enumerate(noms)}
    estimation[:, [indices[n] for n in fen.noeuds]] = demande
    entrees, schema = projeter(
        reference, correspondance, fen.capteurs, np.array(fen.noeuds)[fen.mesure],
        obs, estimation, ref.node["pressure"][noms].to_numpy()[:72],
        ref.link["flowrate"][liens].to_numpy()[:72] * 3600, qv)
    q1 = res.link["flowrate"][[c.segment_1 for c in correspondance]].to_numpy()[:72] * 3600
    q2 = res.link["flowrate"][[c.segment_2 for c in correspondance]].to_numpy()[:72] * 3600
    debit_fuites = res.node["demand"][[c.milieu for c in correspondance]].to_numpy()[:72] * 3600
    actives = np.zeros((72, len(correspondance)), np.float32)
    for k, c in enumerate(correspondance):
        if c.nom in calendrier:
            actives[slice(*calendrier[c.nom]), k] = 1
    cibles = {"q_m3h": np.stack((q1, q2), -1), "fuite_m3h": debit_fuites, "fuite_active": actives}
    if (debit_fuites[actives == 0] > 1e-3).any() or (debit_fuites[actives == 1] <= 0).any():
        raise AssertionError("Debits de fuite et calendrier incoherents")
    residu = float(np.abs(q1 - q2 - debit_fuites).max())
    if residu > 0.01:
        raise AssertionError(f"Bilan aux milieux incorrect : {residu} m3/h")
    for famille in (entrees, cibles, obs, {"dual": qv}):
        for cle, valeurs in famille.items():
            if not np.isfinite(valeurs).all():
                raise AssertionError(f"Valeurs non finies : {cle}")
    rapport = {
        "schema": 1, "graine": graine, "graine_episode": graine_episode,
        "graine_bruit": graine_episode,
        "jour": jour, "fuites_visees_m3h": fuites, "calendrier_fuites": calendrier, "pas": 72, "pas_min": 5,
        "fenetres_12_pas": 61, "bilan_max_abs_m3h": residu, "bruit": provenance_bruit(),
        "conduites": [asdict(c) for c in correspondance], "debitmetres": mesures,
        "schema_entrees": schema, "formes": {k: list(v.shape) for k, v in entrees.items()},
        "provenance": {
            "conditions": f"2018, jour {jour}, premieres six heures ; aucune annotation de fuite",
            "variante": {"graine": var.graine, "reglages": r},
            "dual": "v2 e10.simuler, coupe=True, demandes estimees et mesures de cet episode",
            "causalite": "fenetres et variations causales ; profils de demande historiques OFFLINE",
        },
        "limites": [
            "Episode de six heures ; partition definie uniquement par le manifeste du jeu.",
            "Profils historiques ajustes sur 2018 conserves par decision ; "
            "pas d'independance annuelle des consommations.",
            "Reference : pompe observee et niveau initial observe ; T1 evolue ensuite dans le modele.",
            "Dual : debit pompe et niveau observes imposes ; solveur historique non modifie.",
            "Aucun seuil d'activite choisi : debit de fuite continu conserve.",
            "Fusion brute non normalisee ; attributs statiques physiques a ajouter avant apprentissage.",
        ],
    }
    sortie.mkdir(parents=True)
    np.savez_compressed(sortie / "observations.npz", **obs)
    np.savez_compressed(sortie / "entrees.npz", **entrees)
    np.savez_compressed(sortie / "cibles.npz", **cibles)
    np.savez_compressed(sortie / "reference_dual.npz", demande_estimee=demande, dual=qv,
                        pression=ref.node["pressure"][noms].to_numpy()[:72],
                        debit_m3h=ref.link["flowrate"][liens].to_numpy()[:72] * 3600)
    (sortie / "rapport.json").write_text(json.dumps(rapport, indent=2) + "\n")
    return rapport


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--graine", type=int, default=9)
    ap.add_argument("--conduite", default="p31")
    a = ap.parse_args()
    rapport = fabriquer(a.sortie, a.graine, a.conduite)
    print(json.dumps({k: rapport[k] for k in ("pas", "formes", "bilan_max_abs_m3h")}, indent=2))


if __name__ == "__main__":
    main()
