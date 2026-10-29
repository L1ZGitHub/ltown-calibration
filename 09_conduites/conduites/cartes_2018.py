"""Comparaison exploratoire A/B/C sur six jours fixes, sans nouvelle simulation."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .cartographie import ecrire_cartes
from .donnees import Normalisation
from .entrainer import empreinte, relire_checkpoint
from .sources import Annees
from .zones import bilan_zones, noyau, selection_longueur
from etapes import e00_donnees as D, e05_signatures as S


DATES = ("2018-01-01", "2018-01-15", "2018-04-15", "2018-07-15",
         "2018-10-15", "2018-12-15")
CHAMPS = ("A_bilan", "B_logits", "C_bilan", "C_logits")
BUDGETS = (.01, .05, .10)


def inferer_dates(entrees: Path, checkpoints: Path, normalisation: Path,
                  dates: tuple[str, ...]) -> tuple[dict, np.ndarray, list[str], dict]:
    if not dates or len(set(dates)) != len(dates) \
            or any(not np.datetime64("2018-01-01") <= np.datetime64(d, "D") <
                   np.datetime64("2019-01-01") for d in dates):
        raise ValueError("Dates uniques dans 2018 requises")
    norm = Normalisation.lire(normalisation)
    norm_json = json.loads(normalisation.read_text())
    manifeste = json.loads((entrees / "manifeste.json").read_text())
    if manifeste["schema"] != 2 or manifeste["etat"] != "complet" \
            or manifeste["annee"] != 2018 \
            or manifeste["normalisation_sha256"] != empreinte(normalisation):
        raise ValueError("Entrees preparees incompatibles")
    fichiers = {}
    for f in manifeste["fichiers"]:
        j = f["jour"]
        if not isinstance(j, int) or j in fichiers or f["nom"] != f"jour_{j:03d}.npz":
            raise ValueError("Index journalier du manifeste invalide")
        fichiers[j] = f
    wn = D.reseau()
    conduites = list(wn.pipe_name_list)
    modeles, provenance = {}, {}
    torch.set_num_threads(1)
    for objectif in "ABC":
        fichier = checkpoints / f"{objectif}_meilleur.pt"
        modele, c = relire_checkpoint(fichier)
        if c["objectif"] != objectif or c["modele"]["architecture"] != "gps" \
                or c["modele"]["representation"] != "line" \
                or c["normalisation"] != norm_json \
                or c["graphe"]["noms"] != norm.schema["liens"] \
                or [p["nom"] for p in c["conduites"]] != conduites:
            raise ValueError(f"Checkpoint {objectif} incompatible")
        modeles[objectif] = modele
        provenance[objectif] = {"fichier": str(fichier.resolve()), "sha256": empreinte(fichier)}
    sorties: dict[str, list[np.ndarray]] = {}
    heures = []
    for date in dates:
        jour = int((np.datetime64(date) - np.datetime64("2018-01-01")) / np.timedelta64(1, "D"))
        info = fichiers[jour]
        fichier = entrees / info["nom"]
        if empreinte(fichier) != info["sha256"]:
            raise ValueError(f"Empreinte invalide : {fichier}")
        with np.load(fichier, allow_pickle=False) as data:
            line = data["line"]
        if line.shape != (288, len(norm.schema["liens"]), 37) \
                or not np.isfinite(line).all():
            raise ValueError(f"Entrees invalides : {fichier}")
        for h in range(24):
            x = torch.from_numpy(line[h * 12:(h + 1) * 12][None].copy())
            with torch.inference_mode():
                for objectif, modele in modeles.items():
                    prediction = modele(x, objectif)
                    for cle, valeur in prediction.items():
                        v = valeur[0].cpu().numpy()
                        if not np.isfinite(v).all():
                            raise FloatingPointError(f"Sortie {objectif}/{cle} non finie")
                        if cle == "q_m3h_normalise":
                            v = v * norm.echelle_q
                            cle = "q_m3h"
                        sorties.setdefault(f"{objectif}_{cle}", []).append(v)
            heures.append(np.datetime64(date, "h") + np.timedelta64(h, "h"))
        print(f"A/B/C : {date} termine", flush=True)
    valeurs = {k: np.stack(v) for k, v in sorties.items()}
    heures = np.array(heures)
    return valeurs, heures, conduites, provenance


def comparer(entrees: Path, checkpoints: Path, normalisation: Path, sortie: Path) -> dict:
    if sortie.exists():
        raise FileExistsError(sortie)
    debut = time.perf_counter()
    valeurs, heures, conduites, provenance = inferer_dates(
        entrees, checkpoints, normalisation, DATES)
    wn = D.reseau()
    longueurs = np.array([wn.get_link(p).length for p in conduites])
    distances = S.distances(wn).loc[conduites, conduites].to_numpy()
    w = noyau(distances, longueurs)
    # Les verites publiees n'interviennent qu'apres les predictions.
    _, q = D.fuites_publiees(2018)
    if not q.index.is_unique or not np.isin(heures, q.index.to_numpy()).all() \
            or not set(q.columns).issubset(conduites):
        raise ValueError("Index ou conduites des verites publiees incompatibles")
    debits = q.reindex(index=heures, columns=conduites, fill_value=0.).to_numpy()
    if not np.isfinite(debits).all():
        raise ValueError("Verites publiees non alignees")
    labels = debits > Annees.G.FUITE_ACTIVE
    if labels[:24].any():
        raise ValueError("Le controle du 1er janvier n'est pas sain")
    champs = {}
    for objectif in ("A", "C"):
        v = valeurs[f"{objectif}_q_m3h"]
        valeurs[f"{objectif}_bilan_signe"] = v[..., 0] - v[..., 1]
        champs[f"{objectif}_bilan"] = np.maximum(valeurs[f"{objectif}_bilan_signe"], 0.)
    for objectif in ("B", "C"):
        champs[f"{objectif}_logits"] = valeurs[f"{objectif}_fuite_logits"]
    champs = {k: champs[k] for k in CHAMPS}
    lisses = {k: v @ w.T for k, v in champs.items()}
    rapport = {
        "schema": 1, "etat": "en_cours", "dates_fixees_avant_inference": list(DATES),
        "selection_dates": "1er janvier puis 15 janvier/avril/juillet/octobre/decembre ; "
        "couverture calendaire, pas selection par performance ni representativite statistique",
        "heures": len(heures), "checkpoints": provenance,
        "normalisation_sha256": empreinte(normalisation),
        "manifeste_entrees_sha256": empreinte(entrees / "manifeste.json"),
        "longueur_reseau_m": float(longueurs.sum()),
        "lissage": "moyenne locale exp(-distance/300), tronquee a 300 m ; "
        "poids par longueur, normalises par voisinage ; distance du bareme, pas euclidienne",
        "champs": {
            "A_bilan": "max(Q1-Q2,0) m3/h, indice derive, pas un debit de fuite valide",
            "B_logits": "logits non calibres",
            "C_bilan": "max(Q1-Q2,0) m3/h, tete Q de C",
            "C_logits": "logits non calibres, tete classification de C ; aucune fusion"},
        "selection": "conduites entieres par score decroissant ; sauter une conduite si elle "
        "depasse le budget restant ; egalites par permutation fixe graine 9 ; "
        "longueurs reelles rapportees (budget maximal commun, pas egalite exacte)",
        "controle_sain": "carte moyenne du 1er janvier, figee puis appliquee aux autres jours ; "
        "diagnostic de biais spatial, ni soustraction du fond ni ajustement de seuil",
        "bilans_signes": {
            o: {"fraction_negative": float((valeurs[f"{o}_bilan_signe"] < 0).mean()),
                "quantiles_m3h": np.quantile(valeurs[f"{o}_bilan_signe"],
                                            [0., .01, .5, .99, 1.]).tolist()}
            for o in ("A", "C")},
        "comparaisons": {},
        "avertissements": [
            "2018 retrospectif utilise historiquement pour calibration ; p461 vue en train.",
            "Six jours seulement ; pas un test independant ni une evaluation annuelle.",
            "Budget impose meme pendant les heures saines : les zones ne sont pas des alertes.",
            "Une selection peut couvrir plusieurs fuites ; aucune politique d'intervention.",
            "Lissage et budgets fixes sans optimisation ; pas d'explication physique des mesures.",
            "Aucune comparaison de superiorite au pipeline physique ; test simule non ouvert."],
    }
    masques = {}
    rng = np.random.default_rng(9)
    hasard_scores = rng.random((128, len(conduites)))
    voisins = (distances <= 300).astype(np.int32)
    for nom, brut in champs.items():
        comparaisons = {}
        for mode, carte in (("brut", brut), ("lisse", lisses[nom])):
            stats = {}
            for fraction in BUDGETS:
                budget = fraction * longueurs.sum()
                masque = selection_longueur(carte, longueurs, budget)
                cle = f"{fraction:.0%}"
                if fraction == .05:
                    masques[f"{nom}_{mode}"] = masque
                resultat = bilan_zones(masque, labels, distances, longueurs, heures)
                resultat["par_conduite"] = {
                    conduites[int(p)]: v for p, v in resultat["par_conduite"].items()}
                resultat["par_jour"] = {}
                for i, date in enumerate(DATES):
                    sl = slice(i * 24, (i + 1) * 24)
                    r = bilan_zones(masque[sl], labels[sl], distances, longueurs, heures[sl])
                    resultat["par_jour"][date] = {
                        k: v for k, v in r.items() if k != "par_conduite"}
                controle = np.broadcast_to(carte[:24].mean(0), carte.shape)
                cm = selection_longueur(controle, longueurs, budget)
                r = bilan_zones(cm, labels, distances, longueurs, heures)
                resultat["controle_carte_saine_rappel_300m"] = r["rappel_300m"]
                resultat["controle_carte_saine_longueur_m"] = r["longueur_selectionnee_m"]["moyenne"]
                stats[cle] = resultat
            comparaisons[mode] = stats
        rapport["comparaisons"][nom] = comparaisons
    rapport["hasard"] = {}
    for fraction in BUDGETS:
        masque = selection_longueur(hasard_scores, longueurs, fraction * longueurs.sum())
        couverture = ((masque.astype(np.int32) @ voisins) > 0).mean(0)
        actifs = labels.any(1)
        rappel = ((labels * couverture).sum(1)[actifs] / labels.sum(1)[actifs]).mean()
        rapport["hasard"][f"{fraction:.0%}"] = {
            "rappel_300m": float(rappel), "tirages_cartes": 128,
            "longueur_moyenne_m": float((masque @ longueurs).mean()),
            "fraction_reseau_a_300m": float(couverture @ longueurs / longueurs.sum()),
            "convention": "ordre uniforme des conduites puis meme selection sous budget ; graine 9"}
    sortie.mkdir(parents=True)
    np.savez_compressed(sortie / "cartes.npz", heures=heures, conduites=np.array(conduites),
                        labels=labels, debits_publies_m3h=debits, longueurs_m=longueurs,
                        **valeurs, **{f"brut_{k}": v for k, v in champs.items()},
                        **{f"lisse_{k}": v for k, v in lisses.items()},
                        **{f"selection_5pct_{k}": v for k, v in masques.items()})
    ecrire_cartes(sortie / "cartes.html", wn, conduites, heures, champs, lisses,
                  masques, labels, rapport)
    rapport.update(etat="complet", secondes=time.perf_counter() - debut)
    (sortie / "rapport.json").write_text(json.dumps(rapport, indent=2, allow_nan=False) + "\n")
    return rapport


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--entrees", required=True, type=Path)
    parser.add_argument("--checkpoints", required=True, type=Path)
    parser.add_argument("--normalisation", required=True, type=Path)
    parser.add_argument("--sortie", required=True, type=Path)
    args = parser.parse_args()
    r = comparer(args.entrees, args.checkpoints, args.normalisation, args.sortie)
    for nom, modes in r["comparaisons"].items():
        print(nom, {mode: v["5%"]["rappel_300m"] for mode, v in modes.items()}, flush=True)
