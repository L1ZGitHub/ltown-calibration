"""Reponse A/B/C et dual autour de six debuts de fuite et de six controles temporels."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import pandas as pd

from .cartes_2018 import inferer_dates
from .dynamique import couverture_empreinte, fond_horaire
from .entrainer import empreinte
from .sources import Annees
from .zones import noyau
from etapes import e00_donnees as D, e05_signatures as S


CAS = (("p232", 14), ("p31", 14), ("p158", 14),
       ("p369", 14), ("p654", 14), ("p538", 21))
FRACTIONS = (.10, .20, .30)
HORIZON = 48


def heures_cas(ancre: str) -> tuple[np.ndarray, np.ndarray]:
    t = np.datetime64(ancre, "h")
    avant = t + np.arange(-HORIZON, 0).astype("timedelta64[h]")
    apres = t + np.arange(1, HORIZON + 1).astype("timedelta64[h]")
    return np.concatenate((avant, apres)), np.arange(2 * HORIZON) < HORIZON


def planifier(fuites: pd.DataFrame) -> list[dict]:
    plan = []
    for pipe, recul in CAS:
        debut = pd.Timestamp(fuites.at[pipe, "debut"])
        ancre = debut.floor("h")
        for controle in (False, True):
            t = ancre - pd.Timedelta(days=recul) if controle else ancre
            h, avant = heures_cas(str(t))
            debut_fenetre = pd.Timestamp(h[0])
            fin_fenetre = pd.Timestamp(h[-1]) + pd.Timedelta(hours=1)
            departs = fuites.index[(fuites.debut >= debut_fenetre)
                                   & (fuites.debut < fin_fenetre)].tolist()
            if departs != ([] if controle else [pipe]):
                raise ValueError(f"Debut concurrent dans {pipe}, controle={controle} : {departs}")
            fins = fuites.index[(fuites.fin >= debut_fenetre) & (fuites.fin < fin_fenetre)].tolist()
            if fins:
                raise ValueError(f"Fin concurrente dans {pipe}, controle={controle} : {fins}")
            if h[0] < np.datetime64("2018-01-01") or h[-1] >= np.datetime64("2019-01-01"):
                raise ValueError("Fenetre hors de 2018")
            plan.append({
                "conduite": pipe, "controle": controle, "ancre": str(t),
                "debut_publie": str(debut), "genre": str(fuites.at[pipe, "genre"]),
                "montee_j": float(fuites.at[pipe, "montee_j"]),
                "recul_controle_j": recul if controle else 0,
                "heures": h.astype(str).tolist(), "heures_avant": int(avant.sum())})
    return plan


def _ecrire(chemin: Path, valeur: dict) -> None:
    chemin.write_text(json.dumps(valeur, indent=2, allow_nan=False) + "\n")


def analyser(predictions: Path, plan: list[dict], dual_fichier: Path,
             signatures_fichier: Path) -> tuple[dict, dict]:
    wn = D.reseau()
    conduites = list(wn.pipe_name_list)
    longueurs = np.array([wn.get_link(p).length for p in conduites])
    distances = S.distances(wn).loc[conduites, conduites].to_numpy()
    w = noyau(distances, longueurs)
    with np.load(predictions, allow_pickle=False) as data:
        valeurs = {k: data[k] for k in data.files}
    heures = valeurs["heures"].astype("datetime64[h]")
    if valeurs["conduites"].tolist() != conduites or len(np.unique(heures)) != len(heures):
        raise ValueError("Index des predictions incompatible")
    index = pd.DatetimeIndex(heures)
    # Acces cache uniquement, sans appel aux producteurs pouvant lancer une simulation.
    with np.load(dual_fichier, allow_pickle=True) as data:
        dual = pd.DataFrame(data["q"], index=pd.DatetimeIndex(data["horo"]),
                            columns=data["capteurs"].astype(str)).resample("1h").mean()
    dico = pd.read_csv(signatures_fichier, index_col=0).loc[conduites]
    if not np.isin(heures, dual.index.to_numpy()).all() \
            or set(dual.columns) != set(dico.columns) \
            or not np.isfinite(dico.to_numpy()).all():
        raise ValueError("Cache dual ou signatures incompatibles")
    qv = dual.loc[index, dico.columns].to_numpy()
    if not np.isfinite(qv).all():
        raise ValueError("Debits virtuels horaires non finis")
    champs = {}
    for o in ("A", "C"):
        q = valeurs[f"{o}_q_m3h"]
        champs[f"{o}_bilan"] = np.maximum(q[..., 0] - q[..., 1], 0.)
    for o in ("B", "C"):
        champs[f"{o}_logits"] = valeurs[f"{o}_fuite_logits"]
    if any(v.shape != (len(heures), len(conduites)) or not np.isfinite(v).all()
           for v in champs.values()):
        raise ValueError("Champs appris incompatibles")
    _, fuites_h = D.fuites_publiees(2018)
    if not np.isin(heures, fuites_h.index.to_numpy()).all():
        raise ValueError("Index des fuites publiees incomplet")
    debits = fuites_h.reindex(index=index, columns=conduites, fill_value=0.).to_numpy()
    if not np.isfinite(debits).all():
        raise ValueError("Debits publies non finis")
    resultats, tableaux = [], {"heures": heures, "conduites": np.array(conduites)}
    for numero, cas in enumerate(plan):
        h, avant = heures_cas(cas["ancre"])
        positions = index.get_indexer(pd.DatetimeIndex(h))
        if (positions < 0).any():
            raise ValueError("Predictions manquantes dans une fenetre")
        cible = conduites.index(cas["conduite"])
        apres = ~avant
        local_qv = qv[positions]
        fond_dual = fond_horaire(local_qv, h, avant)
        cartes_duales = {}
        for mode, observations in (("absolu", local_qv), ("variation", local_qv - fond_dual)):
            cartes_duales[mode] = np.stack([
                S.carte(pd.Series(o, index=dico.columns), dico)["explique"].to_numpy()
                for o in observations])
        diagnostics = {}
        for nom in (*champs, "dual_signatures"):
            if nom == "dual_signatures":
                brut = cartes_duales["absolu"]
                variations = cartes_duales["variation"]
            else:
                brut = champs[nom][positions]
                variations = brut - fond_horaire(brut, h, avant)
            absolu, variation = brut @ w.T, variations @ w.T
            fond = fond_horaire(absolu, h, avant)
            cartes = {"absolu": absolu, "variation": variation, "fond_fige": fond}
            local = absolu[:, cible]
            ecart = local - fond[:, cible]
            reference_std = float(ecart[avant].std())
            diag = {
                "valeur_locale_avant": float(local[avant].mean()),
                "valeur_locale_apres": float(local[apres].mean()),
                "hausse_locale_apres": float(ecart[apres].mean()),
                "ecart_type_pre_residuel": reference_std,
                "hausse_en_ecarts_types_pre": float(ecart[apres].mean() / reference_std)
                if reference_std > 1e-12 else None,
                "couverture": {},
            }
            for mode, carte in cartes.items():
                couvert, frontiere = couverture_empreinte(carte, distances, longueurs, FRACTIONS)
                empreinte = couvert @ longueurs / longueurs.sum()
                if not np.allclose(empreinte, np.array(FRACTIONS)[:, None], atol=1e-6, rtol=0.):
                    raise AssertionError("Empreinte moyenne non conforme au budget")
                stats = {}
                for j, fraction in enumerate(FRACTIONS):
                    stats[f"{fraction:.0%}"] = {
                        "avant": float(couvert[j, avant, cible].mean()),
                        "apres": float(couvert[j, apres, cible].mean()),
                        "gain_apres_moins_avant": float(couvert[j, apres, cible].mean()
                                                      - couvert[j, avant, cible].mean()),
                        "empreinte_moyenne": float(empreinte[j].mean()),
                    }
                diag["couverture"][mode] = stats
                prefixe = f"cas{numero:02d}_{nom}_{mode}"
                tableaux[prefixe + "_champ"] = carte.astype(np.float32)
                tableaux[prefixe + "_couverture_cible"] = couvert[:, :, cible]
                tableaux[prefixe + "_frontiere"] = frontiere
            diagnostics[nom] = diag
        debit = debits[positions, cible]
        resultats.append({
            **{k: v for k, v in cas.items() if k != "heures"},
            "debit_cible_avant_m3h": float(debit[avant].mean()),
            "debit_cible_apres_m3h": float(debit[apres].mean()),
            "heures_cible_active_apres": int((debit[apres] > Annees.G.FUITE_ACTIVE).sum()),
            "diagnostics": diagnostics})
        tableaux[f"cas{numero:02d}_heures"] = h
        tableaux[f"cas{numero:02d}_debit_cible_m3h"] = debit
        print(f"Diagnostic {cas['conduite']}, controle={cas['controle']} termine", flush=True)
    synthese = {}
    for nom in (*champs, "dual_signatures"):
        synthese[nom] = {}
        for mode in ("absolu", "variation", "fond_fige"):
            synthese[nom][mode] = {}
            for fraction in FRACTIONS:
                cle = f"{fraction:.0%}"
                groupes = {}
                for controle, label in ((False, "fuites"), (True, "controles")):
                    r = [c["diagnostics"][nom]["couverture"][mode][cle]
                         for c in resultats if c["controle"] == controle]
                    groupes[label] = {k: float(np.mean([v[k] for v in r]))
                                      for k in ("avant", "apres", "gain_apres_moins_avant")}
                groupes["difference_des_gains"] = (
                    groupes["fuites"]["gain_apres_moins_avant"]
                    - groupes["controles"]["gain_apres_moins_avant"])
                synthese[nom][mode][cle] = groupes
    return {"cas": resultats, "synthese": synthese}, tableaux


def executer(entrees: Path, checkpoints: Path, normalisation: Path, sortie: Path,
             reanalyser: bool = False) -> dict:
    debut = time.perf_counter()
    dual_fichier = D.SORTIES / "dual_2018_coupe.npz"
    signatures = D.SORTIES / "signatures_duales_2018_jour1_10m3h_coupe.csv"
    for fichier in (dual_fichier, signatures):
        if not fichier.is_file():
            raise FileNotFoundError(f"Cache physique requis, aucune simulation lancee : {fichier}")
    if reanalyser:
        rapport = json.loads((sortie / "rapport.json").read_text())
        plan = rapport["plan_fige"]
        for nom, fichier in (("dual", dual_fichier), ("signatures", signatures)):
            if rapport["physique"][nom]["sha256"] != empreinte(fichier):
                raise ValueError(f"Cache physique modifie : {fichier}")
        if rapport["predictions_sha256"] != empreinte(sortie / "predictions.npz"):
            raise ValueError("Predictions modifiees")
    else:
        if sortie.exists():
            raise FileExistsError(sortie)
        fuites, _ = D.fuites_publiees(2018)
        plan = planifier(fuites)
        dates = tuple(sorted({str(np.datetime64(h, "D")) for c in plan for h in c["heures"]}))
        rapport = {
            "schema": 1, "etat": "en_cours", "plan_fige": plan, "dates": dates,
            "protocole": {
                "selection": "six conduites fixees avant inference : localisations auparavant "
                "favorables et defavorables ; trois casses et trois progressives ; "
                "analyse retrospective guidee, pas validation independante",
                "temps": "48 heures completes avant et 48 apres ; heure contenant le debut "
                "publie exclue ; debut = premier depassement du seuil publie, pas ouverture connue",
                "controles": "meme heure/meme jour de semaine 14 j avant (21 j pour p538) ; "
                "aucun autre debut ni fin publie dans chaque fenetre, controle non necessairement sain",
                "fond": "moyenne des deux heures de meme heure-du-jour dans les 48 h anterieures, "
                "figee pour toute la fenetre ; aucun instant apres utilise",
                "cartes": "A/C max(Q1-Q2,0), B/C logits ; variations signees non tronquees ; "
                "dual : carte de signatures sur qv absolu ou sur qv moins fond horaire ; "
                "tous lisses par le noyau local 300 m normalise par longueur",
                "empreinte": "prefixes de classement ; fraction de longueur du reseau a <=300m ; "
                "melange aleatoire des deux prefixes encadrant 10/20/30% pour empreinte "
                "EXACTE EN ESPERANCE ; rappel fractionnaire attendu, pas une zone deterministe",
                "egalites": "permutation fixe graine 9 ; une carte plate n'indique pas une fuite",
                "evaluation": "couverture de la nouvelle conduite seule (cible fictive identique "
                "pour le controle) ; memes budgets avant et apres ; moyenne egale par evenement",
                "physique": "comparaison au composant localisation par signatures, amplitude "
                "bornee selon e05_signatures.carte ; pas la chaine de decisions physique complete",
            },
            "physique": {k: {"fichier": str(p.resolve()), "sha256": empreinte(p)}
                         for k, p in (("dual", dual_fichier), ("signatures", signatures))},
            "normalisation_sha256": empreinte(normalisation),
            "manifeste_entrees_sha256": empreinte(entrees / "manifeste.json"),
            "limites": [
                "2018 deja utilise pour calibration ; petit echantillon choisi retrospectivement.",
                "Les debuts publies servent a choisir les periodes, jamais aux entrees du modele.",
                "Fuites progressives : 48 h peuvent etre insuffisantes ; debits reels rapportes.",
                "Deux repetitions par heure-du-jour seulement pour estimer le fond ; pas de "
                "seuil d'alerte, de significativite ni de preuve causale.",
                "Pas d'entrainement, de fusion A/B/C ni d'ouverture du test simule.",
                "Couverture pre-debut et controles sont des diagnostics spatiaux, pas des faux "
                "positifs d'une politique d'intervention."],
        }
        sortie.mkdir(parents=True)
        _ecrire(sortie / "rapport.json", rapport)
        valeurs, heures, conduites, provenance = inferer_dates(
            entrees, checkpoints, normalisation, dates)
        np.savez_compressed(sortie / "predictions.npz", heures=heures,
                            conduites=np.array(conduites), **valeurs)
        rapport.update(checkpoints=provenance,
                       predictions_sha256=empreinte(sortie / "predictions.npz"),
                       heures_inferees=len(heures), secondes_inference=time.perf_counter() - debut)
        _ecrire(sortie / "rapport.json", rapport)
    resultats, tableaux = analyser(sortie / "predictions.npz", plan, dual_fichier, signatures)
    np.savez_compressed(sortie / "diagnostics.npz", **tableaux)
    rapport.update(**resultats, etat="complet", secondes_derniere_execution=time.perf_counter() - debut)
    _ecrire(sortie / "rapport.json", rapport)
    return rapport


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    for nom in ("entrees", "checkpoints", "normalisation", "sortie"):
        parser.add_argument("--" + nom, type=Path, required=True)
    parser.add_argument("--reanalyser", action="store_true")
    args = parser.parse_args()
    r = executer(args.entrees, args.checkpoints, args.normalisation, args.sortie, args.reanalyser)
    for nom, modes in r["synthese"].items():
        print(nom, {mode: valeurs["20%"] for mode, valeurs in modes.items()}, flush=True)
