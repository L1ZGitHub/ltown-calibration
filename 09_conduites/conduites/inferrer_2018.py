"""Inference exploratoire du checkpoint B-GPS sur les mesures reelles de 2018."""
import argparse
import json
from pathlib import Path
import time

import numpy as np
import torch

from .donnees import Normalisation
from .entrainer import empreinte, relire_checkpoint
from .hydraulique import decouper
from .metriques import fuites
from .sources import Annees, M
from etapes import e00_donnees as D


def _contexte_jour(line: np.ndarray, historique: np.ndarray | None) \
        -> tuple[np.ndarray, int]:
    if line.ndim != 3:
        raise ValueError("Entrees journalieres attendues en trois dimensions")
    if historique is None:
        return line, 0
    if historique.shape != (11, *line.shape[1:]):
        raise ValueError("Historique inter-jour incompatible")
    return np.concatenate((historique, line), axis=0), len(historique)


def _indices_cibles(jour: int, longueur: int, offset: int, stride: int) -> range:
    debut_local = 11 if jour == 0 or stride == 12 else 0
    return range(offset + debut_local, offset + longueur, stride)


def inferer(checkpoint: Path, sortie: Path, entrees: Path, jours: int = 365,
            lot: int = 1, stride: int = 12) -> dict:
    if sortie.exists():
        raise FileExistsError(sortie)
    if not 1 <= jours <= 365 or lot < 1 or stride not in (1, 12):
        raise ValueError("Jours [1,365], lot positif et stride 1 ou 12 requis")
    modele, c = relire_checkpoint(checkpoint)
    if c["objectif"] != "B" or c["modele"]["representation"] != "line" \
            or c["modele"]["architecture"] != "gps":
        raise ValueError("Checkpoint attendu : objectif B, GPS sur line graph")
    torch.set_num_threads(1)
    modele.eval()
    root = Path(__file__).resolve().parents[1]
    jeu = root / "sorties" / "pilote_18_bruit_corrige"
    normalisation = Normalisation.lire(jeu / "normalisation.json")
    manifeste_entrees = json.loads((entrees / "manifeste.json").read_text())
    if manifeste_entrees["schema"] != 2 or manifeste_entrees["etat"] != "complet" \
            or manifeste_entrees["annee"] != 2018 \
            or manifeste_entrees["normalisation_sha256"] != empreinte(jeu / "normalisation.json") \
            or manifeste_entrees["jours_demandes"] < jours \
            or len(manifeste_entrees["fichiers"]) < jours:
        raise ValueError("Donnees 2018 preparees absentes, incompletes ou incompatibles")
    fichiers_par_jour = {}
    for info in manifeste_entrees["fichiers"]:
        numero = info.get("jour")
        if not isinstance(numero, int) or numero in fichiers_par_jour \
                or info.get("nom") != f"jour_{numero:03d}.npz":
            raise ValueError("Manifeste des entrees journalieres invalide")
        fichiers_par_jour[numero] = info
    if any(jour not in fichiers_par_jour for jour in range(jours)):
        raise ValueError("Manifeste incomplet pour les jours demandes")
    fen0 = M.fenetre(2018, 0, 1)
    wn0, correspondance = decouper(fen0.wn)
    if list(wn0.node_name_list) != normalisation.schema["noeuds"] \
            or list(wn0.link_name_list) != normalisation.schema["liens"]:
        raise ValueError("Topologie 2018 differente de celle du checkpoint")
    if manifeste_entrees["shape_jour"] != [288, len(wn0.link_name_list), 37]:
        raise ValueError("Dimensions des entrees materialisees incompatibles")
    debut = time.perf_counter()
    buffer = None
    scores_horaires = np.full((jours * 24, len(correspondance)), np.nan, dtype=np.float32)
    for jour in range(jours):
        info = fichiers_par_jour[jour]
        fichier = entrees / info["nom"]
        if empreinte(fichier) != info["sha256"]:
            raise ValueError(f"Empreinte invalide : {fichier}")
        with np.load(fichier, allow_pickle=False) as d:
            line = d["line"]
        if line.shape != tuple(manifeste_entrees["shape_jour"]) or not np.isfinite(line).all():
            raise ValueError(f"Entrees invalides : {fichier}")
        sequence, offset = _contexte_jour(line, buffer)
        jours_scores = np.full((len(line), len(correspondance)), np.nan, dtype=np.float32)
        cibles = _indices_cibles(jour, len(line), offset, stride)
        for debut_lot in range(0, len(cibles), lot):
            ts = cibles[debut_lot:debut_lot + lot]
            x = torch.from_numpy(np.stack([sequence[t - 11:t + 1] for t in ts]))
            with torch.inference_mode():
                z = modele(x, "B")["fuite_logits"].cpu().numpy()
            for k, t in enumerate(ts):
                jours_scores[t - offset] = z[k]
        for h in range(24):
            if stride == 1 and jour == 0 and h == 0:
                continue
            plage = jours_scores[h * 12:(h + 1) * 12]
            valides = np.isfinite(plage).all(axis=1)
            if valides.any():
                scores_horaires[jour * 24 + h] = plage[valides].mean(axis=0)
        buffer = line[-11:].copy()
        if (jour + 1) % 7 == 0 or jour + 1 == jours:
            print(f"Inférence 2018 : {jour + 1}/{jours} jours", flush=True)
    # Les labels publies sont charges seulement apres la production de tous les scores.
    fuites, debits_h = D.fuites_publiees(2018)
    conduites = [x.nom for x in correspondance]
    heures = debits_h.index[:jours * 24]
    if len(debits_h) < jours * 24 or not debits_h.index[:jours * 24].equals(heures) \
            or heures[0] != np.datetime64("2018-01-01T00:00:00"):
        raise ValueError("Index temporel des debits de fuite publies incompatible")
    labels = np.zeros_like(scores_horaires, dtype=np.uint8)
    seuil_actif = Annees.G.FUITE_ACTIVE
    for pipe in fuites.index:
        i = conduites.index(pipe)
        labels[:, i] = (debits_h[pipe].reindex(heures).to_numpy() > seuil_actif).astype(np.uint8)
    eligible = np.isfinite(scores_horaires).all(axis=1)
    if not eligible.any():
        raise ValueError("Aucun score horaire complet")
    scores = scores_horaires[eligible]
    y = labels[eligible]
    metriques = fuites_fn(scores, y)
    sans_p461 = [i for i, p in enumerate(conduites) if p != "p461"]
    metriques_sans_p461 = fuites_fn(scores[:, sans_p461], y[:, sans_p461])
    report = {
        "schema": 1, "etat": "complet", "objectif": "B", "architecture": "GPS line",
        "annee": 2018, "jours_inferes": jours, "pas_modele": "5 min",
        "stride_inference_minutes": stride * 5,
        "score_evalue": (
            "logit de la fenetre terminee a la fin de l'heure"
            if stride == 12 else
            "moyenne horaire des logits des observations de 5 min disponibles"
        ),
        "heures_evaluees": int(eligible.sum()),
        "premiere_heure_incomplete_exclue": stride == 1,
        "modele": str(checkpoint.resolve()), "checkpoint_sha256": empreinte(checkpoint),
        "entrees_materialisees": str(entrees.resolve()),
        "manifeste_entrees_sha256": empreinte(entrees / "manifeste.json"),
        "secondes_inference_seule": time.perf_counter() - debut,
        "jeu_normalisation": str((jeu / "normalisation.json").resolve()),
        "test_pilote_lu_pour_entrainement": False,
        "fuites_publiees_utilisees_uniquement_apres_inference": True,
        "fuites_publiees": len(fuites), "conduites_reelles_deja_vues_en_train": ["p461"],
        "metriques_toutes_conduites": metriques,
        "sensibilite_sans_p461": metriques_sans_p461,
        "seuil_logit": 0.0, "seuil_non_optimise": True,
        "euros_ou_politique_decision": False,
        "avertissement": [
            "Analyse retrospective exploratoire, pas un test independant : 2018 a servi "
            "a la calibration des profils et a des reglages historiques.",
            "Les profils et demandes estimes sont issus du pipeline calibre sur 2018.",
            "La conduite p461 apparait dans les labels simules du train.",
            "Les mesures de 2018 ne sont pas bruitees artificiellement ; les capteurs ont "
            "leur bruit reel, qui peut differer du bruit simule.",
            (
                "Score de fin d'heure descriptif ; une fuite breve dans l'heure peut etre "
                "moins bien representee qu'avec 12 scores."
                if stride == 12 else
                "Moyenne de logits horaire descriptive seulement."
            ) + " Aucune politique causale/euros.",
        ],
        "secondes": time.perf_counter() - debut,
    }
    sortie.mkdir(parents=True)
    np.savez_compressed(sortie / "scores_horaires.npz", scores=scores,
                        labels_publies=y, heures=heures[eligible].to_numpy(),
                        conduites=np.asarray(conduites))
    (sortie / "rapport.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def fuites_fn(scores: np.ndarray, labels: np.ndarray) -> dict:
    return fuites(scores, labels, seuil_logit=0.0)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--entrees", type=Path, required=True)
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--jours", type=int, default=365)
    ap.add_argument("--lot", type=int, default=1)
    ap.add_argument("--stride", type=int, choices=(1, 12), default=12,
                    help="Pas entre fenetres : 12 (horaire) ou 1 (toutes les 5 min)")
    a = ap.parse_args()
    r = inferer(a.checkpoint, a.sortie, a.entrees, a.jours, a.lot, a.stride)
    print(json.dumps({"heures": r["heures_evaluees"],
                      "average_precision": r["metriques_toutes_conduites"]["average_precision_micro"],
                      "secondes": r["secondes"]}, indent=2))


if __name__ == "__main__":
    main()
