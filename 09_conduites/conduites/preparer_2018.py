"""Preparation physique separee des entrees 2018, stockees par jour sans labels."""
import argparse
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path
import time

import numpy as np

from .donnees import Normalisation
from .entrees_2018 import fenetre_jour
from .hydraulique import decouper
from .sources import M
from etapes import e00_donnees as D
from etapes import e10_dual as Dual

_ETAT_JOURS = {}


def empreinte(chemin: Path) -> str:
    h = hashlib.sha256()
    with chemin.open("rb") as fichier:
        for bloc in iter(lambda: fichier.read(1024 * 1024), b""):
            h.update(bloc)
    return h.hexdigest()


def _ecrire_manifeste(chemin: Path, manifeste: dict) -> None:
    temporaire = chemin.with_name(f".{chemin.name}.tmp")
    temporaire.write_text(json.dumps(manifeste, indent=2) + "\n")
    temporaire.replace(chemin)


def _initialiser_jours(mesures, dual, corr, noms_noeuds, normalisation,
                       nombre_liens, sortie) -> None:
    global _ETAT_JOURS
    _ETAT_JOURS = {
        "mesures": mesures, "dual": dual, "corr": corr, "noms_noeuds": noms_noeuds,
        "normalisation": normalisation, "nombre_liens": nombre_liens, "sortie": sortie,
    }


def _preparer_jour(jour: int) -> dict:
    etat = _ETAT_JOURS
    t0 = time.perf_counter()
    line = fenetre_jour(
        jour, etat["mesures"], etat["dual"], etat["corr"], etat["noms_noeuds"],
        etat["normalisation"], etat["nombre_liens"])
    if line.shape != (288, etat["nombre_liens"], 37) or not np.isfinite(line).all():
        raise AssertionError(f"Entrees 2018 invalides jour {jour}")
    nom = f"jour_{jour:03d}.npz"
    fichier = etat["sortie"] / nom
    np.savez_compressed(fichier, line=line.astype(np.float32, copy=False))
    return {
        "jour": jour, "nom": nom, "sha256": empreinte(fichier),
        "secondes_preparation": time.perf_counter() - t0,
        "taille_octets": fichier.stat().st_size,
    }


def preparer(sortie: Path, jours: int = 365, processus: int = 4) -> dict:
    if not 1 <= jours <= 365 or processus < 1:
        raise ValueError("Jours dans [1,365] et nombre de processus positif requis")
    processus = min(processus, os.cpu_count() or 1)
    debut_total = time.perf_counter()
    racine = Path(__file__).resolve().parents[1]
    jeu = racine / "sorties" / "pilote_18_bruit_corrige"
    normalisation_path = jeu / "normalisation.json"
    normalisation = Normalisation.lire(normalisation_path)
    mesures = D.mesures(2018)
    dual = Dual.debits_virtuels(2018, coupe=True, processus=processus)
    fen = M.fenetre(2018, 0, 1)
    wn, corr = decouper(fen.wn)
    if list(wn.node_name_list) != normalisation.schema["noeuds"] \
            or list(wn.link_name_list) != normalisation.schema["liens"]:
        raise ValueError("Topologie de 2018 differente du schema d'entree")
    if dual.shape[0] != 365 * 288:
        raise ValueError("Debit dual de 2018 incomplet")
    empreinte_normalisation = empreinte(normalisation_path)
    shape_jour = [288, len(wn.link_name_list), 37]
    chemin_manifeste = sortie / "manifeste.json"
    deja_faits = {}
    if sortie.exists():
        if not chemin_manifeste.is_file():
            raise FileExistsError(f"Sortie existante sans manifeste : {sortie}")
        manifeste = json.loads(chemin_manifeste.read_text())
        if manifeste.get("schema") != 2 or manifeste.get("etat") != "en_cours" \
                or manifeste.get("annee") != 2018 \
                or manifeste.get("jours_demandes") != jours \
                or manifeste.get("normalisation_sha256") != empreinte_normalisation \
                or manifeste.get("shape_jour") != shape_jour:
            raise ValueError("Sortie existante non reprenable ou incompatible")
        fichiers_existants = manifeste.get("fichiers", [])
        for info in fichiers_existants:
            jour = info.get("jour")
            if not isinstance(jour, int) or not 0 <= jour < jours or jour in deja_faits:
                raise ValueError(f"Jour du manifeste invalide : {jour}")
            nom = f"jour_{jour:03d}.npz"
            fichier = sortie / nom
            if info.get("nom") != nom or not fichier.is_file() \
                    or empreinte(fichier) != info.get("sha256"):
                raise ValueError(f"Fichier journalier incomplet/corrompu : {nom}")
            deja_faits[jour] = info
        manifeste["fichiers"] = [deja_faits[j] for j in sorted(deja_faits)]
    else:
        sortie.mkdir(parents=True)
        manifeste = {
            "schema": 2, "etat": "en_cours", "annee": 2018, "jours_demandes": jours,
            "pas_par_jour": 288, "pas_min": 5,
            "entree": "hydraulique_line_normalisee_float32",
            "shape_jour": shape_jour,
            "labels_charges": False,
            "normalisation": str(normalisation_path.resolve()),
            "normalisation_sha256": empreinte_normalisation,
            "modele_reference": "modele calibre, demandes estimees depuis AMR, "
                                "pompe et niveau observes, tranches 6h",
            "dual": "cache historique coupe, debit pompe/niveau observes",
            "fichiers": [], "secondes": None,
            "avertissements": [
                "Mesures reelles 2018 ; resultats retrospectifs, pas test independant.",
                "Aucune fuite ou label publie n'est lu pendant la preparation.",
                "Les mesures sont conservees au pas de 5 minutes pour construire "
                "des fenetres causales de 12 pas.",
            ],
        }
        _ecrire_manifeste(chemin_manifeste, manifeste)
    a_preparer = [j for j in range(jours) if j not in deja_faits]
    etat_jours = (mesures, dual, corr, list(wn.node_name_list), normalisation,
                  len(wn.link_name_list), sortie)
    if processus == 1:
        _initialiser_jours(*etat_jours)
        resultats = map(_preparer_jour, a_preparer)
        for info in resultats:
            deja_faits[info["jour"]] = info
            manifeste["fichiers"] = [deja_faits[j] for j in sorted(deja_faits)]
            _ecrire_manifeste(chemin_manifeste, manifeste)
            print(f"Preparation 2018 : {len(deja_faits)}/{jours} jours", flush=True)
    elif a_preparer:
        contexte = mp.get_context("fork")
        with contexte.Pool(processus, initializer=_initialiser_jours, initargs=etat_jours) as pool:
            for info in pool.imap_unordered(_preparer_jour, a_preparer, chunksize=1):
                deja_faits[info["jour"]] = info
                manifeste["fichiers"] = [deja_faits[j] for j in sorted(deja_faits)]
                _ecrire_manifeste(chemin_manifeste, manifeste)
                print(f"Preparation 2018 : {len(deja_faits)}/{jours} jours", flush=True)
    manifeste["secondes_calcul_cumule"] = sum(
        x["secondes_preparation"] for x in manifeste["fichiers"])
    manifeste["secondes"] = time.perf_counter() - debut_total
    manifeste["taille_octets"] = sum(x["taille_octets"] for x in manifeste["fichiers"])
    manifeste["etat"] = "complet"
    _ecrire_manifeste(chemin_manifeste, manifeste)
    return manifeste


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--jours", type=int, default=365)
    ap.add_argument("--processus", type=int, default=4)
    a = ap.parse_args()
    r = preparer(a.sortie, a.jours, a.processus)
    print(json.dumps({"etat": r["etat"], "jours": len(r["fichiers"]),
                      "secondes": r["secondes"], "taille_MiB": r["taille_octets"] / 2**20},
                     indent=2))


if __name__ == "__main__":
    main()
