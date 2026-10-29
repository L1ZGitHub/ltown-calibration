"""Pilote borne : 18 episodes, six variantes, sans entrainement."""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from .donnees import Episode, Fenetres, Normalisation, verifier_partitions
from .episode import fabriquer
from .sources import M
from generation import scenarios as S


def planifier(graine: int, candidates: list[str]) -> list[dict]:
    if len(set(candidates)) != len(candidates) or len(candidates) < 4:
        raise ValueError("Au moins quatre identifiants de conduite uniques requis")
    plan = []
    for variante, jour in enumerate((0, 28, 56, 84, 168, 280)):
        partition = "train" if variante < 4 else ("validation" if variante == 4 else "test")
        for categorie, n in enumerate((0, 1, 3)):
            rng = np.random.default_rng([graine, variante, categorie])
            fuites = {str(p): round(float(rng.uniform(2, 20)), 2)
                      for p in rng.choice(candidates, n, replace=False)}
            plan.append({
                "identifiant": f"v{variante}_{('sain', 'une', 'plusieurs')[categorie]}",
                "variante": f"v{variante}", "partition": partition,
                "jour": jour + categorie, "debut_pas": (jour + categorie) * 288,
                "fin_pas_exclue": (jour + categorie) * 288 + 72,
                "graine_variante": graine + variante,
                "graine_episode": int(rng.integers(1_000_000, 2_000_000_000)),
                "fuites_visees_m3h": fuites,
            })
    return plan


def verifier_plan(plan: list[dict]) -> None:
    variantes, identifiants = {}, set()
    for e in plan:
        if e["identifiant"] in identifiants:
            raise ValueError("Identifiant d'episode duplique")
        identifiants.add(e["identifiant"])
        v, partition = e["variante"], e["partition"]
        if v in variantes and variantes[v] != partition:
            raise ValueError("Variante partagee entre partitions")
        variantes[v] = partition
        if e["fin_pas_exclue"] - e["debut_pas"] != 72:
            raise ValueError("Episode attendu de 72 pas")
    for i, a in enumerate(plan):
        for b in plan[i + 1:]:
            if a["partition"] != b["partition"] and max(a["debut_pas"], b["debut_pas"]) \
                    < min(a["fin_pas_exclue"], b["fin_pas_exclue"]):
                raise ValueError("Periodes chevauchantes entre partitions")


def generer(sortie: Path, graine: int = 9000) -> dict:
    if sortie.exists():
        raise FileExistsError(f"Dossier deja present : {sortie}")
    fen = M.fenetre(2018, 0, 1)
    plan = planifier(graine, S.conduites_possibles(fen))
    verifier_plan(plan)
    manifeste = {
        "schema": 1, "etat": "en_cours", "graine": graine, "episodes": plan, "termines": [],
        "but": "pilote de raccordement et futur essai court, pas certification predictive",
        "hypotheses": [
            "Trois scenarios par variante : sain, une fuite, trois fuites.",
            "Fuites constantes en coefficient d'emetteur pendant six heures, "
            "debits vises uniformes 2-20 m3/h, conduites distinctes dans chaque scenario.",
            "Pas d'ouverture dans les fenetres : ce pilote ne teste pas la detection d'un debut.",
            "Profils historiques calibres sur 2018 ; periodes disjointes mais provenance commune.",
            "Aucune optimisation, aucun entrainement, aucun seuil binaire choisi.",
        ],
    }
    sortie.mkdir(parents=True)
    chemin = sortie / "manifeste.json"
    chemin.write_text(json.dumps(manifeste, indent=2) + "\n")
    debut = time.perf_counter()
    for e in plan:
        print(f"Generation {len(manifeste['termines']) + 1}/18 : {e['identifiant']}", flush=True)
        r = fabriquer(sortie / e["identifiant"], graine=e["graine_variante"], jour=e["jour"],
                      fuites=e["fuites_visees_m3h"], graine_episode=e["graine_episode"])
        if r["bilan_max_abs_m3h"] > 0.01:
            raise AssertionError("Bilan du pilote hors tolerance")
        manifeste["termines"].append(e["identifiant"])
        chemin.write_text(json.dumps(manifeste, indent=2) + "\n")
    episodes = [Episode.lire(sortie / e["identifiant"], e["variante"], e["partition"])
                for e in plan]
    verifier_partitions(episodes)
    norm = Normalisation.ajuster([e for e in episodes if e.partition == "train"])
    norm.sauver(sortie / "normalisation.json")
    resume = {}
    for partition in ("train", "validation", "test"):
        selection = [e for e in episodes if e.partition == partition]
        ds = Fenetres(selection, norm, "line")
        resume[partition] = {"episodes": len(selection), "fenetres": len(ds)}
    manifeste.update(etat="complet", secondes=time.perf_counter() - debut, resume=resume)
    chemin.write_text(json.dumps(manifeste, indent=2) + "\n")
    print(json.dumps(resume, indent=2), flush=True)
    return manifeste


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--graine", type=int, default=9000)
    a = ap.parse_args()
    generer(a.sortie, a.graine)


if __name__ == "__main__":
    main()
