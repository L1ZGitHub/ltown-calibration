"""Le jeu du pilote en grand : mêmes épisodes de six heures, beaucoup plus de variantes et de jours.

Même recette que `pilote.planifier` (0, 1 ou 3 fuites constantes, 2-20 m³/h, conduites distinctes),
mêmes contrats (une variante dans une seule partition, périodes disjointes entre partitions). Ce qui
change : le nombre (32 variantes d'apprentissage, 4 de validation, 4 de test, 20 épisodes chacune) et
les jours (apprentissage dans [0, 240), validation dans [250, 300), test dans [310, 362)).

Le jeu « v2 » (`--jeu v2`) : plus de réseaux différents et des fuites qui commencent et s'arrêtent.
200 variantes d'apprentissage, 20 de validation, 20 de test, 4 épisodes chacune ; chaque fuite
commence à un pas tiré dans [0, 60) et une sur deux s'arrête à un pas tiré dans [début + 12, 72)
(au moins une heure active). Mêmes jours, même recette de tailles, même contrat de partitions.
"""
import argparse
import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import numpy as np

from .episode import fabriquer
from .pilote import verifier_plan
from .sources import M
from generation import scenarios as S

PARTITIONS = (("train", 32, 0, 240), ("validation", 4, 250, 300), ("test", 4, 310, 362))
EPISODES_PAR_VARIANTE = 20
CATEGORIES = (0, 1, 3)
JEUX = {
    "v1": {"partitions": PARTITIONS, "par_variante": EPISODES_PAR_VARIANTE, "graine": 9000,
           "prefixe": "v", "calendrier": False},
    "v2": {"partitions": (("train", 200, 0, 240), ("validation", 20, 250, 300), ("test", 20, 310, 362)),
           "par_variante": 4, "graine": 20000, "prefixe": "n", "calendrier": True},
}


def planifier(graine: int, candidates: list[str], jeu: str = "v1") -> list[dict]:
    r = JEUX[jeu]
    plan, v, numero = [], 0, 0
    for partition, n_var, j0, j1 in r["partitions"]:
        for _ in range(n_var):
            rng_v = np.random.default_rng([graine, v])
            jours = rng_v.choice(np.arange(j0, j1), r["par_variante"], replace=False)
            for k, jour in enumerate(sorted(int(j) for j in jours)):
                # v1 : la catégorie tourne dans la variante ; v2 (4 épisodes) : sur tout le jeu
                categorie = (numero if r["calendrier"] else k) % len(CATEGORIES)
                numero += 1
                rng = np.random.default_rng([graine, v, k])
                n = CATEGORIES[categorie]
                fuites = {str(p): round(float(rng.uniform(2, 20)), 2)
                          for p in rng.choice(candidates, n, replace=False)}
                e = {
                    "identifiant": f"{r['prefixe']}{v:02d}_e{k:02d}_{('sain', 'une', 'plusieurs')[categorie]}",
                    "variante": f"{r['prefixe']}{v}", "partition": partition,
                    "jour": jour, "debut_pas": jour * 288, "fin_pas_exclue": jour * 288 + 72,
                    "graine_variante": graine + v,
                    "graine_episode": int(rng.integers(1_000_000, 2_000_000_000)),
                    "fuites_visees_m3h": fuites,
                }
                if r["calendrier"]:
                    calendrier = {}
                    for p in fuites:
                        debut = int(rng.integers(0, 60))
                        fin = int(rng.integers(debut + 12, 72)) if rng.random() < .5 else 72
                        calendrier[p] = [debut, fin]
                    e["calendrier_fuites"] = calendrier
                plan.append(e)
            v += 1
    return plan


def _un(args) -> tuple[str, str | None]:
    sortie, e = args
    ici = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)                            # EPANET écrit ses fichiers ici, un dossier par tâche
        try:
            fabriquer(sortie / e["identifiant"], graine=e["graine_variante"], jour=e["jour"],
                      fuites=e["fuites_visees_m3h"], graine_episode=e["graine_episode"],
                      calendrier=e.get("calendrier_fuites"))
            return e["identifiant"], None
        except Exception as x:                   # un épisode non physique est noté, pas fatal
            return e["identifiant"], f"{type(x).__name__}: {x}"
        finally:
            os.chdir(ici)


def generer(sortie: Path, part: str, processus: int, minutes: float, graine: int | None = None,
            par_partition: int | None = None, jeu: str = "v1") -> dict:
    i, w = (int(x) for x in part.split("/"))
    graine = JEUX[jeu]["graine"] if graine is None else graine
    fen = M.fenetre(2018, 0, 1)
    plan = planifier(graine, S.conduites_possibles(fen), jeu)
    verifier_plan(plan)
    mien = plan[i::w]
    if par_partition:                            # essai : les premiers épisodes de chaque partition
        mien = [e for p, *_ in PARTITIONS for e in [x for x in mien if x["partition"] == p][:par_partition]]
    sortie.mkdir(parents=True, exist_ok=True)
    debut, faits, echecs = time.time(), [], {}
    with mp.get_context("fork").Pool(processus) as pool:
        for ident, err in pool.imap_unordered(_un, [(sortie, e) for e in mien]):
            if err:
                echecs[ident] = err
            else:
                faits.append(ident)
            if (len(faits) + len(echecs)) % 10 == 0:
                print(f"{len(faits) + len(echecs)}/{len(mien)} ({time.time() - debut:.0f} s)", flush=True)
            if time.time() - debut > minutes * 60:
                print("budget atteint : on arrête", flush=True)
                pool.terminate()
                break
    r = {"graine": graine, "jeu": jeu, "part": part, "plan": mien, "termines": sorted(faits), "echecs": echecs,
         "secondes": time.time() - debut}
    (sortie / f"manifeste_{i}.json").write_text(json.dumps(r, indent=1))
    return r


def fusionner(dossiers: list[Path], sortie: Path) -> dict:
    """Un manifeste au format du pilote (`etat` complet) avec les seuls épisodes finis ; les
    dossiers d'épisodes restent où ils sont (`chemins`)."""
    episodes, chemins = [], {}
    for d in dossiers:
        for f in sorted(Path(d).rglob("manifeste_*.json")):
            r = json.loads(f.read_text())
            if "plan" not in r:                  # un manifeste d'une autre sortie (2018)
                continue
            fini = set(r["termines"])
            for e in r["plan"]:
                if e["identifiant"] in fini and (f.parent / e["identifiant"] / "rapport.json").exists():
                    episodes.append(e)
                    chemins[e["identifiant"]] = str(f.parent / e["identifiant"])
    verifier_plan(episodes)
    m = {"schema": 1, "etat": "complet", "episodes": episodes, "chemins": chemins}
    sortie.mkdir(parents=True, exist_ok=True)
    (sortie / "manifeste.json").write_text(json.dumps(m, indent=1))
    return m


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--part", default="0/1")
    ap.add_argument("--processus", type=int, default=4)
    ap.add_argument("--minutes", type=float, default=600, help="budget de temps ; au-delà on s'arrête")
    ap.add_argument("--par-partition", type=int, default=None,
                    help="essai : seulement les N premiers épisodes de chaque partition")
    ap.add_argument("--jeu", choices=sorted(JEUX), default="v1",
                    help="v1 : le jeu livré ; v2 : plus de variantes, fuites qui commencent et s'arrêtent")
    a = ap.parse_args()
    r = generer(a.sortie, a.part, a.processus, a.minutes, par_partition=a.par_partition, jeu=a.jeu)
    print(json.dumps({"termines": len(r["termines"]), "echecs": len(r["echecs"]),
                      "secondes": r["secondes"]}, indent=1))


if __name__ == "__main__":
    main()
