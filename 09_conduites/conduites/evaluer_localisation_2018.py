"""Reevaluation spatiale des scores sauvegardes, sans simulation ni inference."""
import argparse
import json
from pathlib import Path

import numpy as np

from . import sources  # noqa: F401
from .metriques import localisation
from etapes import e00_donnees as D, e05_signatures as S


def evaluer(scores: Path, sortie: Path) -> dict:
    if sortie.exists():
        raise FileExistsError(sortie)
    with np.load(scores, allow_pickle=False) as data:
        s, y = data["scores"], data["labels_publies"]
        conduites = data["conduites"].tolist()
        heures = data["heures"]
    if len(conduites) != s.shape[1] or len(set(conduites)) != len(conduites) \
            or heures.shape != (len(s),):
        raise ValueError("Index des scores incompatible")
    distances = S.distances(D.reseau()).loc[conduites, conduites].to_numpy()
    resultat = localisation(s, y, distances)
    resultat["par_conduite"] = {
        conduites[int(i)]: r for i, r in resultat["par_conduite"].items()}
    rapport = {
        "schema": 1, "scores_source": str(scores.resolve()), "heures_evaluees": len(s),
        "distance": "bareme BattLeDIM : chemin sur conduites seules + demi-longueurs ; "
        "0 pour la meme conduite, inf si deconnectees",
        "metriques": resultat,
        "avertissement": "2018 retrospectif, pas un test independant ; aucun seuil ajuste, "
        "ni politique d'intervention ni comparaison au pipeline physique",
    }
    sortie.parent.mkdir(parents=True, exist_ok=True)
    with sortie.open("x") as fichier:
        json.dump(rapport, fichier, indent=2, allow_nan=False)
    return rapport


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scores", type=Path, required=True)
    parser.add_argument("--sortie", type=Path, required=True)
    args = parser.parse_args()
    rapport = evaluer(args.scores, args.sortie)
    print(json.dumps(rapport, indent=2, allow_nan=False))
