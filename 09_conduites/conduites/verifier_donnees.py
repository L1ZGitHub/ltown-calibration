"""Verifier les deux modeles sur un episode existant, sans entrainer."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from .donnees import Episode, Fenetres, Normalisation
from .graphes import attributs_physiques, construire
from .hydraulique import decouper
from .modeles import Prototype
from .sources import M


def echelles_statiques(x: np.ndarray, continues: int) -> tuple[torch.Tensor, dict]:
    centre = np.zeros(x.shape[1], dtype=np.float32)
    ecart = np.ones(x.shape[1], dtype=np.float32)
    centre[:continues] = x[:, :continues].mean(0)
    ecart[:continues] = x[:, :continues].std(0)
    ecart[ecart < 1e-6] = 1
    return torch.from_numpy((x - centre) / ecart), {
        "centre": centre.tolist(), "ecart": ecart.tolist(),
        "provenance": "topologie et attributs connus du modele calibre uniquement",
    }


def verifier(episode: Path, sortie: Path) -> dict:
    if sortie.exists():
        raise FileExistsError(sortie)
    torch.manual_seed(9)
    torch.set_num_threads(1)
    e = Episode.lire(episode, "variante_raccordement", "train")
    norm = Normalisation.ajuster([e])
    # Le producteur utilise les proprietes calibrees, pas les proprietes cachees de l'episode.
    fen = M.fenetre(2018, 0, 1)
    for nom, c in fen.coefficients.items():
        fen.wn.get_link(nom).roughness = c
    wn, corr = decouper(fen.wn)
    graphes = construire(wn, corr)
    phys = attributs_physiques(wn, corr)
    sn, en = echelles_statiques(phys["noeuds"], 3)
    sl, el = echelles_statiques(phys["liens"], 5)
    # La charge reservoir, separee des indicateurs de type, garde son echelle physique.
    sn[:, 7] /= 100.
    en["ecart"][7] = 100.
    ext = np.asarray(e.schema["extremites_liens"])
    line_statique = torch.cat((sn[ext[:, 0]], sn[ext[:, 1]], sl), -1)
    rapport = {
        "schema": 1, "episode": e.identifiant, "configuration": {
            "attention": "dense", "largeur": 8, "profondeur": 1, "graine": 9,
            "threads": 1, "pas": 12, "mise_a_jour_poids": False,
        }, "statiques": {"noeuds": en, "liens": el,
                         "champs_noeuds": phys["champs_noeuds"],
                         "champs_liens": phys["champs_liens"]},
        "modeles": {},
        "limites": [
            "Un seul episode etiquete train pour verifier la plomberie ; aucune validation predictive.",
            "Aucun poids appris ni sauvegarde ; les valeurs predites ne sont pas interpretees.",
            "Pas de seuil binaire de fuite ni perte supervisee choisis.",
        ],
    }
    for representation, g in graphes.items():
        if g.noms != e.schema["liens" if representation == "line" else "noeuds"]:
            raise ValueError("Ordre des identifiants incompatible")
        ds = Fenetres([e], norm, representation)
        lot = ds[0]
        m = Prototype(
            g, representation, "dense", entrees=lot["x"].shape[-1], largeur=8, profondeur=1,
            entrees_liens=7 if representation == "hydraulique" else 0,
            statiques=line_statique if representation == "line" else sn,
            statiques_liens=sl if representation == "hydraulique" else None)
        liens = lot["liens"][None] if representation == "hydraulique" else None
        sorties = m(lot["x"][None], "C", liens=liens)
        perte = sum(v.square().mean() for v in sorties.values())
        perte.backward()
        if not torch.isfinite(perte) or any(
                not torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None):
            raise AssertionError("Sorties ou gradients non finis")
        rapport["modeles"][representation] = {
            "fenetres": len(ds), "pas_cible_premiere": lot["pas_cible"],
            "formes": {k: list(v.shape) for k, v in sorties.items()}, "gradients_finis": True,
        }
    sortie.mkdir(parents=True)
    norm.sauver(sortie / "normalisation.json")
    (sortie / "rapport.json").write_text(json.dumps(rapport, indent=2) + "\n")
    return rapport


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--episode", type=Path, required=True)
    ap.add_argument("--sortie", type=Path, required=True)
    a = ap.parse_args()
    r = verifier(a.episode, a.sortie)
    print(json.dumps(r["modeles"], indent=2))


if __name__ == "__main__":
    main()
