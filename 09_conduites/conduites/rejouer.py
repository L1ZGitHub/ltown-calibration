"""Rejouer des modèles appris sur tous les épisodes de test, à chaque pas : ce que lisent les carnets.

Pour chaque modèle et chaque pas de chaque épisode de test (pas 11 à 71) : la note « fuite » de
chaque conduite et, pour l'objectif C, les débits prédits des deux moitiés. Pour chaque jeu, une
fois : la vérité (fuites actives, débit de fuite, vrais débits) et les débits du modèle calibré
sans fuite, le point de comparaison des débits prédits.

Usage : python -m conduites.rejouer --jeux D… --sortie S --modele nom=jeu=chemin.pt …
(`chemin` peut contenir `**`, le premier fichier trouvé).
"""
import argparse
import glob
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from .apprendre_gpu import LOT, colonnes_retirees, fenetres_flux
from .donnees import Normalisation
from .entrainer import relire_checkpoint
from .grand_jeu import fusionner


def trouver(chemin: str) -> Path:
    if "*" in chemin:
        trouves = sorted(glob.glob(chemin, recursive=True))
        if not trouves:
            raise FileNotFoundError(chemin)
        return Path(trouves[0])
    return Path(chemin)


def indices_segments(c: dict, schema: dict) -> np.ndarray:
    li = {nom: i for i, nom in enumerate(schema["liens"])}
    return np.array([[li[x["segment_1"]], li[x["segment_2"]]] for x in c["conduites"]])


@torch.no_grad()
def rejouer(chemin_modele: Path, specs: list[dict], chemins: dict) -> tuple[dict, dict]:
    """(prédictions, vérité) pour tous les pas de tous les épisodes de `specs`."""
    modele, c = relire_checkpoint(chemin_modele)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    modele = modele.to(appareil)
    n = c["normalisation"]
    norm = Normalisation(n["schema"], n["statistiques"], n["echelle_q_m3h"], n["episodes_train"])
    retirer = colonnes_retirees(norm.schema, c["retirer"])
    cons = c.get("conservation")
    ds = fenetres_flux(specs, chemins, norm, c["configuration"]["instantane"], retirer,
                       indices_segments(c, norm.schema))
    logits, q = [], []
    for a in range(0, len(ds), LOT):
        lot = [ds[i] for i in range(a, min(a + LOT, len(ds)))]
        s = modele(torch.stack([b["x"] for b in lot]).to(appareil).float(), c["objectif"])
        if "q_m3h_normalise" in s:
            qn = s["q_m3h_normalise"].cpu().numpy()
            if cons:                                     # Δq : on rajoute le débit du modèle calibré
                qn = np.stack([b["q_reference_m3h"].numpy() for b in lot]) + qn * cons["echelle_dq"]
            else:
                qn = qn * norm.echelle_q
            q.append(qn)
        # A n'a pas de tête « fuite » : sa note est le bilan q1 − q2
        logits.append(s["fuite_logits"].cpu().numpy() if "fuite_logits" in s else q[-1][..., 0] - q[-1][..., 1])
    forme = (len(specs), -1, len(c["conduites"]))
    pred = {"logits": np.concatenate(logits).reshape(forme).astype(np.float16)}
    if q:
        pred["q_predit_m3h"] = np.concatenate(q).reshape(*forme, 2).astype(np.float16)
    verite = {
        "fuite_active": np.stack([e.cibles["fuite_active"][11:] for e in ds.episodes]).astype(bool),
        "fuite_m3h": np.stack([e.cibles["fuite_m3h"][11:] for e in ds.episodes]).astype(np.float16),
        "q_vrai_m3h": np.stack([e.cibles["q_m3h"][11:] for e in ds.episodes]).astype(np.float16),
        "q_reference_m3h": np.stack([e.cibles["q_reference_m3h"][11:] for e in ds.episodes]).astype(np.float16),
        "conduites": np.array([x["nom"] for x in c["conduites"]]),
        "pas": np.arange(11, 11 + pred["logits"].shape[1]),
    }
    return pred, verite


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jeux", nargs="+", required=True, help="nom=dossier(s) séparés par des virgules")
    ap.add_argument("--modele", nargs="+", required=True, help="nom=jeu=chemin.pt")
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--exemples", type=int, default=3, help="épisodes copiés par jeu, pour les carnets")
    a = ap.parse_args(argv)
    a.sortie.mkdir(parents=True, exist_ok=True)
    jeux = {}
    for x in a.jeux:
        nom, dossiers = x.split("=", 1)
        m = fusionner([trouver(d) for d in dossiers.split(",")], a.sortie / "manifestes" / nom)
        specs = [e for e in m["episodes"] if e["partition"] == "test"]
        jeux[nom] = (specs, m["chemins"])
        (a.sortie / f"episodes_{nom}.json").write_text(json.dumps(specs, indent=1))
        print(f"jeu {nom} : {len(specs)} épisodes de test", flush=True)
        # quelques épisodes entiers : de préférence des fuites qui s'arrêtent, puis des « une »
        rangs = sorted(specs, key=lambda e: (not any(f < 72 for _, f in e.get("calendrier_fuites", {}).values()),
                                             not e["fuites_visees_m3h"], e["identifiant"]))
        for e in rangs[:a.exemples] + [e for e in specs if not e["fuites_visees_m3h"]][:1]:
            cible = a.sortie / "exemples" / nom / e["identifiant"]
            if not cible.exists():
                shutil.copytree(m["chemins"][e["identifiant"]], cible)
    verites = set()
    for x in a.modele:
        nom, jeu, chemin = x.split("=", 2)
        chemin = trouver(chemin)
        specs, chemins = jeux[jeu]
        pred, verite = rejouer(chemin, specs, chemins)
        np.savez_compressed(a.sortie / f"predictions_{nom}_{jeu}.npz", **pred, modele=str(chemin))
        if jeu not in verites:
            np.savez_compressed(a.sortie / f"verite_{jeu}.npz", **verite)
            verites.add(jeu)
        print(f"{nom} sur {jeu} : {pred['logits'].shape} ({chemin})", flush=True)


if __name__ == "__main__":
    main()
