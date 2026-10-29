"""Controle d'integrite du pilote corrige, sans score predictif ni entrainement."""
import argparse
import json
from pathlib import Path

import numpy as np

from .donnees import Episode, Fenetres, Normalisation, verifier_partitions
from .observations import observer
from .pilote import verifier_plan


def verifier(jeu: Path, ancien: Path) -> dict:
    destination = jeu / "verification_finale.json"
    if destination.exists():
        raise FileExistsError(destination)
    manifeste = json.loads((jeu / "manifeste.json").read_text())
    precedent = json.loads((ancien / "manifeste.json").read_text())
    if manifeste["etat"] != "complet" or manifeste["episodes"] != precedent["episodes"]:
        raise ValueError("Pilote incomplet ou plan different du pilote precedent")
    verifier_plan(manifeste["episodes"])
    episodes = []
    positions, graines = set(), set()
    maximum = 0.
    for spec in manifeste["episodes"]:
        nom = spec["identifiant"]
        dossier = jeu / nom
        r = json.loads((dossier / "rapport.json").read_text())
        if r["graine_bruit"] != spec["graine_episode"] \
                or r["graine_episode"] != spec["graine_episode"] \
                or r["fuites_visees_m3h"] != spec["fuites_visees_m3h"]:
            raise AssertionError(f"Provenance incorrecte : {nom}")
        graines.add(r["graine_bruit"])
        e = Episode.lire(dossier, spec["variante"], spec["partition"])
        e.verifier()
        episodes.append(e)
        q, f = e.cibles["q_m3h"], e.cibles["fuite_m3h"]
        maximum = max(maximum, float(np.abs(q[..., 0] - q[..., 1] - f).max()))
        noms = [c["nom"] for c in r["conduites"]]
        attendus = np.array([p in spec["fuites_visees_m3h"] for p in noms])
        if not np.array_equal(np.any(np.abs(f) > 1e-3, axis=0), attendus):
            raise AssertionError(f"Positions de fuite incorrectes : {nom}")
        positions.update(spec["fuites_visees_m3h"])
        with np.load(ancien / nom / "cibles.npz") as z:
            for k in ("q_m3h", "fuite_m3h"):
                np.testing.assert_array_equal(e.cibles[k], z[k])
        # Le bruit blanc du debitmetre est additif ; sa realisation se reproduit
        # sans connaitre les pressions vraies ni relancer la simulation.
        n = len(q)
        zero = {"capteurs": np.zeros((n, 33)), "debits": np.zeros((n, 3)),
                "niveau": np.zeros(n), "amr": np.zeros((n, 82)), "pompe": np.zeros(n)}
        bruit = observer(zero, r["graine_bruit"])["debits"][:, 1]
        c = r["conduites"][noms.index("p227")]
        segment = 0 if r["debitmetres"]["p227"] == c["segment_1"] else 1
        if r["debitmetres"]["p227"] not in (c["segment_1"], c["segment_2"]):
            raise AssertionError("Debitmetre p227 non raccorde a ses segments")
        with np.load(dossier / "observations.npz") as z:
            np.testing.assert_allclose(z["debits"][:, 1] - q[:, noms.index("p227"), segment],
                                       bruit, atol=1e-10, rtol=1e-10)
    if maximum > 0.01 or len(graines) != len(episodes):
        raise AssertionError("Bilan hydraulique ou unicite des graines incorrect")
    verifier_partitions(episodes)
    norm = Normalisation.lire(jeu / "normalisation.json")
    recalculee = Normalisation.ajuster([e for e in episodes if e.partition == "train"])
    if norm != recalculee:
        raise AssertionError("Normalisation non reproductible sur le train")
    partitions = {}
    for p in ("train", "validation", "test"):
        es = [e for e in episodes if e.partition == p]
        for representation in ("line", "hydraulique"):
            ds = Fenetres(es, norm, representation)
            for x in ds.entrees:
                if not all(np.isfinite(v).all() for v in x.values()):
                    raise AssertionError("Entrees normalisees non finies")
            if not all(np.isfinite(ds[i]["x"].numpy()).all() for i in (0, len(ds) - 1)):
                raise AssertionError("Fenetres non finies")
        partitions[p] = {"episodes": len(es), "fenetres": len(ds)}
    rapport = {
        "etat": "verifie", "episodes": len(episodes),
        "episodes_sains": sum(not s["fuites_visees_m3h"] for s in manifeste["episodes"]),
        "positions_de_fuite_distinctes": len(positions), "partitions": partitions,
        "bilan_max_abs_m3h": maximum, "normalisation_train_uniquement": True,
        "graines_bruit_uniques": len(graines), "bruit_p227_reproduit": True,
        "cibles_identiques_ancien_pilote": True, "plan_identique_ancien_pilote": True,
        "controle_zero_numerique_m3h": 1e-3, "aucun_entrainement": True,
        "test": "controle d'integrite seulement ; aucun score predictif ni selection",
        "duree_generation_secondes": manifeste["secondes"],
        "taille_Mio": sum(p.stat().st_size for p in jeu.rglob("*") if p.is_file()) / 2**20,
    }
    destination.write_text(json.dumps(rapport, indent=2, allow_nan=False) + "\n")
    return rapport


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jeu", type=Path, required=True)
    ap.add_argument("--ancien", type=Path, required=True)
    a = ap.parse_args()
    print(json.dumps(verifier(a.jeu, a.ancien), indent=2))


if __name__ == "__main__":
    main()
