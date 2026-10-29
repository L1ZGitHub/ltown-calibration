"""Test hydraulique, tests de tenseurs et microbenchmark, sans entrainement."""
import argparse
import json
import platform
import statistics
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .graphes import construire
from .hydraulique import liens_mesures, verifier
from .modeles import Prototype


def mesurer(modele: Prototype, x: torch.Tensor, repetitions: int, appareil: str) -> dict:
    temps = []
    if appareil == "cuda":
        torch.cuda.reset_peak_memory_stats()
    for i in range(repetitions + 2):
        modele.zero_grad(set_to_none=True)
        x.grad = None
        if appareil == "cuda":
            torch.cuda.synchronize()
        debut = time.perf_counter()
        sortie = modele(x)
        perte = sum(y.square().mean() for y in sortie.values())
        perte.backward()
        if appareil == "cuda":
            torch.cuda.synchronize()
        if not torch.isfinite(perte):
            raise AssertionError("Perte synthetique non finie")
        if i >= 2:
            temps.append((time.perf_counter() - debut) * 1000)
    # Profil separe : ne pas confondre allocations cumulees et pic resident.
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU], profile_memory=True) as prof:
        modele.zero_grad(set_to_none=True)
        x.grad = None
        sortie = modele(x)
        sum(y.square().mean() for y in sortie.values()).backward()
    allocations = sum(max(0, e.self_cpu_memory_usage) for e in prof.events())
    return {
        "mediane_avant_arriere_ms": statistics.median(temps),
        "mesures_ms": temps,
        "allocations_CPU_cumulees_Mio": allocations / 2**20,
        "pic_CPU_Mio": None,
        "pic_CUDA_alloue_Mio": torch.cuda.max_memory_allocated() / 2**20
        if appareil == "cuda" else None,
        "parametres": sum(p.numel() for p in modele.parameters()),
        "formes": {k: list(v.shape) for k, v in sortie.items()},
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--lot", type=int, default=1)
    ap.add_argument("--largeur", type=int, default=64)
    ap.add_argument("--profondeur", type=int, default=4)
    ap.add_argument("--appareil", choices=("cpu", "cuda"), default="cpu")
    a = ap.parse_args()
    if min(a.repetitions, a.lot, a.largeur, a.profondeur) < 1 or a.largeur % 4:
        ap.error("Parametres positifs requis ; largeur multiple de quatre")
    if a.appareil == "cuda" and not torch.cuda.is_available():
        ap.error("CUDA indisponible")
    if a.sortie.exists():
        ap.error("Le dossier de sortie existe deja ; choisir un nouveau nom")
    torch.set_num_threads(1)
    torch.manual_seed(9)
    wn, correspondance, hydraulique = verifier()
    graphes = construire(wn, correspondance)
    rapport = {
        "schema": 1, "configuration": {**vars(a), "sortie": str(a.sortie), "graine": 9,
                                     "threads": 1, "pas": 12, "entrees_synthetiques": 6},
        "environnement": {"python": platform.python_version(), "torch": torch.__version__,
                          "appareil": a.appareil, "plateforme": platform.platform()},
        "hydraulique": hydraulique, "graphes": {}, "benchmark": {},
        "limites": [
            "Entrees aleatoires : aucune mesure de qualite predictive.",
            "Dense SDPA sans attributs de paire ; creux conditionne par les attributs d'arete.",
            "Sommet global et normalisation LayerNorm communs ; pas une reproduction exacte du 07.",
            "Allocations CPU cumulees du profiler, pas un pic ; pic CPU non mesure.",
            "Aucune normalisation ajustee, aucune donnee d'apprentissage, aucun checkpoint.",
        ],
    }
    mappings = {"schema": 1, "orientation": "debut -> milieu -> fin ; Q en m3/h",
                "conduites": [asdict(c) for c in correspondance], "graphes": {},
                "debitmetres": liens_mesures(correspondance),
                "convention_debitmetres": "p227/p235 cote reservoir, sans duplication"}
    for representation, g in graphes.items():
        mappings["graphes"][representation] = {
            "noms": g.noms, "aretes": g.aretes.tolist(), "attributs": g.attributs.tolist(),
            "segments": g.segments.tolist(), "cibles_fuite": g.cibles_fuite.tolist(),
            "jonctions_partagees": g.jonctions_partagees, "type_lien": g.type_lien,
        }
        rapport["graphes"][representation] = {"sommets": len(g.noms), "aretes": g.aretes.shape[1]}
        x = torch.randn(a.lot, 12, len(g.noms), 6, device=a.appareil, requires_grad=True)
        for attention in ("dense", "creuse"):
            torch.manual_seed(9)
            modele = Prototype(g, representation, attention, largeur=a.largeur,
                               profondeur=a.profondeur).to(a.appareil)
            mesure = mesurer(modele, x, a.repetitions, a.appareil)
            rapport["benchmark"][f"{representation}_{attention}"] = mesure
            print(representation, attention, json.dumps(mesure), flush=True)
            del modele
    a.sortie.mkdir(parents=True)
    (a.sortie / "rapport.json").write_text(json.dumps(rapport, indent=2) + "\n")
    (a.sortie / "correspondances.json").write_text(json.dumps(mappings) + "\n")
    print("Verification terminee :", a.sortie)


if __name__ == "__main__":
    main()
