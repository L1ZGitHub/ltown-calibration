"""Comparaison A/B/C bornee sur train/validation ; jamais d'acces aux tableaux test."""
import argparse
from copy import deepcopy
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .donnees import Episode, Fenetres, Normalisation, verifier_partitions
from .graphes import Graphe, attributs_physiques, construire
from .hydraulique import decouper
from .metriques import debits, fuites
from .modeles import Prototype
from .objectifs import ReglagePertes, perte
from .pilote import verifier_plan
from .sources import M
from .verifier_donnees import echelles_statiques


@dataclass(frozen=True)
class Configuration:
    architecture: str = "gps"
    representation: str = "line"
    attention: str = "dense"
    largeur: int = 8
    profondeur: int = 1
    epoques: int = 1
    lot: int = 1
    taux: float = 1e-3
    graine: int = 9
    threads: int = 1
    fenetres_par_episode: int | None = 2
    instantane: bool = False

    def verifier(self) -> None:
        if self.architecture not in ("gps", "independant") \
                or (self.architecture == "independant" and self.representation != "line") \
                or self.representation not in ("line", "hydraulique") \
                or self.attention not in ("dense", "creuse") \
                or self.largeur < 4 or self.largeur % 4 or self.profondeur < 1 \
                or self.epoques < 1 or self.lot < 1 or self.threads < 1 \
                or not math.isfinite(self.taux) or self.taux <= 0 \
                or (self.fenetres_par_episode is not None and self.fenetres_par_episode < 1):
            raise ValueError("Configuration d'entrainement invalide")


def selection(ds: Fenetres, nombre: int | None) -> list[int]:
    """Meme sous-echantillonnage temporel pour chaque episode, sain ou en fuite."""
    if nombre is not None and nombre < 1:
        raise ValueError("Nombre de fenetres par episode strictement positif requis")
    resultat = []
    for ep in range(len(ds.episodes)):
        indices = [i for i, (e, _) in enumerate(ds.index) if e == ep]
        positions = np.arange(len(indices)) if nombre is None else np.linspace(
            0, len(indices) - 1, min(nombre, len(indices)), dtype=int)
        resultat.extend(indices[int(i)] for i in positions)
    return resultat


def passage(modele: Prototype, chargeur: DataLoader, objectif: str,
            reglage: ReglagePertes, optimiseur: torch.optim.Optimizer | None = None
            ) -> tuple[dict, dict[str, np.ndarray]]:
    apprentissage = optimiseur is not None
    modele.train(apprentissage)
    sommes: dict[str, float] = {}
    nombre = 0
    predictions: dict[str, list[np.ndarray]] = {}
    with torch.set_grad_enabled(apprentissage):
        for lot in chargeur:
            if optimiseur is not None:
                optimiseur.zero_grad(set_to_none=True)
            sorties = modele(lot["x"], objectif,
                             liens=lot["liens"] if modele.representation == "hydraulique" else None)
            termes = perte(sorties, lot, objectif, reglage)
            if not all(torch.isfinite(v).all() for v in termes.values()):
                raise FloatingPointError("Perte non finie")
            if optimiseur is not None:
                termes["totale"].backward()
                if any(not torch.isfinite(p.grad).all() for p in modele.parameters()
                       if p.grad is not None):
                    raise FloatingPointError("Gradient non fini")
                optimiseur.step()
                if any(not torch.isfinite(p).all() for p in modele.parameters()):
                    raise FloatingPointError("Parametre non fini apres mise a jour")
            n = len(lot["x"])
            nombre += n
            for k, v in termes.items():
                sommes[k] = sommes.get(k, 0.) + float(v.detach()) * n
            if not apprentissage:
                valeurs = {**sorties, **{k: lot[k] for k in (
                    "q_normalise", "fuite_active", "fuite_m3h", "pas_cible")}}
                for k, v in valeurs.items():
                    predictions.setdefault(k, []).append(v.detach().cpu().numpy())
                predictions.setdefault("episode", []).append(np.asarray(lot["episode"]))
    if not nombre:
        raise ValueError("Chargeur vide")
    return ({k: v / nombre for k, v in sommes.items()},
            {k: np.concatenate(v) for k, v in predictions.items()})


def relire_checkpoint(chemin: Path) -> tuple[Prototype, dict]:
    # Format compose exclusivement de tenseurs et de types simples, sans pickle de classes.
    c = torch.load(chemin, map_location="cpu", weights_only=True)
    if c["schema"] != 1 or c["convention_q"] != "debut -> milieu -> fin":
        raise ValueError("Checkpoint inconnu")
    modele = Prototype(Graphe(**c["graphe"]), **c["modele"],
                       statiques=c["etat_modele"].get("statiques"),
                       statiques_liens=c["etat_modele"].get("statiques_liens"))
    modele.load_state_dict(c["etat_modele"], strict=True)
    modele.eval()
    return modele, c


def empreinte(chemin: Path) -> str:
    with chemin.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def entrainer(jeu: Path, sortie: Path, config: Configuration) -> dict:
    config.verifier()
    if sortie.exists():
        raise FileExistsError(sortie)
    manifeste = json.loads((jeu / "manifeste.json").read_text())
    if manifeste["schema"] != 1 or manifeste["etat"] != "complet":
        raise ValueError("Jeu incomplet ou format inconnu")
    verifier_plan(manifeste["episodes"])
    specs = [e for e in manifeste["episodes"] if e["partition"] in ("train", "validation")]
    episodes = [Episode.lire(jeu / e["identifiant"], e["variante"], e["partition"]) for e in specs]
    verifier_partitions(episodes)
    train = [e for e in episodes if e.partition == "train"]
    validation = [e for e in episodes if e.partition == "validation"]
    if not train or not validation:
        raise ValueError("Train et validation non vides requis")
    norm = Normalisation.lire(jeu / "normalisation.json")
    reglage = ReglagePertes.ajuster(train, norm)
    torch.set_num_threads(config.threads)
    torch.manual_seed(config.graine)
    fen = M.fenetre(2018, 0, 1)
    for nom, coefficient in fen.coefficients.items():
        fen.wn.get_link(nom).roughness = coefficient
    wn, corr = decouper(fen.wn)
    g = construire(wn, corr)[config.representation]
    schema = train[0].schema
    ni = {nom: i for i, nom in enumerate(wn.node_name_list)}
    extremites = [[ni[wn.get_link(n).start_node_name], ni[wn.get_link(n).end_node_name]]
                  for n in wn.link_name_list]
    if list(wn.node_name_list) != schema["noeuds"] \
            or list(wn.link_name_list) != schema["liens"] \
            or extremites != schema["extremites_liens"] \
            or g.noms != schema["liens" if config.representation == "line" else "noeuds"] \
            or [[c.segment_1, c.segment_2] for c in corr] != [
                [schema["liens"][i], schema["liens"][j]] for i, j in g.segments.tolist()]:
        raise ValueError("Ordre des identifiants graphe/donnees incompatible")
    phys = attributs_physiques(wn, corr)
    sn, en = echelles_statiques(phys["noeuds"], 3)
    sl, el = echelles_statiques(phys["liens"], 5)
    sn[:, 7] /= 100.
    en["ecart"][7] = 100.
    ext = np.asarray(schema["extremites_liens"])
    line = torch.cat((sn[ext[:, 0]], sn[ext[:, 1]], sl), -1)
    datasets = {p: Fenetres(es, norm, config.representation, config.instantane)
                for p, es in (("train", train), ("validation", validation))}
    indices = {p: selection(ds, config.fenetres_par_episode) for p, ds in datasets.items()}
    args_modele = {
        "architecture": config.architecture,
        "representation": config.representation, "attention": config.attention,
        "entrees": datasets["train"][0]["x"].shape[-1],
        "largeur": config.largeur, "profondeur": config.profondeur,
        "entrees_liens": len(schema["champs_liens"]) if config.representation == "hydraulique" else 0,
    }
    initial = Prototype(g, **args_modele,
                        statiques=line if config.representation == "line" else sn,
                        statiques_liens=sl if config.representation == "hydraulique" else None)
    provenance = {
        "jeu": str(jeu.resolve()), "manifeste_sha256": empreinte(jeu / "manifeste.json"),
        "fichiers": {str(p.resolve()): empreinte(p) for e in specs
                     for p in ((jeu / e["identifiant"] / nom)
                               for nom in ("rapport.json", "entrees.npz", "cibles.npz"))},
        "specifications_train_validation": specs,
        "code_sha256": {p.name: empreinte(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
        "avertissement_bruit": "Ancien rapport sans graine_bruit : bruit graine_variante+1, "
                              "commun aux scenarios d'une variante ; pas d'independance du bruit.",
    }
    rapport = {
        "schema": 1, "etat": "en_cours", "configuration": asdict(config),
        "provenance": provenance, "test_lu": False,
        "fenetres": {p: len(v) for p, v in indices.items()}, "objectifs": {},
        "parametres_total": sum(p.numel() for p in initial.parameters()),
        "usage": "verification technique courte" if config.fenetres_par_episode is not None
                 else "pilote train/validation ; pas certification de generalisation",
    }
    ds_val = datasets["validation"]
    colonne_q = [c[0] for c in schema["champs_liens"]].index("debit_reference_m3h")
    q_physique, q_vrai, f_vrai = [], [], []
    for i in indices["validation"]:
        ep, t = ds_val.index[i]
        e = validation[ep]
        q_physique.append(e.entrees["hydraulique_liens"][t, g.segments.numpy(), colonne_q])
        q_vrai.append(e.cibles["q_m3h"][t])
        f_vrai.append(e.cibles["fuite_m3h"][t])
    rapport["reference_physique_validation"] = debits(
        np.stack(q_physique), np.stack(q_vrai), np.stack(f_vrai), seuil_sens_m3h=0.1)
    sortie.mkdir(parents=True)
    norm.sauver(sortie / "normalisation.json")
    chemin_rapport = sortie / "rapport.json"
    chemin_rapport.write_text(json.dumps(rapport, indent=2) + "\n")
    debut = time.perf_counter()
    for objectif in ("A", "B", "C"):
        debut_objectif = time.perf_counter()
        modele = deepcopy(initial)
        optimiseur = torch.optim.Adam(modele.parameters(), lr=config.taux)
        generateur = torch.Generator().manual_seed(config.graine)
        chargeurs = {p: DataLoader(Subset(ds, indices[p]), batch_size=config.lot,
                                  shuffle=p == "train", num_workers=0,
                                  generator=generateur if p == "train" else None)
                     for p, ds in datasets.items()}
        avant, _ = passage(modele, chargeurs["validation"], objectif, reglage)
        historique = []
        meilleur = math.inf
        chemin = sortie / f"{objectif}_meilleur.pt"
        for ep in range(1, config.epoques + 1):
            debut_epoque = time.perf_counter()
            lt, _ = passage(modele, chargeurs["train"], objectif, reglage, optimiseur)
            secondes_train = time.perf_counter() - debut_epoque
            debut_validation = time.perf_counter()
            lv, _ = passage(modele, chargeurs["validation"], objectif, reglage)
            historique.append({"epoque": ep, "train": lt, "validation": lv,
                               "secondes_train": secondes_train,
                               "secondes_validation": time.perf_counter() - debut_validation})
            if lv["totale"] < meilleur:
                meilleur = lv["totale"]
                torch.save({
                    "schema": 1, "objectif": objectif, "epoque": ep,
                    "configuration": asdict(config), "modele": args_modele,
                    "graphe": asdict(g), "conduites": [asdict(c) for c in corr],
                    "etat_modele": modele.state_dict(), "etat_optimiseur": optimiseur.state_dict(),
                    "rng_torch": torch.get_rng_state(), "rng_chargeur": generateur.get_state(),
                    "normalisation": json.loads((sortie / "normalisation.json").read_text()),
                    "statiques": {"noeuds": en, "liens": el,
                                  "champs_noeuds": phys["champs_noeuds"],
                                  "champs_liens": phys["champs_liens"]},
                    "pertes": reglage.configuration(), "provenance": provenance,
                    "indices_fenetres": indices, "historique": historique.copy(),
                    "convention_q": "debut -> milieu -> fin", "versions": {
                        "torch": str(torch.__version__), "numpy": np.__version__},
                }, chemin)
            print(f"{objectif} epoque {ep}: train={lt['totale']:.6f}, "
                  f"validation={lv['totale']:.6f}", flush=True)
        relu, checkpoint = relire_checkpoint(chemin)
        pertes, prediction = passage(relu, chargeurs["validation"], objectif, reglage)
        if not math.isclose(pertes["totale"], meilleur, rel_tol=1e-6, abs_tol=1e-7):
            raise AssertionError("Checkpoint recharge non equivalent")
        diagnostics = {}
        if objectif in ("A", "C"):
            prediction["q_m3h"] = prediction["q_m3h_normalise"] * norm.echelle_q
            diagnostics["q"] = debits(prediction["q_m3h"],
                                     prediction["q_normalise"] * norm.echelle_q,
                                     prediction["fuite_m3h"], seuil_sens_m3h=0.1)
        if objectif in ("B", "C"):
            diagnostics["fuite"] = fuites(prediction["fuite_logits"],
                                         prediction["fuite_active"], seuil_logit=0.)
        prediction["conduites"] = np.asarray([c.nom for c in corr])
        np.savez_compressed(sortie / f"{objectif}_validation.npz", **prediction)
        rapport["objectifs"][objectif] = {
            "secondes": time.perf_counter() - debut_objectif,
            "mises_a_jour": config.epoques * math.ceil(len(indices["train"]) / config.lot),
            "validation_initiale": avant, "historique": historique,
            "meilleure_epoque": checkpoint["epoque"], "validation_rechargee": pertes,
            "checkpoint": chemin.name, "metriques_validation": diagnostics,
        }
        chemin_rapport.write_text(json.dumps(rapport, indent=2, allow_nan=False) + "\n")
    rapport.update(etat="complet", secondes=time.perf_counter() - debut)
    chemin_rapport.write_text(json.dumps(rapport, indent=2, allow_nan=False) + "\n")
    return rapport


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jeu", type=Path, required=True)
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--architecture", choices=("gps", "independant"), default="gps")
    ap.add_argument("--representation", choices=("line", "hydraulique"), default="line")
    ap.add_argument("--attention", choices=("dense", "creuse"), default="dense")
    ap.add_argument("--largeur", type=int, default=8)
    ap.add_argument("--profondeur", type=int, default=1)
    ap.add_argument("--epoques", type=int, default=1)
    ap.add_argument("--lot", type=int, default=1)
    ap.add_argument("--taux", type=float, default=1e-3)
    ap.add_argument("--graine", type=int, default=9)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--fenetres-par-episode", type=int, default=2)
    ap.add_argument("--toutes-fenetres", action="store_true")
    ap.add_argument("--instantane", action="store_true")
    a = ap.parse_args()
    c = Configuration(**{k: v for k, v in vars(a).items()
                         if k not in ("jeu", "sortie", "toutes_fenetres")})
    if a.toutes_fenetres:
        c = Configuration(**{**asdict(c), "fenetres_par_episode": None})
    entrainer(a.jeu, a.sortie, c)


if __name__ == "__main__":
    main()
