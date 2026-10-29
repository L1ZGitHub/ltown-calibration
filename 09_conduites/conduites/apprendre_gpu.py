"""Le prototype des conduites, appris en grand sur GPU, puis passé sur toute l'année 2018.

Mêmes pièces que `entrainer.py` (`Prototype`, `perte`, `ReglagePertes`, `Normalisation`, `Fenetres`,
attributs statiques, format de checkpoint relu par `relire_checkpoint`). Ce qui change, pour tenir
800 épisodes et un GPU :
- la normalisation est calculée en flux (mêmes moyennes et écarts, sommes en float64) ;
- les entrées normalisées sont gardées en float16, les entrées brutes jetées après lecture ;
- toutes les fenêtres d'apprentissage, mélangées ; la validation toutes les `--valider-min`
  minutes : la perte (6 fenêtres par épisode, comme `selection`) et le classement (top-5 à 300 m,
  fin de chaque heure, comme le test) ; un état sauvé à chaque validation (`etapes/`) ; le modèle
  gardé est celui du meilleur classement de validation (`<objectif>_meilleur.pt`), celui de la plus
  basse perte (`<objectif>_meilleur_perte.pt`, le choix de `entrainer`) est aussi passé sur 2018 ;
  arrêt après `--minutes` (lecture des données comprise) ou à l'heure `--fin-utc`.

Variantes (`VARIANTES`) : A, B et C du prototype (A, sans tête « fuite », est rangé par son bilan q1 − q2) ; deux retraits pour l'étape 4 (sans résidu de pression
ni dual ; fenêtre d'un seul pas au lieu d'une heure) ; C avec conservation (`conservation`).
"""
import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset

from .donnees import Episode, Fenetres, Normalisation, masque, verifier_partitions
from .entrainer import Configuration, relire_checkpoint, selection
from .graphes import attributs_physiques, construire
from .grand_jeu import fusionner
from .hydraulique import decouper
from .metriques import localisation
from .modeles import Prototype
from .objectifs import ReglagePertes, perte
from .conservation import (ReglageConservation, colonne_reference, indices_reference, jonctions,
                           perte_conservation)
from .sources import M
from .verifier_donnees import echelles_statiques

VARIANTES = {
    "A": {"objectif": "A", "retirer": [], "instantane": False},
    "B": {"objectif": "B", "retirer": [], "instantane": False},
    "C": {"objectif": "C", "retirer": [], "instantane": False},
    "B_sans_residu_dual": {"objectif": "B", "retirer": ["residu_pression_m", "dual_m3h", "variation_dual_m3h"],
                           "instantane": False},
    "B_instantane": {"objectif": "B", "retirer": [], "instantane": True},
    "C_conservation": {"objectif": "C", "retirer": [], "instantane": False, "conservation": True},
}
LARGEUR, PROFONDEUR, LOT, TAUX = 64, 4, int(os.environ.get("C9_LOT", 16)), 1e-3      # largeur et profondeur par défaut de `Prototype`
FENETRES_VALIDATION = 6


# --------------------------------------------------------------------------------- normalisation
def normalisation_flux(specs: list[dict], chemins: dict) -> Normalisation:
    """`Normalisation.ajuster` sans tout garder en mémoire : sommes, sommes des carrés, nombres."""
    schema, acc, sq, nq = None, {}, 0.0, 0
    for s in specs:
        e = Episode.lire(Path(chemins[s["identifiant"]]), s["variante"], s["partition"])
        schema = schema or e.schema
        if e.schema != schema:
            raise ValueError("schémas différents entre épisodes")
        for cle, champs in (("hydraulique_noeuds", schema["champs_noeuds"]),
                            ("hydraulique_liens", schema["champs_liens"])):
            x = e.entrees[cle]
            a = acc.setdefault(cle, np.zeros((len(champs), 3)))
            for k in range(len(champs)):
                v = x[..., k][masque(x, champs, k)].astype(np.float64)
                a[k] += (v.sum(), np.square(v).sum(), v.size)
        q = e.cibles["q_m3h"].astype(np.float64)
        sq += float(np.square(q).sum())
        nq += q.size
    stats = {}
    for cle, champs in (("hydraulique_noeuds", schema["champs_noeuds"]),
                        ("hydraulique_liens", schema["champs_liens"])):
        centres, ecarts = [], []
        for k, (_, unite) in enumerate(champs):
            s1, s2, n = acc[cle][k]
            if unite == "bool":
                c, d = 0., 1.
            else:
                c = s1 / n
                d = math.sqrt(max(s2 / n - c * c, 0.))
                d = 1. if d < 1e-6 else d
            centres.append(float(c))
            ecarts.append(float(d))
        stats[cle] = {"centre": centres, "ecart": ecarts}
    return Normalisation(schema, stats, max((sq / nq) ** 0.5, 1e-6),
                         [str(Path(chemins[s_["identifiant"]]).resolve()) for s_ in specs])


def colonnes_retirees(schema: dict, noms: list[str]) -> list[int]:
    champs = [c[0] for c in schema["champs_noeuds"]]
    return [champs.index(n) for n in noms]


def fenetres_flux(specs, chemins, norm: Normalisation, instantane: bool, retirer: list[int],
                  idx_ref: np.ndarray | None = None) -> Fenetres:
    """Un `Fenetres` construit épisode par épisode, entrées normalisées en float16, brutes jetées.
    Avec `idx_ref` (conservation) : les débits du modèle calibré des deux demi-conduites, gardés en
    m³/h dans les cibles."""
    ds = Fenetres.__new__(Fenetres)
    ds.episodes, ds.entrees, ds.normalisation = [], [], norm
    ds.representation, ds.instantane = "line", instantane
    for s in specs:
        e = Episode.lire(Path(chemins[s["identifiant"]]), s["variante"], s["partition"])
        x = norm.transformer(e)
        x["hydraulique_noeuds"][..., retirer] = 0.0     # 0 = absent, la convention des masques
        if idx_ref is not None:
            e.cibles["q_reference_m3h"] = e.entrees["hydraulique_liens"][:, idx_ref,
                                                                        colonne_reference(e.schema)].astype(np.float32)
        e.entrees = {}
        ds.episodes.append(e)
        ds.entrees.append({k: v.astype(np.float16) for k, v in x.items()})
    verifier_partitions(ds.episodes)
    ds.index = [(i, t) for i, e in enumerate(ds.episodes) for t in range(11, len(e.cibles["q_m3h"]))]
    return ds


# --------------------------------------------------------------------------------- modèle
def graphe_et_statiques():
    """Comme `entrainer` : rugosités calibrées, réseau coupé, graphe line, attributs statiques."""
    fen = M.fenetre(2018, 0, 1)
    for nom, coefficient in fen.coefficients.items():
        fen.wn.get_link(nom).roughness = coefficient
    wn, corr = decouper(fen.wn)
    g = construire(wn, corr)["line"]
    phys = attributs_physiques(wn, corr)
    sn, en = echelles_statiques(phys["noeuds"], 3)
    sl, el = echelles_statiques(phys["liens"], 5)
    sn[:, 7] /= 100.
    en["ecart"][7] = 100.
    ni = {n: i for i, n in enumerate(wn.node_name_list)}
    ext = np.array([[ni[wn.get_link(n).start_node_name], ni[wn.get_link(n).end_node_name]]
                    for n in wn.link_name_list])
    line = torch.cat((sn[ext[:, 0]], sn[ext[:, 1]], sl), -1)
    return wn, corr, g, line, {"noeuds": en, "liens": el, "champs_noeuds": phys["champs_noeuds"],
                               "champs_liens": phys["champs_liens"]}


def passage(modele, chargeur, objectif, fonction_perte, appareil, optimiseur=None, fin=None):
    """`fonction_perte(sorties, cibles)` : `perte` du prototype ou `perte_conservation`."""
    apprentissage = optimiseur is not None
    modele.train(apprentissage)
    sommes, nombre = {}, 0
    with torch.set_grad_enabled(apprentissage):
        for lot in chargeur:
            x = lot["x"].to(appareil, non_blocking=True).float()
            cibles = {k: lot[k].to(appareil, non_blocking=True)
                      for k in ("q_normalise", "fuite_active", "q_reference_m3h") if k in lot}
            sorties = modele(x, objectif)
            termes = fonction_perte(sorties, cibles)
            if not torch.isfinite(termes["totale"]):
                raise FloatingPointError("Perte non finie")
            if apprentissage:
                optimiseur.zero_grad(set_to_none=True)
                termes["totale"].backward()
                optimiseur.step()
            n = len(x)
            nombre += n
            for k, v in termes.items():
                sommes[k] = sommes.get(k, 0.) + float(v.detach()) * n
            if fin is not None and time.time() > fin:
                break
    return {k: v / max(nombre, 1) for k, v in sommes.items()}


# --------------------------------------------------------------------------------- 2018 et test
def jour_normalise(fichier: Path, schema: dict, norm: Normalisation, retirer: list[int],
                   n_conduites: int) -> np.ndarray:
    """(288, liens, 37) : les entrées line normalisées d'un jour, comme `entrees_2018.fenetre_jour`."""
    with np.load(fichier) as z:
        entrees = {k: z[k] for k in ("hydraulique_noeuds", "hydraulique_liens")}
    n = len(entrees["hydraulique_noeuds"])
    ep = Episode(fichier.stem, "reel_2018", "validation", entrees,
                 {"q_m3h": np.zeros((n, n_conduites, 2), np.float32),
                  "fuite_m3h": np.zeros((n, n_conduites), np.float32)}, schema)
    x = norm.transformer(ep)
    x["hydraulique_noeuds"][..., retirer] = 0.0
    ext = np.asarray(schema["extremites_liens"], dtype=np.int64)
    return np.concatenate((x["hydraulique_noeuds"][:, ext[:, 0]], x["hydraulique_noeuds"][:, ext[:, 1]],
                           x["hydraulique_liens"]), axis=-1)


@torch.no_grad()
def annee_2018(modele, objectif, dossiers: list[Path], norm, retirer, instantane, appareil,
               echelle_q: float) -> dict:
    """Une note par heure : la fenêtre qui finit au dernier pas de l'heure (comme `inferrer_2018`,
    pas de 12) ; en instantané, ce dernier pas seul."""
    fichiers = {}
    for d in dossiers:
        for f in Path(d).rglob("jour_*.npz"):
            fichiers[int(f.stem[5:])] = f
    schema = json.loads(sorted(f for d in dossiers for f in Path(d).rglob("schema.json"))[0].read_text())
    if schema != norm.schema:
        raise ValueError("schéma 2018 différent du jeu")
    n_c = len(modele.segments)
    jours = sorted(fichiers)
    logits = np.full((365 * 24, n_c), np.nan, np.float32)
    bilan = np.full((365 * 24, n_c), np.nan, np.float32)
    modele.eval()
    for j in jours:
        line = jour_normalise(fichiers[j], schema, norm, retirer, n_c)
        x = torch.from_numpy(line).to(appareil).reshape(24, 12, *line.shape[1:])
        if instantane:
            x = x[:, -1:]
        s = modele(x, objectif)
        if "q_m3h_normalise" in s:
            q = s["q_m3h_normalise"].cpu().numpy() * echelle_q
            bilan[j * 24:(j + 1) * 24] = q[..., 0] - q[..., 1]
        # A n'a pas de tête « fuite » : sa carte est le bilan (l'eau perdue au milieu)
        logits[j * 24:(j + 1) * 24] = (s["fuite_logits"].cpu().numpy() if "fuite_logits" in s
                                       else bilan[j * 24:(j + 1) * 24])
    return {"logits": logits, "bilan": bilan, "jours": np.array(jours)}


@torch.no_grad()
def test_simule(modele, objectif, ds: Fenetres, appareil, distances: np.ndarray,
                echelle_bilan: float | None = None) -> dict:
    """Sur des épisodes simulés : fin de chaque heure (pas 11, 23, …), top-k à 300 m. Avec
    `echelle_bilan` (objectif C) : le bilan prédit q1 − q2 (m³/h) contre le vrai débit de fuite —
    pente (droite par l'origine), bruit (99e centile du bilan hors fuite) et part des fuites en cours
    dont le bilan dépasse ce bruit."""
    modele.eval()
    idx = [i for i, (_, t) in enumerate(ds.index) if (t + 1) % 12 == 0]
    scores, labels, bil, vrais = [], [], [], []
    for a in range(0, len(idx), LOT):
        lot = [ds[i] for i in idx[a:a + LOT]]
        x = torch.stack([b["x"] for b in lot]).to(appareil).float()
        s_ = modele(x, objectif)
        labels.append(np.stack([b["fuite_active"].numpy() for b in lot]))
        if echelle_bilan and "q_m3h_normalise" in s_:
            q = s_["q_m3h_normalise"].cpu().numpy()
            bil.append((q[..., 0] - q[..., 1]) * echelle_bilan)
            vrais.append(np.stack([b["fuite_m3h"].numpy() for b in lot]))
        # A n'a pas de tête « fuite » : on range les conduites par leur bilan
        scores.append(s_["fuite_logits"].cpu().numpy() if "fuite_logits" in s_ else bil[-1])
    s, y = np.concatenate(scores), np.concatenate(labels).astype(np.uint8)
    r = localisation(s, y, distances)
    r.pop("par_conduite")
    if bil:
        b, f = np.concatenate(bil).astype(np.float64), np.concatenate(vrais).astype(np.float64)
        bruit = float(np.percentile(b[f <= 0], 99))
        r["bilan"] = {"pente": float((b * f).sum() / max((f * f).sum(), 1e-12)),
                      "bruit_p99_m3h": bruit,
                      "fuites_au_dessus_du_bruit": float((b[f > 0] > bruit).mean()) if (f > 0).any() else None,
                      "bilan_median_fuites_m3h": float(np.median(b[f > 0])) if (f > 0).any() else None,
                      "fuite_mediane_m3h": float(np.median(f[f > 0])) if (f > 0).any() else None}
    return r


@torch.no_grad()
def carte_episode(chemin_modele: Path, dossier: Path, pas: int = -1) -> dict:
    """Un modèle sauvé, relu, sur un épisode fabriqué par `episode.fabriquer` : les logits « fuite »
    de chaque conduite pour la fenêtre qui finit au pas `pas`, et les fuites vraiment actives. Avec
    l'objectif C, aussi les débits prédits des deux moitiés (m³/h, début → milieu → fin), à
    comparer aux vrais débits de l'épisode."""
    modele, c = relire_checkpoint(chemin_modele)
    n = c["normalisation"]
    norm = Normalisation(n["schema"], n["statistiques"], n["echelle_q_m3h"], n["episodes_train"])
    retirer = colonnes_retirees(norm.schema, c["retirer"])
    cons = c.get("conservation")
    idx_ref = np.array([[x["segment_1"], x["segment_2"]] for x in c["conduites"]]) if cons else None
    if cons:
        li = {nom: i for i, nom in enumerate(norm.schema["liens"])}
        idx_ref = np.vectorize(li.get)(idx_ref)
    ds = fenetres_flux([{"identifiant": dossier.name, "variante": "exemple", "partition": "test"}],
                       {dossier.name: str(dossier)}, norm, c["configuration"]["instantane"], retirer, idx_ref)
    b = ds[pas % len(ds)]
    s = modele(b["x"][None].float(), c["objectif"])
    r = {"conduites": [x["nom"] for x in c["conduites"]], "logits": s["fuite_logits"][0].numpy(),
         "fuite_active": b["fuite_active"].numpy().astype(bool), "pas": int(b["pas_cible"]),
         "q_vrai_m3h": b["q_normalise"].numpy() * norm.echelle_q, "fuite_vraie_m3h": b["fuite_m3h"].numpy()}
    if "q_m3h_normalise" in s and cons:                   # Δq : on rajoute le débit du modèle calibré
        r["q_predit_m3h"] = b["q_reference_m3h"].numpy() + s["q_m3h_normalise"][0].numpy() * cons["echelle_dq"]
    elif "q_m3h_normalise" in s:
        r["q_predit_m3h"] = s["q_m3h_normalise"][0].numpy() * norm.echelle_q
    return r


# --------------------------------------------------------------------------------- un travailleur
def travailleur(a) -> None:
    v = VARIANTES[a.variante]
    conservation = v.get("conservation", False)
    appareil = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(9)
    sortie = a.sortie / a.variante
    (sortie / "etapes").mkdir(parents=True, exist_ok=True)
    m = json.loads((a.sortie / "manifeste.json").read_text())
    norm = Normalisation.lire(a.sortie / "normalisation.json")
    specs = {p: [e for e in m["episodes"] if e["partition"] == p] for p in ("train", "validation", "test")}
    retirer = colonnes_retirees(norm.schema, v["retirer"])
    wn, corr, g, line, statiques = graphe_et_statiques()
    if [n for n in wn.link_name_list] != norm.schema["liens"]:
        raise ValueError("liens du graphe et du jeu dans un ordre différent")
    idx_ref = indices_reference(norm.schema, corr) if conservation else None
    t0 = time.time()
    ds = {p: fenetres_flux(specs[p], m["chemins"], norm, v["instantane"], retirer, idx_ref)
          for p in ("train", "validation")}
    reglage = ReglagePertes.ajuster(ds["train"].episodes, norm)
    if conservation:
        retenues, debut, fin_ = jonctions(wn, corr)
        rc = ReglageConservation.ajuster(ds["train"].episodes, retenues, debut, fin_)
        debut, fin_ = debut.to(appareil), fin_.to(appareil)
        fonction = lambda s_, c_: perte_conservation(s_, c_, norm.echelle_q, reglage, rc, debut, fin_)  # noqa: E731
        echelle_bilan = rc.echelle_dq
        print(f"{a.variante} : conservation {rc.configuration()}", flush=True)
    else:
        rc = None
        fonction = lambda s_, c_: perte(s_, c_, v["objectif"], reglage)  # noqa: E731
        echelle_bilan = norm.echelle_q
    print(f"{a.variante} : données lues en {time.time() - t0:.0f} s ; fenêtres {len(ds['train'])} / "
          f"{len(ds['validation'])}", flush=True)
    from etapes import e00_donnees as D, e05_signatures as S5
    conduites = [c.nom for c in corr]
    distances = S5.distances(D.reseau()).loc[conduites, conduites].to_numpy()
    args_modele = {"architecture": "gps", "representation": "line", "attention": "dense",
                   "entrees": ds["train"][0]["x"].shape[-1], "largeur": LARGEUR, "profondeur": PROFONDEUR,
                   "entrees_liens": 0}
    modele = Prototype(g, **args_modele, statiques=line).to(appareil)
    optimiseur = torch.optim.Adam(modele.parameters(), lr=TAUX)
    gen = torch.Generator().manual_seed(9)
    charge_app = DataLoader(ds["train"], batch_size=LOT, shuffle=True, num_workers=2, generator=gen,
                            pin_memory=True, persistent_workers=True)
    charge_val = DataLoader(Subset(ds["validation"], selection(ds["validation"], FENETRES_VALIDATION)),
                            batch_size=LOT, num_workers=1)
    fin = datetime.fromisoformat(a.fin_utc).replace(tzinfo=timezone.utc).timestamp()
    historique, etape = [], 0
    config = Configuration(representation="line", attention="dense", largeur=LARGEUR, profondeur=PROFONDEUR,
                           epoques=0, lot=LOT, taux=TAUX, graine=9, fenetres_par_episode=None,
                           instantane=v["instantane"])
    while time.time() < fin:
        t1 = time.time()
        lt = passage(modele, charge_app, v["objectif"], fonction, appareil, optimiseur,
                     fin=min(fin, t1 + a.valider_min * 60))
        lv = passage(modele, charge_val, v["objectif"], fonction, appareil)
        cv = test_simule(modele, v["objectif"], ds["validation"], appareil, distances, echelle_bilan)
        etape += 1
        historique.append({"etape": etape, "train": lt, "validation": lv,
                           "classement_validation": cv["rappel_top_k"], "bilan_validation": cv.get("bilan"),
                           "secondes": time.time() - t1,
                           "heure": datetime.now(timezone.utc).isoformat(timespec="seconds")})
        print(f"{a.variante} étape {etape} : train {lt['totale']:.4f}, validation {lv['totale']:.4f}, "
              f"top-5 à 300 m {cv['rappel_top_k']['5']:.3f} ({time.time() - t1:.0f} s)", flush=True)
        torch.save({
            "schema": 1, "objectif": v["objectif"], "epoque": etape, "configuration": asdict(config),
            "modele": args_modele, "graphe": asdict(g), "conduites": [asdict(c) for c in corr],
            "etat_modele": {k: t.cpu() for k, t in modele.state_dict().items()},
            "normalisation": json.loads((a.sortie / "normalisation.json").read_text()),
            "statiques": statiques, "pertes": reglage.configuration(), "historique": historique,
            "convention_q": "debut -> milieu -> fin", "variante": a.variante, "retirer": v["retirer"],
            "conservation": rc.configuration() if rc else None,
            "selection": "classement de validation : top-5 a 300 m, fin de chaque heure",
            "versions": {"torch": str(torch.__version__), "numpy": np.__version__}},
            sortie / "etapes" / f"etape_{etape:03d}.pt")
    (sortie / "historique.json").write_text(json.dumps(historique, indent=1))
    # meilleur classement (égalité : la plus tôt) ; et la plus basse perte, le choix d'avant
    choix = {"": max(historique, key=lambda h: (h["classement_validation"]["5"], -h["etape"]))["etape"],
             "_perte": min(historique, key=lambda h: h["validation"]["totale"])["etape"]}
    print(f"{a.variante} : apprentissage fini ; étape gardée {choix['']} (classement), "
          f"{choix['_perte']} (perte)", flush=True)
    del charge_app, ds
    dt = fenetres_flux(specs["test"], m["chemins"], norm, v["instantane"], retirer, idx_ref)
    for suffixe, k in choix.items():
        shutil.copy(sortie / "etapes" / f"etape_{k:03d}.pt", sortie / f"{v['objectif']}_meilleur{suffixe}.pt")
        ici = sortie if not suffixe else a.sortie / f"{a.variante}_selon_perte"
        ici.mkdir(exist_ok=True)
        if suffixe and k == choix[""]:            # même état : mêmes notes
            for f in ("scores_2018.npz", "test_simule.json"):
                shutil.copy(sortie / f, ici / f)
            continue
        etat = torch.load(sortie / "etapes" / f"etape_{k:03d}.pt", map_location=appareil, weights_only=True)
        modele.load_state_dict(etat["etat_modele"])
        r = annee_2018(modele, v["objectif"], a.annee, norm, retirer, v["instantane"], appareil, echelle_bilan)
        np.savez_compressed(ici / "scores_2018.npz", **r, conduites=np.array(conduites), etape=k)
        rt = test_simule(modele, v["objectif"], dt, appareil, distances, echelle_bilan)
        rt["etape"] = k
        (ici / "test_simule.json").write_text(json.dumps(rt, indent=1))
        print(f"{a.variante}{suffixe} : 2018 et test faits ; test top-5 à 300 m {rt['rappel_top_k']['5']:.3f} "
              f"(hasard {rt['hasard_top_k']['5']:.3f}) ; bilan {rt.get('bilan')}", flush=True)


# --------------------------------------------------------------------------------- le lanceur
def ajouter_fuites(fuites: Path) -> None:
    """Si le dossier des données n'a pas les fuites de 2018 (nécessaires aux notes) : un dossier
    temporaire de liens vers les données, plus le fichier des fuites. Les processus lancés ensuite
    le reçoivent par LTOWN_DIR."""
    d = os.environ.get("LTOWN_DIR")
    if not d or (Path(d) / "2018_Leakages.csv").exists():
        return
    src = fuites if fuites.is_file() else sorted(fuites.rglob("2018_Leakages.csv"))[0]
    ecrit = Path(tempfile.mkdtemp(prefix="donnees_"))
    for f in Path(d).iterdir():
        (ecrit / f.name).symlink_to(f)
    shutil.copy(src, ecrit / "2018_Leakages.csv")
    os.environ["LTOWN_DIR"] = str(ecrit)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--jeux", type=Path, nargs="+", required=True)
    ap.add_argument("--annee", type=Path, nargs="+", required=True)
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--variantes", default="B,C")
    ap.add_argument("--variante", default=None, help="interne : un travailleur")
    fin = ap.add_mutually_exclusive_group(required=True)
    fin.add_argument("--minutes", type=float, help="durée de l'apprentissage, lecture des données comprise")
    fin.add_argument("--fin-utc", help="AAAA-MM-JJTHH:MM, heure UTC de fin de l'apprentissage")
    ap.add_argument("--valider-min", type=float, default=8.0)
    ap.add_argument("--fuites", type=Path, default=None,
                    help="2018_Leakages.csv (ou un dossier qui le contient) si les données n'en ont pas")
    ap.add_argument("--apres", default="", help="modules à lancer ensuite, séparés par des virgules (ex. conduites.comparer2018)")
    a = ap.parse_args(argv)
    if a.fuites and not a.variante:
        ajouter_fuites(a.fuites)
    if a.variante:
        return travailleur(a)
    if a.minutes is not None:
        a.fin_utc = (datetime.now(timezone.utc) + timedelta(minutes=a.minutes)).strftime("%Y-%m-%dT%H:%M:%S")
    a.sortie.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    m = fusionner(a.jeux, a.sortie)
    train = [e for e in m["episodes"] if e["partition"] == "train"]
    norm = normalisation_flux(train, m["chemins"])
    if (a.sortie / "normalisation.json").exists():
        (a.sortie / "normalisation.json").unlink()
    norm.sauver(a.sortie / "normalisation.json")
    print(f"épisodes {len(m['episodes'])} (train {len(train)}) ; normalisation en {time.time() - t0:.0f} s",
          flush=True)
    procs = []
    for k, nom in enumerate(a.variantes.split(",")):
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(k % max(torch.cuda.device_count(), 1)),
               "PYTHONWARNINGS": "ignore"}
        cmd = [sys.executable, "-m", "conduites.apprendre_gpu", "--jeux", *map(str, a.jeux),
               "--annee", *map(str, a.annee), "--sortie", str(a.sortie), "--variante", nom,
               "--fin-utc", a.fin_utc, "--valider-min", str(a.valider_min)]
        journal = open(a.sortie / f"journal_{nom}.txt", "w")
        procs.append((nom, subprocess.Popen(cmd, env=env, stdout=journal, stderr=subprocess.STDOUT), journal))
    echecs = []
    for nom, p, j in procs:
        code = p.wait()
        j.close()
        lignes = (a.sortie / f"journal_{nom}.txt").read_text().splitlines()
        print(f"--- {nom} : code {code} ---", *lignes[-(60 if code else 12):], sep="\n", flush=True)
        if code:
            echecs.append(nom)
    if a.apres:                                   # après l'apprentissage : la carte du 08c, puis les notes
        for module in a.apres.split(","):
            subprocess.run([sys.executable, "-m", module, "--sortie", str(a.sortie)], check=False)
    if echecs:
        raise SystemExit(f"échec : {echecs}")


if __name__ == "__main__":
    main()
