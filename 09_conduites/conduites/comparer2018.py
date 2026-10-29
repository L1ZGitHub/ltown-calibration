"""Toutes les cartes horaires de 2018 notées de la même façon, avec les mesures du prototype.

Pour chaque carte (prototype B, C, retraits, 08c), brute et lissée sur 300 m (`zones.noyau`) :
- `fuites` : précision moyenne (AP) et rang moyen de la vraie conduite ;
- `localisation` : part des fuites en cours couvertes par le top-1/5/10 à ≤ 300 m, et le hasard ;
- zones à budget de longueur (1, 5, 10 % du réseau, `zones.selection_longueur`) : part des fuites
  en cours à ≤ 300 m d'une conduite choisie ; contrôles : la carte moyenne du 1er janvier figée
  (comme `cartes_2018`) et des cartes tirées au hasard.
Étiquettes : débits publiés de 2018 > `FUITE_ACTIVE`, lus après toutes les cartes.

Deux vues en plus pour l'empreinte 20 % et le top-5 (colonnes « par fuite » et « hors C ») :
- par fuite : chaque fuite compte une fois (moyenne sur ses heures, puis sur les fuites), au lieu
  d'une fois par heure où elle coule ;
- hors C : les fuites de la zone C (conduites dont toutes les jonctions sont en zone C) retirées des
  étiquettes ; les cartes restent les mêmes.
"""
import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

from . import sources  # noqa: F401
from .metriques import fuites, localisation
from .sources import Annees, M
from .zones import noyau, selection_longueur
from etapes import e00_donnees as D, e05_signatures as S

BUDGETS = (.01, .05, .10)


def rappel_zones(masque: np.ndarray, labels: np.ndarray, proche: np.ndarray) -> float:
    """`zones.bilan_zones` (rappel à 300 m), en produit flottant : une année entière."""
    couverture = (masque.astype(np.float32) @ proche) > 0
    n = labels.sum(1)
    actifs = n > 0
    return float(((couverture & labels).sum(1)[actifs] / n[actifs]).mean())


EMPREINTES = (.10, .20, .30)


def rappel_empreinte(scores: np.ndarray, labels: np.ndarray, proche: np.ndarray, longueurs: np.ndarray,
                     fractions=EMPREINTES, k_max: int = 400, par_fuite: bool = False) -> dict:
    """Comme `dynamique.couverture_empreinte` : conduites par note décroissante jusqu'à ce que le
    réseau à ≤ 300 m d'elles atteigne la part voulue de la longueur ; rappel = part des fuites en
    cours dans cette empreinte (sans l'interpolation à la frontière : la conduite qui franchit le
    seuil est gardée)."""
    actifs = labels.any(1)
    s, y = scores[actifs], labels[actifs]
    ordre = np.argsort(-s, axis=1, kind="stable")
    total = longueurs.sum()
    couvert = np.zeros(s.shape, bool)
    fait = {f: np.zeros(len(s), bool) for f in fractions}
    rappel = {f: np.zeros(len(s)) for f in fractions}
    touche = {f: np.zeros(s.shape, bool) for f in fractions}
    pb = proche.astype(bool)
    lignes = np.arange(len(s))
    for k in range(min(k_max, s.shape[1])):
        couvert |= pb[ordre[:, k]]
        emp = couvert @ longueurs
        for f in fractions:
            neuf = ~fait[f] & (emp >= f * total)
            touche[f][neuf] = couvert[neuf] & y[neuf]
            rappel[f][neuf] = touche[f][neuf].sum(1) / y[neuf].sum(1)
            fait[f] |= neuf
        if all(v.all() for v in fait.values()):
            break
    for f in fractions:                          # au-delà de k_max : la couverture atteinte
        r = ~fait[f]
        touche[f][r] = couvert[r] & y[r]
        rappel[f][r] = touche[f][r].sum(1) / y[r].sum(1)
    del lignes
    if par_fuite:                                # moyenne sur les heures de chaque fuite, puis sur les fuites
        fuites_ = y.any(0)
        return {f: float((touche[f][:, fuites_].sum(0) / y[:, fuites_].sum(0)).mean()) for f in fractions}
    return {f: float(rappel[f].mean()) for f in fractions}


def conduites_zone_c(conduites: list[str]) -> np.ndarray:
    fen = M.fenetre(2018, 0, 1)
    wn = fen.wn
    return np.array([all(fen.zones.get(n, "C") == "C" for n in (wn.get_link(p).start_node_name,
                                                                wn.get_link(p).end_node_name))
                     for p in conduites])


def cartes_disponibles(dossiers: list[Path]) -> dict[str, tuple[np.ndarray, list[str]]]:
    """`dossiers` : des chemins, ou `préfixe=chemin` pour distinguer deux séries de mêmes noms ; un
    chemin peut contenir `*` ou `**` (le premier dossier trouvé)."""
    cartes = {}
    for d in map(str, dossiers):
        prefixe, d = d.split("=", 1) if "=" in d else ("", d)
        if "*" in d:
            trouves = sorted(glob.glob(d, recursive=True))
            if not trouves:
                print(f"rien sous {d}", flush=True)
                continue
            d = trouves[0]
        for f in sorted(Path(d).rglob("scores_2018.npz")):
            z = np.load(f)
            nom = prefixe + f.parent.name
            cartes[f"{nom} · logits"] = (z["logits"], z["conduites"].tolist())
            if "bilan" in z and np.isfinite(z["bilan"]).any():
                cartes[f"{nom} · bilan Q1−Q2"] = (np.maximum(z["bilan"], 0.), z["conduites"].tolist())
        for f in sorted(Path(d).rglob("cartes_08c*_2018.npz")):
            z = np.load(f)
            modele = "donnees" if f.stem == "cartes_08c_2018" else "donnees enrichi"
            for k, nom in (("presence", "présence"), ("attendue", "fuite attendue λ"),
                           ("desequilibre", "déséquilibre")):
                if k not in z:
                    continue
                v = z[k] if k == "presence" else np.maximum(z[k], 0.)
                cartes[f"{prefixe}08c {modele} · {nom}"] = (v, z["conduites"].tolist())
    return cartes


def noter(cartes: dict, sortie: Path) -> pd.DataFrame:
    wn = D.reseau()
    conduites = list(wn.pipe_name_list)
    longueurs = np.array([wn.get_link(p).length for p in conduites])
    distances = S.distances(wn).loc[conduites, conduites].to_numpy()
    proche = (distances <= 300).astype(np.float32)
    w = noyau(distances, longueurs)
    _, q = D.fuites_publiees(2018)
    heures = pd.date_range("2018-01-01", periods=365 * 24, freq="h")
    debits = q.reindex(index=heures, columns=conduites, fill_value=0.).to_numpy()
    labels = debits > Annees.G.FUITE_ACTIVE
    en_c = conduites_zone_c(conduites)
    rng = np.random.default_rng(9)
    rng_vues = np.random.default_rng(10)         # les vues ajoutées ne changent pas les tirages d'avant
    hasard = {}
    for f in BUDGETS:
        r = []
        for _ in range(16):
            m = selection_longueur(rng.random((1, len(conduites))), longueurs, f * longueurs.sum())
            r.append(rappel_zones(np.repeat(m, len(labels), 0), labels, proche))
        hasard[f] = float(np.mean(r))
    lignes = []
    for nom, (v, ordre) in cartes.items():
        v = pd.DataFrame(v, columns=ordre)[conduites].to_numpy()
        ok = np.isfinite(v).all(1)
        for mode, carte in (("brute", v), ("lissée 300 m", np.where(ok[:, None], np.nan_to_num(v) @ w.T, np.nan))):
            c, y = carte[ok], labels[ok]
            r = {"carte": nom, "mode": mode, "heures": int(ok.sum()), "heures avec fuite": int(y.any(1).sum())}
            f_ = fuites(c, y.astype(np.uint8), seuil_logit=0.0)
            r["AP"] = f_["average_precision_micro"]
            r["rang moyen"] = f_["rang_moyen_fuites"]
            loc = localisation(c, y.astype(np.uint8), distances)
            for k in ("1", "5", "10"):
                r[f"top-{k} à 300 m"] = loc["rappel_top_k"][k]
                r[f"hasard top-{k}"] = loc["hasard_top_k"][k]
            controle = np.broadcast_to(c[:24].mean(0), c.shape)   # 1er janvier : les 24 premières heures
            for f in BUDGETS:
                b = f * longueurs.sum()
                r[f"zones {f:.0%}"] = rappel_zones(selection_longueur(c, longueurs, b), y, proche)
                r[f"carte du 1er janvier {f:.0%}"] = rappel_zones(selection_longueur(controle, longueurs, b), y, proche)
                r[f"hasard {f:.0%}"] = hasard[f]
            emp = rappel_empreinte(c, y, proche, longueurs)
            emp_h = rappel_empreinte(rng.random(c.shape), y, proche, longueurs)
            emp_c = rappel_empreinte(np.ascontiguousarray(controle), y, proche, longueurs)
            for f in EMPREINTES:
                r[f"empreinte {f:.0%}"] = emp[f]
                r[f"1er janvier empreinte {f:.0%}"] = emp_c[f]
                r[f"hasard empreinte {f:.0%}"] = emp_h[f]
            alea = rng_vues.random(c.shape)
            r["top-5 à 300 m par fuite"] = np.mean([v_["rappel_top_k"]["5"] for v_ in loc["par_conduite"].values()])
            r["hasard top-5 par fuite"] = np.mean([v_["hasard_top_k"]["5"] for v_ in loc["par_conduite"].values()])
            r["empreinte 20% par fuite"] = rappel_empreinte(c, y, proche, longueurs, (.2,), par_fuite=True)[.2]
            r["hasard empreinte 20% par fuite"] = rappel_empreinte(alea, y, proche, longueurs, (.2,), par_fuite=True)[.2]
            yh = y & ~en_c
            garde = yh.any(1)
            if garde.any():
                lh = localisation(c[garde], yh[garde].astype(np.uint8), distances)
                r["top-5 à 300 m hors C"] = lh["rappel_top_k"]["5"]
                r["hasard top-5 hors C"] = lh["hasard_top_k"]["5"]
                r["empreinte 20% hors C"] = rappel_empreinte(c, yh, proche, longueurs, (.2,))[.2]
                r["hasard empreinte 20% hors C"] = rappel_empreinte(alea, yh, proche, longueurs, (.2,))[.2]
            lignes.append(r)
            n_ = {k: "—" if x is None or not np.isfinite(x) else f"{x:.3f}" for k, x in r.items()
                  if k not in ("carte", "mode")}           # « — » : aucune heure avec fuite
            print(f"{nom:40s} {mode:13s} top-1 {n_['top-1 à 300 m']} (hasard {n_['hasard top-1']}) "
                  f"zones 5 % {n_['zones 5%']} (1er janvier {n_['carte du 1er janvier 5%']}, "
                  f"hasard {hasard[.05]:.3f}) empreinte 20 % {n_['empreinte 20%']} "
                  f"(hasard {n_['hasard empreinte 20%']}) AP {n_['AP']}", flush=True)
            print(f"{'':54s} par fuite : empreinte 20 % {r['empreinte 20% par fuite']:.3f} "
                  f"(hasard {r['hasard empreinte 20% par fuite']:.3f}) top-5 {r['top-5 à 300 m par fuite']:.3f} "
                  f"(hasard {r['hasard top-5 par fuite']:.3f}) ; hors C : empreinte 20 % "
                  f"{r.get('empreinte 20% hors C', np.nan):.3f} top-5 {r.get('top-5 à 300 m hors C', np.nan):.3f}",
                  flush=True)
    t = pd.DataFrame(lignes)
    t.to_csv(sortie / "comparaison_2018.csv", index=False)
    return t


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--dossiers", nargs="*", default=None)
    ap.add_argument("--fuites", type=Path, default=None,
                    help="2018_Leakages.csv (ou un dossier qui le contient) si les données n'en ont pas")
    a = ap.parse_args(argv)
    if a.fuites:                                 # le dossier des données est lu à l'import : on relance
        import os
        import subprocess
        import sys
        from .apprendre_gpu import ajouter_fuites
        avant = os.environ.get("LTOWN_DIR")
        ajouter_fuites(a.fuites)
        if os.environ.get("LTOWN_DIR") != avant:
            reste = [x for x in (argv if argv is not None else sys.argv[1:])]
            i = reste.index("--fuites")
            subprocess.run([sys.executable, "-m", "conduites.comparer2018", *reste[:i], *reste[i + 2:]], check=True)
            return
    cartes = cartes_disponibles(a.dossiers or [a.sortie])
    print("cartes :", list(cartes), flush=True)
    noter(cartes, a.sortie)


if __name__ == "__main__":
    main()
