"""Les entrées hydrauliques de 2018, jour par jour, **avant normalisation**.

Même calcul que `entrees_2018.fenetre_jour` (référence simulée par tranches de six heures, AMR,
dual du cache 2018, `projeter`), mais on garde `hydraulique_noeuds` et `hydraulique_liens` bruts :
la normalisation dépend du jeu d'apprentissage, fabriqué en même temps ailleurs. Aucune fuite lue.
"""
import argparse
import json
import multiprocessing as mp
import os
import tempfile
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .episode import preparer, projeter, prolonger
from .hydraulique import decouper
from .sources import M
from etapes import e00_donnees as D
from etapes import e10_dual as Dual

_ETAT = {}


def jour_brut(jour: int, mesures: dict, dual, correspondance, noms_noeuds: list[str]) -> tuple[dict, dict]:
    fen = M.fenetre(2018, jour, 1)
    index = fen.horo
    capteurs = mesures["pressions"].loc[index, fen.capteurs].to_numpy(dtype=np.float32)
    amr = mesures["amr"].loc[index]
    if list(amr.columns) != [n for n, present in zip(fen.noeuds, fen.mesure) if present]:
        raise ValueError("Identifiants ou ordre AMR differents du schema calibre")
    debits = mesures["debits"].loc[index].loc[:, M.DEBITMETRES].to_numpy(dtype=np.float32)
    niveau = mesures["niveau"].loc[index].iloc[:, 0].to_numpy(dtype=np.float32)
    pompe = fen.pompe[:len(index)].astype(np.float32)
    qv = dual.loc[index, fen.capteurs].to_numpy(dtype=np.float32)
    observations = {
        "capteurs": capteurs, "capteurs_valide": np.ones_like(capteurs, dtype=bool),
        "debits": debits, "debits_valide": np.ones_like(debits, dtype=bool),
        "niveau": niveau, "niveau_valide": np.ones_like(niveau, dtype=bool),
        "amr": amr.to_numpy(dtype=np.float32), "amr_valide": np.ones(amr.shape, dtype=bool),
        "pompe": pompe, "pompe_valide": np.ones_like(pompe, dtype=bool),
    }
    demande = fen.D[:len(index)].copy()
    positions = {n: i for i, n in enumerate(fen.noeuds)}
    for j, n in enumerate(amr.columns):
        demande[:, positions[n]] = observations["amr"][:, j]
    reference_p = np.empty((len(index), len(noms_noeuds)), dtype=np.float32)
    reference_q, dernier_wn = None, None
    for debut in range(0, len(index), 72):
        fin = min(debut + 72, len(index))
        wn, corr = preparer(fen, prolonger(demande[debut:fin]), fen.coefficients,
                            prolonger(pompe[debut:fin]), float(niveau[debut]))
        if corr != correspondance or list(wn.node_name_list) != noms_noeuds:
            raise ValueError("Correspondance hydraulique 2018 instable")
        noms_liens = list(wn.link_name_list)
        if reference_q is None:
            reference_q = np.empty((len(index), len(noms_liens)), dtype=np.float32)
        res = M.R.executer(wn)
        reference_p[debut:fin] = res.node["pressure"][noms_noeuds].to_numpy()[:fin - debut]
        reference_q[debut:fin] = res.link["flowrate"][noms_liens].to_numpy()[:fin - debut] * 3600
        dernier_wn = wn
    demande_schema = np.zeros((len(index), len(noms_noeuds)), dtype=np.float32)
    ni = {n: i for i, n in enumerate(noms_noeuds)}
    demande_schema[:, [ni[n] for n in fen.noeuds]] = demande
    entrees, schema = projeter(
        dernier_wn, correspondance, fen.capteurs,
        np.asarray([n for n, present in zip(fen.noeuds, fen.mesure) if present]),
        observations, demande_schema, reference_p, reference_q, qv)
    return {k: entrees[k] for k in ("hydraulique_noeuds", "hydraulique_liens")}, json.loads(json.dumps(schema))


def _init(mesures, dual, corr, noms, sortie):
    _ETAT.update(mesures=mesures, dual=dual, corr=corr, noms=noms, sortie=sortie)


def _un(jour: int) -> tuple[int, str | None]:
    """Un jour. Si EPANET ne converge pas (réglages stricts du prototype), le jour est abandonné,
    comme un épisode du jeu : ses heures restent sans note. Cela dépend du processeur (un jour qui
    passe sur une machine peut échouer sur une autre)."""
    e = _ETAT
    ici = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            entrees, schema = jour_brut(jour, e["mesures"], e["dual"], e["corr"], e["noms"])
        except Exception as err:                 # noqa: BLE001 — rapporté dans le manifeste
            return jour, f"{type(err).__name__}: {err}"
        finally:
            os.chdir(ici)
    np.savez_compressed(e["sortie"] / f"jour_{jour:03d}.npz", **entrees)
    if not (e["sortie"] / "schema.json").exists():
        (e["sortie"] / "schema.json").write_text(json.dumps(schema))
    return jour, None


def preparer_annee(sortie: Path, jours: range, processus: int) -> dict:
    """Les jours `jours` (consécutifs) ; le dual seulement sur eux, par tranches alignées sur les
    jours (les mêmes tranches que sur l'année entière)."""
    t0 = time.time()
    sortie.mkdir(parents=True, exist_ok=True)
    mesures = D.mesures(2018)
    fd = M.fenetre(2018, jours[0], len(jours))
    dual = pd.DataFrame(Dual.simuler(fd, fd.p_mes, coupe=True, processus=processus),
                        index=fd.horo, columns=fd.capteurs)
    print(f"dual 2018 : {time.time() - t0:.0f} s", flush=True)
    fen = M.fenetre(2018, 0, 1)
    wn, corr = decouper(fen.wn)
    faits, echecs = [], {}
    with mp.get_context("fork").Pool(processus, initializer=_init,
                                     initargs=(mesures, dual, corr, list(wn.node_name_list), sortie)) as pool:
        for j, err in pool.imap_unordered(_un, list(jours)):
            if err:
                echecs[j] = err
            else:
                faits.append(j)
            if (len(faits) + len(echecs)) % 20 == 0:
                print(f"{len(faits) + len(echecs)}/{len(jours)} jours ({time.time() - t0:.0f} s)", flush=True)
    r = {"jours": sorted(faits), "echecs": {str(j): echecs[j] for j in sorted(echecs)},
         "secondes": time.time() - t0}
    (sortie / f"manifeste_{jours[0]:03d}.json").write_text(json.dumps(r))
    return r


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sortie", type=Path, required=True)
    ap.add_argument("--debut", type=int, default=0)
    ap.add_argument("--jours", type=int, default=365)
    ap.add_argument("--processus", type=int, default=4)
    a = ap.parse_args()
    r = preparer_annee(a.sortie, range(a.debut, a.debut + a.jours), a.processus)
    print(len(r["jours"]), "jours faits,", len(r["echecs"]), "abandonnés (EPANET ne converge pas),",
          round(r["secondes"]), "s", flush=True)


if __name__ == "__main__":
    main()
