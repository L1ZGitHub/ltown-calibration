"""Construction des entrees hydrauliques reelles 2018, sans acces aux labels."""
import json

import numpy as np

from .donnees import Episode, Normalisation
from .episode import preparer, projeter, prolonger
from .sources import M


def fenetre_jour(jour: int, mesures: dict, dual, correspondance,
                 noms_noeuds: list[str], normalisation: Normalisation,
                 nombre_liens: int) -> np.ndarray:
    fen = M.fenetre(2018, jour, 1)
    index = fen.horo
    if not index.isin(dual.index).all():
        raise ValueError("Index dual invalide")
    capteurs = mesures["pressions"].loc[index, fen.capteurs].to_numpy(dtype=np.float32)
    amr = mesures["amr"].loc[index]
    if list(amr.columns) != [n for n, present in zip(fen.noeuds, fen.mesure) if present]:
        raise ValueError("Identifiants ou ordre AMR differents du schema calibre")
    debits_df = mesures["debits"].loc[index]
    debits = debits_df.loc[:, M.DEBITMETRES].to_numpy(dtype=np.float32)
    niveau = mesures["niveau"].loc[index].iloc[:, 0].to_numpy(dtype=np.float32)
    pompe = fen.pompe[:len(index)].astype(np.float32)
    qv = dual.loc[index, fen.capteurs].to_numpy(dtype=np.float32)
    if any(not np.isfinite(x).all() for x in (capteurs, debits, niveau, pompe, qv,
                                               amr.to_numpy())):
        raise ValueError(f"Valeur manquante ou non finie au jour {jour}")
    observations = {
        "capteurs": capteurs, "capteurs_valide": np.ones_like(capteurs, dtype=bool),
        "debits": debits, "debits_valide": np.ones_like(debits, dtype=bool),
        "niveau": niveau, "niveau_valide": np.ones_like(niveau, dtype=bool),
        "amr": amr.to_numpy(dtype=np.float32),
        "amr_valide": np.ones(amr.shape, dtype=bool),
        "pompe": pompe, "pompe_valide": np.ones_like(pompe, dtype=bool),
    }
    demande = fen.D[:len(index)].copy()
    positions = {n: i for i, n in enumerate(fen.noeuds)}
    for j, n in enumerate(amr.columns):
        demande[:, positions[n]] = observations["amr"][:, j]
    reference_p = np.empty((len(index), len(noms_noeuds)), dtype=np.float32)
    reference_q = None
    noms_liens = None
    dernier_wn = None
    for debut in range(0, len(index), 72):
        fin = min(debut + 72, len(index))
        wn, corr = preparer(fen, prolonger(demande[debut:fin]),
                            fen.coefficients, prolonger(pompe[debut:fin]),
                            float(niveau[debut]))
        if corr != correspondance or list(wn.node_name_list) != noms_noeuds:
            raise ValueError("Correspondance hydraulique 2018 instable")
        noms_liens = list(wn.link_name_list)
        if reference_q is None:
            reference_q = np.empty((len(index), len(noms_liens)), dtype=np.float32)
        elif reference_q.shape[1] != len(noms_liens):
            raise ValueError("Nombre de liens de reference variable")
        res = M.R.executer(wn)
        reference_p[debut:fin] = res.node["pressure"][noms_noeuds].to_numpy()[:fin - debut]
        reference_q[debut:fin] = res.link["flowrate"][noms_liens].to_numpy()[:fin - debut] * 3600
        dernier_wn = wn
    if reference_q is None or dernier_wn is None or reference_q.shape[1] != nombre_liens:
        raise AssertionError("Simulation hydraulique de reference incomplete")
    demande_schema = np.zeros((len(index), len(noms_noeuds)), dtype=np.float32)
    noeuds_indices = {n: i for i, n in enumerate(noms_noeuds)}
    demande_schema[:, [noeuds_indices[n] for n in fen.noeuds]] = demande
    entrees, schema = projeter(
        dernier_wn, correspondance, fen.capteurs,
        np.asarray([n for n, present in zip(fen.noeuds, fen.mesure) if present]),
        observations, demande_schema, reference_p, reference_q, qv)
    schema = json.loads(json.dumps(schema))
    if schema != normalisation.schema:
        differences = [k for k in schema if schema[k] != normalisation.schema.get(k)]
        raise ValueError(f"Schema 2018 different : {differences}")
    n_pipes = len(correspondance)
    faux_q = np.zeros((len(index), n_pipes, 2), dtype=np.float32)
    ep = Episode(f"2018_j{jour:03d}", "reel_2018", "validation", entrees,
                 {"q_m3h": faux_q, "fuite_m3h": np.zeros((len(index), n_pipes), np.float32)},
                 schema)
    normalisees = normalisation.transformer(ep)
    extremites = np.asarray(schema["extremites_liens"], dtype=np.int64)
    return np.concatenate((
        normalisees["hydraulique_noeuds"][:, extremites[:, 0]],
        normalisees["hydraulique_noeuds"][:, extremites[:, 1]],
        normalisees["hydraulique_liens"],
    ), axis=-1)
