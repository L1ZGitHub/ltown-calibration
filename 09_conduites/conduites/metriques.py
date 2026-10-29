"""Diagnostics physiques et classement multi-fuites ; aucun reglage de seuil."""
from math import comb

import numpy as np
from sklearn.metrics import average_precision_score


def localisation(scores: np.ndarray, actif: np.ndarray, distances: np.ndarray,
                 rayon_m: float = 300., ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    """Couverture de chaque fuite par au moins un candidat du top-k, a distance <= rayon."""
    s, y, d = np.asarray(scores), np.asarray(actif), np.asarray(distances)
    if s.ndim != 2 or s.shape != y.shape or not s.size or not np.isfinite(s).all() \
            or not np.isin(y, (0, 1)).all() or d.shape != (s.shape[1], s.shape[1]) \
            or np.isnan(d).any() or (d < 0).any() or not np.all(np.diag(d) == 0) \
            or not np.allclose(d, d.T) or not np.isfinite(rayon_m) or rayon_m < 0 \
            or not ks or any(not isinstance(k, int) or k < 1 for k in ks):
        raise ValueError("Scores/labels [instants, conduites], distances symetriques >=0 requis")
    y = y.astype(bool)
    n = s.shape[1]
    nombres = y.sum(1)
    positifs = nombres > 0
    sommes = {k: np.zeros(len(s)) for k in ks}
    aleatoire = {k: np.zeros(len(s)) for k in ks}
    par_conduite = {}
    for p in np.flatnonzero(y.any(0)):
        selection = y[:, p]
        lignes = s[selection]
        proches = d[:, p] <= rayon_m
        m = int(proches.sum())
        rappels, hasard = {}, {}
        for k in ks:
            taille = min(k, n)
            coupure = np.partition(lignes, n - taille, axis=1)[:, n - taille]
            hauts = lignes > coupure[:, None]
            egaux = lignes == coupure[:, None]
            touches = (hauts & proches).any(1).astype(float)
            places = taille - hauts.sum(1)
            nb_egaux = egaux.sum(1)
            nb_proches = (egaux & proches).sum(1)
            for i in np.flatnonzero(touches == 0):
                e, b, r = int(nb_egaux[i]), int(nb_proches[i]), int(places[i])
                touches[i] = 1 - (comb(e - b, r) / comb(e, r) if e - b >= r else 0.)
            reference = 1 - (comb(n - m, taille) / comb(n, taille)
                             if n - m >= taille else 0.)
            sommes[k][selection] += touches
            aleatoire[k][selection] += reference
            rappels[str(k)] = float(touches.mean())
            hasard[str(k)] = reference
        par_conduite[str(p)] = {
            "heures_actives": int(selection.sum()), "candidats_dans_rayon": m,
            "rappel_top_k": rappels, "hasard_top_k": hasard,
        }
    return {
        "rayon_m": rayon_m, "instants_avec_fuite": int(positifs.sum()),
        "rappel_top_k": {
            str(k): float((sommes[k][positifs] / nombres[positifs]).mean())
            if positifs.any() else None for k in ks},
        "hasard_top_k": {
            str(k): float((aleatoire[k][positifs] / nombres[positifs]).mean())
            if positifs.any() else None for k in ks},
        "par_conduite": par_conduite,
        "convention": "fraction des fuites couvertes, moyenne par instant positif ; "
        "au moins un candidat a distance <= rayon ; egalites : esperance du tirage uniforme",
        "reference_hasard": "k conduites uniformes sans remise parmi les memes candidates",
        "unite": "localisation horaire retrospective, pas detection d'evenements ni interventions",
    }


def debits(prediction: np.ndarray, vrai: np.ndarray, fuite_m3h: np.ndarray,
           seuil_sens_m3h: float) -> dict:
    p, y, f = (np.asarray(a, dtype=np.float64) for a in (prediction, vrai, fuite_m3h))
    if p.ndim != 3 or p.shape != y.shape or p.shape[-1] != 2 or f.shape != p.shape[:2] \
            or p.size == 0 or not all(np.isfinite(a).all() for a in (p, y, f)) \
            or not np.isfinite(seuil_sens_m3h) or seuil_sens_m3h < 0:
        raise ValueError("Q attendus [instants, conduites, 2], fuite [instants, conduites], seuil >=0")
    err = p - y
    eligibles = np.abs(y) > seuil_sens_m3h
    bilan = p[..., 0] - p[..., 1] - f
    return {
        "mae_m3h": float(np.abs(err).mean()), "rmse_m3h": float(np.square(err).mean() ** .5),
        "mae_par_conduite_m3h": np.abs(err).mean((0, 2)).tolist(),
        "rmse_par_conduite_m3h": np.sqrt(np.square(err).mean((0, 2))).tolist(),
        "seuil_sens_m3h": seuil_sens_m3h, "sens_nombre_eligible": int(eligibles.sum()),
        "sens_exactitude": float((np.sign(p[eligibles]) == np.sign(y[eligibles])).mean())
        if eligibles.any() else None,
        "bilan_milieu_mae_m3h": float(np.abs(bilan).mean()),
        "bilan_milieu_rmse_m3h": float(np.square(bilan).mean() ** .5),
    }


def fuites(scores: np.ndarray, actif: np.ndarray, seuil_logit: float,
           ks: tuple[int, ...] = (1, 5, 10)) -> dict:
    s, y = np.asarray(scores), np.asarray(actif)
    if s.ndim != 2 or s.shape != y.shape or not s.size or not np.isfinite(s).all() \
            or not np.isin(y, (0, 1)).all() or not np.isfinite(seuil_logit) \
            or any(k < 1 for k in ks):
        raise ValueError("Scores finis et labels binaires [instants, conduites] requis")
    y = y.astype(bool)
    alarme = s >= seuil_logit
    tp, fp, fn = int((alarme & y).sum()), int((alarme & ~y).sum()), int((~alarme & y).sum())
    sains = ~y.any(1)
    rangs, rappels = [], {k: [] for k in ks}
    for ligne, labels in zip(s, y):
        if not labels.any():
            continue
        # Rang moyen et fraction attendue de top-k : pas de preference d'ID en cas d'egalite.
        for score in ligne[labels]:
            rangs.append(1 + np.count_nonzero(ligne > score)
                         + (np.count_nonzero(ligne == score) - 1) / 2)
        for k in ks:
            n = min(k, len(ligne))
            coupure = np.partition(ligne, len(ligne) - n)[len(ligne) - n]
            hauts, egaux = ligne > coupure, ligne == coupure
            fraction = (n - hauts.sum()) / egaux.sum()
            rappels[k].append(float((labels[hauts].sum()
                                    + fraction * labels[egaux].sum()) / labels.sum()))
    return {
        "seuil_logit": seuil_logit, "vrais_positifs": tp, "faux_positifs": fp, "faux_negatifs": fn,
        "precision": tp / (tp + fp) if tp + fp else None,
        "rappel": tp / (tp + fn) if tp + fn else None,
        "average_precision_micro": float(average_precision_score(y.ravel(), s.ravel()))
        if y.any() else None,
        "rang_moyen_fuites": float(np.mean(rangs)) if rangs else None,
        "rappel_top_k": {str(k): float(np.mean(v)) if v else None for k, v in rappels.items()},
        "instants_sains": int(sains.sum()),
        "taux_instants_sains_avec_alerte": float(alarme[sains].any(1).mean())
        if sains.any() else None,
        "fausses_conduites_par_instant_sain": float(alarme[sains].sum(1).mean())
        if sains.any() else None,
        "convention_egalites": "rang moyen ; rappel top-k fractionnaire attendu",
        "unite_alertes": "classification par instant, pas interventions ni episodes independants",
    }
