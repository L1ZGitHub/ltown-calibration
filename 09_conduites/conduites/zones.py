"""Cartes de suspicion : lissage local et couverture a budget de longueur fixe."""
import numpy as np


def noyau(distances: np.ndarray, longueurs: np.ndarray, rayon_m: float = 300.) -> np.ndarray:
    d, l = np.asarray(distances), np.asarray(longueurs)
    if l.ndim != 1 or not len(l) or not np.isfinite(l).all() or (l <= 0).any() \
            or d.shape != (len(l), len(l)) or np.isnan(d).any() or (d < 0).any() \
            or not np.allclose(d, d.T) or not (np.diag(d) == 0).all() \
            or not np.isfinite(rayon_m) or rayon_m <= 0:
        raise ValueError("Distances symetriques >=0, diagonale nulle et longueurs positives requises")
    w = np.where(d <= rayon_m, np.exp(-d / rayon_m), 0.) * l[None, :]
    return w / w.sum(axis=1, keepdims=True)


def selection_longueur(scores: np.ndarray, longueurs: np.ndarray, budget_m: float,
                       graine: int = 9) -> np.ndarray:
    """Conduites entieres, par score decroissant ; sauter celles depassant le budget restant."""
    s, l = np.asarray(scores), np.asarray(longueurs)
    if s.ndim != 2 or not s.size or l.shape != (s.shape[1],) \
            or not np.isfinite(s).all() or not np.isfinite(l).all() or (l <= 0).any() \
            or not np.isfinite(budget_m) or budget_m < 0:
        raise ValueError("Scores finis [heures, conduites], longueurs >0 et budget >=0 requis")
    permutation = np.random.default_rng(graine).permutation(len(l))
    ordre = permutation[np.argsort(-s[:, permutation], axis=1, kind="stable")]
    masque = np.zeros(s.shape, dtype=bool)
    restant = np.full(len(s), budget_m, dtype=float)
    lignes = np.arange(len(s))
    for colonne in ordre.T:
        ok = l[colonne] <= restant
        masque[lignes[ok], colonne[ok]] = True
        restant[ok] -= l[colonne[ok]]
    return masque


def bilan_zones(selection: np.ndarray, labels: np.ndarray, distances: np.ndarray,
                longueurs: np.ndarray, heures: np.ndarray, rayon_m: float = 300.) -> dict:
    m, y, d, l, h = (np.asarray(v) for v in
                     (selection, labels, distances, longueurs, heures))
    if m.ndim != 2 or not m.size or y.shape != m.shape or not np.isin(y, (0, 1)).all() \
            or m.dtype != bool or l.shape != (m.shape[1],) or not np.isfinite(l).all() \
            or (l <= 0).any() or d.shape != (m.shape[1], m.shape[1]) \
            or np.isnan(d).any() or (d < 0).any() or h.shape != (len(m),) \
            or not np.isfinite(rayon_m) or rayon_m < 0:
        raise ValueError("Selection, labels, distances, longueurs ou heures incompatibles")
    proche = d <= rayon_m
    couverture = (m.astype(np.int32) @ proche.astype(np.int32)) > 0
    nombres = y.sum(1)
    actifs = nombres > 0
    rappels = (couverture & y.astype(bool)).sum(1)[actifs] / nombres[actifs]
    consecutifs = np.diff(h.astype("datetime64[h]")) == np.timedelta64(1, "h")
    intersection = ((m[1:] & m[:-1]) * l).sum(1)[consecutifs]
    union = ((m[1:] | m[:-1]) * l).sum(1)[consecutifs]
    jaccard = np.divide(intersection, union, out=np.ones_like(union), where=union > 0)
    tailles = m @ l
    empreinte = (couverture @ l) / l.sum()
    return {
        "heures": len(m), "heures_avec_fuite": int(actifs.sum()),
        "rappel_300m": float(rappels.mean()) if actifs.any() else None,
        "longueur_selectionnee_m": {
            "moyenne": float(tailles.mean()), "min": float(tailles.min()),
            "max": float(tailles.max())},
        "fraction_reseau_a_300m": {
            "moyenne": float(empreinte.mean()), "min": float(empreinte.min()),
            "max": float(empreinte.max())},
        "stabilite_jaccard_longueur": float(jaccard.mean()) if len(jaccard) else None,
        "paires_heures_consecutives": int(consecutifs.sum()),
        "par_conduite": {
            str(p): float(couverture[y[:, p].astype(bool), p].mean())
            for p in np.flatnonzero(y.any(0))},
    }
