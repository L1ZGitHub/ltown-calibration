"""Diagnostics temporels et comparaison de cartes a empreinte spatiale controlee."""
import numpy as np


def fond_horaire(cartes: np.ndarray, heures: np.ndarray, avant: np.ndarray) -> np.ndarray:
    x, h, a = np.asarray(cartes), np.asarray(heures), np.asarray(avant)
    if x.ndim != 2 or not x.size or h.shape != (len(x),) or a.shape != (len(x),) \
            or a.dtype != bool or not np.isfinite(x).all() or np.isnat(h).any():
        raise ValueError("Cartes finies, heures et masque avant compatibles requis")
    heures_jour = h.astype("datetime64[h]").astype(np.int64) % 24
    fond = np.empty_like(x, dtype=float)
    for heure in range(24):
        ref = a & (heures_jour == heure)
        if not ref.any():
            raise ValueError("Chaque heure du jour doit etre presente dans le fond anterieur")
        fond[heures_jour == heure] = x[ref].mean(0)
    return fond


def couverture_empreinte(scores: np.ndarray, distances: np.ndarray, longueurs: np.ndarray,
                         fractions: tuple[float, ...] = (.10, .20, .30),
                         graine: int = 9) -> tuple[np.ndarray, np.ndarray]:
    """Prefixes classes ; interpolation aleatoire a la frontiere pour empreinte moyenne exacte."""
    s, d, l = np.asarray(scores), np.asarray(distances), np.asarray(longueurs)
    if s.ndim != 2 or not s.size or not np.isfinite(s).all() or l.shape != (s.shape[1],) \
            or not np.isfinite(l).all() or (l <= 0).any() or d.shape != (len(l), len(l)) \
            or np.isnan(d).any() or (d < 0).any() or not np.allclose(d, d.T) \
            or not (np.diag(d) == 0).all() or not fractions \
            or any(not np.isfinite(f) or not 0 < f <= 1 for f in fractions):
        raise ValueError("Scores, distances, longueurs ou fractions d'empreinte invalides")
    permutation = np.random.default_rng(graine).permutation(len(l))
    ordre = permutation[np.argsort(-s[:, permutation], axis=1, kind="stable")]
    voisinage = d <= 300.
    couvertures = np.empty((len(fractions), len(s), len(l)), dtype=np.float32)
    poids_frontiere = np.empty((len(fractions), len(s)), dtype=float)
    for t, classement in enumerate(ordre):
        couvert = np.zeros(len(l), dtype=bool)
        longueur = 0.
        restants = set(range(len(fractions)))
        for p in classement:
            nouveau = couvert | voisinage[p]
            longueur_nouvelle = float(l[nouveau].sum())
            for j in tuple(restants):
                cible = fractions[j] * l.sum()
                if longueur_nouvelle >= cible:
                    poids = (cible - longueur) / (longueur_nouvelle - longueur)
                    couvertures[j, t] = couvert.astype(float) + poids * (nouveau & ~couvert)
                    poids_frontiere[j, t] = poids
                    restants.remove(j)
            if not restants:
                break
            couvert, longueur = nouveau, longueur_nouvelle
        if restants:
            raise ValueError("Empreinte demandee inaccessible")
    return couvertures, poids_frontiere
