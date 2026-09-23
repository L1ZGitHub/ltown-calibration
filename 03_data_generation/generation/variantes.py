"""Des variantes du modèle calibré : ce qu'on ne connaît pas, tiré au hasard autour de lui.

Le modèle calibré est une estimation. Plusieurs choses y restent incertaines, et une variante en
tire une valeur plausible. Quatre familles, qu'on peut activer séparément :

  conduites     longueur, diamètre et rugosité de chaque conduite, multipliés par un facteur
                tiré dans ±plage. Par défaut ±3 % : c'est la plage dont l'effet aux capteurs
                égale le décalage réel du modèle calibré (1 à 2 cm, mesuré hors de la fenêtre
                d'ajustement, fuites publiées posées). Le carnet b le mesure.
  composition   la part de chaque type de consommateur (résidentiel, commercial) dans les nœuds
                sans compteur. Le volume total ne bouge pas : il est connu par les compteurs.
  decalage      chaque nœud sans compteur consomme un peu plus tôt ou plus tard (±30 min).
  bruit         la consommation d'un nœud n'est pas lisse. Écart relatif réglé pour que les
                capteurs sautillent autant que les vrais d'un pas de 5 min à l'autre (carnet b) ;
                corrélation entre nœuds mesurée sur les 82 compteurs de 2018.

Ce qu'on ne tire jamais : l'altitude (non modifiée par les auteurs du jeu), les nœuds équipés d'un
compteur (leur demande est mesurée), la pompe et le niveau du réservoir (mesurés eux aussi).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import wntr

from . import modele as M
from .modele import R

FAMILLES = ("conduites", "composition", "decalage", "bruit")
PLAGES = {"longueur": 0.03, "diametre": 0.03, "rugosite": 0.03, "composition": 0.20}
DECALAGE_MAX_MIN = 30
CV_BRUIT = 0.40         # écart relatif de la consommation d'un nœud autour de sa forme
RHO_BRUIT = 0.013       # corrélation moyenne entre deux compteurs


@dataclass
class Variante:
    """Un modèle tiré. `kw` se passe tel quel à `modele.simuler` et `scenarios.simuler_fuites`."""
    graine: int
    familles: tuple
    D: np.ndarray
    coefficients: dict
    inp: Path | None = None
    tirages: dict = field(default_factory=dict)       # les facteurs tirés, pour les regarder

    @property
    def kw(self) -> dict:
        return {"D": self.D, "coefficients": self.coefficients, "inp": self.inp}


def _facteur(rng, plage: float, taille):
    """Facteurs autour de 1, uniformes en log entre 1-plage et 1+plage."""
    return np.exp(rng.uniform(np.log(1 - plage), np.log(1 + plage), taille))


def tirer(fen: M.Fenetre, graine: int, familles=FAMILLES, plages=None,
          dossier: Path | None = None, cv_bruit: float = CV_BRUIT) -> Variante:
    """Tire une variante du modèle calibré de la fenêtre `fen`."""
    inconnues = set(familles) - set(FAMILLES)
    if inconnues:
        raise ValueError(f"familles inconnues : {inconnues}, attendu parmi {FAMILLES}")
    plages = {**PLAGES, **(plages or {})}
    rng = np.random.default_rng(graine)
    libres = ~fen.mesure                       # les 700 nœuds sans compteur
    D = fen.D.copy()
    coefs, inp, tirages = fen.coefficients, None, {}

    if "conduites" in familles:
        tuyaux = fen.wn.pipe_name_list
        for nom in ("longueur", "diametre", "rugosite"):
            tirages[nom] = _facteur(rng, plages[nom], len(tuyaux))
        coefs = {p: fen.coefficients[p] * f for p, f in zip(tuyaux, tirages["rugosite"])}
        wn = R.charger_modele()
        for p, fl, fd in zip(tuyaux, tirages["longueur"], tirages["diametre"]):
            wn.get_link(p).length *= float(fl)
            wn.get_link(p).diameter *= float(fd)
        dossier = Path(dossier or M.SORTIES / "variantes")
        dossier.mkdir(parents=True, exist_ok=True)
        inp = dossier / f"variante_{graine:04d}.inp"
        wntr.network.io.write_inpfile(wn, str(inp))

    if "composition" in familles:
        # résidentiel et commercial tirés, industriel fixe ; puis on rend le volume d'origine
        m = np.array([*_facteur(rng, plages["composition"], 2), 1.0], dtype=np.float32)
        tirages["composition"] = dict(zip(R.CATEGORIES, m))
        nouveau = (fen.formes * m) @ fen.bases[libres].T
        D[:, libres] = nouveau * (D[:, libres].sum() / nouveau.sum())

    if "decalage" in familles:
        pas_max = DECALAGE_MAX_MIN // R.PAS_MIN
        s = rng.integers(-pas_max, pas_max + 1, libres.sum())
        tirages["decalage_min"] = s * R.PAS_MIN
        t = np.arange(len(D))[:, None]
        D[:, libres] = np.take_along_axis(D[:, libres], np.clip(t - s, 0, len(D) - 1), axis=0)

    if "bruit" in familles:
        n = int(libres.sum())
        commun = rng.standard_normal((len(D), 1))
        propre = rng.standard_normal((len(D), n))
        bruit = 1 + cv_bruit * (np.sqrt(RHO_BRUIT) * commun + np.sqrt(1 - RHO_BRUIT) * propre)
        D[:, libres] *= np.maximum(bruit, 0.0).astype(D.dtype)

    return Variante(graine=graine, familles=tuple(familles), D=D, coefficients=coefs, inp=inp,
                    tirages=tirages)


def ecarts(fen: M.Fenetre, reference: np.ndarray, variantes: list[np.ndarray]) -> dict:
    """Ce que les variantes changent aux 33 capteurs, par rapport à la référence.

    statique   la moyenne dans le temps de l'écart (un décalage constant par capteur)
    dynamique  ce qui bouge autour de ce décalage

    Chacun est une moyenne quadratique sur les capteurs et les variantes, en mètres.
    """
    E = np.stack([v[:, fen.colonnes] - reference[:, fen.colonnes] for v in variantes])
    statique = E.mean(axis=1)                         # (variantes, 33)
    dynamique = E - statique[:, None, :]
    return {"statique_m": float(np.sqrt((statique ** 2).mean())),
            "dynamique_m": float(np.sqrt((dynamique ** 2).mean()))}
