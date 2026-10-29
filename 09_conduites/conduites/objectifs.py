"""Pertes A/B/C et echelles de reference ajustees sur les seules cibles train."""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F

from .donnees import Episode, Normalisation, verifier_partitions


@dataclass(frozen=True)
class ReglagePertes:
    poids_positif: float
    reference_q: float
    reference_fuite: float
    episodes_train: list[str]
    positifs: int
    negatifs: int

    @classmethod
    def ajuster(cls, episodes: list[Episode], norm: Normalisation) -> "ReglagePertes":
        verifier_partitions(episodes)
        if any(e.partition != "train" for e in episodes):
            raise ValueError("Reglage des pertes sur train uniquement")
        if set(norm.episodes_train) != {e.identifiant for e in episodes}:
            raise ValueError("Provenances train de normalisation et pertes differentes")
        positifs = negatifs = nombre_q = 0
        somme_q = 0.
        for e in episodes:
            if "fuite_active" not in e.cibles:
                raise ValueError("Labels issus du scenario absents ; aucun seuil implicite autorise")
            # Meme population d'instants que Fenetres, pas les onze pas d'amorcage.
            y = e.cibles["fuite_active"][11:]
            positifs += int(y.sum())
            negatifs += int(y.size - y.sum())
            q = e.cibles["q_m3h"][11:].astype(np.float64) / norm.echelle_q
            somme_q += float(np.square(q).sum())
            nombre_q += q.size
        if positifs == 0 or negatifs == 0 or somme_q == 0:
            raise ValueError("Train avec positifs, negatifs et Q non tous nuls requis")
        poids = negatifs / positifs
        # Reference Q : prediction nulle. Reference BCE : logits nuls, p=0.5.
        return cls(poids, somme_q / nombre_q,
                   float(np.log(2) * (negatifs + poids * positifs) / (negatifs + positifs)),
                   [e.identifiant for e in episodes], positifs, negatifs)

    def configuration(self) -> dict:
        return {"schema": 1, **asdict(self), "reference_q_definition": "prediction Q nulle",
                "reference_fuite_definition": "logits nuls, BCE ponderee reduction mean",
                "poids_conjoint": {"q": 0.5, "fuite": 0.5}}


def perte(sorties: dict[str, torch.Tensor], cibles: dict[str, torch.Tensor],
          objectif: str, reglage: ReglagePertes) -> dict[str, torch.Tensor]:
    if objectif not in ("A", "B", "C"):
        raise ValueError(f"Objectif inconnu : {objectif}")
    termes = {}
    for besoin, prediction, cible in (
        (objectif in ("A", "C"), "q_m3h_normalise", "q_normalise"),
        (objectif in ("B", "C"), "fuite_logits", "fuite_active"),
    ):
        if besoin:
            p, y = sorties[prediction], cibles[cible]
            if p.shape != y.shape or p.numel() == 0 or not torch.isfinite(p).all() \
                    or not torch.isfinite(y).all():
                raise ValueError(f"Formes ou valeurs invalides pour {prediction}")
            if cible == "q_normalise":
                termes["q_brute"] = F.mse_loss(p, y)
                termes["q_equilibree"] = termes["q_brute"] / reglage.reference_q
            else:
                if not ((y == 0) | (y == 1)).all():
                    raise ValueError("Labels binaires requis pour BCE")
                termes["fuite_brute"] = F.binary_cross_entropy_with_logits(
                    p, y, pos_weight=p.new_tensor(reglage.poids_positif))
                termes["fuite_equilibree"] = termes["fuite_brute"] / reglage.reference_fuite
    termes["totale"] = (termes["q_equilibree"] if objectif == "A" else
                        termes["fuite_equilibree"] if objectif == "B" else
                        0.5 * (termes["q_equilibree"] + termes["fuite_equilibree"]))
    return termes
