"""C avec conservation : les débits prédits comme écart au modèle calibré sans fuite, et les bilans.

La tête « débits » du prototype prédit, pour chaque demi-conduite, Δq = q − q_référence (le débit du
modèle calibré, une entrée du modèle), à l'échelle de Δq sur l'apprentissage. Trois termes, poids
égaux, comparés chacun à la prédiction nulle (la convention de `objectifs`) :
- Δq lui-même ;
- le bilan aux jonctions : entrées − sorties des demi-conduites, prédit contre vrai (le vrai bilan
  est la demande vraie moins celle du modèle calibré) ; seules les jonctions dont tous les liens sont
  des conduites (pas la pompe, les vannes, les réservoirs) ;
- le bilan aux milieux : q1 − q2, prédit contre vrai, c'est-à-dire le débit de la fuite.
Le total : 0,5 × fuite (la BCE de B) + 0,5 × moyenne des trois termes de débit.
"""
from dataclasses import asdict, dataclass

import numpy as np
import torch
from torch.nn import functional as F


def indices_reference(schema: dict, conduites) -> np.ndarray:
    """(conduites, 2) : la colonne des deux demi-conduites dans `hydraulique_liens`."""
    li = {n: i for i, n in enumerate(schema["liens"])}
    return np.array([[li[c.segment_1], li[c.segment_2]] for c in conduites])


def colonne_reference(schema: dict) -> int:
    return [c[0] for c in schema["champs_liens"]].index("debit_reference_m3h")


def jonctions(wn, conduites) -> tuple[list[str], torch.Tensor, torch.Tensor]:
    """Les jonctions retenues et, pour chaque conduite, l'indice de sa jonction de début et de fin
    (len(retenues) = case poubelle si la jonction n'est pas retenue)."""
    tuyaux = set(wn.pipe_name_list)
    retenues = [n for n in wn.junction_name_list if not n.startswith("m09_")
                and all(x in tuyaux for x in wn.get_links_for_node(n))]
    ni = {n: i for i, n in enumerate(retenues)}
    poubelle = len(retenues)
    debut = torch.tensor([ni.get(c.debut, poubelle) for c in conduites])
    fin = torch.tensor([ni.get(c.fin, poubelle) for c in conduites])
    return retenues, debut, fin


def bilans(q: torch.Tensor, debut: torch.Tensor, fin: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor]:
    """q (lots, conduites, 2), début → milieu → fin : (entrées − sorties aux jonctions, q1 − q2)."""
    z = q.new_zeros(q.shape[0], n + 1)
    z = z.index_add(1, fin, q[..., 1]).index_add(1, debut, -q[..., 0])
    return z[:, :n], q[..., 0] - q[..., 1]


@dataclass(frozen=True)
class ReglageConservation:
    echelle_dq: float
    reference_dq: float
    reference_jonctions: float
    reference_milieux: float
    jonctions: list[str]

    @classmethod
    def ajuster(cls, episodes, retenues, debut, fin) -> "ReglageConservation":
        """Sur l'apprentissage seulement, mêmes instants que les fenêtres (pas 11 et suivants)."""
        dq = [(e.cibles["q_m3h"][11:] - e.cibles["q_reference_m3h"][11:]).astype(np.float64) for e in episodes]
        echelle = max(float(np.sqrt(np.mean([np.square(d).mean() for d in dq]))), 1e-6)
        sj = sm = 0.
        nj = nm = 0
        for d in dq:
            j, m = bilans(torch.from_numpy(d / echelle), debut, fin, len(retenues))
            sj += float(j.square().sum())
            nj += j.numel()
            sm += float(m.square().sum())
            nm += m.numel()
        return cls(echelle, 1.0, max(sj / nj, 1e-6), max(sm / nm, 1e-6), list(retenues))

    def configuration(self) -> dict:
        return {**{k: v for k, v in asdict(self).items() if k != "jonctions"},
                "jonctions_retenues": len(self.jonctions),
                "definition": "Δq = q − q_reference, a l'echelle echelle_dq ; references = prediction nulle",
                "poids": {"fuite": 0.5, "dq": 0.5 / 3, "jonctions": 0.5 / 3, "milieux": 0.5 / 3}}


def perte_conservation(sorties: dict, lot: dict, echelle_q: float, reglage_fuite, rc: ReglageConservation,
                       debut: torch.Tensor, fin: torch.Tensor) -> dict[str, torch.Tensor]:
    p = sorties["q_m3h_normalise"]                           # Δq / echelle_dq
    y = (lot["q_normalise"] * echelle_q - lot["q_reference_m3h"]) / rc.echelle_dq
    if p.shape != y.shape or not torch.isfinite(p).all() or not torch.isfinite(y).all():
        raise ValueError("Formes ou valeurs invalides pour Δq")
    n = len(rc.jonctions)
    jp, mp = bilans(p, debut, fin, n)
    jy, my = bilans(y, debut, fin, n)
    z = sorties["fuite_logits"]
    t = {"dq": F.mse_loss(p, y) / rc.reference_dq,
         "jonctions": F.mse_loss(jp, jy) / rc.reference_jonctions,
         "milieux": F.mse_loss(mp, my) / rc.reference_milieux,
         "fuite_brute": F.binary_cross_entropy_with_logits(
             z, lot["fuite_active"], pos_weight=z.new_tensor(reglage_fuite.poids_positif))}
    t["fuite_equilibree"] = t["fuite_brute"] / reglage_fuite.reference_fuite
    t["totale"] = 0.5 * t["fuite_equilibree"] + 0.5 * (t["dq"] + t["jonctions"] + t["milieux"]) / 3
    return t
