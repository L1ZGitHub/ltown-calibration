"""Mesures bruitees par episode, avant decoupage en fenetres causales."""
import numpy as np

from .sources import Annees, M


def provenance_bruit() -> dict:
    r = Annees.G
    return {
        "producteur": "04bis_detection_localisationv2.etapes.e12_annees_simulees.bruit_capteurs",
        "derive_pression_cm": r.DERIVE_CAPTEUR_CM,
        "derive_tau_h": r.DERIVE_CAPTEUR_TAU_H,
        "arrondi_pression_m": 0.01,
        "debitmetres_ordre": list(M.DEBITMETRES),
        "bruit_debitmetres_m3h": dict(r.BRUIT_DEBITMETRES_M3H),
        "bruit_niveau_m": r.BRUIT_NIVEAU_M,
        "pannes_artificielles": False,
        "bruit_AMR_additionnel": False,
    }


def observer(verite: dict[str, np.ndarray], graine: int) -> dict[str, np.ndarray]:
    """Copie uniquement les observations autorisees ; aucune cible n'est renvoyee."""
    formes = {"capteurs": (33,), "debits": (3,), "niveau": (), "amr": (82,), "pompe": ()}
    if set(formes) - verite.keys():
        raise ValueError(f"Observations absentes : {sorted(set(formes) - verite.keys())}")
    n = len(verite["capteurs"])
    if n < 1:
        raise ValueError("Episode vide")
    mesures = {}
    for cle, forme in formes.items():
        x = np.asarray(verite[cle])
        if x.shape != (n, *forme) or not np.isfinite(x).all():
            raise ValueError(f"{cle} : forme attendue {(n, *forme)} et valeurs finies requises")
        mesures[cle] = x.copy()
    if not np.isin(mesures["pompe"], (0, 1)).all():
        raise ValueError("Etat de pompe attendu : 0 ou 1")
    bruitees = Annees.bruit_capteurs(np.random.default_rng(graine), mesures)
    return {
        **bruitees,
        **{f"{cle}_valide": np.ones(x.shape, dtype=bool) for cle, x in bruitees.items()},
    }
