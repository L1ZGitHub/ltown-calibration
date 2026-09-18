"""Génération de scénarios de fuite étiquetés, à partir du modèle calibré.

Le modèle calibré sait reproduire les 33 capteurs de pression à quelques centimètres près. On
peut donc s'en servir à l'envers : percer une conduite qu'on choisit, à un instant qu'on choisit,
d'un débit qu'on choisit, et enregistrer ce que les capteurs auraient vu. La vérité terrain est
alors exacte par construction — c'est nous qui avons posé la fuite.

**Une fuite est un émetteur.** EPANET écrit `Q = C·p^0,5` : le débit dépend de la pression au
nœud, donc on ne fixe pas un débit mais un coefficient. `reseau.coefficient_emetteur` calcule le
coefficient qui donne le débit visé sous une pression de référence ; le débit réellement délivré
varie ensuite avec la pression, et c'est lui qu'on enregistre comme étiquette.

**On ne simule que ce qui change.** Une fenêtre sans fuite est simulée une fois et sert de témoin,
d'avant-fuite et d'après-réparation. Chaque fuite ne coûte donc que la durée pendant laquelle elle
coule, et non la fenêtre entière. C'est ce qui rend un lot de plusieurs dizaines de scénarios
abordable sur une machine ordinaire.

**Le découpage temporel n'est pas libre.** Le simulateur repart du niveau lu au capteur à chaque
tranche (`niveau_mesure`), donc recoller deux morceaux n'est exact que si la coupure tombe sur une
frontière de tranche. `PAS_TRANCHE` impose ce grain aux dates de début et de fin.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import reseau as R

#: Grain temporel des scénarios : les fuites s'ouvrent et se referment sur une frontière de
#: tranche de 6 h, le pas de réancrage du réservoir retenu par la calibration.
PAS_TRANCHE = R.PAS_JOUR // 4


@dataclass(frozen=True)
class Fuite:
    """Une fuite posée sur une conduite, entre deux pas de la fenêtre.

    `debit_m3h` est le débit **visé sous la pression de référence** du nœud percé, pas le débit
    délivré : celui-ci suit la pression et figure dans l'étiquette sous `debit_reel_m3h`.

    `profil` vaut `"abrupt"` (le débit s'établit d'un coup, une casse) ou `"progressif"` (il monte
    linéairement de zéro au débit visé sur toute la durée, une fuite naissante). Le second est
    découpé en `paliers` segments de coefficient constant.
    """

    conduite: str
    debut: int
    fin: int | None = None
    debit_m3h: float = 10.0
    profil: str = "abrupt"
    paliers: int = 4

    def __post_init__(self):
        if self.profil not in ("abrupt", "progressif"):
            raise ValueError(f"profil inconnu : {self.profil!r}")
        if self.debut % PAS_TRANCHE:
            raise ValueError(f"début {self.debut} hors frontière de tranche ({PAS_TRANCHE})")
        if self.fin is not None:
            if self.fin % PAS_TRANCHE:
                raise ValueError(f"fin {self.fin} hors frontière de tranche ({PAS_TRANCHE})")
            if self.fin <= self.debut:
                raise ValueError(f"fin {self.fin} avant début {self.debut}")


def noeud_perce(wn, conduite: str, noeuds: list[str]) -> str:
    """Le nœud où l'on pose l'émetteur : l'extrémité amont de la conduite qui est une jonction.

    Une fuite est physiquement sur la conduite ; EPANET ne sait la poser qu'à un nœud. On prend
    l'extrémité disponible, ce qui décale la fuite d'une demi-longueur de conduite au plus —
    négligeable devant l'espacement des capteurs.
    """
    lien = wn.get_link(conduite)
    for n in (lien.start_node_name, lien.end_node_name):
        if n in noeuds:
            return n
    raise ValueError(f"aucune extrémité de {conduite} n'est une jonction du modèle")


def _tranches(tableaux: dict, a: int, b: int) -> dict:
    """Découpe les arguments temporels du simulateur sur [a, b], bornes du profil comprises."""
    return {k: (v[a:b + 1] if isinstance(v, np.ndarray) else v) for k, v in tableaux.items()}


def fenetre_propre(D, noeuds: list[str], n_pas: int, **kw) -> np.ndarray:
    """Simule la fenêtre **sans aucune fuite** et renvoie la pression aux 782 jonctions.

    C'est le témoin du lot : il sert de référence pour dimensionner les émetteurs, de portion
    avant-fuite et après-réparation, et d'exemple négatif pour un détecteur.
    """
    kw.pop("tranche_jours", None)          # imposé par PAS_TRANCHE, voir le docstring du module
    morceaux = R.simuler(
        D, noeuds, n_pas,
        lambda res, wn, a, b: res.node["pressure"][noeuds].to_numpy()[:b - a].astype(np.float32),
        tranche_jours=PAS_TRANCHE / R.PAS_JOUR, **kw)
    return np.vstack(morceaux)


def poser(D, noeuds: list[str], n_pas: int, wn, fuite: Fuite, propre: np.ndarray,
          **kw) -> tuple[np.ndarray, np.ndarray]:
    """Rejoue la fenêtre avec `fuite` ouverte, en ne simulant que la durée où elle coule.

    Renvoie `(pressions, debit_reel)` : les pressions aux 782 jonctions sur toute la fenêtre, et
    le débit réellement soutiré au nœud percé, pas par pas, en m³/h (zéro hors de la fuite).
    """
    kw.pop("tranche_jours", None)          # imposé par PAS_TRANCHE, voir le docstring du module
    noeud = noeud_perce(wn, fuite.conduite, noeuds)
    k = noeuds.index(noeud)
    fin = n_pas if fuite.fin is None else min(fuite.fin, n_pas)
    if fuite.debut >= fin:
        raise ValueError("la fuite ne coule à aucun pas de la fenêtre")

    # La pression de référence est celle du nœud sain, moyennée sur la durée de la fuite.
    p_ref = float(np.mean(propre[fuite.debut:fin, k]))
    if fuite.profil == "abrupt":
        bornes = [(fuite.debut, fin, fuite.debit_m3h)]
    else:
        arretes = np.linspace(fuite.debut, fin, fuite.paliers + 1)
        arretes = np.clip(np.round(arretes / PAS_TRANCHE) * PAS_TRANCHE, fuite.debut, fin)
        bornes = [(int(a), int(b), fuite.debit_m3h * (j + 1) / fuite.paliers)
                  for j, (a, b) in enumerate(zip(arretes[:-1], arretes[1:])) if b > a]

    sortie = np.array(propre, copy=True)
    debit = np.zeros(n_pas, dtype=np.float32)
    for a, b, q in bornes:
        c = R.coefficient_emetteur(q, p_ref)
        morceaux = R.simuler(
            D[a:b + 1], noeuds, b - a,
            lambda res, wn_, i, j: (
                res.node["pressure"][noeuds].to_numpy()[:j - i].astype(np.float32),
                res.node["demand"][noeud].to_numpy()[:j - i].astype(np.float32)),
            tranche_jours=PAS_TRANCHE / R.PAS_JOUR, fuites={noeud: c},
            **_tranches(kw, a, b))
        sortie[a:b] = np.vstack([m[0] for m in morceaux])
        # le soutirage total du nœud, moins la demande du modèle, est le débit de la fuite
        total = np.concatenate([m[1] for m in morceaux]) * 3600.0
        debit[a:b] = np.maximum(total - np.asarray(D[a:b, k], dtype=np.float32), 0.0)
    return sortie, debit


def etiquette(wn, noeuds: list[str], fuite: Fuite, debit: np.ndarray, n_pas: int) -> dict:
    """La vérité terrain d'un scénario, sous une forme lisible sans le code qui l'a produite."""
    noeud = noeud_perce(wn, fuite.conduite, noeuds)
    lien = wn.get_link(fuite.conduite)
    coule = debit > 0
    return {
        **asdict(fuite),
        "fin": n_pas if fuite.fin is None else fuite.fin,
        "noeud_perce": noeud,
        "longueur_m": float(lien.length),
        "diametre_mm": float(lien.diameter * 1000),
        "pas_avec_fuite": int(coule.sum()),
        "debit_reel_m3h": float(debit[coule].mean()) if coule.any() else 0.0,
        "debit_reel_max_m3h": float(debit.max()),
        "volume_perdu_m3": float(debit.sum() * R.PAS_MIN / 60.0),
    }


def ecrire(dossier: Path, nom: str, capteurs: list[str], pressions: np.ndarray,
           verite: dict | None, horo: pd.DatetimeIndex, arrondi_cm: bool = True) -> Path:
    """Écrit un scénario : `pressions.parquet` aux 33 capteurs, et `verite.json` s'il y a une fuite.

    `arrondi_cm` reproduit la résolution des capteurs réels de L-Town, qui ne publient que deux
    décimales. Un détecteur entraîné sur des pressions en flottant libre apprendrait un signal qui
    n'existe pas dans les mesures.
    """
    d = dossier / nom
    d.mkdir(parents=True, exist_ok=True)
    p = np.round(pressions, 2) if arrondi_cm else pressions
    pd.DataFrame(p, index=horo[:len(p)], columns=capteurs).to_parquet(d / "pressions.parquet")
    if verite is not None:
        (d / "verite.json").write_text(json.dumps(verite, indent=2, ensure_ascii=False))
    return d


def generer(D, noeuds: list[str], wn, n_pas: int, fuites: list[Fuite], horo: pd.DatetimeIndex,
            dossier: str | Path, *, temoin: str = "temoin", propre: np.ndarray | None = None,
            journal: bool = True, **kw) -> pd.DataFrame:
    """Produit un lot complet : un témoin sans fuite, puis un dossier par fuite, plus l'index.

    Le témoin n'est simulé qu'une fois et sert de référence à toutes les fuites de la fenêtre ;
    `propre` permet de réutiliser un témoin déjà calculé au lieu de le refaire.
    Renvoie l'index du lot, également écrit en `index.csv`.

    `journal` commande l'avancement scénario par scénario ; le `verbeux` du simulateur, qui
    détaille chaque tranche, se passe dans `kw` et reste indépendant des deux.
    """
    dossier = Path(dossier)
    capteurs = R.capteurs()["pressure"]
    colonnes = [noeuds.index(c) for c in capteurs]

    if propre is None:
        if journal:
            print(f"témoin : {n_pas} pas sans fuite", flush=True)
        propre = fenetre_propre(D, noeuds, n_pas, **kw)
    ecrire(dossier, temoin, capteurs, propre[:, colonnes], None, horo)

    lignes = [{"scenario": temoin, "fuite": False, "conduite": "", "debut": 0, "fin": n_pas,
               "debit_m3h": 0.0, "profil": "", "debit_reel_m3h": 0.0, "volume_perdu_m3": 0.0}]
    for i, f in enumerate(fuites, 1):
        nom = f"scenario_{i:04d}"
        if journal:
            print(f"{nom} : {f.conduite}, {f.debit_m3h:.1f} m³/h, {f.profil}", flush=True)
        pressions, debit = poser(D, noeuds, n_pas, wn, f, propre, **kw)
        v = etiquette(wn, noeuds, f, debit, n_pas)
        ecrire(dossier, nom, capteurs, pressions[:, colonnes], v, horo)
        lignes.append({"scenario": nom, "fuite": True,
                       **{c: v[c] for c in ("conduite", "debut", "fin", "debit_m3h", "profil",
                                            "debit_reel_m3h", "volume_perdu_m3")}})

    index = pd.DataFrame(lignes)
    index.to_csv(dossier / "index.csv", index=False)
    return index
