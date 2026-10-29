"""Episodes, normalisation train seulement et fenetres sans franchir leurs bords."""
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


@dataclass
class Episode:
    identifiant: str
    variante: str
    partition: str
    entrees: dict[str, np.ndarray]
    cibles: dict[str, np.ndarray]
    schema: dict

    @classmethod
    def lire(cls, dossier: Path, variante: str, partition: str) -> "Episode":
        rapport = json.loads((dossier / "rapport.json").read_text())
        with np.load(dossier / "entrees.npz") as z:
            entrees = {k: z[k] for k in ("hydraulique_noeuds", "hydraulique_liens")}
        with np.load(dossier / "cibles.npz") as z:
            cibles = {k: z[k] for k in ("q_m3h", "fuite_m3h", "fuite_active") if k in z}
        # Fuites qui commencent et s'arretent : le calendrier est ecrit par l'episode.
        if "fuite_active" in cibles:
            cibles["fuite_active"] = cibles["fuite_active"].astype(np.float32)
        # Contrat du pilote : emetteurs actifs pendant tout l'episode.
        elif "fuites_visees_m3h" in rapport:
            noms = [c["nom"] for c in rapport["conduites"]]
            actifs = np.array([nom in rapport["fuites_visees_m3h"] for nom in noms],
                              dtype=np.float32)
            cibles["fuite_active"] = np.broadcast_to(actifs, cibles["fuite_m3h"].shape).copy()
        return cls(str(dossier.resolve()), variante, partition, entrees, cibles,
                   rapport["schema_entrees"])

    def verifier(self) -> None:
        if self.partition not in ("train", "validation", "test"):
            raise ValueError(f"Partition inconnue : {self.partition}")
        n = len(self.cibles["q_m3h"])
        p = self.cibles["q_m3h"].shape[1]
        attendus = {
            "hydraulique_noeuds": (n, len(self.schema["noeuds"]), len(self.schema["champs_noeuds"])),
            "hydraulique_liens": (n, len(self.schema["liens"]), len(self.schema["champs_liens"])),
            "q_m3h": (n, p, 2), "fuite_m3h": (n, p),
            "fuite_active": (n, p),
        }
        if n < 12:
            raise ValueError("Un episode doit contenir au moins 12 observations")
        for cle, x in {**self.entrees, **self.cibles}.items():
            if cle in attendus and (x.shape != attendus[cle] or not np.isfinite(x).all()):
                raise ValueError(f"Forme ou valeurs invalides : {cle}, attendu {attendus[cle]}")
        if "fuite_active" in self.cibles and not np.isin(self.cibles["fuite_active"], (0, 1)).all():
            raise ValueError("Labels de fuite non binaires")


def verifier_partitions(episodes: list[Episode]) -> None:
    if not episodes:
        raise ValueError("Aucun episode")
    ids: set[str] = set()
    variantes: dict[str, str] = {}
    schema = episodes[0].schema
    for e in episodes:
        e.verifier()
        if e.identifiant in ids:
            raise ValueError(f"Episode duplique : {e.identifiant}")
        ids.add(e.identifiant)
        if e.variante in variantes and variantes[e.variante] != e.partition:
            raise ValueError(f"Variante presente dans plusieurs partitions : {e.variante}")
        variantes[e.variante] = e.partition
        if e.schema != schema:
            raise ValueError("Identifiants, ordre ou schemas differents entre episodes")


def masque(x: np.ndarray, champs: list, colonne: int) -> np.ndarray:
    noms = [nom for nom, _ in champs]
    nom, unite = champs[colonne]
    if unite == "bool":
        v = x[..., colonne]
        if not np.isin(v, (0, 1)).all():
            raise ValueError(f"Masque non binaire : {nom}")
    dependances = {
        "pression_m": ("pression_presente", "pression_valide"),
        "residu_pression_m": ("pression_presente", "pression_valide"),
        "dual_m3h": ("pression_presente", "pression_valide"),
        "variation_dual_m3h": ("pression_presente", "variation_dual_valide"),
        "amr_m3h": ("amr_present", "amr_valide"),
        "niveau_T1_m": ("niveau_present", "niveau_valide"),
        "debit_mesure_m3h": ("debit_present", "debit_valide"),
        "etat_pompe": ("pompe_presente", "pompe_valide"),
    }
    ok = np.ones(x.shape[:-1], dtype=bool)
    for dep in dependances.get(nom, ()):
        ok &= x[..., noms.index(dep)] == 1
    return ok


@dataclass
class Normalisation:
    schema: dict
    statistiques: dict
    echelle_q: float
    episodes_train: list[str]

    @classmethod
    def ajuster(cls, episodes: list[Episode]) -> "Normalisation":
        verifier_partitions(episodes)
        if any(e.partition != "train" for e in episodes):
            raise ValueError("La normalisation accepte uniquement des episodes train")
        schema = episodes[0].schema
        stats = {}
        for cle, champs in (("hydraulique_noeuds", schema["champs_noeuds"]),
                            ("hydraulique_liens", schema["champs_liens"])):
            centres, ecarts = [], []
            for k, (_, unite) in enumerate(champs):
                valeurs = []
                for e in episodes:
                    x = e.entrees[cle]
                    ok = masque(x, champs, k)
                    valeurs.append(x[..., k][ok].astype(np.float64))
                v = np.concatenate(valeurs)
                if unite == "bool":
                    centre, ecart = 0., 1.
                elif not len(v):
                    raise ValueError(f"Aucune observation train pour {cle} colonne {k}")
                else:
                    centre, ecart = float(v.mean()), float(v.std())
                    if ecart < 1e-6:
                        ecart = 1.
                centres.append(centre)
                ecarts.append(ecart)
            stats[cle] = {"centre": centres, "ecart": ecarts}
        somme = sum(float(np.square(e.cibles["q_m3h"].astype(np.float64)).sum()) for e in episodes)
        nombre = sum(e.cibles["q_m3h"].size for e in episodes)
        return cls(schema, stats, max((somme / nombre) ** 0.5, 1e-6),
                   [e.identifiant for e in episodes])

    def transformer(self, e: Episode) -> dict[str, np.ndarray]:
        e.verifier()
        if e.schema != self.schema:
            raise ValueError("Schema incompatible avec la normalisation")
        sorties = {}
        for cle, champs in (("hydraulique_noeuds", self.schema["champs_noeuds"]),
                            ("hydraulique_liens", self.schema["champs_liens"])):
            x = e.entrees[cle]
            s = self.statistiques[cle]
            y = ((x - np.array(s["centre"])) / np.array(s["ecart"])).astype(np.float32)
            for k in range(len(champs)):
                y[..., k] = np.where(masque(x, champs, k), y[..., k], 0.)
            sorties[cle] = y
        return sorties

    def sauver(self, chemin: Path) -> None:
        if chemin.exists():
            raise FileExistsError(chemin)
        chemin.write_text(json.dumps({
            "schema_version": 1, "schema": self.schema, "statistiques": self.statistiques,
            "echelle_q_m3h": self.echelle_q, "episodes_train": self.episodes_train,
            "centrage_q": 0., "convention": "debut -> milieu -> fin",
        }, indent=2) + "\n")

    @classmethod
    def lire(cls, chemin: Path) -> "Normalisation":
        r = json.loads(chemin.read_text())
        if r["schema_version"] != 1 or r["centrage_q"] != 0:
            raise ValueError("Format de normalisation inconnu")
        return cls(r["schema"], r["statistiques"], r["echelle_q_m3h"], r["episodes_train"])


class Fenetres(Dataset):
    def __init__(self, episodes: list[Episode], normalisation: Normalisation,
                 representation: str, instantane: bool = False):
        verifier_partitions(episodes)
        if len({e.partition for e in episodes}) != 1:
            raise ValueError("Un chargeur ne melange pas les partitions")
        if representation not in ("line", "hydraulique"):
            raise ValueError(f"Representation inconnue : {representation}")
        self.episodes, self.normalisation = episodes, normalisation
        self.representation, self.instantane = representation, instantane
        self.entrees = [normalisation.transformer(e) for e in episodes]
        self.index = [(i, t) for i, e in enumerate(episodes)
                      for t in range(11, len(e.cibles["q_m3h"]))]

    def __len__(self) -> int:
        return len(self.index)

    def __getitem__(self, i: int) -> dict:
        ep, t = self.index[i]
        e, x = self.episodes[ep], self.entrees[ep]
        tranche = slice(t if self.instantane else t - 11, t + 1)
        noeuds, liens = x["hydraulique_noeuds"][tranche], x["hydraulique_liens"][tranche]
        if self.representation == "line":
            ext = np.asarray(e.schema["extremites_liens"])
            entree = np.concatenate((noeuds[:, ext[:, 0]], noeuds[:, ext[:, 1]], liens), -1)
        else:
            entree = noeuds
        resultat = {
            "x": torch.from_numpy(entree.copy()), "liens": torch.from_numpy(liens.copy()),
            "q_normalise": torch.from_numpy(
                (e.cibles["q_m3h"][t] / self.normalisation.echelle_q).astype(np.float32)),
            "fuite_m3h": torch.from_numpy(e.cibles["fuite_m3h"][t].astype(np.float32)),
            "episode": e.identifiant, "pas_cible": t,
        }
        if "fuite_active" in e.cibles:
            resultat["fuite_active"] = torch.from_numpy(e.cibles["fuite_active"][t].copy())
        if "q_reference_m3h" in e.cibles:
            resultat["q_reference_m3h"] = torch.from_numpy(e.cibles["q_reference_m3h"][t].copy())
        return resultat
