"""Deux representations du meme reseau decoupe, avec roles d'extremite."""
from dataclasses import dataclass
from itertools import combinations

import numpy as np
import torch
from wntr.network import WaterNetworkModel

from .hydraulique import Conduite


@dataclass
class Graphe:
    noms: list[str]
    aretes: torch.Tensor
    attributs: torch.Tensor
    segments: torch.Tensor
    cibles_fuite: torch.Tensor
    jonctions_partagees: list[str]
    type_lien: list[str]


def attributs_physiques(wn: WaterNetworkModel, correspondance: list[Conduite]) -> dict:
    """Attributs du modele de reference uniquement, jamais de la variante cachee."""
    milieux = {c.milieu for c in correspondance}
    noeuds, liens = [], []
    for nom in wn.node_name_list:
        nd = wn.get_node(nom)
        type_ = nd.node_type
        altitude = nd.base_head if type_ == "Reservoir" else nd.elevation
        noeuds.append([*nd.coordinates, altitude, float(type_ == "Junction"),
                       float(type_ == "Reservoir"), float(type_ == "Tank"),
                       float(nom in milieux), nd.base_head if type_ == "Reservoir" else 0.])
    for nom in wn.link_name_list:
        l = wn.get_link(nom)
        pipe, vanne = l.link_type == "Pipe", l.link_type == "Valve"
        liens.append([
            np.log(l.length) if pipe else 0., l.diameter if pipe or vanne else 0.,
            l.roughness if pipe else 0., l.minor_loss if pipe or vanne else 0.,
            l.initial_setting if vanne else 0.,
            float(pipe), float(l.link_type == "Pump"), float(vanne),
        ])
    return {
        "noeuds": np.array(noeuds, dtype=np.float32),
        "liens": np.array(liens, dtype=np.float32),
        "champs_noeuds": ["x_m", "y_m", "altitude_ou_charge_reservoir_m", "jonction",
                          "reservoir", "bache", "milieu", "charge_reservoir_m"],
        "champs_liens": ["log_longueur_m", "diametre_m", "rugosite_HW", "perte_singuliere",
                         "consigne_vanne", "conduite", "pompe", "vanne"],
    }


def construire(wn: WaterNetworkModel, correspondance: list[Conduite]) -> dict[str, Graphe]:
    noeuds, liens = list(wn.node_name_list), list(wn.link_name_list)
    ni, li = {n: i for i, n in enumerate(noeuds)}, {n: i for i, n in enumerate(liens)}
    milieux = {c.milieu for c in correspondance}
    types = [wn.get_link(n).link_type for n in liens]
    segments = torch.tensor([[li[c.segment_1], li[c.segment_2]] for c in correspondance])
    ext, attrs = [], []
    incidents: dict[str, list[tuple[int, float]]] = {n: [] for n in noeuds}
    for i, nom in enumerate(liens):
        p = wn.get_link(nom)
        debut, fin = ni[p.start_node_name], ni[p.end_node_name]
        ext.extend(((debut, fin), (fin, debut)))
        a = [0.] * 10
        a[5 + ("Pipe", "Pump", "Valve").index(p.link_type)] = 1.
        attrs.extend((a, a.copy()))
        incidents[p.start_node_name].append((i, -1.))
        incidents[p.end_node_name].append((i, 1.))
    dual, da, partagees = [], [], []
    for n, voisins in incidents.items():
        for (i, ri), (j, rj) in combinations(voisins, 2):
            for src, dst, rs, rd in ((i, j, ri, rj), (j, i, rj, ri)):
                dual.append((src, dst))
                a = [rs, rd, float(n in milieux), 0., 0., 1., 0., 0., 0., 0.]
                da.append(a)
                partagees.append(n)
    commun = {"segments": segments, "type_lien": types}
    return {
        "hydraulique": Graphe(noeuds, torch.tensor(ext).T.contiguous(), torch.tensor(attrs),
                             cibles_fuite=torch.tensor([ni[c.milieu] for c in correspondance]),
                             jonctions_partagees=[], **commun),
        "line": Graphe(liens, torch.tensor(dual).T.contiguous(), torch.tensor(da),
                       cibles_fuite=segments.clone(), jonctions_partagees=partagees, **commun),
    }
