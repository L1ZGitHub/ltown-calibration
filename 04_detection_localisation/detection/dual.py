"""Le modèle dual de l'équipe « Under Pressure » (Steffelbauer et al., 2022).

Dans le modèle habituel (le « primal »), on donne la demande de chaque nœud et le modèle rend les
pressions. Dans le modèle dual, chaque capteur de pression est relié à un **réservoir virtuel**
dont la charge est la charge mesurée. Le modèle doit alors trouver quel débit chaque réservoir
virtuel absorbe pour que le réseau sans fuite colle aux mesures. Ce débit est le **débit de fuite
virtuel** `q_v` du capteur, en m³/h : nul si le modèle explique les mesures, positif quand de l'eau
manque près du capteur.

Comme dans l'article, le réseau est **coupé en deux** à la pompe (`separer=True`) : côté A+B, la
pompe est remplacée par son débit mesuré, prélevé au nœud d'aspiration ; côté C, le réservoir T1 est
tenu à son niveau mesuré. Chaque zone n'a alors que des mesures pour frontières.

Écart à l'article : ils relient le réservoir virtuel par une vanne de faible perte de charge ; ici
c'est une conduite d'un mètre, large et lisse, qui joue le même rôle (perte de charge négligeable).
"""
from __future__ import annotations

from multiprocessing import Pool

import numpy as np
import pandas as pd

from .residu import M, R, SORTIES
from generation import scenarios as S                                   # noqa: E402

TR = M.PAS_TRANCHE          # 72 pas de 5 min : une tranche de 6 h


def couper(wn, debit_pompe, niveau) -> None:
    """Coupe le réseau à la pompe : son débit mesuré (m³/h) sort au nœud d'aspiration, et le
    réservoir T1 devient un niveau imposé (le niveau mesuré, m)."""
    pompe = wn.get_link("PUMP_1")
    aspiration = pompe.start_node_name
    wn.remove_link("PUMP_1")
    # WNTR 1.5 s'arrête en route quand la pompe a un profil de vitesse : elle reste dans la liste des
    # pompes et l'écriture du fichier EPANET échoue. On finit le ménage à la main.
    for liste in ("_pumps", "_head_pumps", "_power_pumps"):
        getattr(wn._link_reg, liste).discard("PUMP_1")
    wn.add_pattern("Q_POMPE", list(np.asarray(debit_pompe, float)))
    wn.get_node(aspiration).add_demand(base=1.0 / 3600.0, pattern_name="Q_POMPE")
    t1 = wn.get_node("T1")
    xy, altitude = t1.coordinates, t1.elevation
    sorties = [(nom, wn.get_link(nom)) for nom in wn.get_links_for_node("T1")]
    proprietes = [(nom, lien.end_node_name if lien.start_node_name == "T1" else lien.start_node_name,
                   lien.length, lien.diameter, lien.roughness, lien.minor_loss) for nom, lien in sorties]
    for nom, _ in sorties:
        wn.remove_link(nom)
    wn.remove_node("T1")
    wn.add_pattern("H_T1", list(altitude + np.asarray(niveau, float)))
    wn.add_reservoir("T1", base_head=1.0, head_pattern="H_T1", coordinates=xy)
    for nom, autre, longueur, diametre, rugosite, singuliere in proprietes:
        wn.add_pipe(nom, "T1", autre, length=longueur, diameter=diametre, roughness=rugosite,
                    minor_loss=singuliere)


def tranche(D, pompe, niveau0: float, charges, coefficients: dict, noeuds: list,
            capteurs: list, debit_pompe=None, niveau=None) -> np.ndarray:
    """Une tranche de 6 h du modèle dual. `charges` (73, 33) : la charge imposée à chaque capteur, m.
    Avec `debit_pompe` et `niveau` (73,), le réseau est coupé à la pompe (voir `couper`).
    Rend `q_v` (72, 33) en m³/h, positif quand le réseau perd de l'eau vers le réservoir virtuel."""
    wn = R.preparer(D, noeuds, 6.0, niveau0=niveau0, coefficients=coefficients, pompe=pompe)
    if debit_pompe is not None:
        couper(wn, debit_pompe, niveau)
    for i, c in enumerate(capteurs):
        wn.add_pattern(f"HV_{c}", list(np.asarray(charges[:, i], float)))
        x, y = wn.get_node(c).coordinates
        wn.add_reservoir(f"V_{c}", base_head=1.0, head_pattern=f"HV_{c}", coordinates=(x + 1, y + 1))
        wn.add_pipe(f"PV_{c}", f"V_{c}", c, length=1.0, diameter=0.5, roughness=150.0, minor_loss=0.0)
    res = R.executer(wn)
    q = res.link["flowrate"][[f"PV_{c}" for c in capteurs]].to_numpy()[:TR] * 3600.0
    return (-q).astype(np.float32)          # le débit du lien va du réservoir vers le réseau


def _tranche(args):
    return tranche(*args)


def _arguments(fen, charges: np.ndarray, k: int, debit_pompe=None) -> tuple:
    """Les entrées de la tranche k. La dernière tranche de l'année manque d'un pas : on tient la
    dernière valeur, comme le fait `R.simuler`."""
    a, b = k * TR, (k + 1) * TR + 1
    def bout(x):
        x = x[a:b]
        return np.concatenate([x, np.repeat(x[-1:], TR + 1 - len(x), axis=0)]) if len(x) < TR + 1 else x
    coupe = (None, None) if debit_pompe is None else (bout(debit_pompe), bout(fen.niveau))
    return (bout(fen.D), bout(fen.pompe), float(fen.niveau[a]), bout(charges),
            fen.coefficients, fen.noeuds, fen.capteurs, *coupe)


def debit_pompe_mesure(fen) -> np.ndarray:
    """Le débit mesuré de PUMP_1 sur la fenêtre (n_pas + 1,), m³/h."""
    q = R.charger_debits(fen.annee)["PUMP_1"].interpolate(limit_direction="both").to_numpy()
    q = q[fen.debut:fen.debut + fen.n_pas + 1]
    return np.concatenate([q, np.repeat(q[-1:], fen.n_pas + 1 - len(q))])


def charges_capteurs(fen, pressions: np.ndarray) -> np.ndarray:
    """Pression aux capteurs (n_pas, 33), m -> charge (n_pas + 1, 33) : on ajoute l'altitude, et on
    répète le dernier pas (EPANET lit le profil un pas plus loin que l'horizon)."""
    alt = np.array([fen.wn.get_node(c).elevation for c in fen.capteurs])
    p = pd.DataFrame(pressions).interpolate(limit_direction="both").to_numpy()
    return np.vstack([p, p[-1:]]) + alt


def simuler(fen, pressions: np.ndarray, tranches=None, processus: int = 1,
            separer: bool = True) -> np.ndarray:
    """Le modèle dual sur les tranches demandées (toutes par défaut), capteurs tenus à `pressions`.
    `separer=True` coupe le réseau à la pompe, comme l'article ; `False` garde la pompe du modèle."""
    H = charges_capteurs(fen, pressions)
    ks = range(fen.n_pas // TR + (fen.n_pas % TR > 0)) if tranches is None else tranches
    qp = debit_pompe_mesure(fen) if separer else None
    args = [_arguments(fen, H, k, qp) for k in ks]
    if processus > 1:
        with Pool(processus) as pool:
            morceaux = pool.map(_tranche, args, chunksize=4)
    else:
        morceaux = [_tranche(a) for a in args]
    return np.vstack(morceaux)[:fen.n_pas]


def debits_virtuels(annee: int = 2018, refaire: bool = False, processus: int = 1,
                    separer: bool = True) -> pd.DataFrame:
    """`q_v` de l'année aux 33 capteurs, m³/h, au pas de 5 min (en cache dans `sorties/`).
    Une vingtaine de minutes la première fois avec un processus."""
    f = SORTIES / f"dual_{annee}{'' if separer else '_pompe_modele'}.npz"
    if f.exists() and not refaire:
        d = np.load(f, allow_pickle=True)
        return pd.DataFrame(d["q"], index=pd.DatetimeIndex(d["horo"]),
                            columns=[str(c) for c in d["capteurs"]])
    fen = M.fenetre(annee, 0, 365)
    q = simuler(fen, fen.p_mes, processus=processus, separer=separer)
    SORTIES.mkdir(exist_ok=True)
    np.savez_compressed(f, q=q, capteurs=np.array(fen.capteurs), horo=fen.horo.values)
    return pd.DataFrame(q, index=fen.horo, columns=fen.capteurs)


def signature(conduite: str, debit: float = 10.0, annee: int = 2018, jour: int = 1) -> np.ndarray:
    """Ce qu'une fuite de 1 m³/h sur `conduite` fait aux 33 `q_v` (m³/h par m³/h de fuite).
    Simulée sur une journée : le modèle sans fuite, le modèle avec une fuite de `debit`, puis le dual
    tenu à chacun ; la différence, divisée par `debit`. Quelques secondes."""
    fen = M.fenetre(annee, jour, 1)
    sain = M.simuler(fen)
    avec, _ = S.simuler_fuites(fen, [S.Fuite(conduite, 0, 24, debit)], sain)
    q_sain = simuler(fen, sain[:, fen.colonnes], separer=False)
    q_avec = simuler(fen, avec[:, fen.colonnes], separer=False)
    return (q_avec - q_sain)[TR:].mean(axis=0) / debit       # sans la 1re tranche (mise en route)
