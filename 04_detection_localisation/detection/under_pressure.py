"""La méthode « Under Pressure » de bout en bout : détecter, ajuster la forme, localiser, retirer.

On avance dans le temps. À chaque étape :

1. **détecter** : deux tests par capteur sur `q_v` (le débit virtuel du modèle dual), une fois
   retirées les fuites déjà connues : un CUSUM et, en option, un test du rapport de vraisemblance
   (l'article utilise les deux ; sur nos données, le second ajoute surtout des fausses alarmes, voir
   le carnet d). Les seuils viennent de la première semaine de 2018, sans fuite ;
2. **ajuster la forme** de la nouvelle fuite sur le **bilan de débit** de sa zone (l'eau qui entre
   moins la consommation, moins les fuites connues) : casse brutale (type I) ou fuite qui grandit puis
   plafonne (type II) ;
3. **localiser** par corrélation de Pearson entre la baisse de pression aux capteurs et la
   sensibilité de chaque conduite (le dictionnaire de signatures), sommée sur la fenêtre de la fuite ;
4. **retirer** : la fuite trouvée devient une demande connue ; on cherche la suivante.

Avant de déclarer une nouvelle fuite, on réajuste d'abord les fuites connues de la zone sur les
nouvelles données (l'article : « la meilleure combinaison des débits de fuite au cours du temps »).
Si une fuite connue qui continue de grandir explique la hausse, on met sa courbe à jour, sans
nouvelle alarme.

Les choix de l'article qui ne sont pas des chiffres (seuils « calibrés », vérification visuelle,
fenêtres) sont remplacés par les règles écrites en tête de ce fichier.

Ajout qui n'est pas dans l'article : les **réparations**. Une fuite connue qui est réparée ferait
passer le signal corrigé sous zéro et cacherait les suivantes. Un CUSUM de la baisse sur le bilan de
débit de la zone marque la fin de la fuite connue dont le débit colle le mieux à la baisse.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import dual as Du, lentes as Le, localisation as L, residu as Rs

# ----------------------------------------------------------------------------- réglages
K = 0.5                  # CUSUM : dérive tolérée, en écarts-types de la première semaine
MARGE = 2.0              # seuil = MARGE × le plus grand CUSUM atteint pendant la semaine sans fuite
SEUIL_MIN = 5.0          # ... et jamais moins que 5
FENETRE_RV_H = 72        # rapport de vraisemblance : la hausse est cherchée dans les 72 dernières heures
SEUIL_RV_MIN = 10.0      # ... seuil = MARGE × le plus haut de la semaine sans fuite, et au moins 10
FENETRE_J = 14           # jours après l'alarme, au plus, pour la forme et la localisation
ECART_JOUR = 1.0         # m³/h : la fenêtre s'arrête au premier jour qui s'écarte de la courbe de plus
                         # que ça (ou de 15 % du débit) : une autre fuite ou une réparation commence
DEBIT_MIN = 1.0          # m³/h : en dessous, le bilan de débit ne confirme pas la fuite. L'alarme est
                         # ignorée et le niveau actuel du capteur devient sa nouvelle normale
SEUIL_QV = 1.8           # m³/h (0,5 L/s, l'article) : capteurs retenus pour la corrélation
CORR_MIN = 0.95          # l'article : seules les corrélations au-dessus comptent dans la somme
SEMAINE = slice("2018-01-01", "2018-01-07")


@dataclass
class Fuite:
    """Une fuite trouvée : où, quand, et sa courbe de débit (équations 16 et 17 de l'article)."""
    zone: str
    alarme: pd.Timestamp     # quand le CUSUM a sonné
    capteur: str             # le capteur qui a sonné
    t_s: pd.Timestamp        # début estimé
    t_sa: pd.Timestamp       # fin de la montée (= t_s pour une casse)
    q_s: float               # débit final, m³/h
    puissance: float         # 1 ou 2 : montée en ligne droite ou en parabole
    conduite: str = ""
    fin: pd.Timestamp | None = None     # réparation, si on la voit
    notes: dict = field(default_factory=dict)

    @property
    def type(self) -> str:
        return "I (casse)" if self.t_sa == self.t_s else "II (grandit)"

    def debit(self, index: pd.DatetimeIndex) -> np.ndarray:
        """Le débit de la fuite à chaque heure de `index`, m³/h."""
        h = ((index - self.t_s) / pd.Timedelta(hours=1)).to_numpy(float)
        duree = max((self.t_sa - self.t_s) / pd.Timedelta(hours=1), 1e-9)
        u = np.clip(h / duree, 0.0, 1.0) if self.t_sa > self.t_s else (h >= 0).astype(float)
        q = self.q_s * u ** self.puissance
        if self.fin is not None:
            q[index >= self.fin] = 0.0
        return q


# ----------------------------------------------------------------------------- étape 1 : détecter
def cusum(z: np.ndarray, depart: int, seuil: np.ndarray):
    """CUSUM d'une hausse, colonne par colonne, à partir de la ligne `depart`. `z` est déjà centré et
    réduit. Rend (ligne de l'alarme, colonne, ligne où le cumul était vide pour la dernière fois),
    ou None."""
    s = np.zeros(z.shape[1])
    vide = np.full(z.shape[1], depart)
    for i in range(depart, len(z)):
        s = np.maximum(0.0, s + np.nan_to_num(z[i]) - K)
        vide[s == 0] = i
        j = int(np.argmax(s - seuil))
        if s[j] > seuil[j]:
            return i, j, int(vide[j])
    return None


def rapport_vraisemblance(z: np.ndarray, depart: int, seuil: np.ndarray):
    """Test du rapport de vraisemblance (généralisé) d'une hausse de la moyenne, colonne par colonne.
    À chaque heure t : la hausse a-t-elle commencé à une heure k des FENETRE_RV_H dernières ?
    Pour un bruit gaussien réduit, le log du rapport vaut `max(S, 0)² / (2 n)`, avec S la somme des
    `n` valeurs depuis k. Rend (ligne de l'alarme, colonne, ligne k du début), ou None."""
    c = np.vstack([np.zeros(z.shape[1]), np.cumsum(np.nan_to_num(z), axis=0)])
    for i in range(depart, len(z)):
        k = np.arange(max(depart, i - FENETRE_RV_H + 1), i + 1)
        n = (i + 1 - k)[:, None]
        s = c[i + 1] - c[k]
        g = np.maximum(s, 0.0) ** 2 / (2 * n)
        j = int(np.argmax(g.max(axis=0) - seuil))
        if g[:, j].max() > seuil[j]:
            return i, j, int(k[int(np.argmax(g[:, j]))])
    return None


def plus_haut_rv(z: np.ndarray) -> np.ndarray:
    """Le plus haut rapport de vraisemblance atteint par chaque colonne de `z`."""
    c = np.vstack([np.zeros(z.shape[1]), np.cumsum(np.nan_to_num(z), axis=0)])
    haut = np.zeros(z.shape[1])
    for i in range(len(z)):
        k = np.arange(max(0, i - FENETRE_RV_H + 1), i + 1)
        g = np.maximum(c[i + 1] - c[k], 0.0) ** 2 / (2 * (i + 1 - k)[:, None])
        haut = np.maximum(haut, g.max(axis=0))
    return haut


def calibration_2018() -> dict:
    """Les seuils, pris une fois pour toutes sur la première semaine de 2018 : par capteur pour
    `q_v`, par zone pour le bilan de débit (pour voir les réparations)."""
    qv = horaire(Du.debits_virtuels(2018))
    f = pd.DataFrame({z: horaire(s) for z, s in Le.fuite_par_zone(2018).items()})
    mu, sd, seuil = calibrer(qv)
    seuil_rv = np.maximum(MARGE * plus_haut_rv(((qv.loc[SEMAINE] - mu) / sd).to_numpy()), SEUIL_RV_MIN)
    return {"qv": (mu, sd, seuil), "qv_rv": seuil_rv, "bilan": calibrer(-f)}


def calibrer(x: pd.DataFrame) -> tuple[pd.Series, pd.Series, np.ndarray]:
    """Moyenne, écart-type et seuil de chaque colonne, pris sur la semaine sans fuite."""
    sem = x.loc[SEMAINE]
    mu, sd = sem.mean(), sem.std()
    z = ((sem - mu) / sd).to_numpy()
    s, plus_haut = np.zeros(z.shape[1]), np.zeros(z.shape[1])
    for ligne in z:
        s = np.maximum(0.0, s + ligne - K)
        plus_haut = np.maximum(plus_haut, s)
    return mu, sd, np.maximum(MARGE * plus_haut, SEUIL_MIN)


# ----------------------------------------------------------------------------- étape 2 : la forme
def ajuster_forme(f: pd.Series, t0: pd.Timestamp, fin: pd.Timestamp) -> dict:
    """Ajuste une casse ou une montée sur `f` (bilan de débit corrigé, horaire) entre l'avant-veille
    de `t0` et `fin`. Moindres carrés sur une grille : début à ±24 h de `t0`, fin de montée toutes
    les 6 h, montée droite ou en parabole. Un décalage constant est libre (l'erreur lente du modèle).

    Pour chaque forme `g` de la grille, `f ≈ c + q·g` se résout d'un coup : `q = cov(g, f) / var(g)`."""
    x = f.loc[t0 - pd.Timedelta(days=2):fin].dropna()
    t, y = x.index, x.to_numpy()
    formes, courbes = [], []
    for ts in pd.date_range(t0 - pd.Timedelta(hours=24), t0 + pd.Timedelta(hours=24), freq="1h"):
        for tsa in [ts] + list(pd.date_range(ts + pd.Timedelta(hours=6), fin, freq="6h")):
            for puissance in ((1.0,) if tsa == ts else (1.0, 2.0)):
                formes.append((ts, tsa, puissance))
                courbes.append(Fuite("", ts, "", ts, tsa, 1.0, puissance).debit(t))
    G = np.array(courbes)
    gc, yc = G - G.mean(axis=1, keepdims=True), y - y.mean()
    var = (gc ** 2).sum(axis=1)
    cov = gc @ yc
    q = np.where(var > 0, cov / np.maximum(var, 1e-12), 0.0)
    sse = (yc ** 2).sum() - np.where(q > 0, cov * q, 0.0)       # une fuite ne peut pas être négative
    k = int(np.argmin(sse))
    ts, tsa, puissance = formes[k]
    return {"sse": float(sse[k]), "t_s": ts, "t_sa": tsa, "q_s": float(max(q[k], 0.0)),
            "puissance": puissance}


def fenetre_propre(f: pd.Series, t0: pd.Timestamp, t_a: pd.Timestamp, fin_max: pd.Timestamp):
    """« Une fenêtre qui commence juste avant la fuite et finit avant la suivante » (l'article).
    On part de 2 jours après l'alarme et on ajoute un jour tant que ce jour suit la courbe ajustée."""
    fin = min(t_a + pd.Timedelta(days=2), fin_max)
    forme = ajuster_forme(f, t0, fin)
    while fin + pd.Timedelta(days=1) <= fin_max:
        jour = f.loc[fin:fin + pd.Timedelta(days=1)]
        k = Fuite("", t_a, "", forme["t_s"], forme["t_sa"], forme["q_s"], forme["puissance"])
        base = f.loc[t0 - pd.Timedelta(days=2):t0 - pd.Timedelta(days=1)].mean()
        ecart = abs((jour - base - k.debit(jour.index)).mean())
        if ecart > max(ECART_JOUR, 0.15 * forme["q_s"]):
            break
        fin += pd.Timedelta(days=1)
        forme = ajuster_forme(f, t0, fin)
    return forme, fin


def suit_la_courbe(f: pd.Series, forme: dict, t0: pd.Timestamp, debut: pd.Timestamp, fin: pd.Timestamp) -> bool:
    """Chaque jour entre `debut` et `fin` reste-t-il sur la courbe ajustée (même test que
    `fenetre_propre`) ?"""
    base = f.loc[t0 - pd.Timedelta(days=2):t0 - pd.Timedelta(days=1)].mean()
    k = Fuite("", t0, "", forme["t_s"], forme["t_sa"], forme["q_s"], forme["puissance"])
    for jour in pd.date_range(debut, fin - pd.Timedelta(days=1), freq="1D"):
        x = f.loc[jour:jour + pd.Timedelta(days=1)]
        if abs((x - base - k.debit(x.index)).mean()) > max(ECART_JOUR, 0.15 * forme["q_s"]):
            return False
    return True


def reajuster(connues: list, zone: str, f: pd.Series, t_a: pd.Timestamp, fin: pd.Timestamp):
    """Une fuite connue et active de la zone, réajustée jusqu'à `fin`, explique-t-elle la hausse vue à
    `t_a` ? `f` est le bilan corrigé de toutes les fuites connues. Rend la fuite mise à jour, ou None."""
    for k in sorted([k for k in connues if k.zone == zone and k.fin is None], key=lambda k: k.t_s, reverse=True):
        g = f + pd.Series(k.debit(f.index), index=f.index)          # on lui rend sa propre courbe
        forme = ajuster_forme(g, k.t_s, fin)
        if forme.get("q_s", 0.0) >= k.q_s and suit_la_courbe(g, forme, k.t_s, t_a - pd.Timedelta(days=1), fin):
            k.t_s, k.t_sa, k.q_s, k.puissance = forme["t_s"], forme["t_sa"], forme["q_s"], forme["puissance"]
            return k
    return None


# ----------------------------------------------------------------------------- étape 3 : localiser
def pearson(r: np.ndarray, S: np.ndarray) -> np.ndarray:
    """Corrélation de Pearson entre le vecteur `r` et chaque ligne de `S` (équation 19)."""
    rc = r - r.mean()
    Sc = S - S.mean(axis=1, keepdims=True)
    return Sc @ rc / (np.linalg.norm(Sc, axis=1) * np.linalg.norm(rc) + 1e-12)


def localiser(baisse: pd.DataFrame, qv: pd.DataFrame, t_s, fin, capteurs: list, dico: pd.DataFrame,
              candidates: list) -> tuple[str, dict]:
    """La conduite dont la somme des corrélations (au-dessus de CORR_MIN) est la plus grande.

    `baisse` : baisse de pression corrigée des fuites connues, cm, horaire ; `qv` : débits virtuels
    corrigés. À chaque heure de la fenêtre, on ne garde que les capteurs dont `q_v` a monté d'au moins
    SEUIL_QV (au moins 3 capteurs, sinon l'heure ne compte pas)."""
    avant = slice(t_s - pd.Timedelta(hours=24), t_s)
    b0, q0 = baisse.loc[avant, capteurs].mean(), qv.loc[avant, capteurs].mean()
    S = dico.loc[candidates, capteurs].to_numpy()
    somme, moyenne, heures = np.zeros(len(candidates)), np.zeros(len(candidates)), 0
    for t in baisse.loc[t_s:fin].index:
        r = (baisse.loc[t, capteurs] - b0).to_numpy()
        garde = (qv.loc[t, capteurs] - q0).to_numpy() >= SEUIL_QV
        if garde.sum() < 3:
            continue
        rho = pearson(r[garde], S[:, garde])
        somme += np.where(rho > CORR_MIN, rho, 0.0)
        moyenne += rho
        heures += 1
    if heures == 0:                       # aucun capteur ne bouge assez : toute la zone, une fois
        r = (baisse.loc[t_s:fin, capteurs].mean() - b0).to_numpy()
        rho = pearson(r, S)
        return candidates[int(rho.argmax())], {"heures": 0, "regle": "zone entière"}
    if somme.max() > 0:
        return candidates[int(somme.argmax())], {"heures": heures, "regle": f"somme ρ > {CORR_MIN}"}
    return candidates[int(moyenne.argmax())], {"heures": heures, "regle": "ρ moyen (aucun > seuil)"}


# ----------------------------------------------------------------------------- la boucle
def horaire(x: pd.DataFrame | pd.Series):
    return x.resample("1h").mean()


def chaine(annee: int = 2018, calibration: dict | None = None, verbeux: bool = True,
           tests: tuple = ("cusum",)) -> list[Fuite]:
    """Toute l'année, fuite après fuite. `calibration` : `calibration_2018()`, réutilisée telle
    quelle pour une autre année."""
    qv = horaire(Du.debits_virtuels(annee))
    res = horaire(Rs.residu(annee))
    bilan = {z: horaire(s) for z, s in Le.fuite_par_zone(annee).items()}
    zones = Rs.zones_des_capteurs(qv.columns)
    capteurs_de = {z: [c for c in qv.columns if zones[c] == z] for z in ("AB", "C")}
    dico = L.dictionnaire(2018, 1, 10.0)
    wn = Rs.R.charger_modele()
    vus = set(L.visibles(dico))
    candidates = {z: [p for p in L.conduites_de_zone(wn, z, list(dico.index)) if p in vus] for z in capteurs_de}

    calibration = calibration or calibration_2018()
    mu, sd, seuil = calibration["qv"]
    mu_f, sd_f, seuil_f = calibration["bilan"]
    index = qv.index
    connues: list[Fuite] = []
    signature_dual: dict[str, np.ndarray] = {}
    recalage = np.zeros(qv.shape)          # nouvelles normales des capteurs, après une alarme ignorée
    ignorees = 0

    def corriges():
        """q_v, baisse de pression et bilan, une fois retirées les fuites connues."""
        q, b = qv - recalage, -res.copy()                # baisse = − résidu (cm)
        f = {z: s.copy() for z, s in bilan.items()}
        for k in connues:
            d = k.debit(index)
            q -= np.outer(d, signature_dual[k.conduite])
            b -= np.outer(d / 10.0, dico.loc[k.conduite, res.columns].to_numpy())
            f[k.zone] -= d
        return q, b, f

    depart = depart_rep = int(np.searchsorted(index, pd.Timestamp(SEMAINE.stop) + pd.Timedelta(days=1)))
    if annee != 2018:
        depart = depart_rep = 0
    while True:
        q, b, f = corriges()
        zq = ((q - mu) / sd).to_numpy()
        alarmes = []
        if "cusum" in tests:
            alarmes.append(cusum(zq, depart, seuil))
        if "rv" in tests:
            alarmes.append(rapport_vraisemblance(zq, depart, calibration["qv_rv"]))
        alarmes = [a for a in alarmes if a is not None]
        trouve = min(alarmes, key=lambda a: a[0]) if alarmes else None
        baisse_f = -pd.DataFrame(f)[list(mu_f.index)]            # une réparation fait monter −bilan
        rep = cusum(((baisse_f - mu_f) / sd_f).to_numpy(), depart_rep, seuil_f)
        if trouve is None and rep is None:
            break
        if rep is not None and (trouve is None or rep[0] < trouve[0]):
            i, j, i0 = rep
            zone, t_r = mu_f.index[j], index[i0]
            actives = [k for k in connues if k.zone == zone and k.fin is None and k.t_s < t_r]
            avant = f[zone].loc[t_r - pd.Timedelta(hours=24):t_r].mean()
            apres = f[zone].loc[index[i]:index[i] + pd.Timedelta(hours=24)].mean()
            if actives and avant - apres >= DEBIT_MIN:
                k = min(actives, key=lambda k: abs(k.debit(pd.DatetimeIndex([t_r]))[0] - (avant - apres)))
                k.fin = t_r
                if verbeux:
                    print(f"  {t_r:%d/%m %H:%M} {zone:2s} réparation : le bilan baisse de {avant - apres:4.1f} m³/h "
                          f"-> fin de {k.conduite}", flush=True)
            depart_rep = i + 1
            continue
        i, j, i0 = trouve
        capteur = qv.columns[j]
        zone = zones[capteur]
        t_a, t0 = index[i], index[i0]
        forme, fin = fenetre_propre(f[zone], t0, t_a, min(t_a + pd.Timedelta(days=FENETRE_J), index[-1]))
        depart = i + 1
        if forme.get("q_s", 0.0) < DEBIT_MIN:
            recalage[i:, j] += q.iloc[i0:i + 1, j].mean() - mu.iloc[j]
            ignorees += 1
            continue
        maj = reajuster(connues, zone, f[zone], t_a, fin)
        if maj is not None:
            maj.notes["mises_a_jour"] = maj.notes.get("mises_a_jour", 0) + 1
            if verbeux:
                print(f"  {t_a:%d/%m %H:%M} {zone:2s} {capteur:5s} la fuite connue {maj.conduite} grandit encore : "
                      f"q_s {maj.q_s:5.1f} m³/h", flush=True)
            continue
        conduite, info = localiser(b, q, forme["t_s"], fin, capteurs_de[zone], dico, candidates[zone])
        info["fin_fenetre"] = fin
        k = Fuite(zone, t_a, capteur, forme["t_s"], forme["t_sa"], forme["q_s"], forme["puissance"],
                  conduite, notes=info)
        signature_dual.setdefault(conduite, Du.signature(conduite))
        connues.append(k)
        if verbeux:
            print(f"  {t_a:%d/%m %H:%M} {zone:2s} {capteur:5s} type {k.type:12s} q_s {k.q_s:5.1f} m³/h "
                  f"-> {conduite} ({info['regle']}, {info['heures']} h)", flush=True)
    if verbeux:
        print(f"  {ignorees} alarmes de q_v ignorées : le bilan de débit ne voyait rien")
    return connues
