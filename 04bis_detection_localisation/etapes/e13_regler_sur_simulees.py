"""Régler la chaîne sur les années simulées (étape 12), puis la noter sur 2018.

Jusqu'à l'étape 11, chaque réglage de la chaîne a été choisi en regardant les 14 fuites de 2018,
celles-là mêmes qu'on note. Ici, les mêmes grilles sont essayées sur les 12 années simulées. Pour
chaque réglage, on fait la **somme des euros des 12 années**, ce que rapporterait une année moyenne
fois 12 : les euros d'une année seule ne se comparent pas d'une année à l'autre, mais leur somme est
bien de l'argent. On garde le réglage qui rapporte le plus, puis on le note sur 2018, qui devient
pour ce réglage-là une année de validation.

Ce qui est réglé ici : le couple de CUSUM (étape 06), la fenêtre de fusion W et la confirmation
X / H (étape 09). Le reste de la chaîne (dictionnaire des signatures, σ des cartes, abstention, σ
du dual, pente cm par m³/h) garde ses réglages de 2018 (`A12.reference_2018`).

Les grilles de chaque année sont gardées dans `sorties/simulees_*.csv`. Temps de calcul (mesurés, un
cœur) : lire les signaux des 12 années, une dizaine de secondes ; la grille de l'étape 06, environ une
minute ; celles de l'étape 09, environ 30 secondes ; les tableaux complets (avec le hasard), une
demi-minute par réglage. Les années elles-mêmes doivent exister (étape 12, environ une heure sur 8
cœurs la première fois).
"""
from __future__ import annotations

import itertools
import json
import multiprocessing as mp
from dataclasses import dataclass, field, replace
from functools import cached_property, lru_cache

import numpy as np
import pandas as pd

from . import (e00_donnees as D, e01_residu as R1, e02_cusum as C, e03_bilan as B3, e05_signatures as Sg,
               e06_chaine as Ch, e08_regrouper as Rg, e09_combiner as E9, e11_localiser_dual as L11,
               e12_annees_simulees as A12, reglages as G)
from .e00_donnees import SORTIES

MODES = {"pression": ("pression",), "débit": ("débit",), "les deux": ("pression", "débit")}


# --------------------------------------------------------------------- une année, prête à noter
@dataclass
class Annee:
    """Ce que la chaîne lit d'une année (2018 ou simulée) et de quoi la noter."""
    nom: str
    res: pd.DataFrame                   # résidu aux 33 capteurs, cm, pas de 5 min
    bilan: pd.DataFrame                 # bilan de débit par zone, m³/h, pas de 5 min
    qv: pd.DataFrame | None             # débits virtuels du dual, m³/h
    fuites: pd.DataFrame
    debits: pd.DataFrame
    b: Ch.Bareme
    zones: dict = field(repr=False)
    cache: dict = field(default_factory=dict, repr=False)    # alarmes placées, par réglage (carnet 13)

    @cached_property
    def sig_p(self) -> dict:
        return R1.signal(self.res, self.zones)

    @cached_property
    def sig_d(self) -> dict:
        return B3.signal_debit(self.bilan)

    @cached_property
    def bilan_h(self) -> pd.DataFrame:
        return self.bilan.resample("1h").mean()

    @cached_property
    def res_zone_h(self) -> pd.DataFrame:
        h = self.res.resample("1h").mean()
        return pd.DataFrame({z: h[[c for c in h.columns if self.zones[c] == z]].mean(axis=1) for z in ("AB", "C")})

    @property
    def variante(self) -> str | None:
        return self.nom.split("_")[0] if self.nom.startswith("v") else None


def annee_2018(ref: dict, coeurs: int = 1) -> Annee:
    """2018, depuis les caches des étapes 01, 03 et 10."""
    from . import e10_dual as Du
    res = R1.residu()
    fuites, debits = D.fuites_publiees()
    return Annee("2018", res, B3.bilan_par_zone(), Du.debits_virtuels(coupe=True, processus=coeurs), fuites, debits,
                 Ch.Bareme(D.reseau(), ref["dist"]), R1.zones_des_capteurs(res.columns))


def annees_simulees(ref: dict, coeurs: int = 1) -> dict[str, Annee]:
    """Les 12 années simulées (fabriquées et lues par l'étape 12 si elles n'existent pas encore)."""
    A12.lire_tout(ref, coeurs=coeurs)
    wn = D.reseau()
    out, zones = {}, None
    for t in A12.taches():
        d = A12.dossier_annees() / A12.nom(*t)
        with A12.source(d):
            res, bilan, qv = A12.lire_signaux(d, coeurs)
            fuites, debits = D.fuites_publiees()
            b = Ch.Bareme(wn, ref["dist"])
        zones = zones or R1.zones_des_capteurs(res.columns)
        out[A12.nom(*t)] = Annee(A12.nom(*t), res, bilan, qv, fuites, debits, b, zones)
    return out


# --------------------------------------------------------------------- les règles
def chaine(a: Annee, ref: dict, cusum, debit, abstenir: bool = G.ABSTENIR_SI_PLATE) -> pd.DataFrame:
    """La chaîne (étape 06) sur l'année `a`, avec le couple de CUSUM donné."""
    return Ch.chaine(a.res, a.sig_p, a.sig_d, ref["dico"], D.reseau(), cusum, debit, abstenir)


def confirmer(a: Annee, ch: pd.DataFrame, ref: dict, choix: tuple) -> pd.DataFrame:
    """Les envois de `ch` après confirmation (étape 09) ; `choix` = (mode, X m³/h, H heures)."""
    mode, x, h = choix
    return E9.confirmer(ch, a.bilan_h, a.res_zone_h, ref["pente"], x, h, MODES[mode])


def finale(a: Annee, ref: dict, cusum, debit, choix_confirmer: tuple) -> tuple[pd.DataFrame, pd.DataFrame]:
    """La version finale de l'étape 11 : chaîne avec abstention, carte du dual, confirmer, regrouper.
    Rend (alarmes relocalisées, envois)."""
    ch_d = L11.relocaliser(chaine(a, ref, cusum, debit), a.qv, ref["dico_d"], ref["s_dual"], ref["r_dual"])
    return ch_d, Rg.regrouper(confirmer(a, ch_d, ref, choix_confirmer), ref["dist"])


# --------------------------------------------------------------------- les grilles, en euros
def noter_vite(a: Annee, env: pd.DataFrame) -> dict:
    """N, trouvées, fausses et euros (sans le hasard : il ne sert pas à choisir)."""
    d = a.b.noter(env)
    return {"N": len(env), "fuites": len(a.b.fuites), "trouvées": int((d.type == "vraie").sum()),
            "fausses": int((d.type == "fausse").sum()), "euros": float(d.euros.sum())}


def _cle(x) -> str:
    return f"{x[0]:g}/{x[1]:g}"


def lire_cle(x: str) -> tuple[float, float]:
    """« 3/18 » → (3.0, 18.0)."""
    return tuple(float(v) for v in x.split("/"))


def grille_06(a: Annee, ref: dict) -> pd.DataFrame:
    """Les 9 couples (CANDIDATS_CUSUM × CANDIDATS_DEBIT) de l'étape 06, sans abstention comme à
    l'étape 06, notés sur l'année `a`."""
    lignes = []
    for p in G.CANDIDATS_CUSUM:
        for d in G.CANDIDATS_DEBIT:
            ch = chaine(a, ref, p, d, abstenir=False)
            lignes.append({"pression K/SEUIL": _cle(p), "débit K/SEUIL": _cle(d), **noter_vite(a, Ch.envois(ch))})
    return pd.DataFrame(lignes).set_index(["pression K/SEUIL", "débit K/SEUIL"])


def grille_confirmer(a: Annee, ch: pd.DataFrame, ref: dict, x_m3h=G.CONFIRMER_M3H_ETENDU, h=G.CONFIRMER_H
                     ) -> pd.DataFrame:
    """La grille « confirmer » de l'étape 09 (mode × X × H) sur la chaîne `ch` (avec abstention)."""
    return pd.DataFrame([{"mode": m, "X (m³/h)": x, "H (h)": hh, **noter_vite(a, confirmer(a, ch, ref, (m, x, hh)))}
                         for m in MODES for hh in h for x in x_m3h]).set_index(["mode", "X (m³/h)", "H (h)"])


def grille_fusionner(a: Annee, ch: pd.DataFrame, w_j=G.FUSIONNER_J_ETENDU) -> pd.DataFrame:
    """La grille « fusionner » de l'étape 09 (W) sur la chaîne `ch` (avec abstention)."""
    return pd.DataFrame([{"W (j)": w, **noter_vite(a, E9.fusionner(ch, w))} for w in w_j]).set_index("W (j)")


def en_cache(nom: str, calcul) -> pd.DataFrame:
    """Une grille de toutes les années (colonne `année`), gardée dans `sorties/simulees_<nom>.csv`."""
    f = SORTIES / f"simulees_{nom}.csv"
    if f.exists():
        t = pd.read_csv(f)
        return t.set_index([c for c in t.columns if c not in ("N", "fuites", "trouvées", "fausses", "euros")])
    # (pour refaire une grille après avoir changé un réglage, effacer son fichier)
    t = calcul()
    t.reset_index().to_csv(f, index=False)
    return t


def grilles(ans: dict[str, Annee], une) -> pd.DataFrame:
    """`une(annee)` pour chaque année, en un seul tableau avec une colonne `année` en tête d'index."""
    return pd.concat({n: une(a) for n, a in ans.items()}, names=["année"])


def total(t: pd.DataFrame, garder=None) -> pd.DataFrame:
    """La somme sur les années (toutes, ou celles de `garder`) de chaque réglage de la grille `t`."""
    if garder is not None:
        t = t[t.index.get_level_values("année").isin(garder)]
    s = t.groupby(level=list(range(1, t.index.nlevels))).sum(numeric_only=True)
    s["part trouvée"] = s["trouvées"] / s["fuites"]
    return s[["N", "trouvées", "part trouvée", "fausses", "euros"]]


def meilleur(t: pd.DataFrame, garder=None):
    """Le réglage de la grille `t` qui rapporte le plus d'euros en tout (sur les années `garder`)."""
    return total(t, garder).euros.idxmax()


def sans_une_variante(t: pd.DataFrame) -> pd.Series:
    """Le réglage choisi avec toutes les années, puis en retirant les 3 années d'une variante."""
    noms = sorted(set(t.index.get_level_values("année")))
    out = {"toutes les années": meilleur(t)}
    for v in sorted({n.split("_")[0] for n in noms}):
        reste = [n for n in noms if not n.startswith(v + "_")]
        if reste:
            out[f"sans la variante {v}"] = meilleur(t, reste)
    return pd.Series(out)


def au_bord(choix, grille_valeurs) -> bool:
    """Le choix tombe-t-il au bord de la grille (la plus petite ou la plus grande valeur) ?"""
    return choix in (min(grille_valeurs), max(grille_valeurs))


# --------------------------------------------------------------------- les tableaux complets
def ligne(a: Annee, ch: pd.DataFrame, env: pd.DataFrame, nom: str) -> dict:
    """Une ligne de tableau complète (étape 09) : N, trouvées, fausses par origine, euros, hasard ± σ, z."""
    return {**E9.ligne_origines(a.b, ch, env, a.fuites, a.debits, nom), "fuites": len(a.b.fuites)}


def somme(lignes: list[dict], nom: str) -> dict:
    """Plusieurs années en une ligne : les comptes et les euros s'ajoutent, le hasard aussi (les années
    sont indépendantes, donc les variances s'ajoutent), et z est celui de la somme."""
    t = pd.DataFrame(lignes)
    s = t[["N", "fuites", "trouvées", "fausses", "fausses : détecteur", "fausses : localisation", "répétées",
           "euros", "hasard (€)"]].sum()
    sigma = float(np.sqrt((t["σ (€)"].astype(float) ** 2).sum()))
    return {"règle": nom, **s.to_dict(), "σ (€)": round(sigma), "z": round((s.euros - s["hasard (€)"]) / sigma, 1)}


def tableau(lignes: list[dict]) -> pd.DataFrame:
    """Les lignes prêtes à afficher : trouvées « x / n », hasard « moyenne ± σ »."""
    t = pd.DataFrame(lignes).set_index("règle")
    t["trouvées"] = [f"{int(x)} / {int(n)}" for x, n in zip(t["trouvées"], t["fuites"])]
    t["hasard ± σ (€)"] = [f"{m:,.0f} ± {s:,.0f}".replace(",", " ") for m, s in zip(t["hasard (€)"], t["σ (€)"])]
    return t[["N", "trouvées", "fausses", "fausses : détecteur", "fausses : localisation", "euros",
              "hasard ± σ (€)", "z"]]


def fuites_trouvees(a: Annee, env: pd.DataFrame) -> pd.DataFrame:
    """Fuite par fuite : genre, débit maximal, montée, et le retard de l'alarme qui la trouve (jours ;
    vide si elle n'est pas trouvée)."""
    d = a.b.noter(env)
    v = d[d.type == "vraie"].groupby("fuite").envoi.min()
    t = a.fuites[["genre", "debit_max", "montee_j", "debut"]].copy()
    t["retard (j)"] = ((v.reindex(t.index) - t.debut).dt.total_seconds() / 86400).astype(float)
    return t


def par_classe(regles: dict[str, list[pd.DataFrame]]) -> pd.DataFrame:
    """Pour chaque règle (une liste de `fuites_trouvees`, une par année, mises ensemble) et chaque
    classe de fuite (`A12.classe`) : combien de fuites, la part trouvée et le retard médian (heures
    pour les casses, jours pour les fuites qui grandissent)."""
    out = {}
    for nom, tables in regles.items():
        x = pd.concat(tables)
        x = x.assign(classe=A12.classe(x), k=np.where(x.genre == "casse", 24.0, 1.0))
        g = x.groupby("classe")
        out[nom] = pd.DataFrame({"fuites": g.size(), "part trouvée": g["retard (j)"].apply(lambda r: r.notna().mean()),
                                 "retard médian (h ou j)": (x["retard (j)"] * x.k).groupby(x.classe).median()})
    return pd.concat(out, axis=1).reindex(A12.CLASSES)


# --------------------------------------------------------------------- tous les réglages (carnet 13)
@dataclass(frozen=True)
class Reglages:
    """Tous les réglages qui peuvent changer les alarmes envoyées, dans l'ordre de la chaîne. Par
    défaut : la chaîne finale réglée sur 2018 (étape 11 : dual, confirmer, regrouper à 300 m)."""
    cusum: tuple = G.CHOIX_CUSUM                    # 02 et 06 : (K cm, SEUIL cm·h)
    debit: tuple = G.CHOIX_DEBIT                    # 03 et 06 : (K m³/h, SEUIL m³/h·h)
    pause_h: int = G.PAUSE_H                        # 02 : silence après une alarme (les deux CUSUM)
    attente_h: float = G.ATTENTE_H                  # 05 : alarme de pression, heures observées puis envoi
    semaines_j: float = G.SEMAINES_J                # 05 : alarme de débit, jours avant et après
    amplitude_max: float = G.AMPLITUDE_MAX          # 05 : amplitude maximale d'une signature
    visible_cm: float = G.VISIBLE_CM                # 05 : signature minimale d'une candidate
    sigma: tuple = (G.SIGMA_CM, G.SIGMA_RELATIF)    # 05 : σ de la carte (cm, part)
    abstenir: bool = G.ABSTENIR_SI_PLATE            # 07
    rayon: float | None = G.RAYON                   # 08 : None = pas de regroupement
    combiner: str = "confirmer"                     # 09 : « aucun », « confirmer », « fusionner », « fondre »
    confirmer: tuple = G.CHOIX_CONFIRMER            # 09 : (quelles alarmes, X m³/h, H heures)
    fusionner_j: float = G.CHOIX_FUSIONNER_J        # 09 : W
    fondre: tuple = G.CHOIX_FONDRE                  # 09 : (K m³/h, SEUIL m³/h·h) du signal fondu
    pente: tuple | None = None                      # 01 : ((zone, cm par m³/h), …) ; None = celle de `ref`
    carte: str = "dual"                             # 11 : « pression » ou « dual »
    sigma_dual: tuple | None = None                 # 11 : (m³/h, part) ; None = celui de `ref`

    def avec(self, **x) -> Reglages:
        return replace(self, **x)


# le contexte de chaque étape : ce qui n'existe pas encore à cette étape de la chaîne
ETAPE = {"06": {"abstenir": False, "rayon": None, "combiner": "aucun", "carte": "pression"},
         "07": {"rayon": None, "combiner": "aucun", "carte": "pression"},
         "08": {"combiner": "aucun", "carte": "pression"},
         "09": {"rayon": None, "carte": "pression"},
         "11": {}}


def a_l_etape(r: Reglages, etape: str) -> Reglages:
    """Les réglages `r` tels que la chaîne les voit à `etape` (les étapes suivantes retirées)."""
    return r.avec(**ETAPE[etape])


@lru_cache(maxsize=1)
def _reseau():
    """Le réseau, lu une seule fois (il ne sert ici qu'aux zones des conduites)."""
    return D.reseau()


def _pente(r: Reglages, ref: dict) -> dict:
    return ref["pente"] if r.pente is None else dict(r.pente)


def _lecture(r: Reglages) -> dict:
    return {"attente_h": r.attente_h, "semaines_j": r.semaines_j, "visible_cm": r.visible_cm,
            "sigma": r.sigma[0], "relatif": r.sigma[1], "amplitude_max": r.amplitude_max}


def signal_fondu(a: Annee, pente: dict) -> dict:
    """Le signal fondu de l'étape 09 pour l'année `a` (poids selon le bruit de sa première semaine)."""
    cle = ("fondu", tuple(sorted(pente.items())))
    if cle not in a.cache:
        q_p, q_d = E9.en_m3h(a.sig_p, pente), {z: -x for z, x in a.sig_d.items()}
        a.cache[cle] = E9.signal_fondu(E9.fondre(q_p, q_d, E9.bruits(q_p, q_d)))
    return a.cache[cle]


def placees(a: Annee, ref: dict, r: Reglages, source: str) -> pd.DataFrame:
    """Les alarmes d'une seule source (« pression », « débit » ou « fondu »), placées par la carte de
    pression, puis par celle du dual si `r.carte` le dit. Gardées en mémoire dans `a.cache` : une même
    source sert à beaucoup de réglages de la grille."""
    lu = _lecture(r)
    detecteur = {"pression": r.cusum, "débit": r.debit, "fondu": r.fondre}[source]
    cle = (source, detecteur, r.pause_h, tuple(lu.items()), r.abstenir,
           tuple(sorted(_pente(r, ref).items())) if source == "fondu" else None)
    if cle not in a.cache:
        if source == "fondu":
            al = E9.alarmes_fondues(signal_fondu(a, _pente(r, ref)), *r.fondre)
        else:
            al = Ch.alarmes(a.sig_p if source == "pression" else {}, a.sig_d if source == "débit" else {},
                            r.cusum, r.debit, r.pause_h)
        a.cache[cle] = (_placer_avec_memoire(a, ref, r, al) if len(al)
                        else pd.DataFrame(columns=["alarme", "zone", "source", "envoi", "conduite", "candidates"]))
    ch = a.cache[cle]
    if r.carte == "pression" or len(ch) == 0:
        return ch
    s = (ref["s_dual"], ref["r_dual"]) if r.sigma_dual is None else r.sigma_dual
    cle_d = cle + ("dual", s)
    if cle_d not in a.cache:
        a.cache[cle_d] = L11.relocaliser(ch, a.qv, ref["dico_d"], *s, r.abstenir, r.amplitude_max)
    return a.cache[cle_d]


def _placer_avec_memoire(a: Annee, ref: dict, r: Reglages, al: pd.DataFrame) -> pd.DataFrame:
    """`Ch.placer`, alarme par alarme, avec une mémoire : une alarme de pression ne dépend que de son
    instant et de sa zone (fenêtres de 24 h), une alarme lue sur des semaines aussi de son départ. Deux
    réglages du CUSUM qui sonnent au même instant partagent donc le même placement."""
    lu = _lecture(r)
    mem = a.cache.setdefault(("alarmes", tuple(lu.items()), r.abstenir), {})
    fen = al["fenetre"] if "fenetre" in al else pd.Series(np.where(al.source == "pression", "24h", "semaines"),
                                                          index=al.index)
    cles = [(t, None if f == "24h" else d, z, f) for t, d, z, f in zip(al.alarme, al.depart, al.zone, fen)]
    neuves = [i for i, c in zip(al.index, cles) if c not in mem]
    if neuves:
        nouv = al.loc[neuves]
        placees_ = Ch.placer(a.res, nouv, ref["dico"], _reseau(), r.abstenir, lu)
        faites = {(x.alarme, x.depart, x.zone): x._asdict() for x in placees_.drop(columns="source").itertuples(index=False)}
        for i in neuves:
            x = al.loc[i]
            c = cles[al.index.get_loc(i)]
            mem[c] = faites.get((x.alarme, x.depart, x.zone))
    lignes = []
    for c, (_, x) in zip(cles, al.iterrows()):
        if mem[c] is not None:
            lignes.append({**mem[c], "alarme": x.alarme, "depart": x.depart, "zone": x.zone, "source": x.source})
    ch = pd.DataFrame(lignes, columns=placees_cols()).sort_values("envoi", kind="stable").reset_index(drop=True)
    ch.index.name = "id"
    return ch


def placees_cols() -> list[str]:
    return ["alarme", "depart", "zone", "source", "avant_debut", "avant_fin", "apres_debut", "apres_fin", "envoi", "plate",
            "conduite", "debit_estime", "candidates"]


def chaine_de(a: Annee, ref: dict, r: Reglages) -> pd.DataFrame:
    """Toutes les alarmes placées de l'année avec les réglages `r`, dans l'ordre de `Ch.chaine`."""
    sources = ("fondu",) if r.combiner == "fondre" else ("pression", "débit")
    morceaux = [placees(a, ref, r, s) for s in sources]
    ch = pd.concat([m for m in morceaux if len(m)], ignore_index=True) if any(len(m) for m in morceaux) \
        else morceaux[0]
    ch = ch.sort_values(["envoi", "alarme", "source", "zone"], kind="stable").reset_index(drop=True)
    ch.index.name = "id"
    return ch


def envoyer(a: Annee, ref: dict, r: Reglages) -> tuple[pd.DataFrame, pd.DataFrame]:
    """La chaîne complète avec les réglages `r` : (alarmes placées, alarmes envoyées)."""
    ch = chaine_de(a, ref, r)
    if len(ch) == 0:
        return ch, ch
    if r.combiner == "confirmer":
        mode, x, h = r.confirmer
        env = E9.confirmer(ch, a.bilan_h, a.res_zone_h, _pente(r, ref), x, h, MODES[mode])
    elif r.combiner == "fusionner":
        env = E9.fusionner(ch, r.fusionner_j)
    else:
        env = Ch.envois(ch)
    return ch, env if r.rayon is None else Rg.regrouper(env, ref["dist"], r.rayon)


# --------------------------------------------------------------------- les grilles de réglages
def ecrire(champ: str, v) -> str:
    """Une valeur de réglage en texte court, pour l'index d'une grille (et son fichier)."""
    if v is None:
        return "aucun"
    if champ in ("cusum", "debit", "fondre", "sigma", "sigma_dual"):
        return _cle(v)
    if champ == "confirmer":
        return f"{v[0]} {v[1]:g} {v[2]:g}"
    if champ == "pente":
        return " ".join(f"{z}={p:.4g}" for z, p in v)
    return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def lire(champ: str, s: str):
    """L'inverse de `ecrire`."""
    s = str(s)
    if s == "aucun" and champ in ("rayon", "pente", "sigma_dual"):
        return None
    if champ in ("cusum", "debit", "fondre", "sigma", "sigma_dual"):
        return lire_cle(s)
    if champ == "confirmer":
        m, x, h = s.rsplit(" ", 2)
        return (m, float(x), int(float(h)))
    if champ == "pente":
        return tuple((z, float(p)) for z, p in (x.split("=") for x in s.split()))
    if champ == "abstenir":
        return s == "True"
    if champ in ("combiner", "carte"):
        return s
    return float(s) if champ != "pause_h" else int(float(s))


def grille(a: Annee, ref: dict, base: Reglages, valeurs: dict[str, list]) -> pd.DataFrame:
    """Chaque combinaison des `valeurs` (champ → valeurs essayées), les autres réglages pris dans
    `base`, notée sur l'année `a` (`noter_vite`). Index : un niveau par champ, en texte (`ecrire`)."""
    champs = list(valeurs)
    lignes = []
    for combo in itertools.product(*valeurs.values()):
        r = base.avec(**dict(zip(champs, combo)))
        note = noter_rapide(a, ref, r) if r.combiner in ("aucun", "fondre") and r.rayon is None \
            else noter_vite(a, envoyer(a, ref, r)[1])
        lignes.append({**{c: ecrire(c, v) for c, v in zip(champs, combo)}, **note})
    return pd.DataFrame(lignes).set_index(champs)


def _tableaux(a: Annee, ref: dict, r: Reglages, source: str) -> tuple:
    """Les alarmes envoyées d'une source, en tableaux (pas du barème, rang de la conduite, et de quoi les
    ranger dans l'ordre de `chaine_de`). Gardés dans `a.cache`."""
    ch = placees(a, ref, r, source)
    cle = ("tableaux", id(ch))
    if cle not in a.cache:
        e = ch[ch.conduite.notna()]
        ns = lambda x: pd.DatetimeIndex(x).asi8 if len(x) else np.zeros(0, np.int64)    # noqa: E731
        a.cache[cle] = (np.array([a.b.pas(t) for t in e.envoi], int), np.array([a.b.rang[p] for p in e.conduite], int),
                        ns(e.envoi), ns(e.alarme), e.source.to_numpy(str), e.zone.to_numpy(str), ch)
    return a.cache[cle][:6]


def noter_rapide(a: Annee, ref: dict, r: Reglages) -> dict:
    """Comme `noter_vite(a, envoyer(a, ref, r)[1])`, sans tableaux pandas, quand on envoie toutes les
    alarmes placées (rien à combiner, pas de regroupement) : les grilles des CUSUM en sont 20 fois plus
    rapides."""
    sources = ("fondu",) if r.combiner == "fondre" else ("pression", "débit")
    t = [_tableaux(a, ref, r, s) for s in sources]
    pas, rang, envoi, alarme, source, zone = (np.concatenate([x[i] for x in t]) for i in range(6))
    ordre = np.lexsort((zone, source, alarme, envoi))
    e, typ, _, _ = a.b._noter(rang[ordre], pas[ordre])
    return {"N": len(pas), "fuites": len(a.b.fuites), "trouvées": int((typ == "vraie").sum()),
            "fausses": int((typ == "fausse").sum()), "euros": float(e.sum())}


def choix_de(t: pd.DataFrame, base: Reglages, garder=None) -> Reglages:
    """`base` avec le meilleur réglage de la grille `t` (somme des euros sur les années `garder`)."""
    m = meilleur(t, garder)
    m = m if isinstance(m, tuple) else (m,)
    return base.avec(**{c: lire(c, v) for c, v in zip(t.index.names[1:], m)})


def bords(t: pd.DataFrame, garder=None) -> dict[str, str]:
    """Pour chaque champ numérique de la grille `t` : le meilleur réglage tombe-t-il au bord des
    valeurs essayées ? Rend {champ : « bas » ou « haut »} pour ceux qui y sont."""
    m = meilleur(t, garder)
    m = m if isinstance(m, tuple) else (m,)
    out = {}
    for i, (c, v) in enumerate(zip(t.index.names[1:], m)):
        vals = sorted(set(t.index.get_level_values(i + 1)))
        if c in ("cusum", "debit", "fondre"):
            for j, nom in enumerate(("K", "SEUIL")):
                xs = sorted({lire_cle(x)[j] for x in vals})
                if len(xs) > 1 and lire_cle(v)[j] in (xs[0], xs[-1]):
                    out[f"{c} {nom}"] = "bas" if lire_cle(v)[j] == xs[0] else "haut"
        elif c == "confirmer":
            for j, nom in ((1, "X"), (2, "H")):
                xs = sorted({lire(c, x)[j] for x in vals})
                if len(xs) > 1 and lire(c, v)[j] in (xs[0], xs[-1]):
                    out[f"{c} {nom}"] = "bas" if lire(c, v)[j] == xs[0] else "haut"
        elif c not in ("combiner", "carte", "abstenir", "rayon", "sigma", "sigma_dual", "pente"):
            xs = sorted(float(x) for x in vals)
            if len(xs) > 1 and float(v) in (xs[0], xs[-1]):
                out[c] = "bas" if float(v) == xs[0] else "haut"
        elif c == "rayon":
            xs = sorted(float(x) for x in vals if x != "aucun")
            if v != "aucun" and len(xs) > 1 and float(v) in (xs[0], xs[-1]):
                out[c] = "bas" if float(v) == xs[0] else "haut"
    return out


def grilles_paralleles(ans: dict[str, Annee], une, processus: int = 1) -> pd.DataFrame:
    """Comme `grilles`, une année par processus."""
    if processus <= 1:
        return grilles(ans, une)
    global _TACHE
    _TACHE = (ans, une)
    with mp.get_context("fork").Pool(min(processus, len(ans))) as pool:
        tables = pool.map(_une_annee, list(ans))
    return pd.concat(dict(zip(ans, tables)), names=["année"])


_TACHE = None


def _une_annee(nom: str) -> pd.DataFrame:
    ans, une = _TACHE
    return une(ans[nom])


def en_cache_texte(nom: str, calcul) -> pd.DataFrame:
    """Comme `en_cache`, l'index relu en texte (les grilles de `grille`)."""
    f = SORTIES / f"simulees_{nom}.csv"
    if f.exists():
        t = pd.read_csv(f, dtype=str)
        idx = [c for c in t.columns if c not in ("N", "fuites", "trouvées", "fausses", "euros")]
        for c in ("N", "fuites", "trouvées", "fausses"):
            t[c] = t[c].astype(int)
        t["euros"] = t["euros"].astype(float)
        return t.set_index(idx)
    t = calcul()
    t.reset_index().to_csv(f, index=False)
    return t


# --------------------------------------------------------------------- σ et pente sur les années simulées
def sigmas_simulees(ans: dict[str, Annee], ref: dict) -> dict[str, tuple[float, float]]:
    """σ de la carte de pression (étape 05) et σ du dual (étape 11), réglés comme sur 2018 mais sur les
    fuites des 12 années simulées mises ensemble (première alarme de pression de chaque fuite, CUSUM de
    référence de l'étape 02)."""
    ep, ed = [], []
    for a in ans.values():
        notes = C.detecter_annee(a.sig_p, a.fuites, a.debits)
        ep.append(Sg.ecarts_par_fuite(a.res, notes, ref["dico"]))
        ed.append(L11.ecarts_par_fuite(a.qv, notes, ref["dico_d"]))
    return {"pression": Sg.regler_sigma(pd.concat(ep)), "dual": L11.regler_sigma(pd.concat(ed))}


def pente_simulees(ans: dict[str, Annee]) -> tuple:
    """La pente cm par m³/h de chaque zone (étape 01), sur les jours des 12 années simulées mis ensemble."""
    return tuple((z, float(R1.droite(pd.concat([R1.residu_contre_fuite(a.res, a.zones, a.debits, a.fuites, z)
                                               for a in ans.values()]))["pente_cm_par_m3h"]))
                 for z in ("AB", "C"))


def memes_alarmes(a: Annee, ref: dict, r1: Reglages, r2: Reglages) -> bool:
    """Les deux réglages envoient-ils exactement les mêmes alarmes (conduite et date d'envoi) ?"""
    e1, e2 = envoyer(a, ref, r1)[1], envoyer(a, ref, r2)[1]
    return (len(e1) == len(e2) and list(e1.conduite) == list(e2.conduite)
            and list(e1.envoi) == list(e2.envoi))


# --------------------------------------------------------------------- l'inventaire (carnet 13)
# (réglage, étape, comment il a été choisi, sur quoi, change-t-il les alarmes envoyées ?, réglé sur les simulées ?)
INVENTAIRE = [
    ("CANDIDATS_CUSUM (2/24, 3/12, 3/18)", "02", "front de Pareto : fausses alarmes contre retard médian, toutes les "
     "fuites vues", "2018", "oui : le couple de l'étape 06 est pris parmi eux", "carnet 06 : toute la grille"),
    ("CANDIDATS_DEBIT (1,5/40, 1,5/80, 3/10)", "03", "idem, sur le bilan", "2018", "oui (idem)", "carnet 06 : toute la grille"),
    ("CHOIX_CUSUM, CHOIX_DEBIT (3/18, 3/10)", "06", "euros, parmi les 3 × 3 candidats", "2018",
     "oui", "carnet 06 (déjà en phase 2 parmi les candidats ; ici toute la grille)"),
    ("PAUSE_H (24 h)", "02", "choix : égale au lissage, pas de grille", "aucune donnée", "oui", "carnet 06"),
    ("LISSAGE_H (24 h), REFERENCE_DEPART_H (72 h), la veille (48 → 24 h)", "01, 02, 05", "choix", "aucune donnée",
     "oui", "non : jamais réglés, ni sur 2018 ni ici"),
    ("RETARD_EN_PLUS_J, CROISSANCE, PART", "02", "choix", "aucune donnée",
     "non : ils ne servent qu'à choisir les candidats", "inutile : toute la grille remplace les candidats"),
    ("SIGMA_CM (0,6 cm), SIGMA_RELATIF (0,06)", "05", "écart médian à la bonne conduite (`regler_sigma`)",
     "fuites de 2018", "non : σ ne change pas la conduite la plus probable (vérifié)", "carnet 05"),
    ("PART_NETTE (0,5)", "05", "choix", "aucune donnée", "non : il ne sert qu'à régler σ", "carnet 05 (avec σ)"),
    ("ATTENTE_H (24 h)", "05", "choix (dossier 04)", "aucune grille dans ce dossier", "oui : fenêtre et date d'envoi",
     "carnet 05"),
    ("SEMAINES_J (7 j)", "05", "choix (dossier 04)", "aucune grille dans ce dossier", "oui : fenêtre et date d'envoi",
     "carnet 05"),
    ("AMPLITUDE_MAX (5)", "04", "choix", "aucune donnée", "oui : écart de chaque conduite", "carnet 05"),
    ("VISIBLE_CM (0,5 cm)", "04", "choix", "aucune donnée", "oui : liste des candidates", "carnet 05"),
    ("s'abstenir si la carte est plate", "07", "correction d'un défaut vu sur les alarmes de 2018",
     "2018", "oui", "carnet 07"),
    ("regrouper : la règle et RAYON (300 m)", "08", "règle choisie, rayon pris égal à celui du barème", "2018 (z monte)",
     "oui", "carnet 08 (sans, ou rayon de 0 à 1 000 m et plus)"),
    ("CHOIX_CONFIRMER (pression, 4 m³/h, 24 h)", "09", "euros", "2018", "oui",
     "carnet 09 (phase 2 ; refait ici sur la chaîne de la phase 3)"),
    ("CHOIX_FUSIONNER_J (2 j)", "09", "euros", "2018", "oui", "carnet 09 (phase 2 ; refait ici)"),
    ("CHOIX_FONDRE (2/40) : CUSUM du signal fondu", "09", "front de Pareto puis euros", "2018",
     "oui, si on fond les signaux", "carnet 09"),
    ("RAPIDE_J (2 j)", "09", "choix", "aucune donnée", "oui, si on fond les signaux", "non (choix ; branche écartée)"),
    ("la pente cm par m³/h", "01", "droite résidu contre fuites", "fuites de 2018",
     "seulement pour fondre, ou confirmer une alarme de débit", "carnet 09 (pente des simulées)"),
    ("combiner : rien, confirmer, fusionner ou fondre", "09, 11", "comparaison des lignes", "2018", "oui", "carnets 09 et 11"),
    ("localiser avec la carte de pression ou celle du dual", "11", "comparaison des lignes", "2018", "oui", "carnet 11"),
    ("σ du dual (0,16 m³/h, relatif 0,17)", "11", "comme SIGMA_CM", "fuites de 2018",
     "non : même raison que σ (vérifié)", "carnet 11"),
    ("POUSSE_MIN, MARCHE_MIN", "07, 10", "choix", "aucune donnée", "non : ils servent à classer et à décrire",
     "inutile"),
]


def inventaire() -> pd.DataFrame:
    """Le tableau des réglages de la chaîne : où et sur quoi ils ont été choisis, s'ils changent les
    alarmes envoyées, et où ils sont réglés sur les années simulées."""
    return pd.DataFrame(INVENTAIRE, columns=["réglage", "étape", "choisi par", "choisi sur", "change les alarmes ?",
                                             "réglé sur les simulées"]).set_index("réglage")


# --------------------------------------------------------------------- les tableaux des carnets (phase 3)
def grilles_des_deux(nom: str, a18: Annee, ans: dict[str, Annee], ref: dict, etape: str, valeurs: dict,
                     r18: Reglages, rsim: Reglages, processus: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """La même grille deux fois : sur 2018 autour de la chaîne réglée sur 2018, et sur les 12 années
    simulées autour de la chaîne réglée sur elles. Gardées dans `sorties/simulees_p3_<nom>(_2018).csv`."""
    t18 = en_cache_texte(f"p3_{nom}_2018", lambda: grilles({"2018": a18}, lambda a: grille(
        a, ref, a_l_etape(r18, etape), valeurs)))
    tsim = en_cache_texte(f"p3_{nom}", lambda: grilles_paralleles(ans, lambda a: grille(
        a, ref, a_l_etape(rsim, etape), valeurs), processus))
    return t18, tsim


def cote_a_cote(t18: pd.DataFrame, tsim: pd.DataFrame) -> pd.DataFrame:
    """Chaque réglage d'une grille : sur 2018 (N, fausses, euros) et la somme des 12 années simulées,
    avec l'écart au meilleur de chaque côté."""
    a, b = total(t18), total(tsim)
    t = pd.DataFrame({"2018 : N": a.N, "2018 : trouvées": a["trouvées"], "2018 : fausses": a.fausses,
                      "2018 : euros": a.euros.round(),
                      "simulées : N": b.N, "simulées : trouvées": b["trouvées"], "simulées : fausses": b.fausses,
                      "simulées : euros": b.euros.round()})
    t["2018 : écart au meilleur"] = t["2018 : euros"] - t["2018 : euros"].max()
    t["simulées : écart au meilleur"] = t["simulées : euros"] - t["simulées : euros"].max()
    ordre = list(dict.fromkeys(tsim.index.droplevel(0)))           # l'ordre de la grille, pas l'ordre alphabétique
    return t.reindex(ordre)


def lignes_completes(a18: Annee, ans: dict[str, Annee], ref: dict, versions: dict[str, Reglages], etape: str,
                     processus: int = 1) -> list[dict]:
    """Les lignes complètes (N, trouvées, fausses par origine, euros, hasard ± σ, z) de chaque version à
    `etape` : d'abord notées sur 2018, puis sommées sur les 12 années simulées."""
    def une(a):
        out = []
        for n, r in versions.items():
            ch, env = envoyer(a, ref, a_l_etape(r, etape))
            out.append(ligne(a, ch, env, n))
        return pd.DataFrame(out)
    t = grilles_paralleles(ans, une, processus)
    l18 = [{**x, "règle": f"{x['règle']} → noté sur 2018"} for x in une(a18).to_dict("records")]
    lsim = [somme(t[t["règle"] == n].to_dict("records"), f"{n} → somme des 12 années simulées") for n in versions]
    return l18 + lsim


def classes_par_version(ans: dict[str, Annee], ref: dict, versions: dict[str, Reglages], etape: str) -> pd.DataFrame:
    """`par_classe` de chaque version à `etape`, sur les 12 années simulées."""
    return par_classe({n: [fuites_trouvees(a, envoyer(a, ref, a_l_etape(r, etape))[1]) for a in ans.values()]
                       for n, r in versions.items()})


def reglages_simules() -> Reglages:
    """La chaîne réglée sur les années simulées, étape par étape (`G.REGLE_SIMULEES`, carnet 13)."""
    return Reglages(**G.REGLE_SIMULEES)


def paires(ks, seuils) -> list[tuple[float, float]]:
    """Tous les couples (K, SEUIL)."""
    return [(float(k), float(s)) for k in ks for s in seuils]


def ecrire_reglage(r: Reglages, t: pd.DataFrame) -> str:
    """Les valeurs de `r` pour les champs de la grille `t`, en texte."""
    return ", ".join(f"{c} = {ecrire(c, getattr(r, c))}" for c in t.index.names[1:])


def grilles_p3(g: dict | None = None) -> dict[str, tuple[str, dict, dict]]:
    """Les grilles de la phase 3, dans l'ordre de la chaîne : nom → (étape, valeurs essayées, réglages
    imposés pendant la grille). Les valeurs sont celles de `g`, par défaut `G.GRILLES_SIMULEES` (déjà
    prolongées)."""
    g = G.GRILLES_SIMULEES if g is None else g
    conf = [(m, x, h) for m in MODES for h in g["confirmer_h"] for x in g["confirmer_m3h"]]
    combiner = ["aucun", "confirmer", "fusionner", "fondre"]
    return {
        "cusum": ("06", {"cusum": paires(*g["cusum"]), "debit": paires(*g["debit"]), "pause_h": g["pause_h"]}, {}),
        **{c: ("06", {c: g[c]}, {}) for c in ("attente_h", "semaines_j", "amplitude_max", "visible_cm")},
        "abstenir": ("07", {"abstenir": [False, True]}, {}),
        "rayon": ("08", {"rayon": g["rayon"]}, {}),
        "confirmer": ("09", {"combiner": ["confirmer"], "confirmer": conf}, {"combiner": "confirmer"}),
        "fusionner": ("09", {"combiner": ["fusionner"], "fusionner_j": g["fusionner_j"]}, {"combiner": "fusionner"}),
        "fondre": ("09", {"combiner": ["fondre"], "fondre": paires(*g["fondre"])}, {"combiner": "fondre"}),
        "combiner": ("09", {"combiner": combiner}, {}),
        "finale": ("11", {"carte": ["pression", "dual"], "combiner": combiner, "rayon": g["rayon"]}, {}),
    }


def grilles_etape(nom: str, a18: Annee, ans: dict[str, Annee], ref: dict, r18: Reglages, rsim: Reglages,
                  processus: int = 1) -> tuple[pd.DataFrame, pd.DataFrame]:
    """La grille `nom` de `grilles_p3`, sur 2018 et sur les années simulées (`grilles_des_deux`)."""
    etape, valeurs, x = grilles_p3()[nom]
    return grilles_des_deux(nom, a18, ans, ref, etape, valeurs, r18.avec(**x), rsim.avec(**x), processus)


def resume_des_choix(a18: Annee, ans: dict[str, Annee], ref: dict, r18: Reglages, rsim: Reglages,
                     processus: int = 1) -> pd.DataFrame:
    """Une ligne par grille : le réglage de 2018, le meilleur sur 2018 dans la même grille, celui des
    simulées, s'il est au bord, et combien des 4 « sans une variante » le gardent."""
    lignes = {}
    for nom, (_, _, x) in grilles_p3().items():
        t18, tsim = grilles_etape(nom, a18, ans, ref, r18, rsim, processus)
        champs = [c for c in t18.index.names[1:] if not (x and c == "combiner")]
        tx = lambda r: ", ".join(ecrire(c, getattr(r, c)) for c in champs)        # noqa: E731
        stab = sans_une_variante(tsim)
        lignes[nom] = {"réglé sur 2018": tx(r18), "meilleur sur 2018 (même grille)": tx(choix_de(t18, r18.avec(**x))),
                       "réglé sur les simulées": tx(choix_de(tsim, rsim.avec(**x))),
                       "au bord ?": ", ".join(f"{k} ({v})" for k, v in bords(tsim).items()) or "non",
                       "gardé sans une variante": f"{int((stab.iloc[1:] == stab.iloc[0]).sum())} / {len(stab) - 1}"}
    return pd.DataFrame(lignes).T


def un_par_un(a18: Annee, ans: dict[str, Annee], ref: dict, r18: Reglages, rsim: Reglages, groupes: dict[str, list],
              processus: int = 1) -> pd.DataFrame:
    """La chaîne réglée sur 2018 où l'on ne change qu'un groupe de réglages (sa valeur des simulées) :
    l'écart d'euros sur 2018 et sur la somme des simulées, et sur 2018 en σ du hasard de la chaîne de 2018."""
    versions = {"réglé sur 2018": r18, **{n: r18.avec(**{c: getattr(rsim, c) for c in cs}) for n, cs in groupes.items()}}
    def une(a):
        return pd.DataFrame([noter_vite(a, envoyer(a, ref, r)[1]) for r in versions.values()], index=list(versions))
    t18 = une(a18)
    ts = grilles_paralleles(ans, une, processus).groupby(level=1).sum().reindex(list(versions))
    ch, env = envoyer(a18, ref, r18)
    sigma = float(Ch.hasard(a18.b, env).std())
    out = pd.DataFrame({"valeur sur les simulées": [""] + [", ".join(f"{c} = {ecrire(c, getattr(rsim, c))}" for c in cs)
                                                          for cs in groupes.values()],
                        "2018 : N": t18.N, "2018 : trouvées": t18["trouvées"], "2018 : fausses": t18.fausses,
                        "2018 : euros": t18.euros.round(), "2018 : écart (€)": (t18.euros - t18.euros.iloc[0]).round(),
                        "2018 : écart (σ du hasard)": ((t18.euros - t18.euros.iloc[0]) / sigma).round(2),
                        "simulées : euros": ts.euros.round(), "simulées : écart (€)": (ts.euros - ts.euros.iloc[0]).round()},
                       index=list(versions))
    entiers = [c for c in out.columns if c.endswith("euros") or c.endswith("(€)")]
    return out.astype({c: "Int64" for c in entiers})


def comparer_alarmes(a: Annee, ref: dict, r1: Reglages, r2: Reglages) -> dict:
    """Combien d'alarmes envoyées changent (conduite ou date) entre deux réglages, et l'écart d'euros."""
    e1, e2 = envoyer(a, ref, r1)[1], envoyer(a, ref, r2)[1]
    if len(e1) != len(e2):
        n = abs(len(e1) - len(e2)) + sum(p != q for p, q in zip(e1.conduite, e2.conduite))
    else:
        n = sum(p != q or s != t for p, q, s, t in zip(e1.conduite, e2.conduite, e1.envoi, e2.envoi))
    return {"alarmes qui changent": int(n), "écart (€)": noter_vite(a, e2)["euros"] - noter_vite(a, e1)["euros"]}


def effet_des_sigmas(a18: Annee, ans: dict[str, Annee], ref: dict, base: Reglages, champ: str, essais: dict) -> pd.DataFrame:
    """Pour chaque valeur de σ essayée (`champ` = « sigma » ou « sigma_dual ») : les alarmes qui changent
    et l'écart d'euros par rapport à `base`, sur 2018 et sur la somme des 12 années simulées."""
    lignes = {}
    for nom, v in essais.items():
        x18 = comparer_alarmes(a18, ref, base, base.avec(**{champ: tuple(v)}))
        xs = [comparer_alarmes(a, ref, base, base.avec(**{champ: tuple(v)})) for a in ans.values()]
        lignes[nom] = {"2018 : alarmes qui changent": x18["alarmes qui changent"], "2018 : écart (€)": x18["écart (€)"],
                       "simulées : alarmes qui changent": sum(x["alarmes qui changent"] for x in xs),
                       "simulées : écart (€)": sum(x["écart (€)"] for x in xs)}
    return pd.DataFrame(lignes).T


# --------------------------------------------------------------------- régler toute la chaîne (validation croisée)
# les prolongements possibles de chaque grille, vers le bas et vers le haut, dans l'ordre où on les essaie
PROLONGER = {
    "cusum": {"cusum K": ([0.5], [12.0, 16.0]), "cusum SEUIL": ([1.0], [96.0, 144.0]),
              "debit K": ([0.25], [5.0, 6.0, 8.0]), "debit SEUIL": ([1.25], [240.0]), "pause_h": ([3], [144])},
    "attente_h": {"attente_h": ([2.0, 1.0], [72.0, 96.0])},
    "semaines_j": {"semaines_j": ([2.0, 1.0], [21.0, 28.0])},
    "amplitude_max": {"amplitude_max": ([1.0, 0.5], [])},
    "visible_cm": {"visible_cm": ([], [3.0, 5.0])},
    "rayon": {"rayon": ([], [1500.0, 2000.0])},
    "confirmer": {"confirmer X": ([0.0625, 0.0], [16.0]), "confirmer H": ([0], [168])},
    "fusionner": {"fusionner_j": ([0.03125, 0.0], [10.0, 14.0])},
    "fondre": {"fondre K": ([0.0625], [5.0, 6.0]), "fondre SEUIL": ([5.0], [160.0])},
    "finale": {"rayon": ([], [1500.0, 2000.0])},
}
LECTURE = ("cusum", "attente_h", "semaines_j", "amplitude_max", "visible_cm")


def prolonger(g: dict, axe: str, x) -> None:
    """Ajoute la valeur `x` à la grille `g` le long de `axe` (« cusum K », « confirmer H », « rayon »…)."""
    champ, *quoi = axe.split(" ")
    if champ in ("cusum", "debit", "fondre"):
        ks, ss = (list(v) for v in g[champ])
        g[champ] = (sorted(ks + [x]), ss) if quoi[0] == "K" else (ks, sorted(ss + [x]))
    elif champ == "confirmer":
        cle = "confirmer_m3h" if quoi[0] == "X" else "confirmer_h"
        g[cle] = sorted(list(g[cle]) + [x])
    else:
        g[champ] = sorted(list(g[champ]) + [x], key=lambda v: (v is not None, v if v is not None else 0))


def regler_chaine(ans: dict[str, Annee], ref: dict, processus: int = 1, journal=None) -> tuple[Reglages, dict]:
    """Toute la chaîne réglée sur les années `ans`, comme au carnet 13 : dans l'ordre de la chaîne, chaque
    grille prolongée tant que son meilleur réglage tombe au bord, les CUSUM et la lecture refaits jusqu'à
    ce qu'un tour entier ne change plus rien. Rend (les réglages, les grilles finales)."""
    journal = journal or (lambda *x: None)
    g = {k: (tuple(list(x) for x in v) if isinstance(v, tuple) else list(v)) for k, v in G.GRILLES_SIMULEES.items()}
    reste = {n: {a: [list(b), list(h)] for a, (b, h) in d.items()} for n, d in PROLONGER.items()}
    sig = sigmas_simulees(ans, ref)
    r = Reglages(sigma=tuple(sig["pression"]), sigma_dual=tuple(sig["dual"]), pente=pente_simulees(ans))

    def etape(nom, r):
        while True:
            et, valeurs, x = grilles_p3(g)[nom]
            t = grilles_paralleles(ans, lambda a: grille(a, ref, a_l_etape(r.avec(**x), et), valeurs), processus)
            ajout = False
            for axe, cote in bords(t).items():
                pile = reste.get(nom, {}).get(axe, [[], []])[0 if cote == "bas" else 1]
                if pile:
                    v = pile.pop(0)
                    prolonger(g, axe, v)
                    journal(f"{nom} : {axe} prolongé vers le {cote} ({v})")
                    ajout = True
            if not ajout:
                break
        choix = choix_de(t, r.avec(**x))
        return r.avec(**{c: getattr(choix, c) for c in valeurs if not (x and c == "combiner")})

    for tour in range(1, 6):
        avant = r
        for nom in LECTURE:
            r = etape(nom, r)
        journal(f"tour {tour} : {'point fixe' if r == avant else 'des changements'}")
        if r == avant:
            break
    for nom in ("abstenir", "rayon", "confirmer", "fusionner", "fondre", "combiner", "finale"):
        r = etape(nom, r)
    return r, g


def plis(ans: dict[str, Annee]) -> dict[str, list[str]]:
    """Les plis de la validation croisée : pour chaque variante, les années qu'on garde de côté."""
    return {v: [n for n in ans if n.startswith(v + "_")] for v in sorted({n.split("_")[0] for n in ans})}


def vers_json(r: Reglages) -> dict:
    """Les réglages en dictionnaire (les tuples deviennent des listes), valeurs exactes."""
    return {c: (list(map(list, v)) if c == "pente" and v is not None else list(v) if isinstance(v, tuple) else v)
            for c, v in vars(r).items()}


def depuis_json(d: dict) -> Reglages:
    """L'inverse de `vers_json`."""
    return Reglages(**{c: (tuple(map(tuple, v)) if c == "pente" and v is not None else tuple(v) if isinstance(v, list)
                           else v) for c, v in d.items()})


def regler_un_pli(ans: dict[str, Annee], ref: dict, v: str, processus: int = 1) -> dict:
    """Le pli `v` : toute la chaîne réglée sur les années des autres variantes (`regler_chaine`), gardée
    dans `sorties/simulees_cv_<v>.json` avec les grilles prolongées et la durée du calcul."""
    f = SORTIES / f"simulees_cv_{v}.json"
    if f.exists():
        return json.loads(f.read_text())
    import time
    t0, notes = time.time(), []
    r, _ = regler_chaine({n: a for n, a in ans.items() if n not in plis(ans)[v]}, ref, processus, notes.append)
    d = {"pli": v, "réglé sur": [n for n in ans if n not in plis(ans)[v]], "réglages": vers_json(r),
         "prolongements": [x for x in notes if "prolongé" in x], "tours": [x for x in notes if x.startswith("tour")],
         "secondes": round(time.time() - t0)}
    f.write_text(json.dumps(d, ensure_ascii=False, indent=1))
    return d


CHAMPS_CV = ["cusum", "debit", "pause_h", "attente_h", "semaines_j", "amplitude_max", "visible_cm", "abstenir", "rayon",
             "combiner", "confirmer", "fusionner_j", "fondre", "carte"]


def choix_des_plis(versions: dict[str, Reglages]) -> pd.DataFrame:
    """Les réglages de chaque version côte à côte (une colonne par version), en texte."""
    return pd.DataFrame({n: {c: ecrire(c, getattr(r, c)) for c in CHAMPS_CV} for n, r in versions.items()})


def lignes_cv(a18: Annee, ans: dict[str, Annee], ref: dict, reglages_plis: dict[str, Reglages],
              autres: dict[str, Reglages], processus: int = 1) -> pd.DataFrame:
    """Une ligne complète (`ligne`) par année : la chaîne de chaque pli sur les 3 années gardées de côté
    et sur 2018, les chaînes de `autres` sur les 12 années. Gardé dans `sorties/simulees_cv_lignes.csv`."""
    f = SORTIES / "simulees_cv_lignes.csv"
    if f.exists():
        return pd.read_csv(f)
    pli_de = {n: v for v, ns in plis(ans).items() for n in ns}

    def une(a):
        out = []
        for chaine_, r, pli in [("réglée sur les 3 autres variantes", reglages_plis[pli_de[a.nom]], pli_de[a.nom])] + \
                [(n, r, pli_de[a.nom]) for n, r in autres.items()]:
            ch, env = envoyer(a, ref, r)
            out.append({"pli": pli, "chaîne": chaine_, "année": a.nom, **ligne(a, ch, env, chaine_)})
        return pd.DataFrame(out)
    t = grilles_paralleles(ans, une, processus).reset_index(drop=True)
    l18 = []
    for v, r in reglages_plis.items():
        ch, env = envoyer(a18, ref, r)
        l18.append({"pli": v, "chaîne": "réglée sur les 3 autres variantes", "année": "2018",
                    **ligne(a18, ch, env, "réglée sur les 3 autres variantes")})
    t = pd.concat([t, pd.DataFrame(l18)], ignore_index=True)
    t.to_csv(f, index=False)
    return t


def par_pli(t: pd.DataFrame) -> list[dict]:
    """Les lignes de `lignes_cv` sommées par pli et par chaîne (années gardées de côté), puis la chaîne de
    chaque pli notée sur 2018."""
    out = []
    for v in sorted(t.pli.unique()):
        x = t[(t.pli == v) & (t.année != "2018")]
        for c in x.chaîne.unique():
            out.append({**somme(x[x.chaîne == c].to_dict("records"), f"pli {v} : {c} → 3 années de {v}"), "pli": v})
        y = t[(t.pli == v) & (t.année == "2018")]
        out += [{**r, "règle": f"pli {v} : réglée sur les 3 autres variantes → 2018", "pli": v}
                for r in y.to_dict("records")]
    return out


def ensemble(t: pd.DataFrame) -> list[dict]:
    """Les 12 années gardées de côté mises ensemble (une fois chacune), par chaîne."""
    x = t[t.année != "2018"]
    return [somme(x[x.chaîne == c].to_dict("records"), f"{c} → 12 années gardées de côté") for c in x.chaîne.unique()]


def chaine_retenue() -> Reglages:
    """La chaîne retenue : réglée sur 2018 (étape 11), sans regroupement si CHOIX_REGROUPER est faux."""
    return Reglages() if G.CHOIX_REGROUPER else Reglages(rayon=None)
