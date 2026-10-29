"""Les figures des carnets 1 à 3 : une fonction = une figure.

Les carnets n'appellent que ces fonctions ; tout ce qui est lu vient de `sorties/` et `modeles/`.
Modèle présenté : C (débits + note de fuite). Comparaisons : A (débits seuls), B (note seule),
le 08c et le hasard.
"""
from functools import cache

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

from . import carnets as K, figures as F

S = K.SORTIES
RJ = S / "rejeu"
COUL = {"C · note": "C1", "C · bilan": "C4", "B · note": "C0", "A · bilan": "C2", "08c": "C7", "hasard": "0.6"}
plt.rcParams.update({"figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.titleweight": "bold", "axes.titlesize": 11})


# --------------------------------------------------------------------------------- données
@cache
def reseau():
    wn, conduites, zones = K.reseau()
    return wn, conduites, zones


@cache
def distances():
    wn, conduites, _ = reseau()
    return K.distances(wn, conduites)


@cache
def fuites_2018():
    return K.fuites_2018(reseau()[1])


@cache
def carte_2018(nom: str) -> np.ndarray:
    """Cartes brutes (8760 × 905) : « C · note », « C · bilan », « B · note », « A · bilan », « 08c »."""
    c = reseau()[1]
    A = S / "apprentissage"
    fichier, cle = {"C · note": (A / "C" / "scores_2018.npz", "logits"),
                    "C · bilan": (A / "C" / "scores_2018.npz", "bilan"),
                    "B · note": (A / "B" / "scores_2018.npz", "logits"),
                    "A · bilan": (S / "tour2" / "rA" / "A" / "scores_2018.npz", "bilan"),
                    "08c": (A / "cartes_08c_2018.npz", "presence")}[nom]
    return K.carte_2018(fichier, cle, c)


def a_disponible() -> bool:
    return (S / "tour2" / "rA" / "A" / "scores_2018.npz").exists()


@cache
def rejeu():
    """Le test simulé v1 : vérité et prédictions de C (et de A, B si présents), à chaque pas."""
    v = dict(np.load(RJ / "verite_v1.npz"))
    p = {m: dict(np.load(RJ / f"predictions_{m}_v1.npz")) for m in ("C", "B", "A")
         if (RJ / f"predictions_{m}_v1.npz").exists()}
    ep = pd.DataFrame(K.lire_json(RJ / "episodes_v1.json"))
    return v, p, ep


def cartes_rejeu():
    """Pour chaque carte : (épisodes, pas, conduites), plus haut = plus suspect."""
    _, p, _ = rejeu()
    c = {"C · note": p["C"]["logits"].astype(np.float32),
         "C · bilan": (p["C"]["q_predit_m3h"][..., 0] - p["C"]["q_predit_m3h"][..., 1]).astype(np.float32)}
    if "B" in p:
        c["B · note"] = p["B"]["logits"].astype(np.float32)
    if "A" in p:
        c["A · bilan"] = (p["A"]["q_predit_m3h"][..., 0] - p["A"]["q_predit_m3h"][..., 1]).astype(np.float32)
    return c


def place(carte, k):
    """Place de la 1re conduite à ≤ 300 m de la conduite k dans l'ordre de `carte` (1 = en tête)."""
    return int((distances()[k][np.argsort(-carte)] <= 300).argmax()) + 1


@cache
def places_rejeu():
    """Pour chaque carte : la place de chaque fuite active à chaque pas ; plus le débit de la fuite."""
    v, _, _ = rejeu()
    actif = v["fuite_active"]
    cartes = cartes_rejeu()
    rng = np.random.default_rng(0)
    out = {n: [] for n in [*cartes, "hasard"]}
    debits = []
    for e, t, k in zip(*np.nonzero(actif)):
        for n, c in cartes.items():
            out[n].append(place(c[e, t], k))
        out["hasard"].append(place(rng.random(actif.shape[-1]), k))
        debits.append(float(v["fuite_m3h"][e, t, k]))
    return {n: np.array(x) for n, x in out.items()}, np.array(debits)


def episode_type():
    """L'épisode de test à trois fuites où C fait la médiane (ni le meilleur, ni le pire)."""
    v, _, ep = rejeu()
    c = cartes_rejeu()["C · note"]
    trois = [e for e in range(len(ep)) if len(ep.fuites_visees_m3h[e]) == 3]
    score = {e: np.mean([place(c[e, t], k) <= 5 for t, k in zip(*np.nonzero(v["fuite_active"][e]))])
             for e in trois}
    return sorted(score, key=score.get)[len(score) // 2]


def titre(fig_ou_ax, texte):
    (fig_ou_ax.suptitle if hasattr(fig_ou_ax, "suptitle") else fig_ou_ax.set_title)(texte, fontweight="bold")


# --------------------------------------------------------------------------------- carnet 1
def carte_des_mesures():
    wn, conduites, _ = reseau()
    ex = S / "exemples" / "fuite_p31_p232"
    schema = K.lire_json(ex / "rapport.json")["schema_entrees"]
    x = np.load(ex / "entrees.npz")
    xn, xl = x["hydraulique_noeuds"], x["hydraulique_liens"]
    cn = [c[0] for c in schema["champs_noeuds"]]
    cl = [c[0] for c in schema["champs_liens"]]

    def noeuds(champ):
        idx = np.flatnonzero(xn[0, :, cn.index(champ)] == 1)
        return np.array([wn.get_node(schema["noeuds"][i]).coordinates for i in idx
                         if schema["noeuds"][i] in wn.node_name_list])

    def liens(champ):
        out = []
        for i in np.flatnonzero(xl[0, :, cl.index(champ)] == 1):
            if schema["liens"][i] in wn.link_name_list:
                lien = wn.get_link(schema["liens"][i])
                out.append((np.array(wn.get_node(lien.start_node_name).coordinates)
                            + np.array(wn.get_node(lien.end_node_name).coordinates)) / 2)
        return np.array(out)

    fig, ax = plt.subplots(figsize=(10, 6))
    F.dessiner(ax, wn, conduites, barre=False)
    ax.scatter(*noeuds("amr_present").T, s=14, c="C2", marker="s", label="compteur client (82)", zorder=3)
    ax.scatter(*noeuds("pression_presente").T, s=60, c="C0", edgecolors="k", label="pression (33)", zorder=3)
    ax.scatter(*liens("debit_present").T, s=170, c="C3", marker="^", edgecolors="k", label="débitmètre (3)", zorder=3)
    ax.scatter(*noeuds("niveau_present").T, s=220, c="gold", marker="*", edgecolors="k", label="réservoir T1", zorder=3)
    ax.legend(loc="lower left", fontsize=8)
    titre(ax, "Le réseau n'est mesuré qu'en quelques points ; le modèle calibré donne le reste")
    plt.tight_layout()


def tableau_des_entrees():
    return pd.DataFrame({
        "ce que c'est": ["longueur, diamètre, rugosité, altitude et position des deux bouts",
                         "pression (33 capteurs), débit (3 débitmètres), compteurs clients (82), niveau de T1, pompe",
                         "pression et débit de référence partout, demande estimée partout",
                         "résidu = pression mesurée − pression de référence ; dual = ce même écart traduit en m³/h"],
        "où": ["partout", "seulement aux capteurs", "partout", "aux 33 capteurs de pression"],
    }, index=["1. le réseau fixe", "2. les mesures", "3. le modèle calibré sans fuite", "4. tiré de 2 et 3"])


def coupe_et_cible():
    wn, conduites, _ = reseau()
    from .hydraulique import decouper
    w2, _ = decouper(wn)
    l31 = wn.get_link("p31")
    centre = (np.array(wn.get_node(l31.start_node_name).coordinates)
              + np.array(wn.get_node(l31.end_node_name).coordinates)) / 2
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1, 1.4]})
    for n in w2.link_name_list:
        lien = w2.get_link(n)
        a = np.array(w2.get_node(lien.start_node_name).coordinates)
        b = np.array(w2.get_node(lien.end_node_name).coordinates)
        if min(np.hypot(*(a - centre)), np.hypot(*(b - centre))) > 150:
            continue
        moi = n in ("p31", "s09_p31")
        ax[0].plot([a[0], b[0]], [a[1], b[1]], c="C3" if moi else "0.5", lw=3 if moi else 1.2)
        for nom, p in ((lien.start_node_name, a), (lien.end_node_name, b)):
            ax[0].scatter(*p, s=35 if nom.startswith("m09_") else 12, zorder=3,
                          c="C3" if nom == "m09_p31" else ("C1" if nom.startswith("m09_") else "k"))
    ax[0].set_aspect("equal")
    ax[0].set(xticks=[], yticks=[], title="chaque conduite coupée en deux (orange : les milieux)")
    c = np.load(S / "exemples" / "fuite_p31_p232" / "cibles.npz")
    k = conduites.index("p31")
    t = np.arange(72) * 5 / 60
    ax[1].plot(t, c["q_m3h"][:, k, 0], lw=2, label="q1 : 1re moitié, début → milieu")
    ax[1].plot(t, c["q_m3h"][:, k, 1], lw=2, label="q2 : 2e moitié, milieu → fin (négatif : l'eau remonte)")
    ax[1].plot(t, c["q_m3h"][:, k, 0] - c["q_m3h"][:, k, 1], "k--", label="q1 − q2")
    ax[1].plot(t, c["fuite_m3h"][:, k], ":", c="C3", lw=3, label="la fuite")
    ax[1].set(xlabel="heure de l'épisode", ylabel="m³/h", title="p31 avec une fuite de 8 m³/h au milieu")
    ax[1].legend(fontsize=8, loc="lower left", bbox_to_anchor=(0, 0.12))
    titre(fig, "La cible : les débits q1 et q2 de chaque conduite ; la fuite est q1 − q2")
    plt.tight_layout()


def schema_du_modele():
    fig, ax = plt.subplots(figsize=(13, 3.4))
    boites = [((0.0, 0.7), "1 heure de mesures\n(12 pas de 5 min)\n+ réseau fixe", "0.9"),
              ((2.3, 0.7), "dans le temps\nchaque moitié de conduite\nrésume son heure", "#cfe2f3"),
              ((4.6, 0.7), "dans l'espace\n4 blocs GPS : les voisins\n+ tout le réseau", "#d9ead3"),
              ((6.9, 1.25), "carte des débits\nq1, q2 de chaque conduite\n(A et C)", "#fce5cd"),
              ((6.9, 0.15), "carte de fuite\nune note par conduite\n(B et C)", "#f4cccc")]
    for (x, y), texte, couleur in boites:
        ax.add_patch(FancyBboxPatch((x, y), 1.9, 0.95, boxstyle="round,pad=0.05", fc=couleur, ec="k"))
        ax.text(x + 0.95, y + 0.47, texte, ha="center", va="center", fontsize=9)
    for x0, y0, x1, y1 in ((1.95, 1.17, 2.3, 1.17), (4.25, 1.17, 4.6, 1.17), (6.55, 1.17, 6.9, 1.7),
                           (6.55, 1.17, 6.9, 0.62)):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops={"arrowstyle": "->", "lw": 1.5})
    ax.set(xlim=(-0.1, 9.0), ylim=(0, 2.4))
    ax.axis("off")
    titre(ax, "Une heure de mesures en entrée, deux cartes en sortie")
    plt.tight_layout()


def tableau_des_modeles():
    return pd.DataFrame({
        "ce qu'il prédit": ["les débits q1, q2 de chaque conduite", "une note « fuit / ne fuit pas » par conduite",
                            "les deux"],
        "ce qu'il apprend à minimiser": ["erreur sur les débits", "erreur sur la note (fuites pesées plus fort)",
                                         "moitié l'une, moitié l'autre"],
        "sa carte de fuite": ["le bilan q1 − q2", "la note", "la note, ou le bilan q1 − q2"],
    }, index=["A", "B", "C (le modèle présenté)"])


def tableau_des_donnees():
    m = pd.DataFrame(K.lire_json(S / "apprentissage" / "manifeste.json")["episodes"])
    m["fuites"] = m.fuites_visees_m3h.map(len)
    t = pd.crosstab(m.partition, m.fuites).rename(columns=lambda c: f"{c} fuite(s)")
    t["épisodes"] = t.sum(1)
    t.index = t.index.map({"train": "apprentissage (variantes 1–32, jours 0–240)",
                           "validation": "validation (variantes 33–36, jours 250–300)",
                           "test": "test (variantes 37–40, jours 310–362)"})
    return t


# --------------------------------------------------------------------------------- carnet 2
def apprentissage():
    fig, ax = plt.subplots(figsize=(8, 3.6))
    h = K.lire_json(S / "apprentissage" / "C" / "historique.json")
    e = [x["etape"] * 8 for x in h]
    ax.plot(e, [x["train"]["totale"] for x in h], "o-", label="épisodes d'apprentissage")
    ax.plot(e, [x["validation"]["totale"] for x in h], "o-", label="épisodes de validation (jamais vus)")
    k = int(np.argmin([x["validation"]["totale"] for x in h]))
    ax.scatter(e[k], h[k]["validation"]["totale"], s=220, facecolors="none", edgecolors="k", zorder=3,
               label="modèle gardé")
    ax.set(xlabel="minutes d'apprentissage", ylabel="erreur (1 = prédire zéro)")
    ax.legend(fontsize=8)
    titre(ax, "C apprend vite et sur-apprend vite : le meilleur modèle est celui des 8 premières minutes")
    plt.tight_layout()


def debits_contre_verite():
    v, p, _ = rejeu()
    qv = v["q_vrai_m3h"].astype(np.float32)
    qr = v["q_reference_m3h"].astype(np.float32)
    modeles = {"C": p["C"]["q_predit_m3h"].astype(np.float32)}
    if "A" in p:
        modeles["A"] = p["A"]["q_predit_m3h"].astype(np.float32)
    rng = np.random.default_rng(0)
    ech = rng.choice(qv.size, 20000, replace=False)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8))
    lim = np.percentile(np.abs(qv), 99.5)
    ax[0].scatter(qv.ravel()[ech], modeles["C"].ravel()[ech], s=2, alpha=0.3, c="C1")
    ax[0].plot([-lim, lim], [-lim, lim], "k--", lw=1)
    ax[0].set(xlim=(-lim, lim), ylim=(-lim, lim), xlabel="vrai débit (m³/h)", ylabel="débit prédit par C (m³/h)",
              title="C contre la vérité (20 000 points)")
    bornes = np.logspace(-3, 1.5, 50)
    err = {"modèle calibré sans fuite (recopié)": np.abs(qr - qv)} | {n: np.abs(q - qv) for n, q in modeles.items()}
    for (n, e), c in zip(err.items(), ("0.4", "C1", "C2")):
        ax[1].hist(e.ravel(), bins=bornes, histtype="step", lw=2, color=c,
                   label=f"{n} : médiane {np.median(e):.2f} m³/h")
    ax[1].set(xscale="log", xlabel="erreur absolue (m³/h)", ylabel="moitiés de conduite × pas", title="erreur sur les débits")
    ax[1].legend(fontsize=8)
    titre(fig, "Les débits prédits suivent les vrais, mais moins bien que le modèle calibré qu'on donne en entrée")
    plt.tight_layout()
    return pd.Series({n: float(np.median(e)) for n, e in err.items()}, name="erreur médiane (m³/h)").round(3)


def bilan_contre_fuite():
    v, p, _ = rejeu()
    f = v["fuite_m3h"].astype(np.float32)
    sur = f > 0.01
    modeles = {"C": p["C"]["q_predit_m3h"]} | ({"A": p["A"]["q_predit_m3h"]} if "A" in p else {})
    fig, ax = plt.subplots(1, len(modeles), figsize=(6.5 * len(modeles), 4.8), squeeze=False)
    ligne = {}
    for a, (n, q) in zip(ax[0], modeles.items()):
        b = (q[..., 0] - q[..., 1]).astype(np.float32)
        bruit = np.percentile(np.abs(b[~sur]), 99)
        a.axhspan(-bruit, bruit, color="0.88", zorder=0, label=f"bruit ailleurs (±{bruit:.1f} m³/h)")
        a.scatter(f[sur], b[sur], s=3, alpha=0.3, c=COUL[f"{n} · bilan"], label="conduites qui fuient")
        a.plot([0, 20], [0, 20], "k--", lw=1, label="bilan parfait")
        a.set(xlabel="vraie fuite (m³/h)", ylabel=f"q1 − q2 prédit par {n} (m³/h)", title=n)
        a.legend(fontsize=8)
        ligne[n] = {"fuites dont le bilan sort du bruit": float((b[sur] > bruit).mean()),
                    "pente (1 = parfait)": float((b[sur] * f[sur]).sum() / (f[sur] ** 2).sum())}
    titre(fig, "En valeur, le bilan q1 − q2 ne mesure pas la fuite : il reste presque toujours dans le bruit")
    plt.tight_layout()
    return pd.DataFrame(ligne).T.round(3)


def top_k():
    places, _ = places_rejeu()
    k_ = np.arange(1, 51)
    fig, ax = plt.subplots(figsize=(9, 4.4))
    for n, x in places.items():
        ax.plot(k_, [(x <= k).mean() for k in k_], c=COUL[n], lw=2.5 if n.startswith("C") else 1.5,
                ls="--" if n == "hasard" else "-", label=n)
    ax.axvline(5, c="k", ls=":", lw=1)
    ax.set(xlabel="k : nombre de conduites qu'on accepte de vérifier", ylabel="part des fuites trouvées (à 300 m)",
           ylim=(0, 1))
    ax.legend()
    titre(ax, "Rangés, la note et le bilan de C mènent à la fuite bien mieux que le hasard")
    plt.tight_layout()
    return pd.DataFrame({n: {f"top-{k}": (x <= k).mean() for k in (1, 5, 10)} for n, x in places.items()}).round(2)


def par_taille():
    places, debits = places_rejeu()
    classes = pd.cut(debits, [0, 4, 7, 10, 15, 21], labels=["< 4", "4–7", "7–10", "10–15", "15–20"])
    tab = pd.DataFrame({n: pd.Series(x <= 5).groupby(classes, observed=True).mean() for n, x in places.items()})
    ax = tab.plot.bar(color=[COUL[n] for n in tab.columns], figsize=(9, 3.8), rot=0)
    ax.set(xlabel="débit de la fuite (m³/h)", ylabel="top-5 à 300 m", ylim=(0, 1))
    titre(ax, "Plus la fuite est petite, moins on la trouve")
    plt.tight_layout()
    return tab.round(2)


# --------------------------------------------------------------------------------- carnet 3
def cartes_episode():
    wn, conduites, _ = reseau()
    v, p, ep = rejeu()
    e = episode_type()
    t = v["fuite_active"].shape[1] - 1
    fuites = [conduites[k] for k in np.flatnonzero(v["fuite_active"][e, t])]
    q = p["C"]["q_predit_m3h"][e, t].astype(np.float32)
    fig, ax = plt.subplots(1, 3, figsize=(18, 5))
    vf = v["fuite_m3h"][e, t].astype(np.float32)
    F.dessiner(ax[0], wn, conduites, barre=False, titre="la vérité : où sont les fuites")
    for i, f in enumerate(fuites):
        F.marquer(ax[0], wn, conduites, [f], taille=40 + 12 * vf[conduites.index(f)])
        m = F.segments(wn, [f]).mean(1)[0]
        ax[0].annotate(f"{f} : {vf[conduites.index(f)]:.0f} m³/h", m, xytext=(-90, 25 - 22 * i),
                       textcoords="offset points", fontsize=9, arrowprops={"arrowstyle": "-", "color": "0.5"})
    K.dessiner_carte(ax[1], wn, conduites, K.sigmoide(p["C"]["logits"][e, t]), fuites, top=5,
                     titre="carte de fuite de C (note)")
    K.dessiner_carte(ax[2], wn, conduites, q[:, 0] - q[:, 1], fuites, top=5, cmap="RdBu_r", vmin=-2, vmax=2,
                     titre="carte des débits de C : bilan q1 − q2 (m³/h)")
    ax[1].legend(loc="lower left", fontsize=8)
    titre(fig, f"Épisode de test {ep.identifiant[e]} : la note désigne la région des fuites ; le bilan est dispersé")
    plt.tight_layout()
    return pd.DataFrame({"vraie fuite (m³/h)": [float(v["fuite_m3h"][e, t, conduites.index(f)]) for f in fuites],
                         "q1 − q2 prédit (m³/h)": [float(q[conduites.index(f), 0] - q[conduites.index(f), 1]) for f in fuites]},
                        index=fuites).round(2)


def carte_dq():
    wn, conduites, _ = reseau()
    v, p, _ = rejeu()
    e = episode_type()
    t = v["fuite_active"].shape[1] - 1
    fuites = [conduites[k] for k in np.flatnonzero(v["fuite_active"][e, t])]
    ref = v["q_reference_m3h"][e, t].astype(np.float32)
    vrai = (v["q_vrai_m3h"][e, t].astype(np.float32) - ref).mean(1)
    pred = (p["C"]["q_predit_m3h"][e, t].astype(np.float32) - ref).mean(1)
    lim = np.percentile(np.abs(vrai), 99)
    fig, ax = plt.subplots(1, 2, figsize=(15, 5))
    for a, d, n in ((ax[0], vrai, "vrai"), (ax[1], pred, "prédit par C")):
        K.dessiner_carte(a, wn, conduites, d, fuites, cmap="RdBu_r", vmin=-lim, vmax=lim,
                         titre=f"débit − débit sans fuite, {n} (m³/h)")
    titre(fig, "C retrouve le grand changement sur la conduite principale, pas le chemin de l'eau vers les fuites")
    plt.tight_layout()


def animation_episode():
    wn, conduites, _ = reseau()
    v, p, _ = rejeu()
    e = episode_type()
    q = p["C"]["q_predit_m3h"][e].astype(np.float32)
    b = q[..., 0] - q[..., 1]
    pas = v["pas"]
    return K.animer(wn, conduites, (np.clip(b, -2, 2) + 2) / 4, [np.flatnonzero(x) for x in v["fuite_active"][e]],
                    [f"bilan q1 − q2 de C — {x * 5 // 60} h {x * 5 % 60:02d} (rouge = eau perdue)" for x in pas],
                    cmap="RdBu_r")


def heure_2018():
    """L'heure de 2018 où le plus de fuites coulent."""
    _, debit = fuites_2018()
    return int((debit.to_numpy() > 0.01).sum(1).argmax())


def cartes_2018_une_heure():
    wn, conduites, _ = reseau()
    _, debit = fuites_2018()
    h = heure_2018()
    en_cours = [conduites[i] for i in np.flatnonzero(debit.to_numpy()[h] > 0.01)]
    fig, ax = plt.subplots(1, 2, figsize=(15, 5.2))
    K.dessiner_carte(ax[0], wn, conduites, K.sigmoide(carte_2018("C · note")[h]), en_cours, top=5,
                     titre="carte de fuite de C (note)")
    K.dessiner_carte(ax[1], wn, conduites, carte_2018("C · bilan")[h], en_cours, top=5, cmap="RdBu_r",
                     vmin=-2, vmax=2, titre="carte des débits de C : bilan q1 − q2 (m³/h)")
    ax[0].legend(loc="lower left", fontsize=8)
    titre(fig, f"2018, {debit.index[h]:%d/%m %Hh}, 6 fuites en cours : la note allume des régions, le bilan est du bruit à l'œil")
    plt.tight_layout()
    return pd.DataFrame({"débit publié (m³/h)": [float(debit[p].iloc[h]) for p in en_cours],
                         "q1 − q2 prédit par C (m³/h)": [float(carte_2018("C · bilan")[h, conduites.index(p)])
                                                         for p in en_cours]}, index=en_cours).round(2)


def scores_2018():
    lignes = {"C · note": "tour1 · C · logits", "C · bilan": "tour1 · C · bilan Q1−Q2", "B · note": "tour1 · B · logits",
              "08c": "tour1 · 08c donnees · présence"}
    t = pd.read_csv(S / "tour2" / "tour1" / "comparaison_2018.csv")
    t = t[t["mode"] == "brute"].set_index("carte")
    if a_disponible():
        ta = pd.read_csv(S / "tour2" / "rA" / "comparaison_2018.csv")
        ta = ta[ta["mode"] == "brute"].set_index("carte")
        t.loc["A"] = ta.loc["A · logits"]
        lignes["A · bilan"] = "A"
    vues = {"toutes les heures": "top-5 à 300 m", "par fuite": "top-5 à 300 m par fuite", "hors zone C": "top-5 à 300 m hors C"}
    hasard = {"toutes les heures": "hasard top-5", "par fuite": "hasard top-5 par fuite", "hors zone C": "hasard top-5 hors C"}
    tab = pd.DataFrame({n: {v: t.loc[ligne, c] for v, c in vues.items()} for n, ligne in lignes.items()}).T
    tab.loc["hasard"] = {v: t.loc[lignes["C · note"], c] for v, c in hasard.items()}
    fig, ax = plt.subplots(figsize=(11, 4))
    x = np.arange(len(vues))
    w = 0.8 / len(tab)
    for i, (n, r) in enumerate(tab.iterrows()):
        ax.bar(x + (i - len(tab) / 2 + 0.5) * w, r.to_numpy(), w, color=COUL[n], label=n,
               hatch="//" if n == "hasard" else None)
    ax.set(xticks=x, xticklabels=list(vues), ylabel="top-5 à 300 m", ylim=(0, 0.8))
    ax.legend(fontsize=8, ncol=3)
    titre(ax, "Sur 2018, la note de C trouve la fuite deux fois plus souvent que le hasard ; le bilan, seulement compté par fuite")
    plt.tight_layout()
    return tab.round(2)


def par_fuite_2018():
    wn, conduites, _ = reseau()
    fuites, debit = fuites_2018()
    dist = distances()
    noms = ["C · note", "C · bilan", "B · note"] + (["A · bilan"] if a_disponible() else []) + ["08c"]
    ordre = fuites.sort_values("debit_max", ascending=False).index
    m = np.zeros((len(ordre), len(noms)))
    for i, p in enumerate(ordre):
        k = conduites.index(p)
        on = debit[p].to_numpy() > 0.01
        hasard = len(conduites) / ((dist[k] <= 300).sum() + 1)
        for j, n in enumerate(noms):
            c = carte_2018(n)
            ok = on & np.isfinite(c).all(1)
            m[i, j] = np.median(K.rang_300m(c[ok], k, dist)) / hasard
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(np.log10(m), cmap="RdYlGn_r", vmin=-1.5, vmax=1.5, aspect="auto")
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            ax.text(j, i, "✓" if m[i, j] < 1 else "", ha="center", va="center", fontsize=12)
    ax.set_xticks(range(len(noms)), noms)
    ax.set_yticks(range(len(ordre)), [f"{p} · {fuites.loc[p, 'zone']} · {fuites.loc[p, 'debit_max']:.0f} m³/h" for p in ordre])
    fig.colorbar(im, ax=ax, fraction=0.04, label="log10(place / place du hasard)  — vert = mieux que le hasard")
    titre(ax, "Fuite par fuite : chaque carte place 10 à 11 fuites sur 14 mieux que le hasard")
    plt.tight_layout()
    return pd.Series((m < 1).sum(0), index=noms, name="fuites mieux placées que le hasard (sur 14)")


@cache
def carte_2018_lissee(nom: str) -> np.ndarray:
    """La même carte lissée sur 300 m : chaque conduite prend la moyenne de ses voisines à ≤ 300 m
    (poids exp(−d/300) × longueur), comme dans la règle écrite avant le tour 2."""
    wn, conduites, _ = reseau()
    return carte_2018(nom) @ K.lisseur(distances(), wn, conduites).T


def brut_contre_lisse(fuite="p538"):
    """Une heure où la carte brute de C met `fuite` 1re : l'heure de place lissée médiane parmi elles."""
    wn, conduites, _ = reseau()
    _, debit = fuites_2018()
    dist = distances()
    k = conduites.index(fuite)
    brute, lissee = carte_2018("C · note"), carte_2018_lissee("C · note")
    hs = np.flatnonzero(debit[fuite].to_numpy() > 0.01)
    hs = hs[K.rang_300m(brute[hs], k, dist) == 1]
    pl = K.rang_300m(lissee[hs], k, dist)
    h = hs[np.argsort(pl)[len(pl) // 2]]
    en_cours = [conduites[i] for i in np.flatnonzero(debit.to_numpy()[h] > 0.01)]
    w = K.lisseur(dist, wn, conduites)[k]
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.4))
    K.dessiner_carte(ax[0], wn, conduites, K.sigmoide(brute[h]), en_cours, top=5,
                     titre=f"brute : {fuite} est 1re, le top-5 est réparti")
    K.dessiner_carte(ax[1], wn, conduites, np.where(w > 0, w / w.max(), np.nan), [fuite], cmap="Greens",
                     titre=f"le lissage : la note de {fuite} devient la moyenne\nde ses {(w > 0).sum()} voisines à ≤ 300 m (poids en vert)")
    K.dessiner_carte(ax[2], wn, conduites, K.sigmoide(lissee[h]), en_cours, top=5,
                     titre=f"lissée : {fuite} tombe {K.rang_300m(lissee[h:h + 1], k, dist)[0]}e, le top-5 est groupé")
    fig.legend(*ax[0].get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=9)
    titre(fig, f"2018, {debit.index[h]:%d/%m %Hh} : le lissage regroupe le top-5 sur la plus grande plaque sombre "
               f"et perd {fuite}")
    plt.tight_layout()


def brut_contre_lisse_par_fuite():
    wn, conduites, zones = reseau()
    fuites, debit = fuites_2018()
    dist = distances()
    ordre = fuites.sort_values("debit_max", ascending=False).index
    brute, lissee = carte_2018("C · note"), carte_2018_lissee("C · note")
    lignes = {}
    for p in ordre:
        k = conduites.index(p)
        on = debit[p].to_numpy() > 0.01
        lignes[p] = {"zone": fuites.loc[p, "zone"], "heures": int(on.sum()),
                     "brute": np.median(K.rang_300m(brute[on], k, dist)),
                     "lissée": np.median(K.rang_300m(lissee[on], k, dist)),
                     "hasard": len(conduites) / ((dist[k] <= 300).sum() + 1)}
    t = pd.DataFrame(lignes).T
    y = np.arange(len(t))[::-1]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    for yi, (_, r) in zip(y, t.iterrows()):
        ax.annotate("", xy=(r["lissée"], yi), xytext=(r["brute"], yi),
                    arrowprops=dict(arrowstyle="->", color="C3" if r["lissée"] > r["brute"] else "C2", lw=1.5))
    ax.scatter(t["brute"], y, color="C1", zorder=3, label="brute")
    ax.scatter(t["lissée"], y, color="C1", facecolors="white", zorder=3, label="lissée")
    ax.scatter(t["hasard"], y, marker="|", s=200, color="0.5", label="hasard")
    ax.set(xscale="log", xlabel="place médiane de la fuite (1 = en tête ; échelle log)",
           yticks=y, yticklabels=[f"{p} · {r['zone']} · {r['heures']} h" for p, r in t.iterrows()])
    ax.legend(fontsize=8, loc="upper right")
    titre(ax, "Note de C, 2018 : le lissage recule 8 fuites sur 14 et n'en avance que 3, dont p257 (zone C, toute l'année)")
    plt.tight_layout()
    return t.astype({"brute": int, "lissée": int}).round(1)


def retrait_residu_dual():
    t = pd.read_csv(S / "tour2" / "tour1" / "comparaison_2018.csv")
    t = t[t["mode"] == "brute"].set_index("carte")
    sim = {n: K.lire_json(S / "apprentissage" / d / "test_simule.json")["rappel_top_k"]["5"]
           for n, d in (("B", "B"), ("B sans résidu ni dual", "B_sans_residu_dual"))}
    reel = {"B": t.loc["tour1 · B · logits", "top-5 à 300 m par fuite"],
            "B sans résidu ni dual": t.loc["tour1 · B_sans_residu_dual · logits", "top-5 à 300 m par fuite"]}
    fig, ax = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, d, h, tt in ((ax[0], sim, K.lire_json(S / "apprentissage" / "B" / "test_simule.json")["hasard_top_k"]["5"], "test simulé"),
                        (ax[1], reel, t.loc["tour1 · B · logits", "hasard top-5 par fuite"], "2018, par fuite")):
        a.bar(list(d), list(d.values()), color=["C0", "0.7"])
        a.axhline(h, c="k", ls="--", lw=1, label="hasard")
        a.set(ylim=(0, 1), title=tt, ylabel="top-5 à 300 m")
        a.legend(fontsize=8)
    titre(fig, "Sans le résidu et le dual, le modèle retombe au hasard : tout repose sur le modèle calibré")
    plt.tight_layout()


def zone_c():
    wn, conduites, zones = reseau()
    en_c = K.zone_c(wn, conduites, zones)
    fuites, debit = fuites_2018()
    c = carte_2018("C · note")
    top = np.argsort(-c, axis=1)[:, :5]
    freq = np.zeros(len(conduites))
    np.add.at(freq, top.ravel(), 1)
    freq /= len(c)
    fuite_c = (debit[[p for p in fuites.index if fuites.loc[p, "zone"] == "C"]].to_numpy() > 0.01).any(1)
    part = en_c[top].mean(1)
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.5, 1]})
    F.dessiner(ax[0], wn, conduites, freq, cmap="magma_r", vmin=0, vmax=0.3, epais=2.5,
               titre="part des heures où la conduite est dans le top-5 de C")
    F.marquer(ax[0], wn, conduites, list(fuites.index), label="fuites de 2018")
    ax[0].legend(loc="lower left", fontsize=8)
    ax[1].bar(["sans fuite\nen zone C\n(1–9 janvier)", "avec fuite\nen zone C\n(le reste)"],
              [part[~fuite_c].mean(), part[fuite_c].mean()], color=["0.6", "C1"])
    ax[1].axhline(en_c.mean(), c="k", ls="--", lw=1, label=f"part de la zone C dans le réseau ({en_c.mean():.0%})")
    ax[1].set(ylim=(0, 1), ylabel="part du top-5 dans la zone C")
    ax[1].legend(fontsize=8)
    titre(fig, "La zone C fuit 98 % des heures (p257) : un tiers du top-5 de C y part, sans trouver la conduite")
    plt.tight_layout()
    k = conduites.index("p257")
    on = debit["p257"].to_numpy() > 0.01
    idx = np.flatnonzero(en_c)
    rang = int(np.median((c[on][:, idx] > c[on][:, [k]]).sum(1) + 1))
    return pd.Series({"heures de 2018 avec une fuite en zone C": int(fuite_c.sum()),
                      "place de p257 parmi les 109 conduites de la zone C (médiane)": rang})


def simule_contre_reel():
    t = pd.read_csv(S / "tour2" / "tour1" / "comparaison_2018.csv")
    t = t[t["mode"] == "brute"].set_index("carte")
    sim = K.lire_json(S / "apprentissage" / "C" / "test_simule.json")
    v = {"test simulé": (sim["rappel_top_k"]["5"], sim["hasard_top_k"]["5"]),
         "2018, toutes les heures": (t.loc["tour1 · C · logits", "top-5 à 300 m"], t.loc["tour1 · C · logits", "hasard top-5"]),
         "2018, hors zone C": (t.loc["tour1 · C · logits", "top-5 à 300 m hors C"], t.loc["tour1 · C · logits", "hasard top-5 hors C"])}
    fig, ax = plt.subplots(figsize=(9, 3.6))
    x = np.arange(len(v))
    ax.bar(x - 0.2, [a for a, _ in v.values()], 0.4, color="C1", label="note de C")
    ax.bar(x + 0.2, [b for _, b in v.values()], 0.4, color="0.75", hatch="//", label="hasard")
    for xx, (a, _) in zip(x, v.values()):
        ax.text(xx - 0.2, a + 0.02, f"{a:.2f}", ha="center")
    ax.set(xticks=x, xticklabels=list(v), ylabel="top-5 à 300 m", ylim=(0, 1))
    ax.legend()
    titre(ax, "Ce qui marche sur les simulations marche deux fois moins sur 2018")
    plt.tight_layout()


def bilan_p628():
    _, conduites, _ = reseau()
    _, debit = fuites_2018()
    b = carte_2018("C · bilan")
    k = conduites.index("p628")
    on = np.flatnonzero(debit["p628"].to_numpy() > 0.01)
    fen = slice(max(on[0] - 120, 0), on[-1] + 120)
    h = debit.index[fen]
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(h, debit["p628"].to_numpy()[fen], c="C3", lw=2, label="fuite publiée")
    ax.plot(h, b[fen, k], c="C4", lw=1, label="q1 − q2 prédit par C")
    ax.set(ylabel="m³/h")
    ax.tick_params(axis="x", labelrotation=20)
    ax.legend(fontsize=8)
    titre(ax, "p628 perd jusqu'à 35 m³/h ; le bilan prédit par C reste près de zéro")
    plt.tight_layout()


def tour_2():
    """Les notes du critère écrit avant le tour 2 : cartes lissées sur 300 m, contre le B du tour 1."""
    T2 = S / "tour2"

    def lire(f, n):
        t = pd.read_csv(f)
        return t[t["mode"] == "lissée 300 m"].set_index("carte").loc[n]
    lignes = {
        "B du tour 1 (la référence du critère)": lire(T2 / "tour1" / "comparaison_2018.csv", "tour1 · B · logits"),
        "C du tour 1 (le modèle présenté)": lire(T2 / "tour1" / "comparaison_2018.csv", "tour1 · C · logits"),
        "C choisi sur le top-5 de validation": lire(T2 / "r1" / "comparaison_2018.csv", "C · logits"),
        "C avec conservation des débits": lire(T2 / "r1" / "comparaison_2018.csv", "C_conservation · logits"),
        "C, données v2 (200 variantes, fuites qui s'arrêtent)": lire(T2 / "r2" / "comparaison_2018.csv", "C · logits"),
        "08c (modèle d'avant)": lire(T2 / "tour1" / "comparaison_2018.csv", "tour1 · 08c donnees · présence"),
        "08c + résidu et dual": lire(T2 / "e08" / "comparaison_2018.csv", "08c donnees enrichi · présence"),
    }
    cols = {"empreinte 20 %": "empreinte 20%", "top-5": "top-5 à 300 m",
            "empreinte 20 % hors C": "empreinte 20% hors C", "top-5 hors C": "top-5 à 300 m hors C"}
    return pd.DataFrame({n: {k: r[c] for k, c in cols.items()} for n, r in lignes.items()}).T.round(2)
