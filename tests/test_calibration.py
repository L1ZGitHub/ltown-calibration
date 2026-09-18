"""Tests du dépôt. Ceux qui ont besoin des mesures se désactivent tout seuls si elles manquent.

Deux familles :

* les tests **synthétiques**, qui vérifient la mécanique sur des signaux fabriqués dont on
  connaît la réponse — ils tournent partout, sans données ;
* les tests **sur mesures**, qui vérifient les valeurs structurelles du réseau (nombre de nœuds
  par zone, unités, conservation du volume) et le report d'état entre tranches de simulation.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from calibration import ameliorations as A
from calibration import diagnostics as G
from calibration import profils as P
from calibration import reseau as R
from calibration import generation as Gen
from calibration import rugosite as U

donnees = pytest.mark.skipif(
    not (R.INP.exists() and (R.DONNEES / "2018_SCADA_Pressures.csv").exists()),
    reason="mesures absentes ; voir data/README.md")


# --------------------------------------------------------------------------------- synthétique
def test_decomposition_retrouve_un_produit_exact():
    """Sur un signal construit comme d̄·T·S sans résidu, la décomposition doit être exacte."""
    n = 8 * R.PAS_SEMAINE
    idx = pd.date_range("2018-01-01", periods=n, freq="5min")
    cr = R.creneau(idx)
    saison = 1.0 + 0.5 * np.sin(2 * np.pi * np.arange(R.PAS_SEMAINE) / R.PAS_JOUR)
    tend = 1.0 + 0.2 * np.sin(2 * np.pi * np.arange(n) / n)
    signal = 3.0 * tend * saison[cr]

    d = P.decomposer(signal, cr)

    assert d["moyenne"] == pytest.approx(signal.mean(), rel=1e-9)
    assert d["saison"].mean() == pytest.approx(1.0, abs=1e-12)

    # La reconstruction n'est pas exacte au bit près, et c'est structurel : une moyenne glissante
    # est un filtre passe-bas qui **atténue aussi la tendance qu'elle estime**, d'un facteur
    # sinc(fenêtre / période). Ici, une fenêtre d'une semaine sur une tendance de huit semaines
    # coûte environ 2,5 %, dont il reste quelques pour mille après passage par S. Le seuil ci-
    # dessous borne cet effet ; il ne doit pas être resserré sans changer la méthode.
    ecart = np.abs(d["lisse"] - signal) / signal
    assert np.median(ecart) < 1e-2
    # et la reconstruction doit rester très au-dessus de ce que donnerait la seule moyenne
    assert np.median(ecart) < np.median(np.abs(signal.mean() - signal) / signal) / 20


def test_la_saisonnalite_ignore_les_journees_atypiques():
    """La médiane de créneau doit absorber quelques journées aberrantes, là où une moyenne cède."""
    n = 10 * R.PAS_SEMAINE
    cr = R.creneau(pd.date_range("2018-01-01", periods=n, freq="5min"))
    base = 1.0 + 0.4 * np.cos(2 * np.pi * np.arange(R.PAS_SEMAINE) / R.PAS_JOUR)
    x = base[cr].copy()
    x[:R.PAS_JOUR] *= 20.0                                    # une journée de fuite majeure

    robuste = P.saisonnalite(x, cr, statistique="mediane")
    moyenne = P.saisonnalite(x, cr, statistique="moyenne")

    assert np.abs(robuste - base / base.mean()).max() < 1e-9
    # la moyenne périodique de la référence, elle, encaisse la journée aberrante
    assert np.abs(moyenne - base / base.mean()).max() > 0.05


def test_rmse_se_decompose_en_biais_et_dispersion():
    rng = np.random.default_rng(0)
    mesure = rng.normal(size=(500, 4))
    simule = mesure + 0.3 + rng.normal(scale=0.2, size=(500, 4))

    r = G.resume(simule, mesure)

    assert r["rmse_m"] ** 2 == pytest.approx(r["biais_m"] ** 2 + r["dispersion_m"] ** 2, rel=1e-9)
    assert r["biais_m"] == pytest.approx(0.3, abs=0.02)
    assert r["dispersion_m"] == pytest.approx(0.2, abs=0.02)


def test_creneau_hebdomadaire_tient_compte_du_jour_de_la_semaine():
    idx18 = pd.date_range("2018-01-01", periods=R.PAS_AN, freq="5min")
    assert (R.creneau(idx18) == np.arange(R.PAS_AN) % R.PAS_SEMAINE).all()
    # 2019 commence un mardi : le premier créneau n'est pas 0 mais 288
    assert R.creneau(pd.date_range("2019-01-01", periods=10, freq="5min"))[0] == R.PAS_JOUR


def test_transitions_de_pompe():
    statut = np.array([0, 0, 1, 1, 1, 0, 0, 1.0])
    assert A.transitions(statut).tolist() == [2, 5, 7]


# ------------------------------------------------------------------------------- sur mesures
@donnees
def test_partition_du_reseau():
    wn = R.charger_modele(1)
    z = R.zones(wn)
    assert sum(v == "C" for v in z.values()) == 92
    assert sum(v == "AB" for v in z.values()) == 690
    # les 82 compteurs sont tous derrière le réservoir : le modèle de demande extrapole donc
    # de la zone C vers une zone A+B où aucune consommation n'est mesurée à l'échelle du nœud
    assert all(z[c] == "C" for c in R.charger_amr(2018).columns)


@donnees
def test_les_trois_lignes_de_demande_sont_bien_lues():
    """`Junction.base_demand` ne renvoie que la première des trois catégories : piège vérifié."""
    wn = R.charger_modele(1)
    b = R.bases_nominales(wn)
    assert b.shape == (782, 3)
    assert b.sum().sum() == pytest.approx(176.6, abs=0.2)
    assert wn.get_node("n1").base_demand == 0.0        # n1 est purement industrielle
    assert b.loc["n1"].sum() > 0.6


@donnees
def test_groupes_de_rugosite():
    wn = R.charger_modele(1)
    g = R.groupes_de_rugosite(wn, 6)
    assert len(g) == len(wn.pipe_name_list) == 905
    assert len(set(g.values())) == 6


@donnees
def test_unites_des_compteurs():
    """Compteurs en L/h dans le fichier, m³/h à la sortie : l'ordre de grandeur doit suivre."""
    amr = R.charger_amr(2018)
    assert amr.shape[1] == 82
    assert 10.0 < amr.sum(axis=1).mean() < 30.0        # quelques dizaines de m³/h en zone C
    assert 150.0 < A.demande_ab_mesuree(2018).mean() < 200.0


@donnees
def test_report_detat_entre_tranches():
    """Deux tranches de 12 h doivent donner la même chose que 24 h d'affilée."""
    wn = R.charger_modele(1)
    D, noeuds = P.demandes_nominales(wn, R.horodatage(2018, R.PAS_JOUR + 1))
    entier = R.pressions_simulees(D, noeuds, R.PAS_JOUR, tranche_jours=1, verbeux=False)
    coupe = R.pressions_simulees(D, noeuds, R.PAS_JOUR, tranche_jours=0.5, verbeux=False)
    assert np.abs(entier - coupe).max() < 1e-3


@donnees
def test_une_serie_de_demande_sans_pas_supplementaire():
    """`simuler` doit accepter exactement `n_pas` lignes de demande, pas `n_pas + 1`.

    EPANET lit le profil un pas au-delà de l'horizon ; sur la dernière tranche ce pas n'existe
    pas. Sans garde-fou, le moteur reboucle sur le début du profil sans rien signaler — une
    erreur silencieuse qui n'apparaît qu'en simulant une série complète.
    """
    wn = R.charger_modele(1)
    n = R.PAS_JOUR
    D, noeuds = P.demandes_nominales(wn, R.horodatage(2018, n + 1))
    juste = R.pressions_simulees(D[:n], noeuds, n, tranche_jours=0.5, verbeux=False)
    large = R.pressions_simulees(D, noeuds, n, tranche_jours=0.5, verbeux=False)
    assert juste.shape == large.shape == (n, 33)
    # seul le tout dernier pas peut différer, et de peu
    assert np.abs(juste[:-1] - large[:-1]).max() < 1e-6


@donnees
def test_formes_par_categorie():
    wn = R.charger_modele(1)
    f = P.formes_par_categorie(wn, 2018)
    assert list(f.columns) == list(R.CATEGORIES)
    assert (f >= 0).all().all()
    assert np.allclose(f.mean().to_numpy(), 1.0, atol=1e-9)
    # la tendance annuelle est réelle : l'été consomme plus que l'hiver
    mois = f["Residential"].resample("ME").mean()
    assert mois.max() / mois.min() > 1.1


@donnees
def test_le_recalage_ne_touche_que_la_zone_ab():
    wn = R.charger_modele(1)
    n = 2 * R.PAS_SEMAINE
    D, noeuds = P.demandes_calibrees(wn, 2018, n_pas=n)
    E = A.recaler_zone_ab(D, noeuds, wn, 2018, periode=R.PAS_SEMAINE)

    z = R.zones(wn)
    est_c = np.array([z[x] == "C" for x in noeuds])
    assert np.allclose(D[:, est_c], E[:, est_c])
    # après recalage, la consommation A+B du modèle égale le bilan de masse sur chaque semaine
    mesure = A.demande_ab_mesuree(2018).to_numpy()[:n]
    for a in range(0, n, R.PAS_SEMAINE):
        b = min(a + R.PAS_SEMAINE, n)
        assert E[a:b][:, ~est_c].sum() == pytest.approx(mesure[a:b].sum(), rel=1e-6)


@donnees
def test_ajuster_reservoir_separe_la_section_de_la_demande():
    """Le balayage par simulation doit fixer l'échelle de la zone C à 1, là où la régression
    sur demi-cycles proposait 1,72 — c'est tout l'intérêt de simuler plutôt que de régresser."""
    wn = R.charger_modele(1)
    D, noeuds = P.demandes_calibrees(wn, 2018, n_pas=8 * R.PAS_JOUR)
    t = A.ajuster_reservoir(D, noeuds, wn, 2018, debut=0, n_pas=7 * R.PAS_JOUR,
                            diametres=(15.0, 16.0), echelles=(1.0, 1.6), verbeux=False)
    assert list(t.columns) == ["diametre_m", "echelle_zoneC", "erreur_niveau_m", "sature"]
    assert t.iloc[0].echelle_zoneC == 1.0
    # et l'écart doit être franc, pas marginal
    pire = t[t.echelle_zoneC == 1.6].erreur_niveau_m.min()
    assert pire > 3 * t.iloc[0].erreur_niveau_m


@donnees
def test_section_du_reservoir_par_demi_cycles():
    wn = R.charger_modele(1)
    r = A.section_par_demi_cycles(wn, 2018)
    assert r["n_demi_cycles"] > 50
    assert 13.0 < r["diametre_m"] < 19.0
    # le résultat à retenir n'est pas la valeur mais son encadrement : les deux sens de
    # régression doivent être renvoyés, et ils ne coïncident pas
    bas, haut = r["encadrement_diametre_m"]
    assert haut > bas
    # le volume pompé et le volume consommé sont presque colinéaires sur un remplissage :
    # c'est ce qui empêche de séparer la section de la consommation non comptée
    assert r["correlation_remplissage"] > 0.9


# ------------------------------------------------- bornes physiques et régularisation
def test_huber_manuel_vaut_la_perte_de_huber():
    """La transformation doit valoir l'identité sous le seuil, et croître en √ au-delà.

    C'est ce qui permet d'appliquer Huber aux seuls résidus de mesure et de laisser la pénalité
    de Tikhonov quadratique, au lieu de la faire écraser par la perte robuste.
    """
    from calibration.rugosite import _pseudo_residus_huber as huber
    f = 1.345
    petits = np.array([-1.0, -0.4, 0.0, 0.4, 1.0, f])
    assert np.allclose(huber(petits, f), petits)             # identité sous le seuil
    grands = np.array([10.0, 20.0, 40.0])
    carres = huber(grands, f) ** 2
    ratio = carres / grands                                   # doit tendre vers une constante
    assert np.ptp(ratio) < 0.2 * ratio.mean()
    assert np.all(np.sign(huber(-grands, f)) == -1)            # le signe est conservé


def test_tikhonov_retient_l_estimateur():
    """Un α croissant doit rapprocher la solution du point de départ, et non l'en éloigner."""
    x0 = {"a": 100.0, "b": 100.0}
    cible = np.array([[140.0, 60.0]])                          # une mesure qui tire loin

    def simulateur(c):
        return np.array([[c["a"], c["b"]]])

    ecarts = []
    for alpha in (0.0, 0.1, 10.0):
        sol = U.ajuster(simulateur, cible, x0, tikhonov=alpha, max_nfev=20, verbeux=False)
        v = np.array([sol["coefficients"][k] for k in x0])
        ecarts.append(np.abs(v - np.array([100.0, 100.0])).max())
    assert ecarts[0] > ecarts[1] > ecarts[2]


@donnees
def test_bornes_par_groupe_encadrent_le_fichier():
    """Chaque groupe doit pouvoir atteindre tous les coefficients qu'il porte déjà."""
    wn = R.charger_modele(1)
    groupes = R.groupes_de_rugosite(wn, 6)
    bornes = U.bornes_par_groupe(wn, groupes, marge=0.10)
    for g, (bas, haut) in bornes.items():
        v = [wn.get_link(p).roughness for p, gg in groupes.items() if gg == g]
        assert bas <= min(v) and max(v) <= haut, f"{g} n'encadre pas ses propres conduites"
        assert bas > 60.0 and haut < 160.0, f"{g} n'est pas plus serré que l'encadrement publié"


@donnees
def test_la_premiere_semaine_de_2018_ne_porte_aucune_fuite():
    """La fenêtre du protocole publié est la plus propre de l'année ; octobre est la plus chargée.

    Ce n'est pas une curiosité : un ajustement de rugosité fait sur une fenêtre fuyarde achète la
    fuite avec de la friction, puisque les deux font baisser la pression en aval.
    """
    semaine1 = R.charge_de_fuite(2018, 0, R.PAS_SEMAINE)
    if semaine1 is None:
        pytest.skip("fichier de fuites non fourni")
    octobre = R.charge_de_fuite(2018, 275 * R.PAS_JOUR, R.PAS_SEMAINE)
    assert semaine1["debit_m3h"] == 0.0
    assert octobre["debit_m3h"] > 20.0


@donnees
def test_emetteur_donne_le_debit_vise():
    """`coefficient_emetteur` doit produire le débit demandé, à la baisse de pression près.

    La fuite fait elle-même descendre la pression au nœud, donc le débit obtenu est un peu
    inférieur à la cible : on vérifie qu'il reste à 10 % près, et qu'il est bien au-dessous.
    """
    noeud, cible = "n1", 10.0
    sans = R.executer(R.charger_modele(duree_h=6))
    p_ref = float(sans.node["pressure"][noeud].to_numpy().mean())
    fuite_nulle = float(sans.node["demand"][noeud].to_numpy().mean()) * 3600.0

    wn = R.preparer(None, [], 6.0,
                    fuites={noeud: R.coefficient_emetteur(cible, p_ref)})
    avec = R.executer(wn)
    obtenu = float(avec.node["demand"][noeud].to_numpy().mean()) * 3600.0 - fuite_nulle

    assert 0.90 * cible <= obtenu <= cible


@donnees
def test_add_leak_de_wntr_ne_fait_rien_sous_epanet():
    """Le piège à documenter : `add_leak` ne change pas une simulation EPANET.

    L'API n'est lue que par le simulateur écrit en Python. Sous EPANET la simulation tourne
    normalement — et sans fuite. C'est pour cela que ce dépôt passe par un émetteur.
    """
    wn = R.charger_modele(duree_h=6)
    wn.get_node("n1").add_leak(wn, area=0.01, start_time=0)
    p = R.executer(wn).node["pressure"]["n1"].to_numpy()
    p_ref = R.executer(R.charger_modele(duree_h=6)).node["pressure"]["n1"].to_numpy()

    assert np.allclose(p, p_ref, atol=1e-6)


@donnees
def test_le_bilan_de_la_zone_c_retrouve_les_fuites_publiees():
    """Le bilan de masse de la zone C doit retrouver les fuites que le jeu de données publie.

    Deux des quatorze conduites fuyardes de 2018, `p31` et `p257`, sont à l'intérieur de la
    zone C. Le bilan ne les connaît pas : il ne voit que le débitmètre de la pompe, le capteur de
    niveau et les 82 compteurs. Sur des moyennes journalières, il doit néanmoins les retrouver.
    """
    fuites = R.charger_fuites(2018)
    if fuites is None:
        pytest.skip("fichier de fuites absent")
    wn = R.charger_modele(duree_h=24)
    estimee = A.bilan_zone_c(wn, 2018)["fuite_m3h"].resample("D").mean()
    publiee = fuites[["p31", "p257"]].sum(axis=1).resample("D").mean()

    assert estimee.corr(publiee) > 0.99
    assert abs((estimee - publiee).mean()) < 0.3        # biais, en m³/h
    assert (estimee - publiee).std() < 0.5              # bruit journalier


@donnees
def test_le_critere_physique_ne_melange_pas_deux_coefficients():
    """Le regroupement `"physique"` doit être homogène en coefficient, en diamètre et en zone.

    C'est tout son intérêt : le critère par diamètre seul réunit dans le groupe `D100`
    104 conduites portant 120 et 601 portant 140, donc un seul paramètre pour deux valeurs vraies.
    Et `D100` n'est pas seul dans ce cas : `D150` (13/90) et `D<=75` (2/3) mélangent aussi, soit
    813 conduites sur 905 réparties dans trois groupes qui ne peuvent pas converger.
    """
    wn = R.charger_modele(duree_h=24)
    phys = R.groupes_de_rugosite(wn, critere="physique")
    diam = R.groupes_de_rugosite(wn, n_groupes=6)

    rugosites = pd.Series({p: wn.get_link(p).roughness for p in wn.pipe_name_list})
    par_groupe = rugosites.groupby(pd.Series(phys)).nunique()
    assert (par_groupe == 1).all(), "un groupe physique mélange deux coefficients"

    # et le critère par diamètre, lui, en mélange bien — c'est le défaut qu'on corrige
    assert rugosites.groupby(pd.Series(diam)).nunique()["D100"] == 2
    assert len(set(phys.values())) == 13


@donnees
def test_aucune_jonction_ne_porte_les_trois_categories():
    """Le « mélange de types de consommateurs » de l'équation (2) n'a jamais plus de deux termes.

    Et surtout, la demande industrielle — dont la forme est estimée sur quatre compteurs
    seulement, donc la plus bruitée des trois — vaut exactement zéro dans la zone A+B. Les quatre
    nœuds qui la portent sont en zone C et tous équipés : elle n'est jamais extrapolée.
    """
    wn = R.charger_modele(duree_h=24)
    bases = R.bases_nominales(wn)
    zones = R.zones(wn)
    amr = R.charger_amr(2018)

    n_types = (bases > 0).sum(axis=1)
    assert n_types.max() == 2
    assert (n_types == 2).sum() == 561 and (n_types == 1).sum() == 186

    ab = [n for n, z in zones.items() if z == "AB"]
    assert bases.loc[ab, "Industrial"].sum() == 0.0

    industriels = list(bases.index[bases["Industrial"] > 0])
    assert len(industriels) == 4
    assert all(zones[n] == "C" and n in amr.columns for n in industriels)


@donnees
def test_le_modele_de_demande_retrouve_la_consommation_reelle_de_ab():
    """Le modèle de demande n'a pas d'écart de niveau : ce qui lui manquait était la fuite.

    Le bilan `p227 + p235 − PUMP_1` mesure tout ce qui sort vers A+B, fuites comprises ; le modèle
    ne reconstruit que ce qui est consommé. Une fois les fuites publiées retirées, les deux se
    rejoignent — alors que les 82 compteurs sont tous en zone C, qui porte 12 % de la demande.
    """
    fuites = R.charger_fuites(2018)
    if fuites is None:
        pytest.skip("fichier de fuites absent")
    wn = R.charger_modele(duree_h=24)
    zones = R.zones(wn)
    en_c = [p for p in fuites.columns
            if zones.get(wn.get_link(p).start_node_name) == "C"
            and zones.get(wn.get_link(p).end_node_name) == "C"]

    D, noeuds = P.demandes_calibrees(wn, 2018)
    ab = np.array([zones[n] == "AB" for n in noeuds])
    modele = float(D[:R.PAS_AN, ab].sum(axis=1, dtype=float).mean())

    bilan = A.demande_ab_mesuree(2018).mean()
    fuite_ab = fuites[[p for p in fuites.columns if p not in en_c]].sum(axis=1).mean()

    assert abs(modele / (bilan - fuite_ab) - 1) < 0.01     # à 1 % de la consommation réelle
    assert modele / bilan - 1 < -0.05                      # et bien en dessous du bilan brut


# ------------------------------------------------------------------ génération de scénarios
def test_une_fuite_refuse_une_date_hors_grille():
    """Les dates doivent tomber sur une frontière de tranche, sinon le recollement serait faux."""
    Gen.Fuite(conduite="p232", debut=Gen.PAS_TRANCHE, fin=2 * Gen.PAS_TRANCHE)

    with pytest.raises(ValueError, match="frontière de tranche"):
        Gen.Fuite(conduite="p232", debut=Gen.PAS_TRANCHE + 1)
    with pytest.raises(ValueError, match="avant début"):
        Gen.Fuite(conduite="p232", debut=2 * Gen.PAS_TRANCHE, fin=Gen.PAS_TRANCHE)
    with pytest.raises(ValueError, match="profil inconnu"):
        Gen.Fuite(conduite="p232", debut=0, profil="lent")


@donnees
def test_le_scenario_est_exact_avant_la_fuite_et_delivre_le_debit_vise():
    """Les deux propriétés dont dépend l'étiquetage.

    Avant l'ouverture, le scénario doit être **bit pour bit** le témoin : c'est ce qui autorise à
    ne simuler que la durée de la fuite. Et le débit soutiré doit être celui qu'on a demandé, sans
    quoi l'étiquette ne décrirait pas ce qui a été simulé.
    """
    wn = R.charger_modele()
    noeuds = wn.junction_name_list
    n = 2 * Gen.PAS_TRANCHE
    D, _ = P.demandes_nominales(wn, R.horodatage(2018, n + 1))

    propre = Gen.fenetre_propre(D, noeuds, n, niveau0=3.5, verbeux=False)
    f = Gen.Fuite(conduite="p232", debut=Gen.PAS_TRANCHE, debit_m3h=10.0)
    pressions, debit = Gen.poser(D, noeuds, n, wn, f, propre, niveau0=3.5, verbeux=False)

    assert np.array_equal(pressions[:f.debut], propre[:f.debut])
    assert not np.array_equal(pressions[f.debut:], propre[f.debut:])
    assert np.all(debit[:f.debut] == 0)

    v = Gen.etiquette(wn, noeuds, f, debit, n)
    assert v["noeud_perce"] in noeuds
    assert v["pas_avec_fuite"] == n - f.debut
    assert 0.90 * f.debit_m3h <= v["debit_reel_m3h"] <= 1.10 * f.debit_m3h


@donnees
def test_une_fuite_progressive_monte_par_paliers():
    """Le profil progressif doit produire un débit croissant, pas un échelon."""
    wn = R.charger_modele()
    noeuds = wn.junction_name_list
    n = 4 * Gen.PAS_TRANCHE
    D, _ = P.demandes_nominales(wn, R.horodatage(2018, n + 1))

    propre = Gen.fenetre_propre(D, noeuds, n, niveau0=3.5, verbeux=False)
    f = Gen.Fuite(conduite="p232", debut=0, fin=n, debit_m3h=20.0,
                  profil="progressif", paliers=4)
    _, debit = Gen.poser(D, noeuds, n, wn, f, propre, niveau0=3.5, verbeux=False)

    quarts = [debit[i * n // 4:(i + 1) * n // 4].mean() for i in range(4)]
    assert quarts == sorted(quarts)
    assert quarts[0] == pytest.approx(f.debit_m3h / 4, rel=0.15)
    assert quarts[-1] == pytest.approx(f.debit_m3h, rel=0.15)


@donnees
def test_generer_ecrit_un_lot_relisible(tmp_path):
    """Un lot doit se relire sans le code qui l'a produit : parquet, json et index."""
    wn = R.charger_modele()
    noeuds = wn.junction_name_list
    n = 2 * Gen.PAS_TRANCHE
    horo = R.horodatage(2018, n + 1)
    D, _ = P.demandes_nominales(wn, horo)

    index = Gen.generer(D, noeuds, wn, n,
                        [Gen.Fuite(conduite="p232", debut=Gen.PAS_TRANCHE, debit_m3h=8.0)],
                        horo, tmp_path, niveau0=3.5, journal=False, verbeux=False)

    assert list(index["fuite"]) == [False, True]
    assert (tmp_path / "index.csv").exists()
    p = pd.read_parquet(tmp_path / "scenario_0001" / "pressions.parquet")
    assert p.shape == (n, 33)
    assert list(p.columns) == R.capteurs()["pressure"]
    # les capteurs réels ne publient que deux décimales : le lot doit en faire autant
    assert np.allclose(p.to_numpy(), np.round(p.to_numpy(), 2))

    import json as _json
    v = _json.loads((tmp_path / "scenario_0001" / "verite.json").read_text())
    assert v["conduite"] == "p232" and v["debut"] == Gen.PAS_TRANCHE
    assert not (tmp_path / "temoin" / "verite.json").exists()
