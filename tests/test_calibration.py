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

    s = P.saisonnalite(x, cr)

    assert np.abs(s - base / base.mean()).max() < 1e-9


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
