"""Tous les réglages des étapes, au même endroit.

Chaque réglage est décrit dans `REGLAGES` : sa valeur, son unité, ce qu'il contrôle, et d'où il
vient :

- « réglé sur 2018 » : choisi en regardant les fuites de 2018, l'année qu'on note ensuite ;
- « article » : repris de l'article d'Under Pressure ;
- « choix » : un choix raisonnable, pas réglé sur les fuites.

Un carnet affiche ceux qu'il utilise avec `tableau(["K", "SEUIL", ...])`.
"""
from __future__ import annotations

import pandas as pd

# --------------------------------------------------------------------- étape 00 : les données
ANNEE = 2018
FUITE_ACTIVE = 0.01          # m³/h
MONTEE_FRACTION = 0.9        # part du débit maximal qui marque la fin de la montée
MONTEE_CASSE_J = 1.0         # jours

# --------------------------------------------------------------------- étape 01 : le résidu
SEMAINE_CALME = ("2018-01-01", "2018-01-07")
CAPTEUR_CALME = "n410"
LISSAGE_H = 24               # heures

# --------------------------------------------------------------------- étape 02 : le CUSUM
K = 2.0                      # cm
SEUIL = 24.0                 # cm·h
PAUSE_H = 24                 # heures
REFERENCE_DEPART_H = 72      # heures
CROISSANCE = 0.25            # m³/h
PART = 0.5
FUITE_EXEMPLE = "p673"
GRILLE_K = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
GRILLE_SEUIL = (12.0, 18.0, 24.0, 36.0, 48.0)
CANDIDATS_CUSUM = ((2.0, 24.0), (3.0, 12.0), (3.0, 18.0))     # (K, SEUIL) : étape 02, partie 4
RETARD_EN_PLUS_J = 1.0       # jours
AVANT_H = (48, 24)           # heures avant l'alarme : la « veille »
APRES_H = 6                  # heures avant l'alarme : le « maintenant »

# --------------------------------------------------------------------- étape 03 : le bilan de débit
K_DEBIT = 1.5                # m³/h
SEUIL_DEBIT = 40.0           # m³/h·h
GRILLE_K_DEBIT = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)
GRILLE_SEUIL_DEBIT = (10.0, 20.0, 40.0, 80.0)
CANDIDATS_DEBIT = ((1.5, 40.0), (1.5, 80.0), (3.0, 10.0))    # (K, SEUIL) : étape 03, partie 3

# --------------------------------------------------------------------- étape 04 : lire une fuite
FENETRE_J = 14               # jours
ECART_JOUR = 1.0             # m³/h
ECART_RELATIF = 0.15
DEBIT_DICO = 10.0            # m³/h
VISIBLE_CM = 0.5             # cm
SIGMA_CM = 0.6               # cm
SIGMA_RELATIF = 0.06
AMPLITUDE_MAX = 5.0

# --------------------------------------------------------------------- étape 05 : les signatures
ATTENTE_H = 24               # heures
SEMAINES_J = 7               # jours
PART_NETTE = 0.5
N_TEST = 500                 # fuites simulées par débit
N_PAIRES = 1000
N_REALISTES = 500
TIRAGES_BRUIT = 100          # tirages de bruit par fuite réelle resimulée
DEBITS_TEST = (2.0, 5.0, 10.0, 20.0, 40.0)    # m³/h
DEBITS_PAIRE = (5.0, 10.0, 20.0)              # m³/h
TAILLES_DICO = (2.0, 5.0, 10.0, 20.0, 40.0)   # m³/h
PAS_BRUIT_H = 6              # heures
CALME_M3H = 0.5              # m³/h
GRAINE = 0

# --------------------------------------------------------------------- étape 06 : la chaîne et le barème
PRIX_EQUIPE = 500.0          # €
TIRAGES = 200
CHOIX_CUSUM = (3.0, 18.0)    # (K cm, SEUIL cm·h) : choisi en euros à l'étape 06
CHOIX_DEBIT = (3.0, 10.0)    # (K m³/h, SEUIL m³/h·h) : choisi en euros à l'étape 06

# --------------------------------------------------------------------- étape 07 : les fausses alarmes
POUSSE_MIN = 0.5             # m³/h
ABSTENIR_SI_PLATE = True

# --------------------------------------------------------------------- étape 09 : combiner pression et débit
FUSIONNER_J = (1.0, 2.0, 3.0, 5.0, 7.0)          # jours
CONFIRMER_H = (24, 48)                            # heures
CONFIRMER_M3H = (0.5, 1.0, 2.0, 4.0)              # m³/h
GRILLE_K_FUSION = (0.5, 1.0, 1.5, 2.0, 3.0, 4.0)  # m³/h
GRILLE_SEUIL_FUSION = (10.0, 20.0, 40.0, 80.0)    # m³/h·h
REFERENCE_FUSION = (3.0, 10.0)                    # (K m³/h, SEUIL m³/h·h)
RAPIDE_J = 2.0                                    # jours
CHOIX_FUSIONNER_J = 2.0                           # jours
CHOIX_FONDRE = (2.0, 40.0)                        # (K m³/h, SEUIL m³/h·h) : le candidat du signal fondu choisi en euros
CHOIX_CUSUM_SIMULEES = (3.0, 12.0)      # (K cm, SEUIL cm·h) : le meilleur en euros sur les 12 années simulées
CHOIX_DEBIT_SIMULEES = (3.0, 10.0)      # (K m³/h, SEUIL m³/h·h) : idem
CHOIX_CONFIRMER_SIMULEES = ("pression", 4.0, 24)   # idem, sur la chaîne réglée sur les années simulées
CHOIX_FUSIONNER_J_SIMULEES = 0.5        # jours : idem (sur la grille prolongée)
CONFIRMER_M3H_ETENDU = (0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)   # m³/h : X = 4 était au bord sur 2018
FUSIONNER_J_ETENDU = (0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0)     # jours : W = 1 était au bord sur les simulées (grille 1 à 7 j)
CHOIX_CONFIRMER = ("pression", 4.0, 24)            # (quelles alarmes, X m³/h, H heures)
# --------------------------------------------------------------------- carnet 13 : tous les réglages sur les années simulées
SIGMA_SIMULEES = (1.02, 0.21)          # (cm, part) : σ de la carte de pression, comme l'étape 05, sur les fuites simulées
SIGMA_DUAL_SIMULEES = (0.69, 0.36)     # (m³/h, part) : σ du dual, comme l'étape 11, sur les fuites simulées
PENTE_SIMULEES = (("AB", -0.8876450549314115), ("C", -2.8045067296459583))   # cm par m³/h : étape 01, jours simulés
GRILLES_SIMULEES = {
    "cusum": ((1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0),
              (2.0, 3.0, 4.0, 6.0, 8.0, 12.0, 18.0, 24.0, 36.0, 48.0, 72.0)),        # (K cm, SEUIL cm·h)
    "debit": ((0.5, 1.0, 1.5, 2.0, 3.0, 4.0), (2.5, 5.0, 10.0, 20.0, 40.0, 80.0, 120.0, 160.0)),   # (K m³/h, SEUIL)
    "pause_h": [6, 12, 24, 48, 72, 96],
    "attente_h": [3.0, 6.0, 12.0, 24.0, 36.0, 48.0],
    "semaines_j": [3.0, 5.0, 7.0, 10.0, 14.0],
    "amplitude_max": [2.0, 3.0, 5.0, 8.0, 1e6],          # 1e6 : pas de limite
    "visible_cm": [0.0, 0.25, 0.5, 1.0, 2.0],
    "rayon": [None, 0.0, 100.0, 200.0, 300.0, 400.0, 600.0, 1000.0],   # None : pas de regroupement
    "fusionner_j": [0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 3.0, 5.0, 7.0],
    "fondre": ((0.125, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0), (10.0, 20.0, 40.0, 80.0, 120.0)),
    "confirmer_m3h": [0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0],
    "confirmer_h": [6, 12, 24, 48, 72, 96, 120],
}
REGLE_SIMULEES = {"cusum": (8.0, 6.0), "debit": (3.0, 10.0), "pause_h": 48, "attente_h": 24.0, "semaines_j": 7.0,
                  "amplitude_max": 1e6, "visible_cm": 0.25, "sigma": SIGMA_SIMULEES, "abstenir": True, "rayon": None,
                  "combiner": "confirmer", "confirmer": ("débit", 0.25, 96), "fusionner_j": 0.125, "fondre": (0.25, 80.0),
                  "pente": PENTE_SIMULEES, "carte": "pression", "sigma_dual": SIGMA_DUAL_SIMULEES}
CHOIX_REGROUPER = False      # la chaîne retenue : celle de l'étape 11, sans le regroupement à 300 m (carnet 13)
# --------------------------------------------------------------------- étape 10 : le modèle dual
MARCHE_MIN = 2.0             # m³/h
# --------------------------------------------------------------------- étape 11 : localiser avec le dual
SIGMA_DUAL_04 = 1.0          # m³/h
# --------------------------------------------------------------------- étape 12 : les années simulées
N_VARIANTES = 4              # réseaux « vrais » tirés autour du modèle calibré
ANNEES_PAR_VARIANTE = 3      # années simulées sur chaque réseau
GRAINE_ANNEES = 2026         # graine des réseaux, des consommations, du bruit et des fuites
CONSO_ECART = 0.05           # écart-type du facteur de consommation de l'année, par type de consommateur
CONSO_LISSAGE_J = 14.0       # jours : le facteur varie lentement (largeur du lissage)
N_FUITES_AN = (8, 20)        # nombre de fuites qui commencent dans l'année
PART_ABRUPTES = 0.5          # part des fuites « abruptes » (casses) ; le reste grandit
DEBIT_FUITE_M3H = (1.0, 40.0)        # m³/h : plus petit et plus grand débit d'une fuite (A+B)
DEBIT_FUITE_C_M3H = (1.0, 10.0)      # m³/h : zone C (elle vit sur la pompe et le réservoir)
DEBUT_FUITE_J = (7, 357)     # jours : la première semaine reste sans fuite, comme en 2018
MONTEE_J = (14, 180)         # jours, log-uniforme : durée de la montée d'une fuite qui grandit
DUREE_FUITE_J = (7, 120)     # jours, log-uniforme : de l'ouverture complète à la réparation
PART_NON_REPAREES = 0.3      # part des fuites encore ouvertes à la fin de l'année
CLASSES_CASSE_M3H = (10, 20)     # m³/h : bornes des classes de taille des casses (tableaux)
CLASSES_MONTEE_J = (30, 90)      # jours : bornes des classes de montée des fuites qui grandissent
COEF_DECHARGE = 0.75         # orifice : Q = 0,75 · A · √(2 g p)
NIVEAU_DEPART_M = 3.5        # m : niveau de T1 au 1er janvier
# bruit et réseau vrai, mesurés sur la semaine 1 de 2018 contre les 12 années sans bruit de mesure
SECTION_T1_PART = 0.926      # section vraie de T1 / section du fichier de réseau
CONSO_AB_ECART = 0.008       # écart-type du facteur lent sur la consommation de A+B
CONSO_AB_TAU_H = 48.0        # heures : durée de vie de ce facteur
DERIVE_CAPTEUR_CM = 0.4      # cm : dérive lente propre à chaque capteur de pression
DERIVE_CAPTEUR_TAU_H = 6.0   # heures : durée de vie de cette dérive
BRUIT_DEBITMETRES_M3H = {"p227": 0.76, "p235": 0.0, "PUMP_1": 0.0}   # bruit blanc au pas de 5 min
BRUIT_NIVEAU_M = 0.0         # m : bruit blanc du capteur de niveau (en plus de l'arrondi au cm)
# --------------------------------------------------------------------- le barème (distance)
RAYON = 300.0                # m
PRIX_EAU = 0.80              # €/m³

REGLAGES = {
    "ANNEE": (ANNEE, "", "l'année étudiée", "choix (2019 n'est pas regardée)"),
    "FUITE_ACTIVE": (FUITE_ACTIVE, "m³/h", "débit au-dessus duquel une fuite publiée est « en cours »",
                     "choix"),
    "MONTEE_FRACTION": (MONTEE_FRACTION, "", "la montée d'une fuite finit quand elle atteint cette part "
                        "de son débit maximal", "choix"),
    "MONTEE_CASSE_J": (MONTEE_CASSE_J, "jours", "une fuite qui monte en moins de ça est une casse, "
                       "sinon une fuite qui grandit", "choix"),
    "SEMAINE_CALME": ("1–7 janv.", "", "la semaine sans aucune fuite, pour mesurer le bruit",
                      "fichier des fuites (aucune fuite en cours)"),
    "CAPTEUR_CALME": (CAPTEUR_CALME, "", "le capteur montré pour la semaine calme", "choix"),
    "LISSAGE_H": (LISSAGE_H, "h", "moyenne glissante du signal de zone : retire le cycle de la journée",
                  "choix (une journée)"),
    "K": (K, "cm", "baisse tolérée chaque heure avant que le CUSUM cumule", "réglé sur 2018"),
    "SEUIL": (SEUIL, "cm·h", "cumul qui déclenche l'alarme", "réglé sur 2018"),
    "PAUSE_H": (PAUSE_H, "h", "silence après une alarme, le temps que le lissage absorbe la marche",
                "choix (= LISSAGE_H)"),
    "REFERENCE_DEPART_H": (REFERENCE_DEPART_H, "h", "la référence de départ du CUSUM est la moyenne de "
                           "ces premières heures", "choix"),
    "CROISSANCE": (CROISSANCE, "m³/h", "de combien une fuite doit avoir grandi pour qu'une alarme la "
                   "« voie »", "choix"),
    "PART": (PART, "", "la fuite vue doit faire au moins cette part de ce qui a grandi dans la zone",
             "choix"),
    "FUITE_EXEMPLE": (FUITE_EXEMPLE, "", "la casse montrée pas à pas", "choix"),
    "GRILLE_K": ("1 ; 1,5 ; 2 ; 2,5 ; 3 ; 4", "cm", "les K essayés", "choix"),
    "GRILLE_SEUIL": ("12 ; 18 ; 24 ; 36 ; 48", "cm·h", "les seuils essayés", "choix"),
    "CANDIDATS_CUSUM": (" ; ".join(f"{k:g}/{s:g}" for k, s in CANDIDATS_CUSUM), "K cm / SEUIL cm·h",
                        "les réglages gardés pour la suite ; le choix final se fait en euros",
                        "choisis à l'étape du barème (euros), sur 2018, dans l'échantillon"),
    "RETARD_EN_PLUS_J": (RETARD_EN_PLUS_J, "jours", "retard médian qu'on accepte de perdre pour "
                         "moins de fausses alarmes (choix du candidat le plus prudent)", "choix"),
    "AVANT_H": ("48 h → 24 h", "", "la « veille » à laquelle on compare, pour trouver le capteur qui "
                "baisse le plus", "choix"),
    "APRES_H": (APRES_H, "h", "les dernières heures avant l'alarme, comparées à la veille", "choix"),
    "K_DEBIT": (K_DEBIT, "m³/h", "hausse du bilan tolérée chaque heure avant que le CUSUM cumule",
                "réglé sur 2018"),
    "SEUIL_DEBIT": (SEUIL_DEBIT, "m³/h·h", "cumul du bilan qui déclenche l'alarme", "réglé sur 2018"),
    "GRILLE_K_DEBIT": ("0,5 ; 1 ; 1,5 ; 2 ; 3 ; 4", "m³/h", "les K essayés sur le bilan", "choix"),
    "GRILLE_SEUIL_DEBIT": ("10 ; 20 ; 40 ; 80", "m³/h·h", "les seuils essayés sur le bilan", "choix"),
    "CANDIDATS_DEBIT": (" ; ".join(f"{k:g}/{s:g}" for k, s in CANDIDATS_DEBIT),
                        "K m³/h / SEUIL m³/h·h",
                        "les réglages du bilan gardés pour la suite ; le choix final se fait en euros",
                        "choisis à l'étape du barème (euros), sur 2018, dans l'échantillon"),
    "FENETRE_J": (FENETRE_J, "jours", "la fenêtre de la forme s'arrête au plus tard ce nombre de jours "
                  "après l'alarme", "article"),
    "ECART_JOUR": (ECART_JOUR, "m³/h", "un jour qui s'écarte de la courbe de plus que ça arrête la "
                   "fenêtre (une autre fuite ou une réparation commence)", "choix"),
    "ECART_RELATIF": (ECART_RELATIF, "", "… ou de plus que cette part du débit de la fuite", "choix"),
    "DEBIT_DICO": (DEBIT_DICO, "m³/h", "le débit de la fuite simulée sur chaque conduite (le "
                   "dictionnaire des signatures)", "choix"),
    "VISIBLE_CM": (VISIBLE_CM, "cm", "une conduite dont la signature est plus petite que ça ne se voit "
                   "pas aux capteurs : elle n'est pas candidate", "choix"),
    "SIGMA_CM": (SIGMA_CM, "cm", "l'écart toléré par capteur entre l'observation et la signature",
                 "bruit sur 24 h (étape 01)"),
    "SIGMA_RELATIF": (SIGMA_RELATIF, "", "… ou cette part de la taille de l'observation, si c'est plus",
                      "choix"),
    "AMPLITUDE_MAX": (AMPLITUDE_MAX, "", "amplitude maximale de la signature (5 × 10 = 50 m³/h)", "choix"),
    "ATTENTE_H": (ATTENTE_H, "h", "alarme de pression : on observe les 24 h qui suivent, puis on envoie "
                  "la conduite", "choix (dossier 04)"),
    "SEMAINES_J": (SEMAINES_J, "jours", "alarme de débit : on compare la semaine avant le départ du CUSUM "
                   "à la semaine après l'alarme, puis on envoie", "choix (dossier 04)"),
    "PART_NETTE": (PART_NETTE, "", "pour régler SIGMA_RELATIF, on ne garde que les fuites dont l'écart "
                   "restant fait moins de cette part de la baisse observée", "choix"),
    "N_TEST": (N_TEST, "", "nombre de fuites simulées pour chaque débit, pour les vérifications", "choix"),
    "N_PAIRES": (N_PAIRES, "", "nombre de paires de fuites simulées", "choix"),
    "N_REALISTES": (N_REALISTES, "", "nombre de fuites simulées avec des débits tirés comme ceux des "
                    "vraies fuites à leur première alarme", "choix"),
    "TIRAGES_BRUIT": (TIRAGES_BRUIT, "", "tirages du vrai bruit pour chaque fuite de 2018 resimulée", "choix"),
    "TAILLES_DICO": ("2 ; 5 ; 10 ; 20 ; 40", "m³/h", "débits des dictionnaires « plusieurs tailles »",
                     "choix (= DEBITS_TEST)"),
    "PAS_BRUIT_H": (PAS_BRUIT_H, "h", "on prend un échantillon du vrai bruit toutes les PAS_BRUIT_H heures",
                    "choix"),
    "CALME_M3H": (CALME_M3H, "m³/h", "un instant est « calme » si les fuites publiées de chaque zone ont "
                  "changé de moins que ça entre la veille et les 24 h suivantes", "choix ; le fichier des "
                  "fuites ne sert ici qu'à choisir les instants, pas à localiser"),
    "DEBITS_TEST": ("2 ; 5 ; 10 ; 20 ; 40", "m³/h", "débits essayés pour vérifier le dictionnaire",
                    "choix"),
    "DEBITS_PAIRE": ("5 ; 10 ; 20", "m³/h", "débits tirés pour les paires de fuites", "choix"),
    "GRAINE": (GRAINE, "", "graine du tirage au hasard (simulations, hasard du barème)", "choix"),
    "PRIX_EQUIPE": (PRIX_EQUIPE, "€", "coût d'une alarme fausse (une équipe envoyée pour rien)",
                    "barème officiel"),
    "TIRAGES": (TIRAGES, "", "nombre de tirages du hasard", "choix"),
    "CHOIX_CUSUM": (f"{CHOIX_CUSUM[0]:g}/{CHOIX_CUSUM[1]:g}", "K cm / SEUIL cm·h",
                    "le réglage du CUSUM de pression utilisé à partir de l'étape 06",
                    "choisi en euros à l'étape 06, sur 2018, dans l'échantillon"),
    "CHOIX_DEBIT": (f"{CHOIX_DEBIT[0]:g}/{CHOIX_DEBIT[1]:g}", "K m³/h / SEUIL m³/h·h",
                    "le réglage du CUSUM de débit utilisé à partir de l'étape 06",
                    "choisi en euros à l'étape 06, sur 2018, dans l'échantillon"),
    "POUSSE_MIN": (POUSSE_MIN, "m³/h", "une fausse alarme vient du détecteur si aucune fuite de la zone "
                   "n'a grandi d'au moins ça entre les deux intervalles comparés", "choix (dossier 04)"),
    "ABSTENIR_SI_PLATE": (ABSTENIR_SI_PLATE, "", "quand aucune conduite n'explique la baisse (carte "
                          "plate), ne rien envoyer", "correction de l'étape 07"),
    "FUSIONNER_J": (" ; ".join(f"{x:g}" for x in FUSIONNER_J), "jours", "fusionner : une alarme de "
                    "pression et une de débit de la même zone à moins de W jours n'en font qu'une ; les W "
                    "essayés", "choix ; W choisi en euros à l'étape 09, sur 2018, dans l'échantillon"),
    "CONFIRMER_H": (" ; ".join(f"{x:g}" for x in CONFIRMER_H), "h", "confirmer : on regarde l'autre "
                    "signal jusqu'à H heures après l'alarme ; les H essayés", "choix"),
    "CONFIRMER_M3H": (" ; ".join(f"{x:g}" for x in CONFIRMER_M3H), "m³/h", "confirmer : l'autre signal "
                      "doit avoir bougé d'au moins X m³/h (la pression est convertie avec la pente de "
                      "l'étape 01) ; les X essayés", "choix ; X choisi en euros à l'étape 09, sur 2018"),
    "GRILLE_K_FUSION": ("0,5 ; 1 ; 1,5 ; 2 ; 3 ; 4", "m³/h", "les K essayés sur le signal fusionné",
                        "choix"),
    "GRILLE_SEUIL_FUSION": ("10 ; 20 ; 40 ; 80", "m³/h·h", "les seuils essayés sur le signal fusionné",
                            "choix"),
    "REFERENCE_FUSION": (f"{REFERENCE_FUSION[0]:g}/{REFERENCE_FUSION[1]:g}", "K m³/h / SEUIL m³/h·h",
                         "le point de départ du choix des candidats du signal fusionné (le réglage du "
                         "débit choisi à l'étape 06)", "choix"),
    "RAPIDE_J": (RAPIDE_J, "jours", "une alarme du signal fusionné dont le CUSUM a monté en moins de ça "
                 "est localisée comme une casse (24 h), sinon sur des semaines", "choix"),
    "CHOIX_FONDRE": (f"{CHOIX_FONDRE[0]:g}/{CHOIX_FONDRE[1]:g}", "K m³/h / SEUIL m³/h·h", "le réglage du CUSUM du "
                     "signal fondu (étape 09, partie 5) ; la chaîne ne fond pas les signaux",
                     "choisi en euros à l'étape 09 parmi les candidats du front, sur 2018, dans l'échantillon"),
    "CHOIX_FUSIONNER_J": (CHOIX_FUSIONNER_J, "jours", "le W de la fusion utilisé après l'étape 09",
                          "choisi en euros à l'étape 09, sur 2018, dans l'échantillon"),
    "CHOIX_CUSUM_SIMULEES": (f"{CHOIX_CUSUM_SIMULEES[0]:g}/{CHOIX_CUSUM_SIMULEES[1]:g}", "K cm / SEUIL cm·h",
                             "le CUSUM de pression réglé sur les années simulées", "le couple qui rapporte le plus "
                             "d'euros en tout sur les 12 années simulées, dans la grille de l'étape 06 (carnet 06)"),
    "CHOIX_DEBIT_SIMULEES": (f"{CHOIX_DEBIT_SIMULEES[0]:g}/{CHOIX_DEBIT_SIMULEES[1]:g}", "K m³/h / SEUIL m³/h·h",
                             "le CUSUM du bilan réglé sur les années simulées", "idem"),
    "CHOIX_CONFIRMER_SIMULEES": (f"{CHOIX_CONFIRMER_SIMULEES[0]}, X = {CHOIX_CONFIRMER_SIMULEES[1]:g}, "
                                 f"H = {CHOIX_CONFIRMER_SIMULEES[2]}", "", "la confirmation réglée sur les années "
                                 "simulées", "le meilleur en euros en tout sur les 12 années simulées, sur la chaîne "
                                 "réglée sur elles (carnet 09)"),
    "CHOIX_FUSIONNER_J_SIMULEES": (CHOIX_FUSIONNER_J_SIMULEES, "jours", "le W de la fusion réglé sur les années "
                                   "simulées", "idem (carnet 09)"),
    "CONFIRMER_M3H_ETENDU": (" ; ".join(f"{x:g}" for x in CONFIRMER_M3H_ETENDU), "m³/h", "confirmer : les X "
                             "essayés pour le réglage sur les années simulées, et pour refaire celui de 2018 sur la "
                             "même grille", "CONFIRMER_M3H prolongé : X = 4 m³/h, le choix de 2018, en était le bord"),
    "FUSIONNER_J_ETENDU": (" ; ".join(f"{x:g}" for x in FUSIONNER_J_ETENDU), "jours", "fusionner : les W essayés "
                           "pour le réglage sur les années simulées, et pour refaire celui de 2018 sur la même grille",
                           "FUSIONNER_J prolongé vers le bas : W = 1 jour, le choix sur les années simulées, en était le "
                           "bord"),
    "CHOIX_CONFIRMER": (f"{CHOIX_CONFIRMER[0]}, X = {CHOIX_CONFIRMER[1]:g}, H = {CHOIX_CONFIRMER[2]}",
                        "m³/h, h", "la confirmation utilisée après l'étape 09 : seules les alarmes de pression "
                        "sont confirmées, par une hausse du bilan d'au moins X en H heures",
                        "choisi en euros à l'étape 09, sur 2018, dans l'échantillon (X au bord de la grille)"),
    "CHOIX_REGROUPER": (CHOIX_REGROUPER, "", "la chaîne retenue regroupe-t-elle les alarmes proches (étape 08) ? "
                        "Non : la chaîne retenue est celle de l'étape 11 (dual et confirmation par le bilan) sans "
                        "regroupement",
                        "phase 3 du carnet 13, sur les 12 années simulées : « sans regroupement » choisi à l'étape 08, "
                        "et encore en retirant tour à tour chacune des 4 variantes ; sur 2018 cela coûte 3 000 € "
                        "(0,13 σ, z +5,7 → +4,7), un écart dans le bruit d'une année ; sur les 12 années, ne pas "
                        "regrouper rapporte 2 604 480 € contre 1 574 392 €"),
    "MARCHE_MIN": (MARCHE_MIN, "m³/h", "on ne mesure la marche d'une fuite que si elle grandit d'au moins "
                   "ça en un jour (les casses)", "choix (dossier 04)"),
    "SIGMA_DUAL_04": (SIGMA_DUAL_04, "m³/h", "l'écart toléré par capteur pour la carte du dual dans le "
                      "dossier 04 (jamais réglé)", "dossier 04 (valeur par défaut)"),
    "N_VARIANTES": (N_VARIANTES, "", "nombre de réseaux « vrais » tirés autour du modèle calibré (conduites "
                    "±1 %, composition des consommateurs, décalage horaire)", "choix (coût de calcul)"),
    "ANNEES_PAR_VARIANTE": (ANNEES_PAR_VARIANTE, "", "années simulées sur chaque réseau : fuites, bruit et "
                            "consommation changent d'une année à l'autre", "choix (coût de calcul)"),
    "GRAINE_ANNEES": (GRAINE_ANNEES, "", "graine des années simulées", "choix"),
    "CONSO_ECART": (CONSO_ECART, "", "écart-type du facteur qui modifie la consommation de chaque type de "
                    "consommateur d'une année à l'autre", "choix (les profils nominaux sont annoncés à 10 % "
                    "près par l'énoncé de BattLeDIM)"),
    "CONSO_LISSAGE_J": (CONSO_LISSAGE_J, "jours", "le facteur de consommation varie lentement : bruit lissé "
                        "sur cette largeur", "choix"),
    "N_FUITES_AN": (f"{N_FUITES_AN[0]} à {N_FUITES_AN[1]}", "", "nombre de fuites qui commencent dans une "
                    "année simulée (tiré uniformément)", "choix (non donné par l'article)"),
    "PART_ABRUPTES": (PART_ABRUPTES, "", "part des fuites abruptes (casse d'un coup) ; les autres grandissent "
                      "(« incipient »)", "choix ; les deux types viennent du générateur de BattLeDIM"),
    "DEBIT_FUITE_M3H": (f"{DEBIT_FUITE_M3H[0]:g} à {DEBIT_FUITE_M3H[1]:g}", "m³/h", "débit d'une fuite à son "
                        "maximum, sous la pression médiane de son nœud, en zone A+B. Le diamètre du trou est tiré "
                        "uniformément entre ceux qui donnent ces deux débits : environ 40 % des fuites sous "
                        "10 m³/h, 25 % entre 10 et 20, 35 % au-dessus", "choix large, des fuites qu'aucun "
                        "détecteur ne voit (1 m³/h, sous le bruit du bilan) aux grosses casses ; il donne assez de "
                        "fuites dans chaque classe de taille ; la liste de 2018 n'est pas utilisée"),
    "DEBIT_FUITE_C_M3H": (f"{DEBIT_FUITE_C_M3H[0]:g} à {DEBIT_FUITE_C_M3H[1]:g}", "m³/h", "la même chose en "
                          "zone C", "choix : la zone C est alimentée par une pompe de 44 m³/h (débit mesuré "
                          "quand elle tourne) qui sert aussi sa consommation ; plus de 10 m³/h de fuite y "
                          "viderait le réservoir"),
    "DEBUT_FUITE_J": (f"{DEBUT_FUITE_J[0]} à {DEBUT_FUITE_J[1]}", "jours", "jour de l'année où une fuite "
                      "commence (uniforme) ; la première semaine reste sans fuite", "choix (la chaîne mesure "
                      "son bruit sur la première semaine)"),
    "MONTEE_J": (f"{MONTEE_J[0]} à {MONTEE_J[1]}", "jours", "fuite qui grandit : son diamètre monte "
                 "linéairement de 0 au maximum pendant cette durée (log-uniforme), puis reste fixe. Une fuite "
                 "qui commence tard grandit encore à la fin de l'année", "montée linéaire : générateur de "
                 "BattLeDIM ; durée : choix large, de deux semaines à six mois"),
    "DUREE_FUITE_J": (f"{DUREE_FUITE_J[0]} à {DUREE_FUITE_J[1]}", "jours", "durée entre l'ouverture complète "
                      "(le début pour une casse, la fin de la montée sinon) et la réparation (log-uniforme)",
                      "choix (non donné par l'article)"),
    "CLASSES_CASSE_M3H": (f"{CLASSES_CASSE_M3H[0]} et {CLASSES_CASSE_M3H[1]}", "m³/h", "bornes des classes de "
                          "taille des casses dans les tableaux par classe", "choix"),
    "CLASSES_MONTEE_J": (f"{CLASSES_MONTEE_J[0]} et {CLASSES_MONTEE_J[1]}", "jours", "bornes des classes de "
                         "montée des fuites qui grandissent dans les tableaux par classe", "choix"),
    "PART_NON_REPAREES": (PART_NON_REPAREES, "", "part des fuites jamais réparées dans l'année", "choix (l'énoncé "
                          "dit que certaines fuites ne sont pas réparées)"),
    "COEF_DECHARGE": (COEF_DECHARGE, "", "coefficient de l'orifice : Q = C · A · √(2 g p)",
                      "générateur de BattLeDIM (code public)"),
    "NIVEAU_DEPART_M": (NIVEAU_DEPART_M, "m", "niveau de T1 au début de l'année simulée", "choix"),
    "SECTION_T1_PART": (SECTION_T1_PART, "", "section du réservoir T1 dans le réseau vrai, en part de celle du "
                        "fichier de réseau (201 m²) ; la chaîne garde celle du fichier, comme sur 2018",
                        "mesuré sur la semaine 1 de 2018 : le bilan horaire de la zone C contient −7,4 % du terme "
                        "de stockage (0,1 % dans les années sans ce réglage) ; le résidu de pression de la zone C "
                        "suit la dérive du niveau depuis le dernier recalage (corrélation 0,95, pente 6,8 %)"),
    "CONSO_AB_ECART": (CONSO_AB_ECART, "", "écart-type d'un facteur lent sur la consommation vraie de la zone A+B, "
                       "que les compteurs (tous en zone C) ne voient pas", "ajusté sur la semaine 1 de 2018 : le "
                       "bilan A+B sur 24 h (0,85 m³/h contre 0,51 sans ce facteur) et le résidu moyen de A+B sur "
                       "24 h (0,60 cm contre 0,38) ; les deux bougent ensemble (corrélation −0,99)"),
    "CONSO_AB_TAU_H": (CONSO_AB_TAU_H, "h", "durée de vie de ce facteur", "ajusté sur la semaine 1 de 2018 : "
                       "12 h à 4 jours s'ajustent aussi bien sur une semaine ; on prend le milieu"),
    "DERIVE_CAPTEUR_CM": (DERIVE_CAPTEUR_CM, "cm", "écart-type d'une dérive lente propre à chaque capteur de "
                          "pression", "ajusté sur la semaine 1 de 2018 : l'écart de chaque capteur à la moyenne de "
                          "sa zone, en moyenne horaire (0,64 cm contre 0,55) et sur 24 h (0,23 contre 0,11)"),
    "DERIVE_CAPTEUR_TAU_H": (DERIVE_CAPTEUR_TAU_H, "h", "durée de vie de cette dérive", "ajusté avec "
                             "DERIVE_CAPTEUR_CM (grille 1 à 24 h)"),
    "BRUIT_DEBITMETRES_M3H": (", ".join(f"{k} {v:g}" for k, v in BRUIT_DEBITMETRES_M3H.items()), "m³/h",
                              "bruit blanc de chaque débitmètre au pas de 5 min", "mesuré sur la semaine 1 de "
                              "2018 : la variance des différences d'un pas à l'autre en plus de celle des années "
                              "sans bruit (0 si 2018 est sous leur médiane)"),
    "BRUIT_NIVEAU_M": (BRUIT_NIVEAU_M, "m", "bruit blanc du capteur de niveau de T1, en plus de l'arrondi au "
                       "cm", "mesuré comme les débitmètres : 2018 et les années sans bruit sont identiques"),
    "SIGMA_SIMULEES": ("1,02 cm ; 0,21", "cm, part", "σ et partie relative de la carte de pression, réglés sur les "
                       "années simulées", "comme l'étape 05 (écart médian à la bonne conduite), sur les fuites des 12 "
                       "années simulées ; ne change pas les alarmes (carnet 05)"),
    "SIGMA_DUAL_SIMULEES": ("0,69 m³/h ; 0,36", "m³/h, part", "σ et partie relative de la carte du dual, réglés sur les "
                            "années simulées", "comme l'étape 11, sur les fuites des 12 années simulées ; ne change pas "
                            "les alarmes (carnet 11)"),
    "PENTE_SIMULEES": ("A+B −0,888 ; C −2,80", "cm par m³/h", "la pente résidu contre fuite, pour fondre les signaux "
                       "ou confirmer une alarme de débit par la pression",
                       "la droite de l'étape 01 sur les jours des 12 années simulées (2018 : −0,90 et −2,90)"),
    "GRILLES_SIMULEES": ("voir reglages.py", "", "les grilles essayées pour chaque réglage, sur 2018 et sur les années "
                         "simulées (carnets 05 à 13)", "choix, prolongées tant que le meilleur tombait au bord, d'un côté "
                         "ou de l'autre"),
    "REGLE_SIMULEES": ("CUSUM 8/6 et 3/10, pause 48 h, lecture 24 h et 7 j, sans limite d'amplitude, visible 0,25 cm, "
                       "abstention, carte de pression, confirmer les alarmes de débit (0,25 m³/h en 96 h), sans "
                       "regroupement", "",
                       "la chaîne entière réglée sur les années simulées (carnet 13)",
                       "chaque réglage : le plus d'euros en tout sur les 12 années simulées, dans l'ordre de la chaîne, "
                       "les CUSUM et la lecture refaits jusqu'à ce qu'un tour ne change plus rien"),
    "RAYON": (RAYON, "m", "distance maximale pour qu'une alarme trouve une fuite", "barème officiel"),
    "PRIX_EAU": (PRIX_EAU, "€/m³", "prix de l'eau perdue", "barème officiel"),
}


def tableau(noms: list[str]) -> pd.DataFrame:
    """Les réglages `noms`, en tableau : valeur, unité, rôle, origine."""
    lignes = {n: dict(zip(["valeur", "unité", "ce qu'il contrôle", "origine"], REGLAGES[n]))
              for n in noms}
    return pd.DataFrame.from_dict(lignes, orient="index")
