"""Acces explicite aux producteurs existants, sans modifier leurs fichiers."""
import sys
from pathlib import Path

RACINE = Path(__file__).resolve().parents[2]
for dossier in ("03_data_generation", "07_profondeur_gnn", "04bis_detection_localisation"):
    chemin = str(RACINE / dossier)
    if chemin not in sys.path:
        sys.path.insert(0, chemin)

from generation import modele as M  # noqa: E402
from profondeur import graphe as G, modeles as Mo  # noqa: E402
from etapes import e12_annees_simulees as Annees  # noqa: E402
