"""Calibration d'un modèle hydraulique de L-Town à partir des mesures d'une année.

Le dépôt contient deux choses :

* `profils` et `rugosite` — la reproduction de la calibration décrite par l'équipe
  « Under Pressure » : demande décomposée en produit d'effets, mélange de types de consommateurs,
  puis six groupes de rugosité ajustés par Levenberg-Marquardt ;
* `ameliorations` — quatre leviers supplémentaires lus dans les capteurs (état de la pompe,
  ancrage du niveau du réservoir, section réelle du réservoir, niveau de demande par bilan de
  masse).

`reseau` porte les accès aux données et la simulation par tranches, `diagnostics` les mesures
d'erreur. Les carnets de `notebooks/` enchaînent le tout.
"""
from . import ameliorations, diagnostics, profils, reseau, rugosite   # noqa: F401

__all__ = ["reseau", "profils", "rugosite", "ameliorations", "diagnostics"]
