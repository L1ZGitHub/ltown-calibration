# Calibration d'un modèle hydraulique de L-Town

Reproduction d'une méthode de calibration publiée, puis mesure de ce qu'il reste à gagner.

Le réseau L-Town est un réseau de distribution d'eau de taille moyenne — 782 jonctions,
905 conduites, 43 km — diffusé avec une année complète de mesures au pas de 5 minutes :
33 capteurs de pression, 3 débitmètres, un niveau de réservoir et 82 compteurs communicants.
On dispose donc du modèle **et** de ce que le réseau a réellement fait. C'est le cadre idéal
pour poser la question qui nous intéresse ici :

> Un modèle hydraulique livré avec des valeurs de conception, jusqu'où peut-on le rapprocher
> de la réalité, et quel paramètre paie vraiment ?

Deux carnets y répondent.

| Carnet | Contenu |
|---|---|
| `notebooks/01_calibration_under_pressure.ipynb` | La méthode de référence, reproduite pas à pas : modèle de demande en produit d'effets, mélange de types de consommateurs, six groupes de rugosité ajustés par Levenberg-Marquardt. |
| `notebooks/02_ameliorations.ipynb` | Quatre leviers que cette méthode laisse de côté, tous lus dans les capteurs : l'état réel de la pompe, l'ancrage du niveau du réservoir, sa section réelle, et le niveau de demande donné par le bilan de masse. |

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate     # .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

Déposez les mesures dans `data/raw/` (voir [`data/README.md`](data/README.md)), ou pointez
`LTOWN_DIR` vers le dossier où elles se trouvent. Puis :

```bash
pytest -q                       # vérifie l'installation et les invariants du réseau
jupyter lab notebooks/
```

Les carnets sont versionnés **avec leurs sorties** : ils se lisent sans être exécutés. Les
relancer prend environ cinq minutes chacun, dont l'essentiel dans l'ajustement des rugosités —
chaque évaluation de la fonction de coût est une simulation hydraulique complète.

Ils tournent par défaut sur une **fenêtre d'une semaine** et non sur l'année entière : une année
de pressions aux 782 jonctions pèse 330 Mo. Trois variables en tête de chaque carnet contrôlent
cela — `DEBUT` (le pas de départ), `N_PAS` (la fenêtre d'évaluation, à mettre à `reseau.PAS_AN`
pour l'année complète) et `N_ENTR` (la fenêtre d'ajustement des rugosités).

## La méthode de référence

Elle se lit en deux équations.

**Un compteur.** Chaque compteur communicant est décomposé en produit d'effets :

```
d(t) = d̄ · T(t) · S(t) · R(t)
```

`d̄` moyenne annuelle, `T(t)` tendance lente obtenue par moyenne glissante sur une semaine (elle
porte donc la saisonnalité annuelle), `S(t)` profil hebdomadaire obtenu par médianes de créneau,
`R(t)` résidu — **jeté**. On ne cherche pas à prédire le bruit de consommation, seulement la
partie reproductible.

**Un nœud non mesuré.** Les 82 compteurs sont tous dans la même zone ; les 700 autres jonctions
n'ont aucune mesure. Le modèle les reconstruit comme un **mélange** de types de consommateurs :

```
d̂ᵢ(t) = Σⱼ d̄ᵢⱼ · Tⱼ(t) · Sⱼ(t)
```

`d̄ᵢⱼ` est la demande nominale du nœud `i` pour le type `j`, lue dans le fichier de réseau — où
chaque jonction porte trois lignes de demande, une par type. Les formes `Tⱼ·Sⱼ` sont obtenues en
résolvant cette même équation à l'envers là où l'on a la mesure.

**Les rugosités** sont ensuite ajustées par moindres carrés non linéaires, non pas conduite par
conduite mais par **six groupes** de diamètre comparable. Le carnet 2 explique pourquoi ce
regroupement n'est pas une commodité mais une nécessité.

## Ce que le second carnet ajoute

Le modèle livré pilote la pompe par deux consignes de niveau du réservoir. La vraie pompe ne les
suit pas, et son débit **est mesuré** : le cycle simulé se désynchronise donc du cycle réel sans
aucune raison — alors que la pompe prélève un cinquième de tout ce qui entre dans le réseau, et
que son démarrage fait descendre toutes les pressions d'un tiers de mètre. Une fois la pompe
commandée par la mesure, plus rien ne régule le niveau du réservoir, qui dérive jusqu'à saturer
contre le trop-plein : il faut le réancrer sur son capteur. Enfin, la consommation totale de la
grande zone est directement **mesurée** par différence entre les débitmètres : aucun modèle de
demande ne bat une mesure.

Restent deux paramètres que l'on croit calibrables. Les **rugosités** ne donnent presque aucune
prise dans un réseau où la conduite médiane perd quatre millimètres de charge : ajustées, elles
retirent du biais et ne touchent pas à la dynamique, ce qu'une fenêtre de transfert et une
évaluation sur l'année entière montrent chacune à leur manière. La **section du réservoir**, elle,
demande deux passes : le bilan volumique par demi-cycle de pompe paraît la mesurer mais confond
en réalité la section et la consommation de la zone alimentée (colinéaires à 0,997) ; simuler la
trajectoire du niveau au lieu de régresser des demi-cycles lève l'ambiguïté — à condition de
choisir une fenêtre où le réservoir ne sature pas, faute de quoi le critère devient monotone et
désigne le bord de la grille. Le résultat reste négatif, pour une raison différente de celle
qu'on attendait : une fois le niveau réancré sur son capteur, la section n'a plus d'effet.

Chaque levier est pris seul, puis cumulé, et le **biais** est à chaque fois séparé de la
**dispersion**. La distinction décide de la lecture : un biais constant s'annule dans toute
statistique de variation, si bien qu'une calibration qui ne fait que retirer un décalage améliore
la RMSE sans rien changer à ce qui se voit dans la dynamique. Deux des quatre leviers sont dans ce
cas.

## Comparaison au chiffre publié

La référence indique une **RMSE de 6 cm** sur ses 33 capteurs, obtenue en calibrant six groupes de
rugosité sur la **première semaine de 2018** et en évaluant sur cette même semaine. C'est donc un
chiffre *dans l'échantillon*, sur la semaine la plus facile de l'année. Le protocole est
reproductible tel quel, et le voici :

| première semaine de 2018, 33 capteurs | biais | dispersion | RMSE |
|---|---|---|---|
| modèle livré | −0,080 | 0,170 | 0,188 m |
| modèle de demande, commandes du fichier | +0,069 | 0,196 | 0,208 m |
| ⤷ + six groupes de rugosité ajustés sur la semaine | +0,004 | 0,192 | 0,192 m |
| configuration complète (pompe, ancrage, bilan de masse) | +0,059 | 0,075 | 0,095 m |
| ⤷ + six groupes de rugosité ajustés sur la semaine | −0,002 | 0,062 | **0,062 m** |
| | | | *publié : 0,060 m* |

**La reproduction retombe sur le chiffre publié**, à deux millimètres près. Deux observations qui
comptent plus que cette coïncidence :

* **la rugosité seule n'y suffit pas.** Ajustée par-dessus le modèle de demande et les commandes
  du fichier — la configuration la plus proche de ce que décrit la référence — elle plafonne à
  0,192 m. Les 6 cm ne s'atteignent qu'une fois les conditions aux limites prises dans les
  capteurs ;
* **trois des six groupes s'arrêtent sur une borne** (`aux_bornes` le signale). Ce sont les trois
  plus petits groupes, 5 à 16 conduites : sans information, un paramètre libre va au bout de sa
  contrainte. La référence n'a pas ce problème, parce que ses groupes sont formés par matériau et
  âge — une information dont on ne dispose pas ici, et qui manque vraiment.

## Résultats sur l'année complète

Les carnets travaillent sur une fenêtre d'une semaine pour rester rapides à relancer. Voici les
mêmes variantes évaluées sur les **105 120 pas de l'année 2018**, aux 33 capteurs de pression :

| | biais | dispersion | RMSE |
|---|---|---|---|
| modèle livré, sans calibration | +0,310 | 0,389 | **0,497 m** |
| + modèle de demande | +0,302 | 0,361 | 0,470 |
| + pompe mesurée, réancrage quotidien du réservoir | +0,272 | 0,266 | 0,380 |
| + niveau de demande par bilan de masse | +0,148 | 0,221 | 0,266 |
| + réancrage toutes les 6 h | +0,122 | 0,158 | 0,199 m |
| + six rugosités ajustées sur la **seule semaine 1** | +0,018 | 0,157 | **0,158 m** |

**−68 % de RMSE, −60 % de dispersion.** Quatre remarques de lecture :

* **la calibration de rugosité tient sur l'année, mais seulement sur la RMSE.** Six coefficients
  réglés sur une semaine de janvier retirent le biais de 12 cm sur les douze mois et laissent la
  dispersion inchangée à un demi pour cent près (0,1578 → 0,1573). Les écarts de ±13 à ±28 % que
  montre le tableau de transfert sur des fenêtres de sept jours se compensent sur l'année :
  c'était du bruit d'ajustement, ni gain ni perte ;

* la dispersion suit la RMSE ici, ce qui n'allait pas de soi — les deux leviers qui n'agissaient
  que sur le biais sur une semaine (bilan de masse, rugosité) ne portent pas ce résultat ;
* le modèle de demande rapporte **moins sur l'année** (−7 % de dispersion) que sur une semaine
  d'été (−20 %). Son apport est saisonnier, et le chiffrer sur une fenêtre choisie le surestime ;
* **le pas de réancrage est le plus gros levier restant.** Passer de vingt-quatre à six heures
  retire encore un quart de la RMSE, et ne coûte que du temps de calcul.

Le calcul prend une vingtaine de minutes :

```python
from calibration import reseau as R, profils as P, ameliorations as A, diagnostics as G
wn = R.charger_modele(24); noeuds = wn.junction_name_list
niv, pompe = R.charger_niveau(2018).to_numpy(), A.statut_pompe(2018)
D = A.recaler_zone_ab(P.demandes_calibrees(wn, 2018)[0], noeuds, wn, 2018)
S = R.pressions_simulees(D, noeuds, R.PAS_AN, tranche_jours=0.25,
                         pompe=pompe, niveau_mesure=niv, niveau0=float(niv[0]))
print(G.resume(S, R.charger_pressions(2018).to_numpy()))
```

## Organisation

```
calibration/
├── reseau.py         accès aux données, topologie, zones, simulation par tranches
├── profils.py        les deux équations du modèle de demande
├── rugosite.py       groupes de conduites, Levenberg-Marquardt, diagnostic d'identifiabilité
├── ameliorations.py  pompe mesurée, niveau ancré, section du réservoir, bilan de masse
└── diagnostics.py    biais / dispersion / RMSE, par capteur et agrégés
```

Les fonctions sont écrites pour être appelées depuis un carnet : elles prennent des tableaux et
rendent des tableaux, ne dessinent rien d'elles-mêmes (sauf les deux aides de `diagnostics`) et
n'écrivent aucun fichier sans qu'on le leur demande.

## Notes d'implémentation

* **Une année ne se simule pas d'un coup.** Le moteur garde tous ses résultats en mémoire.
  `reseau.simuler` découpe en tranches de quelques jours et reporte le niveau du réservoir et le
  statut de la pompe d'une tranche à la suivante ; c'est vérifié par un test (deux tranches de
  12 h contre 24 h d'affilée, écart inférieur au millimètre).
* **Trois pièges d'unités** — modèle en m³/h contre moteur en m³/s, compteurs en L/h contre
  débitmètres en m³/h, CSV à virgule décimale — sont traités à un seul endroit, dans `reseau`.
* **Le créneau hebdomadaire se calcule par jour de la semaine**, jamais par `pas % 2016` : deux
  années consécutives ne commencent pas le même jour, et un profil hebdomadaire transféré à
  l'aveugle se retrouve décalé d'une journée entière.
