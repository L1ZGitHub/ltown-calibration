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

Restent deux paramètres que l'on croit calibrables et qui ne le sont pas ici — la section du
réservoir, indissociable de la consommation non comptée de la zone qu'elle alimente, et les
rugosités, dans un réseau où la conduite médiane perd quatre millimètres de charge. Le carnet
établit les deux par une mesure directe, sans lancer d'optimisation, puis vérifie sur une fenêtre
de transfert ce qu'un ajustement de rugosité a réellement appris.

Chaque levier est pris seul, puis cumulé, et le **biais** est à chaque fois séparé de la
**dispersion**. La distinction décide de la lecture : un biais constant s'annule dans toute
statistique de variation, si bien qu'une calibration qui ne fait que retirer un décalage améliore
la RMSE sans rien changer à ce qui se voit dans la dynamique. Deux des quatre leviers sont dans ce
cas.

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
