# Calibration d'un modèle hydraulique de L-Town

Reproduction d'une méthode de calibration publiée, puis mesure de ce qu'il reste à gagner.

Le réseau L-Town est un réseau de distribution d'eau de taille moyenne — 782 jonctions,
905 conduites, 43 km — diffusé avec une année complète de mesures au pas de 5 minutes :
33 capteurs de pression, 3 débitmètres, un niveau de réservoir et 82 compteurs communicants.
On dispose donc du modèle **et** de ce que le réseau a réellement fait. C'est le cadre idéal
pour poser la question qui nous intéresse ici :

> Un modèle hydraulique livré avec des valeurs de conception, jusqu'où peut-on le rapprocher
> de la réalité, et quel paramètre paie vraiment ?

Six carnets y répondent, et les deux premiers existent en deux découpages.

| Carnet | Contenu |
|---|---|
| `notebooks/01_calibration_under_pressure.ipynb` | La méthode de référence, reproduite pas à pas : modèle de demande en produit d'effets, mélange de types de consommateurs, six groupes de rugosité ajustés par moindres carrés sous contraintes. |
| `notebooks/01bis_calibration_pas_a_pas.ipynb` | Le même contenu, en deux fois plus de cellules : une idée par cellule, une figure à la fois, des commentaires courts. Mêmes sorties. |
| `notebooks/02_ameliorations.ipynb` | Quatre leviers que cette méthode laisse de côté, tous lus dans les capteurs : l'état réel de la pompe, l'ancrage du niveau du réservoir, sa section réelle, et le niveau de demande donné par le bilan de masse. |
| `notebooks/02bis_ameliorations_pas_a_pas.ipynb` | Le second carnet dans le même découpage fin. Mêmes sorties. |
| `notebooks/03_generer_des_fuites.ipynb` | À quoi sert le modèle calibré : poser une fuite par émetteur, mesurer la baisse de pression aux 33 capteurs, et la comparer à l'erreur du modèle. |
| `notebooks/04_bilan_par_zone.ipynb` | Sans simuler : la zone C a toute sa frontière instrumentée, donc son bilan de masse donne directement le débit de fuite. La zone A+B, non — et on mesure de combien elle en est loin, puis on s'en sert pour valider le modèle de demande du carnet 1. |
| `notebooks/05_regroupement.ipynb` | Le résultat négatif du carnet 2 venait-il du paramètre ou du regroupement ? Comparaison des groupes par diamètre et par zone × coefficient × diamètre, sur trois fenêtres. |
| `notebooks/06_le_modele_calibre.ipynb` | **Le carnet final, et il se lit seul.** Onze parties qui reprennent tout depuis le réseau : la métrique, les deux équations de demande, la zone C et son bilan de masse, la zone A+B et le piège du recalage, la pompe et le réservoir, les treize groupes de rugosité, puis cinq variantes comparées sur l'année. Aucun renvoi à un autre carnet, aucune conclusion admise sans être remesurée sur la page. Se termine sur la fonction qui livre le modèle calibré. |

Les carnets `bis` ne sont pas un résumé ni une suite : c'est **le même code et les mêmes
résultats**, découpés plus finement. Pour découvrir le travail, commencez par eux ; les versions
courtes se relisent plus vite une fois qu'on sait ce qu'on y cherche.

Le carnet 6 est **autosuffisant** : il refait tous les calculs dont il parle et ne renvoie à aucun
autre fichier. Les cinq premiers montrent comment chaque conclusion a été trouvée, souvent en
passant par des impasses ; le sixième montre l'état final du raisonnement et livre le modèle. Si
vous n'en lisez qu'un, lisez celui-là.

Deux documents accompagnent le code : [`GUIDE.md`](GUIDE.md) explique le travail dans l'ordre où
il a été fait, code à l'appui, et se lit d'une traite ; [`CORRECTIONS.md`](CORRECTIONS.md) liste
les endroits où le texte des carnets demande à être repris.

## Démarrage

```bash
python -m venv .venv && source .venv/bin/activate     # .venv\Scripts\activate sous Windows
pip install -r requirements.txt
```

Déposez les mesures dans `data/raw/` (voir [`data/README.md`](data/README.md)), ou pointez
`LTOWN_DIR` vers le dossier où elles se trouvent. Puis :

```bash
pytest -q                       # 19 tests : installation et invariants du réseau
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
porte donc la saisonnalité annuelle), `S(t)` profil hebdomadaire obtenu par moyennes de créneau,
`R(t)` résidu — **jeté**. On ne cherche pas à prédire le bruit de consommation, seulement la
partie reproductible.

**Un nœud non mesuré.** Les 82 compteurs sont tous dans la même zone ; les 700 autres jonctions
n'ont aucune mesure. Le modèle les reconstruit comme un **mélange** de types de consommateurs :

```
d̂ᵢ(t) = Σⱼ d̄ᵢⱼ · Tⱼ(t) · Sⱼ(t)
```

`d̄ᵢⱼ` est la demande nominale du nœud `i` pour le type `j`, lue dans le fichier de réseau — où
chaque jonction porte une ligne de demande par type présent, jamais les trois. Les formes `Tⱼ·Sⱼ`
sont obtenues en résolvant cette même équation à l'envers là où l'on a la mesure.

Ce modèle passe un contrôle qu'on ne lui demande nulle part ailleurs. Les 82 compteurs sont tous
en zone C, qui porte 12 % de la demande ; appliquées aux 690 nœuds non mesurés de A+B, les formes
qu'on en tire retrouvent la consommation réelle de la zone à **+0,2 %** sur l'année — 157,0 m³/h
contre 156,7 une fois les fuites publiées retirées du bilan des débitmètres (carnet 4, §8).

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
évaluation sur l'année entière montrent chacune à leur manière. Pire : refaire le même ajustement
sur deux fenêtres également propres, à trois jours d'écart, donne des coefficients qui vont d'une
borne à l'autre de l'intervalle autorisé — le paramètre n'est pas identifié, et poser des bornes
serrées ne fait que le remplacer par la borne. La **section du réservoir**, elle,
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

**La reproduction retombe sur le chiffre publié**, à deux millimètres près. Trois observations
qui comptent plus que cette coïncidence :

* **cette semaine ne porte aucune fuite.** C'est la seule de l'année dans ce cas — sur les
  51 autres, le débit de fuite va jusqu'à un cinquième de la consommation de la zone A+B. Le
  chiffre est donc obtenu sur la fenêtre la plus propre qui soit, ce qui rend la comparaison
  légitime (c'est le protocole de la référence) mais ne dit rien des autres semaines ;

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

**−68 % de RMSE, −60 % de dispersion.** Cinq remarques de lecture :

* **ce chiffre annuel est un majorant de l'erreur de modèle, pas une mesure de celle-ci.** L'année
  porte des fuites — 18,2 m³/h en moyenne, soit 10 % de la consommation de la zone A+B — et le
  recalage sur le bilan de masse les absorbe dans la demande : leur volume est réparti sur 690
  nœuds au lieu de sortir en un point. Le carnet 4 chiffre l'opération : le modèle de demande était
  juste à 0,2 % près avant le recalage, qui lui ajoute donc exactement le volume des fuites. Cette mauvaise localisation est spatiale, donc elle se retrouve dans la
  dispersion. Chercher à faire descendre ce chiffre plus bas reviendrait d'ailleurs à absorber le
  signal de fuite, c'est-à-dire à détruire ce qu'un détecteur cherche ;

* **la calibration de rugosité tient sur l'année, mais seulement sur la RMSE.** Six coefficients
  réglés sur la première semaine retirent le biais de 12 cm sur les douze mois et laissent la
  dispersion inchangée à un demi pour cent près (0,1578 → 0,1573). Les écarts de −14 % à +10 % que
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
├── reseau.py         accès aux données, topologie, zones, regroupements, simulation
├── profils.py        les deux équations du modèle de demande
├── rugosite.py       groupes de conduites, ajustement sous contraintes, identifiabilité
├── ameliorations.py  pompe mesurée, niveau ancré, section du réservoir, bilans de masse
└── diagnostics.py    biais / dispersion / RMSE, par capteur et agrégés
```

Le carnet 3 écrit ses scénarios dans `data/generes/`, ignoré par git.

Les fonctions sont écrites pour être appelées depuis un carnet : elles prennent des tableaux et
rendent des tableaux, ne dessinent rien d'elles-mêmes (sauf les deux aides de `diagnostics`) et
n'écrivent aucun fichier sans qu'on le leur demande.

## Notes d'implémentation

* **Une année ne se simule pas d'un coup.** Le moteur garde tous ses résultats en mémoire.
  `reseau.simuler` découpe en tranches de quelques jours et reporte le niveau du réservoir et le
  statut de la pompe d'une tranche à la suivante ; c'est vérifié par un test (deux tranches de
  12 h contre 24 h d'affilée, écart inférieur au millimètre).
* **Une fuite se pose par un émetteur**, jamais par `wn.add_leak` : cette API de WNTR n'est lue
  que par le simulateur écrit en Python et ne change **rien** sous le moteur EPANET — la
  simulation tourne, sans fuite, et sans prévenir. `reseau.coefficient_emetteur` donne le
  coefficient correspondant à un débit visé ; deux tests couvrent les deux points.
* **Trois pièges d'unités** — modèle en m³/h contre moteur en m³/s, compteurs en L/h contre
  débitmètres en m³/h, CSV à virgule décimale — sont traités à un seul endroit, dans `reseau`.
* **Le créneau hebdomadaire se calcule par jour de la semaine**, jamais par `pas % 2016` : deux
  années consécutives ne commencent pas le même jour, et un profil hebdomadaire transféré à
  l'aveugle se retrouve décalé d'une journée entière.
