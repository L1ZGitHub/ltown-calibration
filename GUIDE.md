# Guide de reprise

Ce document explique le dépôt en détail : ce que fait chaque fonction, ce que fait chaque
cellule des deux carnets, d'où vient chaque chiffre publié, et ce que l'on peut réécrire sans
casser la cohérence entre le texte et les sorties.

Le `README` s'adresse à quelqu'un qui découvre le sujet. Ce guide-ci s'adresse à quelqu'un qui
va **modifier** le dépôt.

Ordre de lecture conseillé si vous partez de zéro :

1. §1 — le résultat en une page (5 min) ;
2. §3 — les six conventions du code, sans lesquelles le reste paraît arbitraire (10 min) ;
3. §5 et §6 — les carnets cellule par cellule, avec le carnet ouvert à côté (1 h) ;
4. §4 — les modules, en s'y référant au fur et à mesure ;
5. §9 — **à lire avant toute réécriture** : les endroits où le texte actuel ne dit pas ce que
   le code fait.

---

## 1. Le résultat en une page

**La question.** Un modèle hydraulique livré avec des valeurs de conception, jusqu'où peut-on le
rapprocher de la réalité, et quel paramètre paie vraiment ?

**Le terrain.** Le réseau L-Town : 782 jonctions, 905 conduites, 43,2 km, une année de mesures au
pas de 5 minutes (105 120 pas). 33 capteurs de pression, 3 débitmètres, 1 capteur de niveau,
82 compteurs communicants. On a donc le modèle **et** ce que le réseau a réellement fait.

**Ce qui est reproduit (carnet 1).** Une méthode de calibration publiée, en deux équations plus un
ajustement :

| étape | idée | fonction |
|---|---|---|
| équation (1) | décomposer chaque compteur en `d̄ · T(t) · S(t) · R(t)`, jeter `R` | `profils.decomposer` |
| équation (2) | reconstruire les 700 nœuds sans mesure comme un **mélange** de types de consommateurs | `profils.formes_par_categorie`, `demandes_calibrees` |
| rugosités | six groupes de conduites, moindres carrés non linéaires bornés | `rugosite.ajuster` |

**Ce qui est ajouté (carnet 2).** Quatre leviers, tous lus dans des capteurs déjà installés :

| # | levier | mesure | verdict |
|---|---|---|---|
| 1 | état réel de la pompe | débitmètre `PUMP_1` | **décisif** |
| 2 | réancrage du niveau du réservoir | capteur `T1` | **décisif**, et indissociable du 1 |
| 3 | section réelle du réservoir | simulation de la trajectoire de niveau | **nul** — identifiable mais sans effet |
| 4 | niveau de la demande A+B | bilan de masse aux débitmètres | **biais seulement** |

**Les chiffres.** Sur l'année complète, aux 33 capteurs :

| | biais | dispersion | RMSE |
|---|---|---|---|
| modèle livré | +0,310 | 0,389 | **0,497 m** |
| + modèle de demande | +0,302 | 0,361 | 0,470 |
| + pompe mesurée, réancrage quotidien | +0,272 | 0,266 | 0,380 |
| + bilan de masse | +0,148 | 0,221 | 0,266 |
| + réancrage toutes les 6 h | +0,122 | 0,158 | 0,199 |
| + six rugosités réglées sur la semaine 1 | +0,018 | 0,157 | **0,158 m** |

Sur le protocole de la référence — six groupes ajustés sur la première semaine de 2018, évalués
sur cette même semaine — la reproduction donne **0,062 m** contre **0,060 m** publiés.

**Les trois idées qui portent tout le dépôt**, et qu'il faut préserver en réécrivant :

1. **`RMSE² = biais² + dispersion²`.** Un biais constant s'annule dans toute statistique de
   variation. Une calibration qui ne fait que recentrer améliore la RMSE sans améliorer le
   modèle. Deux des quatre leviers, et l'ajustement de rugosité, sont dans ce cas.
2. **Une condition aux limites lue dans un capteur vaut un modèle entier.** Le plus gros levier
   unique n'est pas le modèle de consommation, c'est l'état de la pompe.
3. **Un paramètre sans information ne reste pas inerte** : il part au bout de sa contrainte et se
   met au service de l'erreur qui domine la fenêtre d'ajustement.

---

## 2. Les données

Rien n'est redistribué dans le dépôt. `data/raw/` attend :

```
L-TOWN.inp                  modèle EPANET, unités CMH (m³/h)
dataset_configuration.yaml  listes officielles de capteurs
2018_SCADA_Pressures.csv    33 colonnes, m
2018_SCADA_Flows.csv        3 colonnes (p227, p235, PUMP_1), m³/h
2018_SCADA_Levels.csv       1 colonne (T1), m
2018_SCADA_Demands.csv      82 colonnes, L/h
2019_*.csv                  facultatif
```

`LTOWN_DIR` pointe ailleurs si les fichiers vivent ailleurs. `reseau.verifier_donnees()` lève une
erreur lisible si l'un manque — c'est la première ligne de chaque carnet.

**Quatre pièges, tous traités dans `reseau` et nulle part ailleurs :**

* CSV européens : `sep=";"`, `decimal=","`. Sans ces deux arguments, `pandas` lit des chaînes de
  caractères **sans prévenir** ;
* compteurs en **L/h**, débitmètres en **m³/h** : `charger_amr` divise par 1000 ;
* `.inp` en m³/h, `wntr` en SI (m³/s) : tout ce qui sort du module est en m³/h, la division par
  3600 se fait au moment d'écrire dans le modèle ;
* pressions arrondies à **deux décimales** — c'est le plancher de ce qu'on peut espérer mesurer,
  et il faut s'en souvenir quand on compare des pertes de charge de quelques millimètres.

**Géographie du problème** (le point le plus important de tout le dépôt) : les 82 compteurs sont
**tous** dans la zone C, celle qui est derrière le réservoir, et cette zone ne porte que **11,1 %**
de la demande nominale. Les 690 jonctions de la grande zone n'ont aucune mesure de consommation.
Le modèle de demande n'interpole donc pas entre des points connus : il **extrapole** d'une zone
vers une autre.

---

## 3. Les conventions du code

Six décisions traversent tout le dépôt. Les connaître évite de lire chaque fonction comme un cas
particulier.

**(a) Une unité, partout.** Tout ce qui sort de `calibration` est en m³/h et en mètres. Aucune
conversion ailleurs que dans `reseau`.

**(b) La simulation se fait par tranches.** Une année ne se simule pas d'un coup : le moteur garde
tous ses résultats en mémoire. `reseau.simuler` découpe en tranches de quelques jours et **reporte
l'état** — niveau du réservoir et statut de la pompe au dernier pas — d'une tranche à la suivante.
Vérifié par un test : deux tranches de 12 h donnent les mêmes pressions que 24 h d'affilée à moins
d'un millimètre près.

**(c) C'est la fonction `extraire` qui borne la mémoire.** `simuler(D, noeuds, n_pas, extraire, …)`
appelle `extraire(res, wn, debut, fin)` une fois par tranche et ne garde que ce qu'elle renvoie.
Une année de pressions aux 782 jonctions pèse 330 Mo en float32 ; aux 33 capteurs, 14 Mo. **Ne
jamais renvoyer `res`.** `pressions_simulees` est le raccourci qui garde les 33 capteurs.

**(d) Le profil de demande porte un pas de plus que l'horizon.** EPANET lit le profil un pas
au-delà de la fin de la tranche. D'où le `+ 1` partout : `D[DEBUT:DEBUT + N_PAS + 1]` pour
`N_PAS` pas simulés. Si le profil est trop court, EPANET **reboucle silencieusement sur son
début** — pas d'erreur, juste un résultat faux. `preparer` lève une `ValueError` explicite, et
`simuler` tient la dernière valeur sur l'ultime tranche d'une série de longueur exacte. Un test
de non-régression protège ce point (`test_une_serie_de_demande_sans_pas_supplementaire`).

**(e) Le créneau hebdomadaire se calcule par jour de la semaine**, jamais par `pas % 2016` :
2018 commence un lundi et 2019 un mardi. Un profil hebdomadaire transféré à l'aveugle se
retrouverait décalé d'une journée entière.

**(f) Les carnets travaillent sur une semaine, pas sur l'année.** Trois variables en tête de
chaque carnet :

| variable | valeur | rôle |
|---|---|---|
| `DEBUT` | `181 * R.PAS_JOUR` = 52 128 | 1er juillet, la saison de forte demande |
| `N_PAS` | `7 * R.PAS_JOUR` = 2 016 | fenêtre d'**évaluation** ; mettre `R.PAS_AN` pour l'année |
| `N_ENTR` | `2 * R.PAS_JOUR` = 576 | fenêtre d'**ajustement** des rugosités |

**Attention en réécrivant** : certaines cellules évaluent sur **3 jours** et non sur `N_PAS`
(les fonctions locales `evaluer`, `croiser`). C'est pourquoi le « modèle livré » vaut 0,4740 m
dans une cellule et 0,4535 m dans une autre. Ce ne sont pas deux valeurs contradictoires, ce sont
deux fenêtres. Voir §8.

---

## 4. Les cinq modules

### 4.1 `reseau.py` — accès, topologie, simulation (300 lignes)

**Constantes.** `PAS_MIN = 5`, `PAS_JOUR = 288`, `PAS_SEMAINE = 2016`, `PAS_AN = 105 120`,
`CATEGORIES = ("Residential", "Commercial", "Industrial")`.

**Lecture des mesures.** `charger_scada(annee, genre)` est le point d'entrée unique ;
`charger_amr` (÷1000), `charger_debits`, `charger_niveau`, `charger_pressions` (dans l'ordre du
fichier de configuration) en dérivent. `capteurs()` lit les listes officielles dans le YAML.
`horodatage(annee, n_pas)` construit l'index. `creneau(index)` donne le créneau hebdomadaire.

**Topologie.**

* `graphe(wn)` — tous les liens en graphe non orienté ;
* `zones(wn)` — **déduit** la partition, ne la lit pas dans une liste écrite à la main : on coupe
  les deux arêtes qui traversent le réservoir (`n54`→`T1` par la pompe, `T1`→`n343` par `p239`) et
  la composante connexe de `n343` est la zone C. Résultat : **92** jonctions en C, **690** en A+B ;
* `bases_nominales(wn)` — 782 × 3 en m³/h. **Le piège** : `Junction.base_demand` ne renvoie que la
  **première** des trois lignes de demande, donc 0 pour une jonction purement industrielle. Il
  faut parcourir `demand_timeseries_list`. Totaux : Residential 106,09, Commercial 66,55,
  Industrial 3,94 m³/h ;
* `groupes_de_rugosite(wn, n_groupes=6)` — regroupe les 905 conduites par **classe de diamètre**.
  L-Town a sept diamètres très inégalement répartis ; pour six groupes, les deux plus petites
  classes fusionnent. Résultat : `D100` 705, `D150` 103, `D200` 64, `D160` 16, `D225` 12,
  `D<=75` 5. **Ce déséquilibre explique tout le §6 du carnet 2** ;
* `section_reservoir(wn)` — section de T1 telle qu'écrite dans le modèle : 201,1 m².

**Simulation.**

* `executer(wn)` — un passage EPANET dans un fichier temporaire privé, nettoyé derrière lui
  (indispensable si plusieurs simulations tournent en parallèle) ;
* `preparer(D, noeuds, heures, …)` — construit le modèle d'une tranche. Arguments utiles :
  * `niveau0` — niveau initial de T1,
  * `statut_pompe` — statut initial reporté de la tranche précédente,
  * `pompe` — **profil d'état mesuré** : retire tous les contrôles du fichier et pilote la pompe
    par un profil de vitesse (vitesse nulle = fermée). C'est le levier 1,
  * `rugosite` (facteur **multiplicatif**) et `coefficients` (valeur **absolue**) — mutuellement
    exclusifs, une `ValueError` si les deux sont passés. La référence travaille en absolu ; le
    balayage à un paramètre travaille en facteur,
  * `diametre_T1` — pour le levier 3 ;
* `simuler(...)` — la boucle de tranches, décrite en §3 ;
* `pressions_simulees(...)` — le raccourci (T, 33) en float32.

### 4.2 `profils.py` — les deux équations (178 lignes)

* `tendance(x, fenetre=2016)` — `T(t)`, moyenne glissante centrée, bords tenus. Une fenêtre
  d'exactement une semaine annule le cycle hebdomadaire ; ce qui reste est la dérive lente.
  **Limite documentée dans la docstring** : le filtre atténue ce qu'il estime d'un facteur
  `sinc(fenêtre/période)` — 0,1 % pour une saisonnalité annuelle, déjà 2,5 % pour une variation
  de deux mois. La tendance restituée est donc systématiquement un peu plate.
* `saisonnalite(x, creneaux, n, statistique="moyenne")` — `S`, profil de 2016 valeurs renormalisé
  à 1. **Le défaut est la moyenne**, qui reproduit la référence. `"mediane"` est une variante
  robuste proposée en option : chaque créneau ne dispose que de 52 observations dans l'année, si
  bien qu'une poignée de journées atypiques suffit à déplacer la moyenne. *(Voir §9 : plusieurs
  endroits du texte annoncent encore la médiane.)*
* `decomposer(serie, creneaux, …)` — équation (1) sur une série. Renvoie `moyenne`, `tendance`,
  `saison`, `residu`, et `lisse` = `d̄·T·S`, la seule partie que la méthode conserve. Un compteur
  muet (moyenne ≤ 0) renvoie des tableaux neutres au lieu de diviser par zéro.
* `decomposer_amr(annee)` — applique (1) aux 82 compteurs (~15 s).
* `formes_par_categorie(wn, annee, decomposition)` — équation (2) **à l'envers**. Avec `W` la
  matrice 82 × 3 des bases nominales des nœuds équipés et `A` les séries lissées, on résout
  `W·m(t) = A(t)` aux moindres carrés à chaque pas. La pseudo-inverse est calculée une fois et
  appliquée par un seul produit matriciel : l'année entière coûte moins d'une seconde. Les formes
  sont écrêtées à zéro puis renormalisées à une moyenne de 1.
* `demandes_calibrees(wn, annee, formes, n_pas, dtype=np.float32)` — équation (2) dans le bon
  sens, (T × 782) en m³/h. **La plus grosse matrice du dépôt** : 330 Mo en float32, le double en
  float64. Le produit est fait directement dans le type demandé, sans intermédiaire en double
  précision ; `n_pas` permet de n'en calculer qu'un début.
* `demandes_nominales(wn, index)` — le point de départ : bases × profils du fichier. Ces profils
  sont **strictement hebdomadaires** et ne portent aucune saisonnalité annuelle. À noter : le
  profil `P-Industrial` ne contient **qu'un seul point**, il est donc constant — d'où le
  `m[pas % len(m)]` qui sert aussi bien un profil de 2016 points qu'un profil de 1.

### 4.3 `rugosite.py` — l'ajustement (231 lignes)

La docstring de module reproduit le critère publié en entier :

```
min_{x_L ≤ x ≤ x_U}  ½ Σ_j Σ_i H_κ( ([S·y(t_j,x)]_i − z_i^j) / σ_ij )  +  α‖x − x_0‖²
```

* `facteurs_par_conduite` / `coefficients_par_conduite` — traduisent `{groupe: valeur}` en
  `{conduite: valeur}`, respectivement pour un facteur et pour un coefficient absolu ;
* `coefficients_initiaux(wn, groupes)` — coefficient de départ de chaque groupe : la moyenne de
  ses conduites **pondérée par longueur**. Nécessaire parce que les groupes par diamètre mélangent
  parfois deux coefficients d'origine (sur les 705 conduites de 100 mm, 104 sont à 120 et 601 à
  140). Résultat : `D100` 137,1, `D150` 137,7, `D160` 140,0, `D200` 140,0, `D225` 140,0,
  `D<=75` 135,3 ;
* `residus(simule, mesure, sous_ech=12, sigma=None)` — vecteur aplati, **sous-échantillonné à
  l'heure**. Chaque évaluation du critère coûte une simulation complète, et une fenêtre de deux
  jours donne déjà 48 × 33 = 1 584 résidus pour six paramètres ;
* `ajuster(simulateur, mesure, depart, bornes=(60,160), huber=1.345, tikhonov=0.0, sigma=None,
  sous_ech=12, max_nfev=25)` — le cœur. `simulateur` est une **fermeture** qui prend
  `{groupe: coefficient}` et rend les pressions aux capteurs : toute la configuration hydraulique
  y est enfermée, ce qui permet d'employer la même routine avant et après un changement de
  conditions aux limites. Renvoie `coefficients`, `depart`, `rmse_depart`, `rmse`, `aux_bornes`,
  `n_evaluations`, `secondes`, `trace`.
  * **`rmse` porte sur les seuls résidus de mesure**, pas sur le critère optimisé — sinon la
    régularisation la gonflerait artificiellement ;
  * **`aux_bornes`** nomme les groupes arrêtés sur une contrainte. S'il n'est pas vide, les
    données tiraient plus loin que ce que la physique autorise : c'est une information, pas un
    détail d'exécution ;
  * le solveur est `least_squares(method="trf")` — une **région de confiance réfléchissante** et
    non Levenberg-Marquardt au sens strict, parce que ce dernier n'accepte pas de bornes. *(Voir
    §9 : trois endroits du texte disent encore « Levenberg-Marquardt ».)* ;
  * **`tikhonov` vaut 0 par défaut, donc la régularisation est désactivée**, et `sigma=None`
    désactive la pondération par capteur. Aucun des deux carnets ne les active. *(Voir §9.)*
* `balayage(...)` — balaye **un seul** facteur appliqué à tous les groupes. C'est le contrôle de
  l'ajustement à six paramètres : si la courbe à un paramètre atteint le même minimum, les cinq
  autres n'ont rien apporté ;
* `pertes_de_charge(...)` — le diagnostic d'identifiabilité. **La perte est calculée comme la
  différence de charge entre les deux extrémités**, et non lue dans la colonne `headloss` du
  simulateur : celle-ci est normalisée par la longueur pour les conduites, et les confondre fausse
  le diagnostic d'un facteur propre à chaque conduite ;
* `concentration(pertes)` — quelle part de la friction totale est portée par les n % de conduites
  les plus chargées. C'est cette lecture qui décide du nombre de groupes utiles.

### 4.4 `ameliorations.py` — les quatre leviers (245 lignes)

* `statut_pompe(annee, seuil=1.0)` — 1,0 en marche, 0,0 à l'arrêt, lu au débitmètre. La pompe
  débite ~44 m³/h ou zéro : la séparation est franche, le seuil n'a pas besoin d'être fin ;
* `transitions(statut)` — indices des basculements ;
* `demande_ab_mesuree(annee)` — `p227 + p235 − PUMP_1`. La consommation totale de A+B, **fuites
  comprises** ;
* `demande_c_estimee(wn, annee)` — extrapolation des 82 compteurs aux 92 jonctions de la zone C,
  au prorata des demandes nominales ;
* `demi_cycles(wn, annee, min_pas=12, min_delta=0.05)` — découpe l'année en intervalles où la
  pompe ne change pas d'état et fait le bilan de chacun : `(Δh, V_pompe, V_zoneC)`. Les
  demi-cycles trop courts ou de trop faible amplitude sont écartés, leur Δh étant dominé par
  l'arrondi du capteur de niveau ;
* `section_par_demi_cycles(wn, annee)` — **première passe, celle qui échoue**. Régresse
  `A·Δh = V_net` dans les deux sens, ce qui **encadre** l'estimateur, puis laisse la consommation
  de la zone C libre d'un facteur d'échelle. La clé à regarder est `correlation_remplissage` ;
* `ajuster_reservoir(D, noeuds, wn, annee, …)` — **deuxième passe, celle qui marche**. Simule la
  trajectoire complète du niveau **sans jamais la réancrer** — c'est l'accumulation libre de
  l'écart qui porte l'information — et balaye (diamètre, échelle zone C). L'asymétrie exploitée :
  pompe à l'arrêt la pente du niveau vaut `−Q_C/A`, un simple rapport ; pompe en marche elle vaut
  `(Q_pompe − Q_C)/A`, et `Q_pompe` est **mesuré**. **Vérifier la colonne `sature`** avant de
  croire le résultat ;
* `recaler_zone_ab(D, noeuds, wn, annee, periode=2016)` — levier 4. Ne touche que le **niveau**
  d'ensemble, par blocs : la répartition entre nœuds et la forme temporelle restent celles du
  modèle de demande. Renvoie une copie.

**La réserve du levier 4**, écrite dans la docstring de module et à reprendre partout où le
niveau de demande est discuté : le débit d'entrée contient tout ce qui sort du réseau, **fuites
comprises**. Recaler la demande sur le bilan de masse absorbe donc les fuites dans la demande.
C'est sans conséquence pour calibrer un modèle hydraulique, et rédhibitoire si l'on compte
ensuite chercher des fuites dans le résidu : dans ce second cas, il faut ajuster le recalage sur
une période saine et le transporter.

### 4.5 `diagnostics.py` — la mesure (101 lignes)

* `par_capteur(simule, mesure)` — biais, dispersion, RMSE, écart maximal, capteur par capteur ;
* `resume(simule, mesure)` — les mêmes, agrégés ;
* `tableau(variantes, mesure, reference)` — une ligne par variante, avec les gains relatifs sur la
  dispersion et sur la RMSE. `reference` est la première variante par défaut, d'où l'importance
  de l'**ordre** du dictionnaire passé ;
* `erreur_de_pas(simule, mesure)` — écart sur la variation d'un pas au suivant, la statistique
  que le biais ne touche pas ;
* `tracer_capteur`, `tracer_par_capteur` — les deux seules fonctions du dépôt qui dessinent.

La dispersion est calculée comme `√(RMSE² − biais²)`, avec un `max(…, 0)` de sécurité contre les
arrondis.

---

## 5. Carnet 1, cellule par cellule

`notebooks/01_calibration_under_pressure.ipynb` — 30 cellules, 7 figures. Fenêtre : juillet,
7 jours. Versionné **avec ses sorties** : il se lit sans être exécuté.

| # | type | contenu | sortie |
|---|---|---|---|
| 0 | md | titre, les trois temps de la méthode, pré-requis données | — |
| 1 | code | imports, `ANNEE`/`DEBUT`/`N_PAS`/`N_ENTR`, `verifier_donnees` | `fenêtre : 2016 pas … à partir du pas 52128` |
| 2 | md | §1 « Le réseau et les mesures » | — |
| 3 | code | charge le modèle, les capteurs, les zones, les bases, les compteurs | 782 jonctions / 905 conduites / 43,2 km ; 33+3+1 capteurs, 82 compteurs ; 92 C / 690 A+B ; bases 106,09 / 66,55 / 3,94 |
| 4 | code | où sont les compteurs par rapport à la demande | zone C = **11,1 %** de la demande, nœuds équipés **10,0 %**, **0** compteur hors zone C |
| 5 | md | la conséquence : le modèle **extrapole**, il n'interpole pas. Rappelle les deux entrées, la pompe, les trois réducteurs | — |
| 6 | md | §2 « Le point de départ » | — |
| 7 | code | simule le modèle livré sur la fenêtre | biais **0,3412** / disp **0,3290** / RMSE **0,4740** / max 2,2851 |
| 8 | code | figure : mesure contre simulation sur `n54` et `n1`, 3 jours | figure |
| 9 | md | pourquoi le modèle livré ne peut pas suivre l'année | — |
| 10 | code | figure : les profils du fichier ; consommation A+B mesurée par mois | amplitude **136,8 à 201,1 m³/h**, soit **±18 %** ; profils du fichier 0 % ; profil industriel constant |
| 11 | md | §3, équation (1) posée terme à terme | — |
| 12 | code | décompose **un** compteur (le plus chargé) ; figure 2×2 | `d̄ = 1,835 m³/h`, variance portée par `d̄·T·S` : **11,1 %** |
| 13 | md | pourquoi 11 % n'est pas un échec : les résidus s'annulent en sommant | — |
| 14 | code | applique (1) aux 82 compteurs (~15 s) | par compteur **42,1 %**, sur la somme **71,1 %** |
| 15 | code | figure : tendance moyenne et profil hebdomadaire moyen | figure |
| 16 | md | §4, équation (2) : c'est un **mélange**, pas un produit | — |
| 17 | code | inverse (2) sur les 82 nœuds mesurés ; figures semaine + mois | rapport max/min mensuel : Residential **1,30**, Commercial **1,32**, Industrial **1,36** |
| 18 | md | la forme résidentielle porte la saisonnalité ; **réserve** : estimé sur une zone, appliqué à une autre | — |
| 19 | code | applique (2) aux 782 nœuds ; contrôle contre le bilan de masse | figure |
| 20 | md | la forme est retrouvée, **le niveau non** — renvoyé au carnet 2 | — |
| 21 | code | simule avec le modèle de demande | biais **0,2022** / disp **0,2526** / RMSE **0,3235** ; gain dispersion **−23,2 %** |
| 22 | code | figure : dispersion par capteur, deux variantes | figure |
| 23 | md | §6, la résistance de Hazen-Williams, les six groupes, le critère complet | — |
| 24 | code | forme les six groupes | `D100` 705, `D150` 103, `D200` 64, `D160` 16, `D225` 12, `D<=75` 5 |
| 25 | code | **l'ajustement** sur `N_ENTR` = 2 jours | **67** simulations en **64 s** ; RMSE d'ajustement 0,3342 → **0,2368** ; coefficients 88,5 / 156,5 / 160,0 / 127,3 / 160,0 / 160,0 ; **aux bornes : `D225`, `D<=75`** |
| 26 | code | évalue les coefficients sur la fenêtre de 7 jours | biais **−0,0067** / disp **0,2630** / RMSE **0,2631** |
| 27 | md | §7 « Bilan » | — |
| 28 | code | reprend les trois variantes sur **3 jours** de janvier et de juillet | tableau croisé (voir §8) |
| 29 | md | la lecture en trois constats | — |

**Le point à ne pas perdre en réécrivant la cellule 26 puis la 29** : la dispersion passe de
0,2526 à **0,2630**, elle se **dégrade** légèrement. Seul le biais s'effondre (0,2022 → −0,0067).
C'est exactement le constat n° 3 de la cellule 29, et c'est la thèse du carnet 2.

**Coût réel de la cellule 25** : `max_nfev=12` compte les évaluations *principales*. Chaque
colonne du jacobien en demande une de plus, d'où les 67 simulations effectives. Le commentaire
dans la cellule le dit ; ne le supprimez pas, c'est la question que tout lecteur se pose.

---

## 6. Carnet 2, cellule par cellule

`notebooks/02_ameliorations.ipynb` — 36 cellules, 5 figures. Même fenêtre.

| # | type | contenu | sortie |
|---|---|---|---|
| 0 | md | titre, le tableau des quatre leviers, l'annonce du verdict | — |
| 1 | code | imports et préparation ; recalcule `formes` et `D_calibre` (~15 s) | `prêt` |
| 2 | md | §1, les deux contrôles du fichier cités tels quels | — |
| 3 | code | compare l'état simulé à l'état mesuré | marche **65,0 %** mesuré contre **49,5 %** simulé ; **14** basculements de part et d'autre ; **22,5 %** des pas en désaccord |
| 4 | code | figure : les deux états superposés sur 3 jours | figure |
| 5 | code | ce que vaut la pompe **en mètres**, dans les mesures seules | entrant **217** contre **178 m³/h** (**+22 %**) ; la pompe prélève **44 m³/h** = **20 %** ; pression moyenne 46,099 contre 46,452 m (**−0,353 m**) ; `n54` **−0,913**, `n410` −0,781, `n429` −0,772, `n342` −0,582 |
| 6 | md | le mécanisme, mesuré et non supposé ; puis §2, le réservoir qui dérive | — |
| 7 | code | trois trajectoires de niveau : libre, ancrée 24 h, ancrée 6 h | mesuré 2,39–3,90 m ; **sans ancrage** saturé **20 %**, écart **0,49 m** ; **24 h** saturé 9 %, écart 0,37 m ; **6 h** saturé **2 %**, écart **0,11 m** |
| 8 | md | la saturation, et pourquoi les leviers 1 et 2 sont indissociables ; puis §3 | — |
| 9 | code | le **tableau croisé 2 × 2**, deux saisons, sur **3 jours** | janvier 0,2162 → 0,1623 → 0,1229 → **0,0697** ; juillet 0,3090 → 0,2285 → 0,2289 → **0,1802** (dispersions) |
| 10 | md | **la lecture qui compte** : les gains se composent sans s'additionner | — |
| 11 | code | §4, la section par bilan volumique ; figure du nuage | modèle **201,1 m²** (16,00 m) ; erreur sur volumes **207,9** (16,27 m) ; erreur sur niveaux **414,0** (**22,96 m**) |
| 12 | md | les deux estimateurs sont **incompatibles** — du simple au double | — |
| 13 | code | l'explication en un nombre | corrélation **0,997** ; à échelle libre : section 228,4 m², facteur **×1,72** |
| 14 | md | l'aliasing, puis comment le **contourner** : l'asymétrie des deux régimes | — |
| 15 | code | `ajuster_reservoir` sur **janvier** | meilleur **16,0 m / ×1,0** à **0,2020 m**, saturé 0,5 % ; à ×1,4 l'erreur est **treize fois pire** et le réservoir sature la moitié du temps |
| 16 | md | l'échelle se fixe **sans ambiguïté à 1** ; le diamètre a un minimum intérieur mais le critère est plat | — |
| 17 | code | le même balayage en **juillet** et en **octobre** | juillet 14 m 0,457 → 17 m 0,592, **saturé 20 %** ; octobre 14 m 0,419 → 17 m 0,563, **saturé 14 %** — **monotones, sans minimum intérieur** |
| 18 | md | l'artefact de saturation, et la leçon générale : vérifier que l'optimum est **intérieur** *et* que le régime est celui qu'on croit | — |
| 19 | code | est-ce que la section change quelque chose, une fois le niveau ancré ? | section du modèle 0,1888 contre section estimée 0,1903 (dispersion) — **rien** |
| 20 | md | levier 3 **à écarter, mais pas pour la raison qu'on croyait** ; année : 0,1992 contre 0,2000 m | — |
| 21 | code | §5, le recalage sur le bilan de masse | biais 0,1832 → **0,1515** ; dispersion 0,1888 → **0,1882** |
| 22 | md | biais seulement — et la **réserve sur les fuites** | — |
| 23 | code | §6, le **balayage à un seul facteur** | minimum de RMSE à **×0,95** (0,1903 m), où la dispersion vaut 0,1752 contre 0,1784 à ×1,00 |
| 24 | code | figure : RMSE, dispersion et \|biais\| contre le facteur | figure |
| 25 | md | **la figure qui résume le problème** : le minimum de RMSE tombe là où le biais change de signe | — |
| 26 | code | les pertes de charge réelles | médiane **4,11 mm** ; 9 conduites sur 10 sous **44,37 mm** ; maximum **456 mm** ; 10 % des conduites portent **56,4 %** de la friction ; tête de classement `p110`, `p105`, `p235`, `p479`, `p480` |
| 27 | md | le réseau est presque sans friction, et la friction qui existe est concentrée | — |
| 28 | code | l'ajustement à six coefficients, **dans les bonnes conditions aux limites** | **73** simulations en **83 s** ; 0,2376 → **0,1412** ; 72,0 / 160,0 / 160,0 / 150,1 / 160,0 / 160,0 ; **quatre groupes aux bornes** |
| 29 | md | le seul test qui tranche : **est-ce que ça transfère ?** | — |
| 30 | code | les trois jeux sur la fenêtre d'ajustement et sur deux périodes jamais vues | six coefficients : **−13,2 %** de dispersion en juillet, **−3,0 %** en octobre, **+28,2 %** en janvier |
| 31 | md | la signature du sur-ajustement ; l'encadré sur les garde-fous ; les trois règles pratiques | — |
| 32 | md | §7 « Bilan » | — |
| 33 | code | le tableau cumulé sur la fenêtre de 7 jours | 0,4740 → 0,3235 → 0,2631 → 0,2417 → 0,2023 (RMSE) |
| 34 | code | figure : `n54` avant et après, 2 jours | figure |
| 35 | md | les quatre choses à retenir, le tableau annuel, et « ce qui resterait à essayer » | — |

**Les deux cellules qui portent le carnet** sont la 10 et la 31. La 10 établit que mesurer les
améliorations une par une, dans un modèle où un autre facteur est encore faux, **sous-estime leur
valeur commune**. La 31 établit qu'un paramètre sans information se met au service de l'erreur
qui domine la fenêtre d'ajustement. Tout le reste est de la mesure.

**Le chiffre le plus fragile du carnet** est le « ×1,72 » de la cellule 13 : c'est un artefact, et
le texte de la cellule 14 le dit. Si vous réécrivez la 13, ne la laissez pas ressembler à un
résultat.

---

## 7. Les tests

`pytest -q` — **15 tests**, dont 5 purement synthétiques (qui tournent sans les données) et 10
marqués `@donnees` (sautés si `data/raw/` est vide). Ce sont eux qui disent si un remaniement a
cassé quelque chose.

**Synthétiques :**

| test | ce qu'il protège |
|---|---|
| `test_decomposition_retrouve_un_produit_exact` | l'équation (1) retrouve un produit fabriqué. **Tolérance 1e-2 et non 1e-3** : l'écart résiduel est l'atténuation en `sinc` de la moyenne glissante, expliquée en commentaire, et un contrôle relatif vérifie qu'on fait mieux que la moyenne seule |
| `test_la_saisonnalite_ignore_les_journees_atypiques` | la médiane et la moyenne divergent là où il y a quelque chose à voir |
| `test_rmse_se_decompose_en_biais_et_dispersion` | l'identité `RMSE² = biais² + dispersion²` |
| `test_creneau_hebdomadaire_tient_compte_du_jour_de_la_semaine` | le piège du `% 2016` |
| `test_transitions_de_pompe` | la détection des basculements |

**Sur données :** partition en zones (92/690), les **trois** lignes de demande par jonction, les
groupes de rugosité, les unités des compteurs (L/h contre m³/h), le report d'état entre tranches,
**la série de demande sans pas supplémentaire** (non-régression du rebouclage silencieux
d'EPANET), les formes par catégorie, le recalage qui ne touche que la zone A+B, le fait que
`ajuster_reservoir` sépare bien la section de la demande, et la section par demi-cycles.

Si vous ne deviez en relire qu'un avant de modifier `reseau.simuler`, c'est
`test_une_serie_de_demande_sans_pas_supplementaire` : le bug qu'il protège ne produit aucune
erreur, seulement des résultats faux.

---

## 8. Où vit chaque chiffre publié

Deux fenêtres circulent dans le dépôt, et c'est la source d'erreur la plus probable si vous
recopiez un chiffre d'une section à l'autre.

**Fenêtre de 7 jours** (`N_PAS`) — carnet 1 cellules 7/21/26, carnet 2 cellules 19/21/30/33 :

| variante | biais | dispersion | RMSE |
|---|---|---|---|
| modèle livré | 0,3412 | 0,3290 | 0,4740 |
| + modèle de demande | 0,2022 | 0,2526 | 0,3235 |
| + rugosités (carnet 1, commandes du fichier) | −0,0067 | 0,2630 | 0,2631 |
| + pompe mesurée et niveau ancré | 0,1832 | 0,1888 | 0,2631 |
| + bilan de masse | 0,1515 | 0,1882 | 0,2417 |
| + rugosité globale ×0,95 | 0,0589 | 0,1936 | 0,2023 |
| + six coefficients ajustés | −0,0019 | 0,1634 | 0,1634 |

**Fenêtre de 3 jours** — carnet 1 cellule 28, carnet 2 cellule 9. Le modèle livré y vaut
**0,4535 m** en juillet et **0,2591 m** en janvier. Ce ne sont pas les mêmes nombres que
ci-dessus, et c'est normal.

> ⚠️ Deux lignes portent la même RMSE de **0,2631 m** par pure coïncidence : « rugosités du
> carnet 1 » et « pompe mesurée et niveau ancré ». Leurs biais et dispersions n'ont rien à voir
> (−0,0067 / 0,2630 contre 0,1832 / 0,1888). Ne les confondez pas en réécrivant.

**Année complète** — pas dans les carnets, reproduit par le bloc de code du `README` (une
vingtaine de minutes) :

```python
from calibration import reseau as R, profils as P, ameliorations as A, diagnostics as G
wn = R.charger_modele(24); noeuds = wn.junction_name_list
niv, pompe = R.charger_niveau(2018).to_numpy(), A.statut_pompe(2018)
D = A.recaler_zone_ab(P.demandes_calibrees(wn, 2018)[0], noeuds, wn, 2018)
S = R.pressions_simulees(D, noeuds, R.PAS_AN, tranche_jours=0.25,
                         pompe=pompe, niveau_mesure=niv, niveau0=float(niv[0]))
print(G.resume(S, R.charger_pressions(2018).to_numpy()))
```

**Comparaison au chiffre publié** — `README`, section dédiée. Protocole de la référence : six
groupes ajustés sur la **première semaine de 2018**, évalués sur cette même semaine, 33 capteurs.
Résultat : **0,062 m** contre **0,060 m** publiés. Deux observations qui comptent plus que la
coïncidence : la rugosité seule plafonne à **0,192 m** (les 6 cm ne s'atteignent qu'une fois les
conditions aux limites prises dans les capteurs), et **trois des six groupes s'arrêtent sur une
borne**.

**Si vous relancez les carnets**, les chiffres peuvent bouger de quelques millimètres selon la
version d'EPANET et le chemin de l'optimiseur. Les carnets sont versionnés avec leurs sorties :
après toute modification de code, il faut les **réexécuter** et **relire les affirmations du
texte**, qui sont chiffrées presque partout.

---

## 9. Incohérences connues entre le texte et le code

Relevées en relisant le dépôt ligne à ligne. Aucune n'affecte les résultats — ce sont des
affirmations du texte que le code ne soutient pas. **À traiter en priorité si vous reformulez.**

### 9.1 La régularisation est annoncée comme active ; elle ne l'est pas

`rugosite.ajuster` a `tikhonov=0.0` par défaut, ce qui **désactive** le terme `α‖x − x₀‖²`, et
`sigma=None`, ce qui désactive la pondération par capteur. **Aucun des deux carnets ne les
active** : les deux appellent `U.ajuster(simulateur, mesure_entr, depart, max_nfev=12,
verbeux=False)`.

Or le carnet 1, cellule 23, écrit : « *Les trois garde-fous sont actifs par défaut dans
`rugosite.ajuster`.* » Et la docstring de `rugosite.py` : « *`ajuster` les reproduit tous les
trois.* »

Ce qui tourne réellement : les **bornes** (60–160) et la **perte de Huber** (`f_scale=1.345`).
Sur quatre ingrédients du critère publié, deux sont actifs.

Deux issues possibles, au choix :

* **corriger le texte** — dire que `ajuster` *implémente* les trois mais que la régularisation
  est laissée à 0 dans les carnets, les bornes suffisant à contenir l'estimateur ;
* **corriger le code** — passer un `tikhonov` non nul dans les deux carnets, et réexécuter. Cela
  changera les coefficients, donc les tableaux de transfert, donc le texte du §6.

La première est honnête et coûte cinq minutes ; la seconde est plus fidèle à la référence et
coûte une réexécution complète.

### 9.2 « Médiane » contre « moyenne »

`profils.saisonnalite` a pour défaut `statistique="moyenne"` — c'est délibéré, c'est ce que fait
la référence, et la docstring de la fonction le dit correctement. Mais trois endroits annoncent
encore la médiane :

| fichier | ligne | texte |
|---|---|---|
| `README.md` | 56 | « `S(t)` profil hebdomadaire obtenu par **médianes de créneau** » |
| `calibration/profils.py` | 11 | « la **saisonnalité hebdomadaire** (médianes périodiques…) » |
| `notebooks/01…ipynb` | cellule 11 | « obtenue par **médiane** de créneau […] La médiane, et non la moyenne : … » |
| `notebooks/01…ipynb` | cellule 12 | titre de figure « S — profil hebdomadaire (médianes de créneau) » |

La cellule 11 du carnet 1 argumente même explicitement *en faveur* de la médiane, alors que la
figure juste en dessous est produite à la moyenne. Correction : remplacer par « moyennes de
créneau », et garder l'argument sur la médiane comme **variante proposée** — c'est bien ce que
dit la docstring de `saisonnalite`. Le titre de figure de la cellule 12 est du code : le changer
suppose de réexécuter la cellule (ou de corriger le seul titre à la main).

### 9.3 « Levenberg-Marquardt »

Le solveur employé est `least_squares(method="trf")`, une région de confiance réfléchissante, et
la docstring de `rugosite.py` explique pourquoi : Levenberg-Marquardt n'accepte pas de bornes,
alors que le critère en pose. Trois endroits disent malgré tout « Levenberg-Marquardt » :
`README.md` lignes 18 et 178, `calibration/__init__.py` ligne 7, et le titre du §6 du carnet 1
(« *Les rugosités : six groupes et Levenberg-Marquardt* »).

Formulation de remplacement : « moindres carrés non linéaires sous contraintes de boîte ».

### 9.4 Le bilan du carnet 2 contredit son propre §4

Cellule 35, point 4 : « *Deux paramètres réputés calibrables ne le sont pas ici : la section du
réservoir, **aliasée** avec la consommation non comptée de la zone qu'elle alimente…* »

Mais tout le §4 démontre l'inverse. La cellule 16 conclut « **l'aliasing est levé** », et la
cellule 20 insiste : « *à écarter, **mais pas pour la raison qu'on croyait**. La section n'est pas
« non identifiable » : elle l'est […] Elle est simplement **sans effet** une fois le niveau
réancré.* »

Le point 4 du bilan raconte donc la conclusion de la *première* passe, celle que le carnet a
explicitement dépassée. C'est la seule contradiction franche du dépôt. Correction : reformuler en
« la section du réservoir, identifiable mais sans effet une fois le niveau réancré ».

### 9.5 Le tableau d'ouverture du carnet 2 annonce la mauvaise méthode

Cellule 0, ligne du levier 3 : mesure utilisée = « bilan volumique par demi-cycle ». C'est la
méthode qui **échoue**. Celle qui aboutit est la simulation de la trajectoire de niveau.

Même cellule, deux lignes plus bas : « *Deux d'entre eux transforment le résultat. Un troisième
corrige un biais […]. Le quatrième ne donne rien* ». C'est exact, mais « le troisième » désigne le
levier **4** et « le quatrième » le levier **3** — l'ordre du tableau et l'ordre de la phrase ne
coïncident pas. À désambiguïser.

### 9.6 « Ce qui resterait à essayer » ne cite pas le plus gros levier

La fin du carnet 2 établit que **le pas de réancrage est le plus gros levier restant** — passer de
24 h à 6 h retire encore un quart de la RMSE sur l'année, pour un coût qui n'est que du temps de
calcul. Mais la section « Ce qui resterait à essayer », juste en dessous, propose la répartition
spatiale de la demande et les réducteurs de pression, sans mentionner le réancrage.

La piste des réducteurs mérite en outre d'être rétrogradée : elle est justifiée par le fait
qu'« *une consigne fausse de quelques centimètres déplacerait en bloc toute une sous-zone — ce qui
ressemble beaucoup à ce que fait le biais résiduel* ». Or après ajustement des rugosités, le biais
annuel n'est plus que de **+0,018 m** : il n'y a presque plus de biais à expliquer. L'erreur
restante est **dynamique**, et les réducteurs sont un candidat pour le biais.

Ordre défendable : (1) descendre le pas de réancrage sous 6 h ; (2) la répartition spatiale de la
demande sur les 690 nœuds, que le bilan de masse ne contraint pas ; (3) les réducteurs.

### 9.7 Ce que le chiffre annuel contient et que le texte ne dit pas

Ni le `README` ni le carnet 2 ne signalent que **la première semaine de 2018 ne porte aucune
fuite**, alors que sur le reste de l'année le débit de fuite moyen représente une part
appréciable de la consommation de la zone A+B.

Deux conséquences, à ajouter en une phrase chacune :

* la comparaison **0,062 contre 0,060 m** est faite sur la semaine la plus propre de l'année.
  C'est le protocole de la référence, la comparaison est donc légitime — mais elle ne dit rien
  des 51 autres semaines ;
* sur l'année, `recaler_zone_ab` cale la demande sur un débit d'entrée qui **contient les
  fuites** : leur volume est réparti sur 690 nœuds au lieu de sortir en un point. Cette
  mauvaise localisation est **spatiale**, donc elle se retrouve dans la dispersion. Le **0,158 m**
  annuel est par conséquent un **majorant** de l'erreur de modèle, et non une mesure de
  celle-ci — et chercher à le faire baisser davantage reviendrait à absorber le signal de fuite.

La réserve est déjà écrite dans la docstring de `ameliorations.py` ; elle manque simplement là où
les chiffres annuels sont présentés.

---

### 9.8 Le test de report d'état est décrit avec les mauvaises durées

`reseau.simuler` annonce dans sa docstring : « *deux tranches de 24 h donnent les mêmes pressions
que 48 h d'affilée à 1e-3 m près* ». Le test qui le vérifie
(`test_report_detat_entre_tranches`) compare en réalité **deux tranches de 12 h à 24 h
d'affilée** — c'est-à-dire `tranche_jours=0.5` contre `tranche_jours=1` sur `PAS_JOUR` pas. Le
`README` (« deux tranches de 12 h contre 24 h d'affilée ») est exact ; c'est la docstring qui
dérive.

---

## 10. Si vous reformulez

**Ce qui est négociable** — le ton, le découpage en sections, les titres, les métaphores, l'ordre
des arguments, la longueur. Les textes actuels sont denses et assez péremptoires ; rien
n'interdit de les aérer.

**Ce qui ne l'est pas, sauf à réexécuter :**

* tout nombre cité dans un texte en markdown. Ils viennent tous d'une sortie de cellule, et il y
  en a beaucoup ;
* les comparaisons **biais contre dispersion**. Elles ne sont pas un ornement de vocabulaire :
  c'est ce qui distingue « la RMSE s'améliore » de « le modèle s'améliore ». Si une phrase dit
  simplement « c'est mieux », elle a perdu l'essentiel ;
* la distinction **produit** (équation 1, effets temporels d'un même compteur) contre **mélange**
  (équation 2, somme sur les types de consommateurs). C'est le contresens le plus facile à
  introduire en reformulant ;
* les deux écarts assumés par rapport à la référence, listés dans la docstring de `profils.py` :
  les trois types de consommateurs du fichier au lieu d'une classification automatique, et
  l'option moyenne/médiane. Les retirer transformerait une reproduction honnête en une
  reproduction annoncée comme exacte ;
* les **résultats négatifs** — la section du réservoir sans effet, les rugosités qui n'agissent
  que sur le biais, les groupes arrêtés sur une borne. Ce sont eux qui font la valeur du carnet 2 ;
  les adoucir le viderait.

**Méthode pratique.** Pour une correction purement rédactionnelle dans un carnet, on peut éditer
le champ `source` de la cellule markdown dans le `.ipynb` sans rien réexécuter : une cellule
markdown n'a pas de sortie. Dès qu'une cellule **de code** change — y compris un simple titre de
figure — il faut réexécuter le carnet et vérifier que les affirmations du texte tiennent toujours.

---

## 11. Limites de fond

À distinguer des incohérences du §9 : celles-ci sont des propriétés du travail, pas des défauts
de rédaction.

1. **Le modèle de demande extrapole d'une zone à l'autre.** Les 82 compteurs sont tous en zone C,
   qui porte 11 % de la demande. Rien dans les données ne garantit que les 690 nœuds de A+B aient
   les mêmes habitudes de consommation.
2. **Les groupes de rugosité sont formés par diamètre, faute de mieux.** La référence les forme
   par matériau et âge — une information que le fichier de réseau ne porte pas. C'est une limite
   de la reproduction, pas un choix, et elle explique en partie pourquoi trois groupes s'arrêtent
   sur une borne.
3. **Les six coefficients sont réglés sur une seule fenêtre.** Sur sept jours, le transfert donne
   −13 %, −3 % et +28 % de dispersion selon la saison. Sur l'année, ces écarts se compensent
   (0,1578 → 0,1573) : c'était du bruit d'ajustement. Le conclure demandait d'évaluer sur l'année.
4. **La tendance est estimée par un filtre qui l'atténue** d'un facteur `sinc(fenêtre/période)`.
   Négligeable à l'échelle annuelle, 2,5 % à l'échelle de deux mois.
5. **Le recalage sur le bilan de masse absorbe les fuites dans la demande** (§9.7).
6. **Tout est mesuré sur une seule année et un seul réseau.** La seconde année disponible n'a
   servi à rien ici.

---

## 12. Formules et lexique

**Équation (1), un compteur :**
```
d(t) = d̄ · T(t) · S(t) · R(t)
```
`d̄` moyenne annuelle ; `T` tendance (moyenne glissante d'une semaine, porte la saisonnalité
annuelle) ; `S` profil hebdomadaire renormalisé à 1 ; `R` résidu, **jeté**.

**Équation (2), un nœud sans mesure :**
```
d̂ᵢ(t) = Σⱼ d̄ᵢⱼ · Tⱼ(t) · Sⱼ(t)
```
Somme sur les types de consommateurs `j`. `d̄ᵢⱼ` est lu dans le fichier de réseau ; les formes
`Tⱼ·Sⱼ` sont obtenues en résolvant cette équation à l'envers sur les 82 nœuds mesurés.

**Résistance de Hazen-Williams :**
```
R = 10,674 · L / (C^1,852 · D^4,871)
```

**Critère d'ajustement des rugosités :**
```
min_{x_L ≤ x ≤ x_U}  ½ Σⱼ Σᵢ H_κ( ([S·y(tⱼ,x)]ᵢ − z^j_ᵢ) / σ_ij )  +  α‖x − x₀‖²
```

**Bilan volumique du réservoir, sur un demi-cycle :**
```
A · Δh = ∫ (Q_pompe − Q_zoneC) dt
```

**Décomposition de l'erreur :**
```
RMSE² = biais² + dispersion²
```

**Lexique :**

| terme | sens dans ce dépôt |
|---|---|
| biais | moyenne de l'écart `simulé − mesuré`, en m |
| dispersion | `√(RMSE² − biais²)`, la part qui survit à un recentrage |
| créneau | position dans la semaine, de 0 à 2015, **calculée par jour de la semaine** |
| tranche | bloc de simulation, de quelques heures à quelques jours |
| ancrage / réancrage | repartir, à chaque tranche, du niveau **lu au capteur** |
| saturation | part des pas où le niveau est collé à 0 ou à 4 m |
| zone C | les 92 jonctions derrière le réservoir — là où sont tous les compteurs |
| zone A+B | les 690 autres, alimentées directement, sans aucune mesure par nœud |
| aliasing | deux paramètres colinéaires que les données ne peuvent pas séparer |
| aux bornes | groupe dont le coefficient s'est arrêté sur une contrainte : un diagnostic |
