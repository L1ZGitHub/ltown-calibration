# Corrections à apporter

Huit endroits où le texte du dépôt n'énonce pas ce que le code fait. Relevés en relisant ligne à
ligne. **Aucune n'affecte les résultats** — ce sont des affirmations rédactionnelles que le code
ne soutient pas.

Pour comprendre le travail lui-même, voir [`GUIDE.md`](GUIDE.md).

---

## 1. La régularisation est annoncée comme active ; elle ne l'est pas

`rugosite.ajuster` a `tikhonov=0.0` par défaut, ce qui **désactive** le terme `α‖x − x₀‖²`, et
`sigma=None`, ce qui désactive la pondération par capteur. **Aucun des deux carnets ne les
active** : les deux appellent

```python
U.ajuster(simulateur, mesure_entr, depart, max_nfev=12, verbeux=False)
```

Or le carnet 1, cellule 23, écrit : « *Les trois garde-fous sont actifs par défaut dans
`rugosite.ajuster`.* » Et la docstring de `rugosite.py` : « *`ajuster` les reproduit tous les
trois.* »

Ce qui tourne réellement : les **bornes** (60–160) et la **perte de Huber** (`f_scale=1.345`).
Sur quatre ingrédients du critère publié, deux sont actifs.

**Deux issues possibles :**

* **corriger le texte** — dire que `ajuster` *implémente* les trois mais que la régularisation est
  laissée à 0 dans les carnets, les bornes suffisant à contenir l'estimateur. Cinq minutes ;
* **corriger le code** — passer un `tikhonov` non nul dans les deux carnets et réexécuter. Plus
  fidèle à la référence, mais cela change les coefficients, donc le tableau de transfert du §6 du
  carnet 2, donc plusieurs paragraphes. Compter une heure.

**Fichiers** : `calibration/rugosite.py` (docstring de module), `notebooks/01…ipynb` cellule 23.

---

## 2. « Médiane » contre « moyenne »

`profils.saisonnalite` a pour défaut `statistique="moyenne"`. C'est délibéré — c'est ce que fait
la référence — et la docstring de la fonction le dit correctement. Mais quatre endroits annoncent
encore la médiane :

| fichier | endroit | texte |
|---|---|---|
| `README.md` | ligne 56 | « `S(t)` profil hebdomadaire obtenu par **médianes de créneau** » |
| `calibration/profils.py` | ligne 11 | « la **saisonnalité hebdomadaire** (médianes périodiques…) » |
| `notebooks/01…ipynb` | cellule 11 | « obtenue par **médiane** de créneau […] La médiane, et non la moyenne : … » |
| `notebooks/01…ipynb` | cellule 12 | titre de figure « S — profil hebdomadaire (médianes de créneau) » |

La cellule 11 argumente même explicitement *en faveur* de la médiane, alors que la figure juste en
dessous est produite à la moyenne.

**Correction** : remplacer par « moyennes de créneau », et garder l'argument sur la médiane comme
**variante proposée** — c'est exactement ce que dit la docstring de `saisonnalite`.

⚠️ Le titre de figure de la cellule 12 est du **code** : le changer impose de réexécuter la cellule
(ou de corriger le seul titre à la main dans le `.ipynb`).

---

## 3. « Levenberg-Marquardt »

Le solveur employé est `least_squares(method="trf")`, une région de confiance réfléchissante. La
docstring de `rugosite.py` explique pourquoi : Levenberg-Marquardt n'accepte pas de bornes, alors
que le critère en pose.

Trois endroits disent malgré tout « Levenberg-Marquardt » :

* `README.md`, lignes 18 et 178 ;
* `calibration/__init__.py`, ligne 7 ;
* `notebooks/01…ipynb`, titre du §6 (« *Les rugosités : six groupes et Levenberg-Marquardt* »).

**Formulation de remplacement** : « moindres carrés non linéaires sous contraintes de boîte ».

---

## 4. Le bilan du carnet 2 contredit son propre §4

Cellule 35, point 4 :

> *Deux paramètres réputés calibrables ne le sont pas ici : la section du réservoir, **aliasée**
> avec la consommation non comptée de la zone qu'elle alimente…*

Mais tout le §4 démontre l'inverse. La cellule 16 conclut « **l'aliasing est levé** », et la
cellule 20 insiste :

> *à écarter, **mais pas pour la raison qu'on croyait**. La section n'est pas « non identifiable » :
> elle l'est […] Elle est simplement **sans effet** une fois le niveau réancré.*

Le point 4 du bilan raconte donc la conclusion de la *première* passe, celle que le carnet a
explicitement dépassée. **C'est la seule contradiction franche du dépôt**, et elle coûte une
phrase à corriger.

**Correction** : « la section du réservoir, identifiable mais sans effet une fois le niveau
réancré ».

---

## 5. Le tableau d'ouverture du carnet 2 annonce la mauvaise méthode

Cellule 0, ligne du levier 3 : mesure utilisée = « bilan volumique par demi-cycle ». C'est la
méthode qui **échoue**. Celle qui aboutit est la simulation de la trajectoire de niveau.

Même cellule, deux lignes plus bas :

> *Deux d'entre eux transforment le résultat. Un troisième corrige un biais […]. Le quatrième ne
> donne rien.*

C'est exact, mais « le troisième » désigne le levier **4** et « le quatrième » le levier **3** :
l'ordre du tableau et l'ordre de la phrase ne coïncident pas. À désambiguïser.

---

## 6. « Ce qui resterait à essayer » ne cite pas le plus gros levier

La fin du carnet 2 établit que **le pas de réancrage est le plus gros levier restant** — passer de
24 h à 6 h retire encore un quart de la RMSE annuelle, pour un coût qui n'est que du temps de
calcul. Mais la section « Ce qui resterait à essayer », juste en dessous, propose la répartition
spatiale de la demande et les réducteurs de pression, **sans mentionner le réancrage**.

La piste des réducteurs mérite en outre d'être rétrogradée. Elle est justifiée ainsi :

> *une consigne fausse de quelques centimètres déplacerait en bloc toute une sous-zone — ce qui
> ressemble beaucoup à ce que fait le biais résiduel.*

Or après ajustement des rugosités, le biais annuel n'est plus que de **+0,018 m**. Il n'y a
presque plus de biais à expliquer ; l'erreur restante est **dynamique**, et les réducteurs sont un
candidat pour le biais.

**Ordre défendable** : (1) descendre le pas de réancrage sous 6 h ; (2) la répartition spatiale de
la demande sur les 690 nœuds ; (3) les réducteurs.

---

## 7. Ce que le chiffre annuel contient, et que le texte ne dit pas

Ni le `README` ni le carnet 2 ne signalent que **la première semaine de 2018 ne porte aucune
fuite**, alors que sur le reste de l'année le débit de fuite moyen représente une part
appréciable de la consommation de la zone A+B.

Deux conséquences, une phrase chacune :

* la comparaison **0,062 contre 0,060 m** est faite sur la semaine la plus propre de l'année.
  C'est le protocole de la référence, la comparaison est donc légitime — mais elle ne dit rien des
  51 autres semaines ;
* sur l'année, `recaler_zone_ab` cale la demande sur un débit d'entrée qui **contient les fuites**.
  Leur volume se retrouve réparti sur 690 nœuds au lieu de sortir en un point ; cette mauvaise
  localisation est **spatiale**, donc elle se retrouve dans la dispersion. Le **0,158 m** annuel est
  par conséquent un **majorant** de l'erreur de modèle, et non une mesure de celle-ci — et chercher
  à le faire baisser davantage reviendrait à absorber le signal de fuite.

La réserve est déjà écrite dans la docstring de `ameliorations.py` ; elle manque simplement là où
les chiffres annuels sont présentés.

---

## 8. Le test de report d'état est décrit avec les mauvaises durées

`reseau.simuler` annonce dans sa docstring :

> *deux tranches de 24 h donnent les mêmes pressions que 48 h d'affilée à 1e-3 m près*

Le test qui le vérifie (`test_report_detat_entre_tranches`) compare en réalité **deux tranches de
12 h à 24 h d'affilée** — `tranche_jours=0.5` contre `tranche_jours=1` sur `PAS_JOUR` pas. Le
`README` (« deux tranches de 12 h contre 24 h d'affilée ») est exact ; c'est la docstring qui
dérive.

---

## Avant de réécrire quoi que ce soit

**Ce qui est négociable** : le ton, le découpage, les titres, les métaphores, l'ordre des
arguments, la longueur. Les textes actuels sont denses et assez péremptoires ; rien n'interdit de
les aérer.

**Ce qui ne l'est pas, sauf à réexécuter :**

* **tout nombre cité dans un texte en markdown.** Ils viennent tous d'une sortie de cellule, et il
  y en a beaucoup ;
* **les comparaisons biais contre dispersion.** Ce n'est pas un ornement de vocabulaire : c'est ce
  qui distingue « la RMSE s'améliore » de « le modèle s'améliore ». Si une phrase dit simplement
  « c'est mieux », elle a perdu l'essentiel ;
* **la distinction produit / mélange.** Équation (1) : un produit, les effets temporels d'un même
  compteur. Équation (2) : une somme, les types de consommateurs d'un même nœud. C'est le
  contresens le plus facile à introduire en reformulant ;
* **les deux écarts assumés** par rapport à la référence (annexe du `GUIDE`). Les retirer
  transformerait une reproduction honnête en une reproduction annoncée comme exacte ;
* **les résultats négatifs** — la section du réservoir sans effet, les rugosités qui n'agissent que
  sur le biais, les groupes arrêtés sur une borne. Ce sont eux qui font la valeur du carnet 2 ;
  les adoucir le viderait.

**Méthode pratique.** Une correction purement rédactionnelle dans un carnet s'édite directement
dans le champ `source` de la cellule markdown du `.ipynb`, sans rien réexécuter : une cellule
markdown n'a pas de sortie. Dès qu'une cellule **de code** change — y compris un simple titre de
figure — il faut réexécuter le carnet et vérifier que les affirmations du texte tiennent toujours.
