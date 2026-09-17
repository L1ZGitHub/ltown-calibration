# Journal des corrections

Onze endroits où le texte du dépôt n'énonçait pas ce que le code faisait, un défaut de code, et
une conclusion que les données ne soutenaient pas. Tous sont corrigés ; ce document garde la trace
de ce qui a été trouvé, parce que trois de ces écarts changeaient un résultat et pas seulement une
phrase.

Pour comprendre le travail lui-même, voir [`GUIDE.md`](GUIDE.md).

---

## Les trois qui changeaient un résultat

### A. La régularisation était annoncée comme active ; elle ne l'était pas

`rugosite.ajuster` a `tikhonov=0.0` par défaut — ce qui désactive le terme `α‖x − x₀‖²` — et
`sigma=None`, ce qui désactive la pondération par capteur. Aucun carnet ne les activait. Le
carnet 1 écrivait pourtant « *les trois garde-fous sont actifs par défaut* », et la docstring de
module « *`ajuster` les reproduit tous les trois* ».

Ce n'était pas qu'une formule : avec la régularisation activée, le chiffre qui servait alors de
signature du sur-ajustement — **+28 % de dispersion en janvier** — devenait bien plus modeste. Le texte décrivait une méthode dont les résultats
auraient été différents.

**Corrigé** : le texte dit maintenant lesquels sont actifs et pourquoi, et le carnet 2 mesure ce
que la régularisation change.

### B. La pénalité de Tikhonov traversait la perte de Huber

`least_squares(loss="huber")` applique la perte robuste à **tout** le vecteur qu'on lui rend, donc
aussi aux lignes de régularisation qu'on y concatène. Au-delà de quelques unités d'écart, la
pénalité croissait donc en `|x − x₀|` et non en `(x − x₀)²` — ce n'était plus du Tikhonov mais du
L1, affaibli précisément là où on le voudrait fort. C'est aussi ce qui rendait α peu intuitif :
il fixait à la fois la force de la pénalité et de quel côté du seuil de Huber on se trouvait.

**Corrigé** : `_pseudo_residus_huber` applique Huber à la main sur les seuls résidus de mesure, et
`least_squares` travaille en moindres carrés ordinaires ; la pénalité reste quadratique. La
transformation conserve le signe, pour que le jacobien numérique reste correct, et vaut l'identité
sous le seuil — **à `tikhonov=0` le résultat est rigoureusement inchangé**, ce qui a été vérifié
coefficient par coefficient avant d'aller plus loin.

### C. La fuite avait bon dos : le vrai problème est que le paramètre n'est pas identifiable

Le §7 du carnet 2 affirmait qu'un ajustement réglé sur une fenêtre fuyarde « achète la fuite avec
de la rugosité », et que régler sur une fenêtre propre suffisait à l'éviter. La première moitié est
une hypothèse raisonnable ; la seconde est fausse, et le test qui le montre n'avait pas été fait.

Il tient en un ajustement de plus : refaire **le même réglage sur une seconde fenêtre sans fuite**,
trois jours après la première. Les deux ne sont pas d'accord — D100, qui couvre 705 conduites sur
905, vaut 82 sur l'une et 137 sur l'autre ; D160 passe d'une extrémité à l'autre de l'intervalle
autorisé. Et la fenêtre propre répond comme la fenêtre **fuyarde**, pas comme l'autre fenêtre
propre : la fuite n'est donc pas ce qui pilote le résultat.

Le contrôle symétrique le confirme. Sous bornes à ±10 %, les deux réglages tombent sur le même
vecteur — celui des bornes — et le gain de 13,8 % que la fenêtre propre obtenait en juillet n'y
survit pas mieux (−3,5 %) que celui de la fenêtre fuyarde (−0,1 %). Les deux étaient le même
artefact.

**Corrigé** : le carnet 2 exécute désormais les trois ajustements et le contrôle, et le §7 comme la
section 14 du guide énoncent le résultat tel qu'il est — sur ce réseau la rugosité n'est pas
identifiable, quelle que soit la fenêtre. L'hypothèse de la fuite y est présentée pour ce qu'elle
était : une hypothèse, mise à l'épreuve et écartée.

---

## Les neuf autres

| | quoi | où | correction |
|---|---|---|---|
| 1 | « médianes de créneau » alors que le défaut est la **moyenne** | `README`, `profils.py`, carnet 1 ×2 | moyenne, la médiane redevient l'option documentée qu'elle est |
| 2 | « Levenberg-Marquardt » alors que le solveur est une région de confiance (LM n'accepte pas de bornes) | `README` ×2, `__init__.py`, titre du §6 du carnet 1 | « moindres carrés sous contraintes de boîte » |
| 3 | le bilan du carnet 2 disait la section du réservoir « aliasée » alors que son propre §4 conclut que **l'aliasing est levé** et qu'elle est simplement sans effet | carnet 2, bilan | reformulé : identifiable, mais sans effet une fois le niveau réancré |
| 4 | le tableau d'ouverture annonçait le levier 3 par le bilan volumique — la méthode qui **échoue** | carnet 2, cellule 0 | « trajectoire simulée du niveau » |
| 5 | « un troisième… le quatrième » désignaient les leviers 4 et 3 | carnet 2, cellule 0 | leviers nommés par leur numéro |
| 6 | « la forme résidentielle porte une saisonnalité marquée, la commerciale beaucoup moins » — les sorties disent 1,30 contre **1,32** | carnet 1, cellule 18 | amplitudes comparables, ce qui les sépare est la **phase** : pic en juillet contre septembre |
| 7 | le profil industriel présenté comme un résultat alors qu'il est estimé sur **4 nœuds** | carnet 1, cellule 18 | réserve explicite ; c'est le bruit de quatre compteurs |
| 8 | « ce qui resterait à essayer » omettait le pas de réancrage, pourtant désigné comme le plus gros levier restant, et surestimait les réducteurs | carnet 2, bilan | réordonné ; les réducteurs agiraient sur un biais qui ne vaut plus que +0,018 m |
| 9 | la docstring de `simuler` décrivait son test avec les mauvaises durées (24 h/48 h au lieu de 12 h/24 h) | `reseau.py` | corrigé, avec le nom du test |

---

## Ce que la relecture a fait apparaître en plus

Deux choses qui ne sont pas des corrections mais des manques, et qui ont donné lieu à du code
nouveau plutôt qu'à des phrases :

**La charge de fuite de chaque fenêtre n'était nulle part.** Elle décide pourtant de la lecture de
tout ajustement : les fenêtres employées dans les carnets vont de **0 m³/h** (la première semaine
de janvier, celle du protocole publié) à **34 m³/h** (octobre, un cinquième de la consommation de
la zone A+B). `reseau.charge_de_fuite` la donne, et le carnet 2 l'affiche à côté de chaque
transfert. Les fuites ne servent à aucun calcul de calibration — seulement à savoir sur quoi on a
réglé.

**L'encadrement (60, 160) de la référence ne contraint rien ici.** Le fichier ne contient que deux
coefficients, 120 et 140, si bien que ces bornes autorisent −56 % à +17 % autour du point de
départ. `rugosite.bornes_par_groupe` construit un encadrement à partir des valeurs que chaque
groupe porte réellement. C'est une hypothèse sur la construction du jeu de données, énoncée comme
telle.

---

## Avant de réécrire quoi que ce soit

**Négociable** : le ton, le découpage, les titres, les métaphores, la longueur.

**Pas négociable, sauf à réexécuter :**

* **tout nombre cité dans un texte en markdown** — ils viennent tous d'une sortie de cellule ;
* **les comparaisons biais contre dispersion.** Ce n'est pas du vocabulaire : c'est ce qui
  distingue « la RMSE s'améliore » de « le modèle s'améliore ». Une phrase qui dit seulement
  « c'est mieux » a perdu l'essentiel ;
* **la distinction produit / mélange.** Équation (1) : un produit, les effets temporels d'un même
  compteur. Équation (2) : une somme, les types de consommateurs d'un même nœud ;
* **les deux écarts assumés** par rapport à la référence. Les retirer transformerait une
  reproduction honnête en une reproduction annoncée comme exacte ;
* **les résultats négatifs** — section du réservoir sans effet, rugosités qui n'agissent que sur le
  biais, groupes arrêtés sur une borne. Ce sont eux qui font la valeur du carnet 2.

**Méthode.** Une correction purement rédactionnelle s'édite directement dans le champ `source` de
la cellule markdown du `.ipynb`, sans rien réexécuter : une cellule markdown n'a pas de sortie. Dès
qu'une cellule **de code** change — y compris un titre de figure — il faut réexécuter le carnet et
vérifier que les affirmations du texte tiennent toujours.
