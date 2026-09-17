# Comprendre ce dépôt

Ce document raconte le travail dans l'ordre où il a été fait, avec le code sous les yeux.
Il se lit en une fois, du début à la fin — chaque section suppose la précédente et rien d'autre.

Le `README` résume les résultats. Celui-ci explique comment on y arrive.
Les corrections à apporter au texte des carnets sont dans [`CORRECTIONS.md`](CORRECTIONS.md).

---

## 1. Le problème

Un réseau de distribution d'eau, c'est des conduites, des consommateurs au bout, et de la
pression dedans. La pression en un point dépend de deux choses : **combien on consomme** en aval
(plus on tire, plus la pression chute) et **combien les conduites résistent** au passage de l'eau.

Un modèle hydraulique, c'est cette physique écrite dans un fichier. On lui donne la consommation
de chaque nœud à chaque instant, il calcule la pression partout. Le moteur de calcul s'appelle
EPANET ; on l'appelle depuis Python par la bibliothèque `wntr`.

Le problème, c'est que le fichier est rempli avec des valeurs de **conception**, pas de mesure.
La consommation d'un nœud, personne ne l'a mesurée : on a mis une valeur plausible. La résistance
d'une conduite, on l'a prise dans un catalogue. Le modèle est donc cohérent mais faux, et la
question qui nous occupe est :

> jusqu'où peut-on le rapprocher de la réalité, et quel paramètre paie vraiment ?

**Calibrer**, c'est exactement ça : ajuster les paramètres qu'on ne connaît pas pour que les
pressions calculées ressemblent aux pressions mesurées.

### Le terrain

Le réseau s'appelle L-Town. 782 jonctions, 905 conduites, 43,2 km. Ce qui le rend exploitable,
c'est qu'il est diffusé avec **une année entière de mesures au pas de 5 minutes** — 105 120 pas :

* **33 capteurs de pression**, en mètres ;
* **3 débitmètres** : `p227` et `p235`, les deux conduites qui amènent toute l'eau du réseau, et
  `PUMP_1`, une pompe interne ;
* **1 capteur de niveau**, sur le réservoir `T1` ;
* **82 compteurs communicants** chez des consommateurs, qui donnent leur consommation réelle.

On a donc le modèle **et** ce que le réseau a réellement fait. C'est rare, et c'est ce qui permet
de vérifier chaque idée au lieu d'en discuter.

### Comment on mesure qu'on a progressé

On simule, on compare aux 33 capteurs, et on regarde l'écart. L'indicateur naturel est la RMSE —
la racine de la moyenne des carrés de l'écart. Mais la RMSE seule est **trompeuse**, et tout le
dépôt repose sur cette remarque, alors autant la poser tout de suite.

Imaginez deux modèles :

* le premier donne une pression toujours **30 cm trop haute**, mais qui suit parfaitement chaque
  variation — chaque montée du matin, chaque creux de la nuit ;
* le second est bien centré en moyenne, mais **se trompe de 30 cm dans un sens ou dans l'autre**
  au hasard.

Les deux ont la même RMSE. Le premier est pourtant excellent : il suffit de lui retrancher 30 cm.
Le second est inutilisable. L'écart moyen s'appelle le **biais**, ce qui reste une fois le biais
retiré s'appelle la **dispersion**, et les deux se séparent exactement :

```
RMSE² = biais² + dispersion²
```

C'est ce que calcule `diagnostics.resume` :

```python
e = (simule - mesure).ravel()
biais = e.mean()
rmse  = np.sqrt((e ** 2).mean())
dispersion = np.sqrt(rmse ** 2 - biais ** 2)
```

**Pourquoi c'est décisif ici** : un biais constant disparaît dès qu'on regarde une *variation* —
une différence entre deux instants, une dérive, un écart à une moyenne. Donc une calibration qui
ne fait que retirer un décalage améliore la RMSE **sans rien améliorer du tout**. On verra plus
loin que deux des leviers testés sont précisément dans ce cas, et on ne peut s'en apercevoir
qu'en affichant les deux colonnes séparément. C'est pour cette raison que tous les tableaux du
dépôt en ont trois : biais, dispersion, RMSE.

---

## 2. Ce qu'on a vraiment sous la main

Avant de calibrer quoi que ce soit, il faut savoir où sont les mesures par rapport à ce qu'on
veut estimer. C'est ce que fait la première page du carnet 1, et le résultat commande tout le
reste.

```python
wn = R.charger_modele(duree_h=24)
zones = R.zones(wn)
bases = R.bases_nominales(wn)
amr   = R.charger_amr(2018)
```

```
782 jonctions, 905 conduites, 43.2 km
capteurs : 33 pressions, 3 débits, 1 niveau, 82 compteurs communicants
zones    : 92 jonctions derrière le réservoir (C), 690 alimentées directement (A+B)
```

Le réseau se sépare en deux. Une pompe remplit un réservoir `T1`, qui alimente une petite zone
de 92 jonctions appelée **zone C**. Les 690 autres, la **zone A+B**, sont alimentées directement
par les deux conduites d'entrée.

`zones()` ne lit pas cette partition dans une liste écrite à la main, elle la **déduit** : on
coupe les deux arêtes qui traversent le réservoir et on regarde ce qui reste connecté.

```python
g = graphe(wn)
g.remove_edge("n54", "T1")     # la pompe
g.remove_edge("T1", "n343")    # la sortie du réservoir
zone_c = next(c for c in nx.connected_components(g) if "n343" in c)
```

Et maintenant le chiffre qui compte :

```
la zone C porte 11.1% de la demande nominale, les nœuds équipés 10.0%
compteurs situés hors de la zone C : 0
```

**Les 82 compteurs sont tous dans la zone C**, qui ne représente qu'un neuvième de la
consommation du réseau. Les 690 jonctions de la grande zone n'ont **aucune mesure de
consommation**.

Le modèle de demande qu'on va construire ne va donc pas interpoler entre des points connus. Il va
**extrapoler** d'une zone vers une autre. C'est une hypothèse forte, et il faudra la juger sur
pièces.

### Trois pièges d'unités, réglés une fois pour toutes

Ils sont tous traités dans `reseau.py` et nulle part ailleurs. Si un bilan ne tombe pas juste,
c'est presque toujours l'un des trois :

* les CSV sont européens : `sep=";"`, `decimal=","`. Sans ces deux arguments, `pandas` lit des
  **chaînes de caractères sans rien signaler** ;
* les compteurs sont en **L/h**, les débitmètres en **m³/h**. D'où le `/1000` dans
  `charger_amr` ;
* le fichier de réseau est en m³/h, `wntr` travaille en SI (m³/s). Tout ce qui sort du module est
  en m³/h ; la division par 3600 se fait au moment d'écrire dans le modèle.

Dernier détail qui resservira : **les pressions sont arrondies à deux décimales**. C'est le
plancher de ce qu'on peut espérer distinguer.

---

## 3. Le point de départ : à quoi ressemble l'erreur

Avant d'améliorer, on mesure. Le fichier contient déjà, pour chaque jonction, une consommation de
base et un profil hebdomadaire. C'est un modèle complet, utilisable tel quel. Simulons-le sur une
semaine de juillet :

```python
D_nominal, _ = P.demandes_nominales(wn, horo[DEBUT:DEBUT + N_PAS + 1])
S_nominal = R.pressions_simulees(D_nominal, noeuds, N_PAS, niveau0=float(niveau[DEBUT]))
print(G.resume(S_nominal, mesure))
```

```
biais_m        0.3412
dispersion_m   0.3290
rmse_m         0.4740
max_m          2.2851
```

Presque un demi-mètre d'erreur, à peu près moitié biais moitié dispersion. C'est la référence à
battre.

### Pourquoi ce modèle ne *peut pas* suivre l'année

Il y a une raison structurelle, et on peut la voir sans rien modéliser. Les profils du fichier
sont **hebdomadaires** : ils se répètent à l'identique du 1er janvier au 31 décembre.

Or la consommation réelle, elle, on sait la mesurer directement. Les deux conduites d'entrée
amènent toute l'eau du réseau ; la pompe en prélève une partie pour le réservoir. Donc par
différence :

```python
conso_ab = debits["p227"] + debits["p235"] - debits["PUMP_1"]
```

C'est la consommation totale de la zone A+B, mesurée, sans modèle. Sa moyenne par mois :

```
amplitude annuelle mesurée : 136.8 à 201.1 m³/h (soit ±18% autour de la moyenne)
amplitude annuelle des profils du fichier : 0 % — ils sont strictement hebdomadaires
```

**±18 % de variation saisonnière, contre 0 % dans le modèle.** Voilà un trou qu'aucun réglage de
rugosité ne bouchera : il manque une information, pas un paramètre.

C'est exactement ce que le modèle de demande vient combler.

---

## 4. Un compteur, ça ne se prédit pas — et ce n'est pas grave

On veut la consommation de chaque jonction, tous les 5 minutes. On ne la mesure que sur 82 nœuds.
Première question : qu'est-ce qu'on peut tirer d'un compteur ?

Prenons le plus chargé du lot. Sa courbe monte le matin, retombe la nuit, remonte le lundi, et
dérive lentement au fil des saisons. La méthode suppose que ces trois effets se **multiplient** :

```
d(t) = d̄ · T(t) · S(t) · R(t)
```

`d̄` c'est le niveau moyen sur l'année. `T` la dérive lente. `S` le rythme de la semaine. `R` ce
qui reste. Voici comment on les sépare, c'est six lignes (`profils.decomposer`) :

```python
x = d / moyenne                    # on travaille en relatif : moyenne 1
T = tendance(x)                    # la dérive lente
S = saisonnalite(x / T, creneaux)  # le rythme hebdomadaire, une fois la dérive retirée
lisse = moyenne * T * S[creneaux]  # la reconstruction, sans le résidu
```

**`tendance`**, c'est une moyenne glissante — une ligne :

```python
uniform_filter1d(x, size=2016, mode="nearest")
```

2016, c'est le nombre de pas de 5 minutes dans une semaine. Le choix n'est pas approximatif :
moyenner sur *exactement* une semaine fait disparaître le cycle hebdomadaire, puisque chaque
fenêtre contient chaque jour une fois. Ce qui survit est donc la dérive lente, et rien d'autre.

**`saisonnalite`**, c'est une moyenne par *créneau*. Un créneau, c'est une position dans la
semaine : « mardi 8 h 05 » est le créneau n° 1 633. Il y en a 2016, et chacun revient 52 fois dans
l'année. On moyenne ces 52 valeurs :

```python
s = [x[creneaux == k].mean() for k in range(2016)]
s = s / s.mean()                   # renormalisé : le profil vaut 1 en moyenne
```

Le tout donne `lisse` — la série *sans son résidu*. C'est la seule partie que la méthode garde.

Et voilà le résultat sur ce compteur :

```
d̄ = 1,835 m³/h   |   part de variance portée par d̄·T·S : 11,1 %
```

**11 %.** On vient de jeter 89 % de ce que fait ce compteur. À première vue, la méthode ne marche
pas.

Elle marche. Ce qu'on a jeté, c'est le comportement d'un foyer particulier — quelqu'un rentre plus
tôt, prend une douche, part en week-end. C'est imprévisible, et ça n'a aucune raison d'être
corrélé avec le voisin. Donc en additionnant des centaines de nœuds, ces écarts se compensent,
tandis que la dérive et le rythme hebdomadaire, eux, s'additionnent.

Ça se vérifie en deux lignes :

```
variance portée par d̄·T·S, compteur par compteur : 42,1 %
variance portée par d̄·T·S, sur la somme des 82    : 71,1 %
```

42 % en moyenne sur un compteur isolé, **71 % dès qu'on les somme**. C'est exactement l'effet
annoncé, et c'est ce qui rend la suite possible : on ne cherche pas à prédire un foyer, on cherche
ce qui reste vrai quand on en regarde beaucoup.

> **Une limite à connaître.** La moyenne glissante n'estime pas la dérive sans la déformer : elle
> l'**atténue**, d'un facteur `sinc(fenêtre/période)`. Négligeable pour une variation annuelle
> (0,1 %), déjà 2,5 % pour une variation de deux mois. La tendance restituée est donc toujours un
> peu plate. C'est le prix d'un estimateur qui ne suppose aucune forme, et c'est documenté dans la
> docstring de `tendance`.

---

## 5. Des 82 compteurs vers les 782 nœuds

On sait décrire un compteur. Il faut maintenant fabriquer une série pour les 700 jonctions qui
n'en ont pas. Comment ?

L'idée est de ne pas traiter les nœuds comme interchangeables. Le fichier de réseau porte, pour
chaque jonction, **trois lignes de consommation** — une par type de consommateur : résidentiel,
commercial, industriel. Un nœud avec beaucoup de commerces n'a pas le même rythme qu'un nœud
résidentiel. Donc on écrit la consommation d'un nœud comme un **mélange** de types :

```
d̂ᵢ(t) = Σⱼ d̄ᵢⱼ · Tⱼ(t) · Sⱼ(t)
```

Attention à ne pas confondre les deux équations : l'équation (1) est un **produit** (les effets
temporels d'un même compteur se multiplient), l'équation (2) est une **somme** (les types de
consommateurs d'un même nœud s'additionnent). C'est le contresens le plus facile à faire.

Les poids `d̄ᵢⱼ` sont donnés — ils sont dans le fichier :

```python
lignes = {}
for n in wn.junction_name_list:
    ligne = dict.fromkeys(CATEGORIES, 0.0)
    for d in wn.get_node(n).demand_timeseries_list:      # les trois lignes
        ligne[str(d.pattern_name).replace("P-", "")] = d.base_value * 3600.0
    lignes[n] = ligne
```

> **Piège** : `Junction.base_demand` ne renvoie que la **première** des trois lignes. Une jonction
> purement industrielle renverrait donc 0. Il faut parcourir `demand_timeseries_list`. Un test
> protège ce point.

Ce qu'on ne connaît pas, ce sont les **formes** `Tⱼ·Sⱼ` de chaque type. Mais là où on a la mesure —
les 82 nœuds équipés — l'équation (2) se retourne. On a 82 équations et 3 inconnues à chaque pas,
donc largement de quoi résoudre :

```python
W = R.bases_nominales(wn).loc[A.columns].to_numpy()      # 82 × 3, les poids connus
formes = (np.linalg.pinv(W) @ A.to_numpy().T).T          # T × 3, les formes cherchées
formes = np.clip(formes, 0.0, None)                      # une demande négative n'a pas de sens
formes /= formes.mean(axis=0)                            # relues comme des multiplicateurs
```

La pseudo-inverse est calculée **une seule fois** et appliquée en un produit matriciel : l'année
entière coûte moins d'une seconde.

Résultat — le rapport entre le mois le plus fort et le mois le plus faible, par type :

```
Residential    1.30
Commercial     1.32
Industrial     1.36
```

Une saisonnalité de 30 %, là où le modèle livré en avait zéro.

Il ne reste qu'à appliquer l'équation dans le bon sens, sur les 782 nœuds :

```python
M = formes.to_numpy()[:n_pas].astype(np.float32)   # T × 3
W = bases.to_numpy().T.astype(np.float32)          # 3 × 782
D = M @ W                                          # T × 782, en m³/h
```

> **Pourquoi `float32`.** Une année × 782 nœuds, c'est 330 Mo en float32 et 660 en float64. C'est
> la plus grosse matrice du dépôt. Le produit est fait directement dans le type voulu, sans
> intermédiaire en double précision — une version antérieure allouait 657 Mo pour rien.

**Et la réserve, qu'il faut garder en tête :** ces formes sont estimées sur 82 compteurs d'**une
seule zone**, celle qui porte 11 % de la demande, et on va les appliquer à 690 jonctions d'une
autre. Rien dans les données ne garantit que les habitudes de consommation y soient les mêmes.

---

## 6. Ce que le modèle de demande change

On resimule, avec les mêmes conditions que la section 3, en ne changeant que la demande :

```
                   biais_m  dispersion_m  rmse_m  gain_dispersion
modèle livré        0.3412        0.3290  0.4740           0.0000
modèle de demande   0.2022        0.2526  0.3235          -0.2324
```

**−23 % de dispersion.** Le gain est réel, et il porte sur la bonne colonne : on a bien amélioré
la dynamique, pas seulement recentré.

Un contrôle indépendant confirme *et* nuance. Si on compare la demande totale du modèle à ce que
les débitmètres mesurent, mois par mois : la **forme** saisonnière est retrouvée, mais il reste un
**écart de niveau** — le modèle est systématiquement en dessous de la mesure.

Une partie de cet écart est attendue (le débit d'entrée contient aussi ce qui fuit, et un réseau
réel fuit), mais pas forcément toute. Retenez ce point : on y reviendra en section 12, et il
deviendra un levier à part entière.

---

## 7. Les rugosités : six groupes et un optimiseur

Deuxième volet de la méthode. On a traité la consommation ; reste la **résistance des conduites**.

Une conduite résiste au passage de l'eau, et cette résistance s'écrit :

```
R = 10,674 · L / (C^1,852 · D^4,871)
```

`L` sa longueur et `D` son diamètre sont connus. `C` est le **coefficient de Hazen-Williams** : il
vaut environ 140 pour une conduite neuve et lisse, moins pour une vieille conduite entartrée.
C'est un paramètre qu'on ne peut pas aller mesurer, donc un candidat naturel à la calibration.

Mais il y a 905 conduites et 33 capteurs. Les calibrer une par une n'a aucun sens. La méthode les
**regroupe** — conduites de même matériau, âge, diamètre — et n'ajuste qu'un coefficient par
groupe. Ici les groupes sont formés par classe de diamètre, faute d'information sur le matériau :

```
D100     705
D150     103
D200      64
D160      16
D225      12
D<=75      5
```

**Ce déséquilibre va tout expliquer** — un groupe de 5 conduites, c'est un paramètre libre avec
presque rien pour le contraindre. Gardez-le en tête.

### Le critère

Ce n'est pas une simple somme de carrés. C'est un moindres carrés pondéré, **sous contraintes de
boîte**, avec perte de **Huber** et **régularisation de Tikhonov** :

```
min_{x_L ≤ x ≤ x_U}  ½ Σⱼ Σᵢ H_κ( ([S·y(tⱼ,x)]ᵢ − z^j_ᵢ) / σ_ij )  +  α‖x − x₀‖²
```

Chacun des ingrédients sert à quelque chose :

* les **bornes** (60 à 160) interdisent les coefficients qu'aucune conduite réelle ne porterait ;
* la **perte de Huber** empêche quelques pas de temps aberrants de gouverner l'ajustement ;
* la **régularisation** retient l'estimateur près du modèle livré là où les données ne
  contraignent rien.

En Python, c'est un appel :

```python
sol = least_squares(critere, x0, method="trf", bounds=(60.0, 160.0),
                    loss="huber", f_scale=1.345, diff_step=0.02, max_nfev=25)
```

> **Sur le solveur** : `method="trf"` est une région de confiance réfléchissante, et non
> Levenberg-Marquardt au sens strict. La raison est simple : LM n'accepte pas de bornes, et le
> critère ci-dessus en pose.
>
> **Sur la régularisation** : `tikhonov` vaut **0 par défaut**, donc le terme `α‖x−x₀‖²` est
> désactivé, et `sigma=None` désactive la pondération par capteur. Le carnet 1 reproduit la
> référence avec les bornes et Huber seulement ; le carnet 2 mesure ce que la régularisation
> change, et ce que donne un encadrement construit sur le fichier plutôt que sur la publication.
>
> **Huber ne doit pas écraser la pénalité.** `least_squares(loss="huber")` applique la perte
> robuste à tout le vecteur qu'on lui rend — lignes de régularisation comprises. `ajuster`
> applique donc Huber à la main, sur les seuls résidus de mesure, pour que `α‖x−x₀‖²` reste
> quadratique. Voir `CORRECTIONS.md`, défaut B.

**Sur quelle fenêtre ajuster ?** Sur la **première semaine de l'année**, et non sur la fenêtre
d'évaluation. C'est ce que fait la référence, et la section 14 montre que c'est aussi la seule
semaine de 2018 qui ne porte aucune fuite. Les carnets ont donc deux variables distinctes :
`DEBUT` (évaluation, juillet) et `DEBUT_ENTR` (ajustement, semaine 1).

`simulateur` est une **fermeture** : elle prend un dictionnaire `{groupe: coefficient}` et rend les
pressions aux capteurs. Toute la configuration hydraulique y est enfermée, ce qui permettra
d'employer exactement la même routine plus tard, dans des conditions aux limites différentes.

```python
def simulateur(coefficients):
    return R.pressions_simulees(D_calibre[DEBUT:DEBUT + N_ENTR + 1], noeuds, N_ENTR,
                                coefficients=U.coefficients_par_conduite(groupes, coefficients),
                                niveau0=float(niveau[DEBUT]))
```

### Le résultat

```
55 simulations en 53 s
RMSE sur la fenêtre d'ajustement : 0.2455 → 0.2099 m

D100     112.1
D150     137.1
D160     158.3
D200     124.8
D225      70.0
D<=75    157.6

groupes arrêtés sur une borne : aucun
```

Deux choses à relever tout de suite.

**Le coût.** On a demandé `max_nfev=12` et il y a eu **55 simulations**. C'est normal : chaque
colonne du jacobien demande une évaluation supplémentaire, et `max_nfev` ne compte que les
évaluations principales. Chaque évaluation est une simulation hydraulique complète.

**`aux_bornes`.** Ici, aucun groupe ne touche sa contrainte. Mais regardez D225 : 70, contre 140 au
départ, alors que ce groupe ne compte que **12 conduites**. Un paramètre auquel les données ne
disent rien ne reste pas au repos, et la section 14 montre le même ajustement rendre quatre
groupes sur six collés à leur borne — sur une autre fenêtre, avec le même code. C'est pour cela que `ajuster` renvoie cette liste.

### Et sur les pressions ?

```
                      biais_m  dispersion_m  rmse_m
modèle de demande      0.2022        0.2526  0.3235
+ rugosités ajustées   0.0195        0.2584  0.2591
```

Regardez bien. Le **biais s'effondre** : 0,2022 → 0,0195, il ne reste presque rien. La
**dispersion, elle, se dégrade légèrement** : 0,2526 → 0,2584.

La RMSE s'améliore beaucoup (−19 %). Le modèle, lui, ne s'est pas amélioré du tout.

C'est le premier cas concret de ce qu'annonçait la section 1 : **ajuster la rugosité a servi de
constante d'étalonnage**, pas de paramètre physique. On y reviendra en section 13 pour comprendre
pourquoi, et pour le prouver.

---

## 8. Changer de point de vue

Fin de la reproduction. On est passé de 0,4740 à 0,3235 m de RMSE, dont l'essentiel du gain vient
du modèle de demande, et une part importante de ce qui reste est du **biais**.

La question naturelle serait : comment raffiner le modèle de consommation ? Le second carnet prend
le problème par l'autre bout. Au lieu de raffiner un modèle, il va regarder **ce que les capteurs
disent déjà et que le modèle n'écoute pas**.

Quatre choses, en l'occurrence. Les voici dans l'ordre où elles ont été trouvées.

---

## 9. La pompe suit une consigne, pas la réalité

Le fichier de réseau pilote la pompe par deux règles :

```
LINK PUMP_1 CLOSED IF NODE T1 ABOVE 3.9000
LINK PUMP_1 OPEN   IF NODE T1 BELOW 2.4000
```

Autrement dit : remplir le réservoir quand il descend sous 2,40 m, arrêter à 3,90 m. C'est une
règle plausible. C'est aussi une **hypothèse**, et son débit est mesuré — donc on peut la vérifier
au lieu de la supposer.

```python
statut_mesure = (R.charger_debits(2018)["PUMP_1"].to_numpy() > 1.0).astype(float)
```

(Le seuil à 1 m³/h n'a pas besoin d'être fin : la pompe débite 44 m³/h ou zéro.)

```
pompe en marche : 65.0% du temps mesuré, 49.5% simulé
basculements sur la fenêtre : 14 mesurés, 14 simulés
états en désaccord : 22.5% des pas
```

Même nombre de cycles, mais **un pas sur cinq en désaccord**. La vraie pompe est commandée
autrement, et le cycle simulé dérive du cycle réel sans aucune raison.

### Est-ce que c'est grave ?

C'est la vraie question, et on peut y répondre **sans simuler quoi que ce soit** — uniquement en
comparant les mesures selon que la pompe tourne ou non :

```
débit entrant : 217 m³/h pompe en marche, 178 m³/h à l'arrêt (+22 %)
la pompe prélève à elle seule 44 m³/h, soit 20 % de ce qui entre

pression moyenne aux 33 capteurs : 46.099 m en marche, 46.452 m à l'arrêt (écart -0.353 m)

capteurs les plus sensibles :
n54    -0.913
n410   -0.781
n429   -0.772
```

Voilà le mécanisme, mesuré. La pompe n'est pas un détail du réservoir : elle prélève **un
cinquième de tout ce qui entre dans le réseau**. Quand elle démarre, le débit dans les conduites
de transport augmente d'autant, la perte de charge augmente avec, et **toutes** les pressions
descendent — d'un tiers de mètre en moyenne, de près d'un mètre près de l'arrivée.

Se tromper d'état un pas sur cinq revient donc à injecter un tiers de mètre d'erreur un pas sur
cinq. C'est **du même ordre que l'erreur totale qu'on cherche à expliquer**.

### Le remède

Aucun modèle. On retire les deux contrôles et on impose l'état lu au débitmètre, sous forme d'un
profil de vitesse (vitesse nulle = pompe fermée) :

```python
for nom in list(wn.control_name_list):
    wn.remove_control(nom)
wn.add_pattern("POMPE_MESUREE", list(pompe))
pm = wn.get_link("PUMP_1")
pm.speed_timeseries.pattern_name = "POMPE_MESUREE"
pm.initial_status = LinkStatus.Open
```

---

## 10. Mais alors le réservoir dérive

Retirer les contrôles retire aussi **la seule chose qui régulait le niveau du réservoir**.

La pompe étant désormais commandée de l'extérieur, le niveau devient l'intégrale libre de l'écart
entre ce qui entre (mesuré) et ce que le modèle croit consommer (estimé). Une petite erreur de
demande, même constante, s'accumule sans rien pour la rattraper.

Sur deux semaines de juillet :

```
mesuré                     : de 2.39 à 3.90 m
sans ancrage               : de 2.84 à 4.00 m, saturé sur 20% des pas, écart moyen 0.49 m
```

Le niveau part et vient **buter contre le trop-plein** à 4 m, où il reste collé un pas sur cinq.
Un réservoir saturé ne joue plus son rôle : il absorbe ou fournit ce qu'on veut sans que son
niveau bouge, et la zone qu'il alimente devient fausse.

Le remède est d'une ligne. Le niveau est mesuré lui aussi : il suffit, à chaque tranche de
simulation, de **repartir de la valeur lue au capteur** au lieu de la valeur simulée.

```python
if niveau_mesure is not None:
    niveau = float(niveau_mesure[debut])
```

Et son efficacité se règle d'un seul paramètre — la longueur des tranches. Plus elles sont
courtes, moins l'erreur a le temps de s'accumuler :

```
sans ancrage               : saturé 20% des pas, écart moyen 0.49 m
réancré chaque jour        : saturé  9% des pas, écart moyen 0.37 m
réancré toutes les 6 h     : saturé  2% des pas, écart moyen 0.11 m
```

L'ancrage ne supprime pas la saturation, il la **borne**. Et le seul prix à payer est du temps de
calcul.

**Les leviers 1 et 2 sont indissociables** : le premier sans le second ne tient pas.

---

## 11. Les gains ne s'additionnent pas

Il est tentant de tester une amélioration à la fois. Voyons ce que ça donne. On croise les deux
modèles de demande et les deux jeux de conditions aux limites — quatre simulations, deux
saisons :

```
                                                   biais_m  dispersion_m  rmse_m
janvier modèle livré      contrôles du fichier     -0.1428        0.2162  0.2591
                          pompe mesurée + ancrage  -0.1141        0.1623  0.1984
        modèle de demande contrôles du fichier      0.0536        0.1229  0.1341
                          pompe mesurée + ancrage   0.0486        0.0697  0.0849
juillet modèle livré      contrôles du fichier      0.3320        0.3090  0.4535
                          pompe mesurée + ancrage   0.3014        0.2285  0.3782
        modèle de demande contrôles du fichier      0.2050        0.2289  0.3072
                          pompe mesurée + ancrage   0.1847        0.1802  0.2580
```

Suivez la colonne `dispersion_m`.

**En juillet**, les gains se multiplient presque exactement : chacun retire environ un quart de la
dispersion (0,3090 → 0,2285 et 0,3090 → 0,2289), et le couple en retire un peu plus de quarante
pour cent (→ 0,1802).

**En janvier**, le couple fait nettement **mieux** que cette composition. Pris seuls, les deux
gains laissaient attendre un peu moins de la moitié de la dispersion ; ensemble, on en garde un
tiers (0,2162 → 0,0697).

Autrement dit : **corriger la pompe ne fait pas que retirer sa propre erreur, ça rend le modèle de
demande plus rentable qu'il ne l'était.** C'est logique une fois dit — tant que la pompe bascule au
mauvais moment, elle injecte dans le résidu un écart qui recouvre partiellement ce que le modèle
de demande apporterait.

La leçon de méthode vaut au-delà de ce réseau : **mesurer les améliorations une par une, dans un
modèle où un autre facteur est encore mal spécifié, sous-estime leur valeur commune.** Un tableau
croisé coûte quatre simulations au lieu de deux, et évite d'écarter un raffinement au vu d'un gain
isolé décevant.

Et un constat qui resservira : le plus gros levier unique de ce tableau n'est pas le modèle de
consommation, c'est **une condition aux limites lue dans un capteur déjà installé**.

---

## 12. La section du réservoir : une mesure qui n'en est pas une

Le diamètre du réservoir inscrit dans le modèle est **16,00 m**. Un nombre rond, donc une valeur
de catalogue plutôt qu'une mesure. Et il devrait se retrouver dans les données sans jamais
inverser le modèle hydraulique.

Le raisonnement est élémentaire. Sur tout intervalle où la pompe **ne change pas d'état**, le
réservoir n'a qu'une entrée (la pompe, mesurée) et qu'une sortie (la zone C, estimée aux
compteurs). La conservation du volume donne :

```
A · Δh = V_pompe − V_zoneC
```

Chaque demi-cycle de pompe fournit un point, et il y en a plusieurs centaines dans l'année. On
ajuste une droite par l'origine, sa pente est la section. Facile.

### Première passe : ça échoue, et proprement

Une droite peut s'ajuster dans deux sens. Régresser `V_net` sur `Δh` suppose que l'erreur est sur
les volumes ; régresser `Δh` sur `V_net` suppose qu'elle est sur les niveaux. Les deux hypothèses
sont défendables — le capteur de niveau est arrondi au centimètre, et l'estimation de la zone C
n'est pas exacte. Faisons les deux :

```
section du modèle           : 201.1 m²  (diamètre 16.00 m)
estimée, erreur sur volumes : 207.9 m²  (diamètre 16.27 m)
estimée, erreur sur niveaux : 414.0 m²  (diamètre 22.96 m)
```

**Du simple au double.** Le nuage de points était pourtant convaincant, et chaque estimateur pris
isolément aurait donné une barre d'erreur étroite et rassurante.

L'explication tient en un nombre :

```
corrélation entre volume pompé et volume consommé, sur un remplissage : 0.997
si l'on laisse la consommation de la zone C libre d'un facteur d'échelle :
   section 228.4 m², facteur sur la consommation ×1.72
```

Voilà. Sur un demi-cycle de **remplissage**, le volume pompé et le volume consommé sont tous deux
proportionnels à la durée du cycle. Ils sont donc **colinéaires à 0,997**, et deux régresseurs
colinéaires ne se séparent pas, quelle que soit la quantité de données. La droite mesure en
réalité une *combinaison* de la section et de la consommation de la zone, et laisser cette
dernière libre absorbe l'écart sans difficulté.

C'est ce qu'on appelle un **aliasing** : deux paramètres que les données ne peuvent pas
distinguer.

> Le « ×1,72 » n'est donc **pas un résultat**. C'est l'artefact qui révèle le problème.

### Deuxième passe : contourner l'aliasing au lieu de le subir

Le découpage en demi-cycles jette une information, et c'est précisément celle qui manque. Les deux
régimes ne sont **pas symétriques** :

* pompe **à l'arrêt**, la pente du niveau vaut `−Q_C / A` — ça ne donne qu'un *rapport* ;
* pompe **en marche**, elle vaut `(Q_pompe − Q_C) / A`, et `Q_pompe`, lui, est **mesuré**.

Les deux régimes pris **ensemble** séparent donc `A` de `Q_C`. Encore faut-il les faire travailler
ensemble : plutôt que d'ajuster une droite sur des demi-cycles indépendants, on simule la
trajectoire **complète** du niveau — sans jamais la réancrer, puisque c'est l'accumulation libre
de l'écart qui porte l'information — et on balaye les deux paramètres.

```python
for k in echelles:                      # facteur d'échelle sur la demande de la zone C
    E = np.array(base, copy=True)
    E[:, est_c] *= float(k)
    for d in diametres:
        h = simuler(E, ..., diametre_T1=d, pompe=pompe)   # pas de niveau_mesure : c'est le but
        erreur = np.sqrt(((h - niveau_mesure) ** 2).mean())
```

Sur une fenêtre de janvier :

```
 diametre_m  echelle_zoneC  erreur_niveau_m   sature
       16.0            1.0         0.201988 0.005456
       17.0            1.0         0.213216 0.003472
       15.0            1.0         0.213927 0.006944
       14.0            1.0         0.264053 0.008681
       17.0            1.4         2.729589 0.497520
       ...
```

**Le facteur d'échelle se fixe sans ambiguïté à 1** : dès 1,4, l'erreur est treize fois pire et le
réservoir sature la moitié du temps. Le ×1,72 de la première passe était donc bien un artefact de
colinéarité, et non une consommation cachée. **L'aliasing est levé** — et il l'est parce qu'on a
simulé au lieu de régresser, pas parce qu'on a ajouté des données.

Le diamètre, lui, a un minimum **intérieur**, à 16 m — la valeur du fichier. Mais le critère est
plat : trois mètres de diamètre ne déplacent l'erreur que d'une dizaine de pour cent. C'est une
confirmation, pas une mesure de précision.

### Le piège, et il est vicieux

Cette fenêtre a été choisie en janvier pour une raison. Le même balayage à deux autres saisons :

```
juillet  14 m : 0.457   15 m : 0.504   16 m : 0.551   17 m : 0.592   | saturé 20% du temps
octobre  14 m : 0.419   15 m : 0.473   16 m : 0.521   17 m : 0.563   | saturé 14% du temps
```

L'erreur **décroît sans minimum** jusqu'au bord de la grille. Le balayage réclame le plus petit
réservoir qu'on lui propose, et il le réclamerait quelle que soit la grille.

Pourquoi ? Regardez la colonne `sature` : le réservoir passe 13 à 20 % du temps collé au
trop-plein. Or **un réservoir plus petit se remplit et se vide plus vite, donc sature moins**.
L'optimiseur poursuit la saturation, pas la physique.

**Et c'est là la leçon qui dépasse le réservoir.** Le critère ne prévient pas qu'il est devenu vide
de sens. Il ne s'aplatit pas — ce serait trop commode, on le verrait. Il continue de produire un
minimum bien net, simplement il est au bord. Un paramètre estimé par optimisation demande donc
**deux** vérifications distinctes :

1. que l'optimum soit **intérieur** ;
2. que le régime simulé soit **celui qu'on croit**.

C'est pour ça que `ajuster_reservoir` renvoie la saturation **à côté** de l'erreur, et pas
seulement le meilleur point.

### Et finalement, est-ce que ça change quelque chose ?

Non.

```
                   biais_m  dispersion_m  rmse_m
section du modèle   0.1832        0.1888  0.2631
section estimée     0.1834        0.1903  0.2643
```

Sur l'année entière, même conclusion : 0,1992 m avec la section du fichier, 0,2000 m avec la
section estimée.

Le levier est donc **à écarter, mais pas pour la raison qu'on croyait**. La section n'est pas « non
identifiable » : elle l'est, pourvu qu'on simule au lieu de régresser et qu'on choisisse une
fenêtre où le réservoir respire. Elle est simplement **sans effet** — parce que le levier 2 réancre
déjà le niveau sur son capteur à chaque tranche, et a donc pris d'avance tout ce que la section
aurait apporté.

C'est le genre de conclusion qu'il vaut mieux établir en une demi-heure que supposer pendant un
mois.

---

## 13. Le niveau de la demande est mesuré, pas à modéliser

Souvenez-vous de la section 6 : le modèle de demande retrouve la *forme* saisonnière mais reste
systématiquement **en dessous** en niveau.

Or ce niveau, on le mesure. C'est le `conso_ab` de la section 3 : `p227 + p235 − PUMP_1`. Aucun
modèle ne battra une mesure directe.

Le recalage ne touche donc **que le niveau d'ensemble**, par blocs d'une semaine. À l'intérieur
d'un bloc, la répartition entre nœuds et la forme temporelle restent celles du modèle :

```python
for a in range(0, len(D), periode):
    b = min(a + periode, len(D))
    modele = float(E[a:b][:, est_ab].sum())
    cible  = float(np.nansum(mesure[a:b]))
    E[a:b][:, est_ab] *= cible / modele
```

```
                   biais_m  dispersion_m  rmse_m
modèle de demande   0.1832        0.1888  0.2631
+ bilan de masse    0.1515        0.1882  0.2417
```

**Le biais tombe, la dispersion ne bouge pas.** Et c'est exactement ce qu'on devait attendre : un
niveau de demande trop bas déplace toutes les pressions dans le même sens, donc produit un
*décalage*, et corriger un décalage ne change rien à la dynamique.

Le levier vaut donc pour ce qu'il est — un modèle mieux centré, utile dès qu'on compare des
pressions absolues — et pas pour ce qu'il n'est pas.

> ⚠️ **Une réserve importante, et elle compte.** Le débit d'entrée contient **tout** ce qui sort du
> réseau, y compris ce qui fuit. Recaler la demande sur le bilan de masse **absorbe donc les
> fuites dans la demande**.
>
> C'est sans conséquence pour calibrer un modèle hydraulique. C'est rédhibitoire si l'on compte
> ensuite chercher des fuites dans le résidu : le volume d'une fuite se retrouve réparti sur
> 690 nœuds au lieu de sortir en un point, et le signal qu'on cherchait est dilué. Dans ce cas, il
> faut ajuster le recalage sur une **période saine** et le transporter, sans jamais le laisser
> suivre la mesure en continu.

---

## 14. Retour sur les rugosités : pourquoi si peu de prise

On a laissé une question ouverte en section 7 : l'ajustement des rugosités agit sur le biais et
pas sur la dynamique. Reprenons-la maintenant qu'on a de bonnes conditions aux limites.

### Le contrôle le plus simple d'abord

Avant d'ajuster six paramètres, balayons-en **un seul** — un facteur unique appliqué à toutes les
conduites :

```
 facteur   rmse   biais  dispersion
    0.85 0.2675 -0.1344      0.2312
    0.90 0.1924 -0.0219      0.1912
    0.95 0.1903  0.0742      0.1752
    1.00 0.2376  0.1569      0.1784
    1.05 0.2997  0.2287      0.1937
    1.10 0.3618  0.2913      0.2146
```

Voilà la figure qui résume tout le problème. La RMSE a un minimum net, à ×0,95. Et ce minimum
tombe **exactement là où le biais change de signe** (+0,074 à ×0,95, −0,022 à ×0,90). La
dispersion, elle, bouge à peine sur toute la plage.

Autrement dit : ajuster la rugosité ne fait pas mieux coller le modèle au réseau. Ça déplace
toutes les pressions **en bloc** jusqu'à compenser une erreur de niveau qui n'a rien à voir avec
la friction. C'est un paramètre physique utilisé comme une constante d'étalonnage — et on ne peut
le voir qu'en séparant le biais de la dispersion.

### Pourquoi si peu de prise ? Il suffit de regarder

La question se tranche par une mesure directe, sans aucune optimisation : **de combien la rugosité
peut-elle bouger une pression ?** Réponse : de ce que les conduites perdent réellement en charge.

```python
H = res.node["head"]
perte = abs(H[conduite.start_node_name] - H[conduite.end_node_name])
```

> **Piège** : ne surtout pas lire la colonne `headloss` du simulateur. Pour les conduites, elle est
> **normalisée par la longueur**, et la confondre avec une perte en mètres fausse le diagnostic
> d'un facteur propre à chaque conduite. Le contrôle est immédiat : la somme des pertes le long
> d'un chemin doit rendre l'écart de charge entre ses extrémités.

```
perte de charge médiane d'une conduite : 4.11 mm
9 conduites sur 10 sont sous            : 44.37 mm
perte maximale observée                 : 456 mm

part de la friction totale portée par les conduites les plus chargées :
 1 % des conduites     0.129
 5 % des conduites     0.378
10 % des conduites     0.564
```

**Quatre millimètres.** La conduite médiane perd quatre millimètres de charge. Changer son
coefficient de 10 % la déplacerait d'une fraction de millimètre — alors que les capteurs sont
arrondis au centimètre et que l'erreur à expliquer se compte en dizaines de centimètres.

Et la friction qui existe est **très concentrée** : un dixième des conduites en porte plus de la
moitié. Les premières du classement sont `p110`, `p105`, `p235`, `p479`, `p480` — les conduites
d'arrivée et de transport, celles-là mêmes dont le débit dépend de l'état de la pompe.

Les centaines de conduites restantes ne portent donc **aucune information exploitable**. Les
regrouper en six familles plutôt qu'en une seule ajoute cinq paramètres qui n'ont presque rien à
identifier.

### Le test qui tranche : est-ce que ça transfère ?

Ajustons les six coefficients sur **juillet**, la fenêtre qu'on évalue, dans les bonnes conditions
aux limites. C'est le choix naturel, et c'est celui que ce dépôt faisait avant de mesurer ce que
juillet contient :

```
D100      72.0
D150     160.0
D160     160.0
D200     150.0
D225     160.0
D<=75    160.0
groupes arrêtés sur une borne : ['D150', 'D160', 'D225', 'D<=75']
```

**Quatre groupes sur six au bout de leur contrainte**, et le plus gros, D100, divisé par deux. Sur
la fenêtre d'ajustement, la RMSE tombe spectaculairement.

Mais un seul test tranche pour une calibration : **est-ce que ça transfère ?** Jugeons les mêmes
coefficients sur des périodes qui n'ont servi à rien. Variation de dispersion, sur trois fenêtres
de sept jours :

| évaluée sur | variation de dispersion |
|---|---|
| **juillet** — la fenêtre d'ajustement | **−13,2 %** |
| octobre — la saison voisine | −3,0 % |
| **semaine 1** — jamais vue, et sans fuite | **+10,0 %** |

Le gain s'évapore à mesure qu'on s'éloigne de la fenêtre d'ajustement, puis **change de signe**.
C'est la signature du sur-ajustement, et elle n'est lisible que parce qu'on a pris la peine
d'évaluer ailleurs.

Les coefficients disent pourquoi : quatre groupes sur une borne, et ce sont les plus petits.
**Un paramètre sans information ne reste pas inerte** — il part au bout de sa contrainte et se met
au service de l'erreur qui domine la fenêtre. La borne l'empêche de partir à l'infini ; elle ne
l'empêche pas d'être inutile.

> **Et c'est là que les garde-fous de la méthode prennent leur sens.** Bornes, perte de Huber,
> régularisation : ce ne sont pas des précautions d'écriture, ils portent une part du résultat.
> Retirez-les, avec un budget d'itérations plus large, et l'optimiseur part chercher des
> coefficients **multipliés par une trentaine** sur les conduites de transport — des conduites
> rendues parfaitement lisses. Ce jeu améliore la dispersion de 16 % en juillet et la dégrade de
> 46 % en janvier.

### Mais d'abord : qu'est-ce que la fenêtre d'ajustement contenait ?

Avant de conclure au sur-ajustement, une question qu'on aurait dû poser plus tôt. Le jeu de données
publie les fuites qu'il contient. On ne s'en sert pour aucun calcul — seulement pour savoir sur
quoi on a réglé (`reseau.charge_de_fuite`) :

| fenêtre | fuite moyenne | part de la conso A+B | rôle |
|---|---|---|---|
| semaine 1 (protocole publié) | **0,0 m³/h** | 0 % | comparaison au 6 cm |
| janvier | ~0,0 | 0 % | transfert |
| **juillet** | **12,3** | **7 %** | **ajustement** |
| octobre | 34,3 | 20 % | transfert |

La première semaine est donc la seule fenêtre propre de l'année, et c'est déjà celle du protocole
publié : c'est sur elle qu'on règle. L'idée derrière ce choix est qu'une fuite et un excès de
friction produisent **le même effet** — une baisse de pression en aval — et qu'un optimiseur qui
n'a que la rugosité sous la main ne peut pas les distinguer : il paierait la fuite avec de la
rugosité.

C'est une hypothèse. Mettons-la à l'épreuve, parce qu'elle ne survit pas au test.

### Le même ajustement sur trois fenêtres

Un coefficient de rugosité est censé décrire une conduite, pas une semaine. Refaisons donc
exactement le même ajustement sur trois fenêtres, dont **deux sans fuite** : la semaine 1, et
janvier trois jours plus tard.

| groupe | conduites | départ | **semaine 1** (propre) | **janvier** (propre) | **juillet** (12 m³/h) |
|---|---|---|---|---|---|
| **D100** | **705** | 137 | **82** | **137** | **72** |
| D150 | 103 | 138 | 160 | 137 | 160 |
| D160 | 16 | 140 | **160** | **75** | 160 |
| D200 | 64 | 140 | 146 | 128 | 150 |
| D225 | 12 | 140 | 160 | 76 | 160 |
| D<=75 | 5 | 135 | 160 | 109 | 160 |

**Les deux fenêtres propres ne sont pas d'accord entre elles.** D100, qui couvre 705 conduites sur
905, vaut 82 sur l'une et 137 sur l'autre. D160 passe d'une extrémité à l'autre de l'intervalle
autorisé, sur les mêmes seize conduites, à trois jours d'écart.

Et l'écart ne sépare pas le propre du fuyard : **la semaine 1 répond comme juillet**, c'est
janvier qui est à part. L'explication par la fuite ne tient donc pas telle quelle. Ce qu'on
observe est plus simple et plus grave — un paramètre qui change autant selon la fenêtre **n'est
pas mesuré**. Le tableau des pertes de charge ci-dessus l'annonçait : le critère est presque plat,
donc l'optimiseur suit le bruit de la fenêtre.

Le transfert raconte la même chose (variation de dispersion, en %) :

| évaluée sur ↓ — réglée sur → | semaine 1 | janvier | juillet |
|---|---|---|---|
| semaine 1 | −3,1 | **−10,6** | **+10,0** |
| juillet | −13,8 | **+4,6** | −13,2 |
| octobre | −4,1 | **+8,7** | −3,0 |

Chaque réglage aide les fenêtres qui ressemblent à la sienne et dégrade les autres. Que la
semaine 1 les améliore toutes les trois est une coïncidence heureuse, pas une propriété du
réglage.

### Ce que les garde-fous valent, mesuré

D'où vient alors ce −13 % que deux fenêtres sur trois trouvent en juillet ? Interdisons au modèle
de trop s'éloigner du fichier et regardons ce qu'il en reste. Deux façons de le dire — une
contrainte dure (bornes à ±10 % de ce que le groupe porte déjà) et une pénalité molle (Tikhonov) :

| réglage | semaine 1 | juillet | octobre |
|---|---|---|---|
| semaine 1, tel quel (60–160, α = 0) | −3,1 | **−13,8** | −4,1 |
| juillet, tel quel (60–160, α = 0) | +10,0 | **−13,2** | −3,0 |
| **semaine 1 + bornes à ±10 %** | −12,8 | **−3,5** | +4,4 |
| **juillet + bornes à ±10 %** | −11,7 | **−0,1** | +9,0 |
| juillet + régularisation α = 0,01 | −11,6 | +2,5 | +9,0 |
| juillet + régularisation α = 0,1 | −10,6 | +2,0 | +5,7 |

Il n'en reste presque rien — **et pas seulement pour juillet**. Sous bornes, les deux fenêtres
tombent sur le même vecteur (D100 à 108, tout le reste collé à 154, c'est-à-dire sur les bornes) :
ce n'est plus la donnée qui décide, c'est la contrainte. Et le −13,8 % que la fenêtre **propre**
obtenait sur juillet ne survit pas mieux que le −13,2 % de la fenêtre fuyarde.

**La conclusion est négative, et plus large que la fuite.** Il n'y a pas assez de friction dans ce
réseau pour identifier six coefficients, quelle que soit la fenêtre. Le §15 le confirme sur douze
mois : l'ajustement y déplace la dispersion de 0,3 %.

Et l'encadrement publié ne contraint rien ici : le fichier ne contient que **deux** coefficients,
120 sur 119 conduites et 140 sur 786, si bien que (60, 160) autorise −56 % à +17 % autour du
départ. `rugosite.bornes_par_groupe` construit un encadrement à partir de ce que chaque groupe
porte réellement — sous l'hypothèse, à énoncer comme telle, que les paramètres du jeu de données
ont été écartés d'au plus 10 % de leur valeur vraie.

**Bornes ou Tikhonov ?** Ils n'encodent pas le même statut de connaissance. Une borne encode un
**fait** et ne se règle pas ; α encode une **préférence** et demanderait d'être réglé — sur une
fenêtre, dont on vient de voir qu'elle décide du résultat.

**Quatre règles pratiques en sortent**, qui ne valent pas que pour ce réseau :

1. **un paramètre se refait sur plusieurs fenêtres avant d'être cru.** S'il change de valeur de
   l'une à l'autre, il n'est pas identifié, et aucune borne ne le rendra identifiable — elle ne
   fera que le remplacer par la borne ;
2. une calibration se juge **sur une période qui n'a pas servi à la régler**, et de préférence
   dans un autre régime de fonctionnement ;
3. des paramètres physiquement invraisemblables sont **un diagnostic, pas un détail** — ils
   signalent que le modèle compense une erreur qui n'est pas celle qu'on croit corriger ;
4. le nombre de paramètres **se décide avant l'optimisation**, en mesurant ce que les données
   peuvent contraindre. Ici, un tableau de pertes de charge qui coûte une journée de simulation.

---

## 15. Ce que ça donne sur l'année

Tout ce qui précède a été mesuré sur des fenêtres d'une semaine, pour rester rapide à relancer.
Voici les mêmes variantes sur les **105 120 pas de l'année**, aux 33 capteurs :

| | biais | dispersion | RMSE |
|---|---|---|---|
| modèle livré | +0,310 | 0,389 | **0,497 m** |
| + modèle de demande | +0,302 | 0,361 | 0,470 |
| + pompe mesurée, réancrage quotidien | +0,272 | 0,266 | 0,380 |
| + bilan de masse | +0,148 | 0,221 | 0,266 |
| + réancrage toutes les 6 h | +0,122 | 0,158 | 0,199 |
| + six rugosités réglées sur la semaine 1 | +0,018 | 0,157 | **0,158 m** |

**−68 % de RMSE, −60 % de dispersion.** Quatre lectures :

* **la calibration de rugosité tient sur l'année, mais seulement sur la RMSE.** Six coefficients
  réglés sur une semaine de janvier retirent le biais de 12 cm sur douze mois et laissent la
  dispersion inchangée à un demi pour cent près (0,1578 → 0,1573). Les écarts de −14 % à +10 % du
  tableau de transfert se compensent sur l'année : c'était du bruit d'ajustement, ni gain ni perte ;
* **le modèle de demande rapporte moins sur l'année** (−7 % de dispersion) que sur la semaine de
  juillet (−20 %). Son apport est saisonnier, et le chiffrer sur une fenêtre choisie le surestime ;
* **le pas de réancrage est le plus gros levier restant.** Passer de 24 h à 6 h retire encore un
  quart de la RMSE, et ne coûte que du temps de calcul. C'est le paramètre à régler en premier sur
  un autre réseau ;
* il n'a jamais été descendu **sous 6 h** sur une année complète. C'est la première chose à
  essayer.

### Comparaison au chiffre publié

La méthode de référence annonce une **RMSE de 6 cm** sur ses 33 capteurs, obtenue en calibrant six
groupes de rugosité sur la **première semaine de 2018** et en évaluant sur cette même semaine.
C'est donc un chiffre *dans l'échantillon*. Le protocole est reproductible tel quel :

| première semaine de 2018, 33 capteurs | biais | dispersion | RMSE |
|---|---|---|---|
| modèle livré | −0,080 | 0,170 | 0,188 m |
| modèle de demande, commandes du fichier | +0,069 | 0,196 | 0,208 m |
| ⤷ + six groupes de rugosité | +0,004 | 0,192 | 0,192 m |
| configuration complète (pompe, ancrage, bilan de masse) | +0,059 | 0,075 | 0,095 m |
| ⤷ + six groupes de rugosité | −0,002 | 0,062 | **0,062 m** |
| | | | *publié : 0,060 m* |

**La reproduction retombe sur le chiffre publié**, à deux millimètres près. Deux observations qui
comptent plus que cette coïncidence :

* **la rugosité seule n'y suffit pas.** Ajustée par-dessus le modèle de demande et les commandes
  du fichier — la configuration la plus proche de ce que décrit la référence — elle plafonne à
  0,192 m. Les 6 cm ne s'atteignent qu'une fois les conditions aux limites prises dans les
  capteurs ;
* **trois des six groupes s'arrêtent sur une borne.** Ce sont les trois plus petits, 5 à
  16 conduites. La référence n'a pas ce problème, parce que ses groupes sont formés par matériau
  et âge — une information dont on ne dispose pas ici, et qui manque vraiment.

---

## 16. Travailler dans ce dépôt

Cette section est de la référence. Elle ne se lit utilement qu'une fois le reste compris.

### Les cinq modules

```
calibration/
├── reseau.py         accès aux données, topologie, zones, simulation par tranches
├── profils.py        les deux équations du modèle de demande (sections 4 et 5)
├── rugosite.py       groupes, critère d'ajustement, diagnostic d'identifiabilité (7 et 14)
├── ameliorations.py  pompe, ancrage, section du réservoir, bilan de masse (9 à 13)
└── diagnostics.py    biais / dispersion / RMSE, par capteur et agrégés (section 1)
```

Elles prennent des tableaux et rendent des tableaux, ne dessinent rien d'elles-mêmes (sauf les
deux aides de `diagnostics`) et n'écrivent aucun fichier sans qu'on le leur demande.

### Cinq choses à savoir avant de modifier quoi que ce soit

**(a) Une année ne se simule pas d'un coup.** Le moteur garde tous ses résultats en mémoire.
`reseau.simuler` découpe en tranches et **reporte l'état** — niveau du réservoir et statut de la
pompe — d'une tranche à la suivante. Un test le vérifie : deux tranches de 12 h donnent les mêmes
pressions que 24 h d'affilée, à moins d'un millimètre près.

**(b) C'est la fonction `extraire` qui borne la mémoire.** `simuler(D, noeuds, n_pas, extraire)`
appelle `extraire(res, wn, debut, fin)` une fois par tranche et ne garde que ce qu'elle renvoie.
Une année de pressions aux 782 jonctions pèse 330 Mo ; aux 33 capteurs, 14 Mo. **Ne jamais
renvoyer `res`.**

**(c) Le profil de demande porte un pas de plus que l'horizon.** D'où le `+ 1` partout :
`D[DEBUT:DEBUT + N_PAS + 1]` pour `N_PAS` pas simulés. EPANET lit le profil un pas au-delà de la
fin de la tranche, et **si le profil est trop court il reboucle silencieusement sur son début** —
pas d'erreur, juste un résultat faux. `preparer` lève une exception explicite, `simuler` tient la
dernière valeur sur l'ultime tranche, et un test de non-régression protège le tout. Si vous ne
relisez qu'un test avant de toucher à `simuler`, c'est celui-là.

**(d) Le créneau se calcule par jour de la semaine**, jamais par `pas % 2016`. 2018 commence un
lundi et 2019 un mardi : un profil hebdomadaire transféré à l'aveugle serait décalé d'une journée
entière.

**(e) Les carnets tournent sur une semaine, pas sur l'année.** Trois variables en tête de chaque
carnet :

| variable | valeur | rôle |
|---|---|---|
| `DEBUT` | `181 * R.PAS_JOUR` | 1er juillet, la saison de forte demande |
| `N_PAS` | `7 * R.PAS_JOUR` | fenêtre d'**évaluation** ; mettre `R.PAS_AN` pour l'année |
| `N_ENTR` | `2 * R.PAS_JOUR` | fenêtre d'**ajustement** des rugosités |

⚠️ **Certaines cellules évaluent sur 3 jours et non sur `N_PAS`** (les fonctions locales `evaluer`
et `croiser`). C'est pourquoi le « modèle livré » vaut 0,4740 m dans une cellule et 0,4535 m dans
une autre. Ce ne sont pas deux valeurs contradictoires, ce sont deux fenêtres.

### Les tests

`pytest -q` — **19 tests**, dont 7 synthétiques (qui tournent sans les données) et 12 marqués
`@donnees` (sautés si `data/raw/` est vide). Environ une minute.

Les synthétiques vérifient que l'équation (1) retrouve un produit fabriqué, que la médiane et la
moyenne divergent là où il y a quelque chose à voir, l'identité `RMSE² = biais² + dispersion²`, le
piège du `% 2016`, la détection des basculements de pompe, le fait que la transformation de Huber
vaut l'identité sous son seuil et conserve le signe, et qu'un α croissant rapproche bien la
solution de son point de départ.

Ceux sur données vérifient la partition en zones (92/690), les **trois** lignes de demande par
jonction, les groupes de rugosité, les unités des compteurs, le report d'état entre tranches, la
série de demande sans pas supplémentaire, les formes par catégorie, le recalage qui ne touche que
la zone A+B, le fait qu'`ajuster_reservoir` sépare bien la section de la demande, la section par
demi-cycles, que les bornes par groupe encadrent les coefficients que le groupe porte déjà, et que
la première semaine de 2018 ne porte aucune fuite quand octobre en porte beaucoup.

### Relancer

Les carnets sont versionnés **avec leurs sorties** : ils se lisent sans être exécutés. Les
relancer prend environ cinq minutes chacun, dont l'essentiel dans l'ajustement des rugosités.

L'année complète n'est dans aucun carnet — elle prend une vingtaine de minutes :

```python
from calibration import reseau as R, profils as P, ameliorations as A, diagnostics as G
wn = R.charger_modele(24); noeuds = wn.junction_name_list
niv, pompe = R.charger_niveau(2018).to_numpy(), A.statut_pompe(2018)
D = A.recaler_zone_ab(P.demandes_calibrees(wn, 2018)[0], noeuds, wn, 2018)
S = R.pressions_simulees(D, noeuds, R.PAS_AN, tranche_jours=0.25,
                         pompe=pompe, niveau_mesure=niv, niveau0=float(niv[0]))
print(G.resume(S, R.charger_pressions(2018).to_numpy()))
```

**Après toute modification de code, il faut réexécuter les carnets et relire le texte** : les
affirmations y sont chiffrées presque partout.

---

## 17. Ce qu'il faut retenir, et ce qui reste

**Quatre conclusions :**

1. **Une condition aux limites lue dans un capteur vaut un modèle entier.** Prendre l'état de la
   pompe au débitmètre et réancrer le niveau sur son capteur tient en deux lignes et retire plus
   de dispersion que toute la décomposition des compteurs.
2. **Les gains se composent, et pas en s'additionnant.** Mesurer les améliorations une par une,
   dans un modèle où un autre facteur est encore faux, sous-estime leur valeur commune.
3. **Séparer le biais de la dispersion change les conclusions.** Le recalage sur le bilan de masse
   et l'ajustement des rugosités agissent presque exclusivement sur le biais. Jugés à la RMSE, ils
   paraissent utiles ; jugés sur la dynamique, ils ne le sont pas.
4. **Deux paramètres réputés calibrables ne paient pas ici** : la section du réservoir, identifiable
   mais sans effet une fois le niveau réancré, et les rugosités, dans un réseau où la conduite
   médiane perd quatre millimètres de charge. Dans les deux cas, le diagnostic s'établit par une
   mesure directe et rapide, **sans lancer d'optimisation**.
5. **La fenêtre d'ajustement compte plus que la façon de contraindre le paramètre.** Réglées sur
   une semaine portant 12 m³/h de fuite, les rugosités achètent la fuite avec de la friction et
   rendent absurde le groupe qui porte 705 conduites sur 905. Aucune borne, aucune régularisation
   ne rattrape ce choix : elles n'en annulent que le gain apparent. Avant de contraindre un
   paramètre, regarder ce que la fenêtre contient.

**Ce qui reste à essayer**, dans cet ordre :

1. **descendre le pas de réancrage sous 6 h** sur l'année complète. C'est le plus gros levier
   identifié, il n'a jamais été poussé plus loin, et il ne coûte que du temps de calcul ;
2. **refaire l'ajustement des rugosités sur la première semaine de 2018**, la seule de l'année
   sans fuite, avec les bornes construites sur le fichier. C'est le seul réglage de friction qui
   ne soit pas contaminé ;
3. **estimer la répartition spatiale de la demande** sur les 690 nœuds de la zone A+B. Le bilan de
   masse en contraint la *somme*, pas la répartition, et l'erreur qui subsiste est majoritairement
   dynamique ;
4. **regarder les réducteurs de pression.** Ils fixent la charge de tout l'aval et leur consigne
   est, comme le diamètre du réservoir, un nombre rond dans le fichier. À garder pour la fin : ils
   déplaceraient une sous-zone en bloc, donc ils agiraient sur le **biais** — or après ajustement
   des rugosités le biais annuel n'est plus que de +0,018 m. Il n'y a presque plus rien à y gagner.

**Six limites de fond**, à distinguer des corrections de texte de `CORRECTIONS.md` :

1. **le modèle de demande extrapole d'une zone à l'autre.** Les 82 compteurs sont tous en zone C,
   qui porte 11 % de la demande. Rien ne garantit que les 690 nœuds de A+B aient les mêmes
   habitudes ;
2. **les groupes de rugosité sont formés par diamètre, faute de mieux.** La référence les forme par
   matériau et âge — une information que le fichier ne porte pas. C'est une limite de la
   reproduction, pas un choix, et elle explique en partie les groupes arrêtés sur une borne ;
3. **les six coefficients ne sont pas identifiés.** Deux fenêtres également propres, à trois jours
   d'écart, donnent des coefficients qui vont d'une borne à l'autre (section 14) ; sur sept jours
   le transfert varie de −14 % à +10 % selon la saison, et sur l'année ces écarts se compensent à
   0,3 % près. Ce qu'on publie ici, c'est donc un recentrage, pas une mesure de friction ;
4. **la tendance est estimée par un filtre qui l'atténue** de `sinc(fenêtre/période)` — négligeable
   à l'échelle annuelle, 2,5 % à deux mois ;
5. **le recalage sur le bilan de masse absorbe les fuites dans la demande** (section 13). Le
   chiffre annuel de 0,158 m est donc un **majorant** de l'erreur de modèle, pas une mesure de
   celle-ci ;
6. **tout est mesuré sur une seule année et un seul réseau.** La seconde année disponible n'a servi
   à rien ici.

---

## Annexe — deux écarts assumés par rapport à la référence

À connaître avant de comparer des chiffres. Ils sont écrits dans la docstring de `profils.py` :

1. La référence identifie le nombre de motifs distincts et les compteurs aberrants par
   **classification automatique** des séries. Ici on prend directement les trois types de
   consommateurs que le fichier porte déjà, avec leurs poids par nœud — l'équation (2) demande
   exactement cette structure, et elle est fournie. C'est plus simple et moins général.
2. `saisonnalite` propose la **moyenne** périodique de la référence (valeur par défaut) et, en
   option, la **médiane**, plus robuste : chaque créneau ne dispose que de 52 observations, si
   bien qu'une poignée de journées atypiques suffit à déplacer la moyenne — et le but de l'étape
   est justement de renvoyer ces journées dans le résidu. Le choix se mesure plutôt qu'il ne se
   tranche : sur des compteurs propres les deux coïncident, et ils divergent exactement là où il y
   a quelque chose à voir.

## Annexe — lexique

| terme | sens dans ce dépôt |
|---|---|
| biais | moyenne de l'écart `simulé − mesuré`, en m |
| dispersion | `√(RMSE² − biais²)`, la part qui survit à un recentrage |
| créneau | position dans la semaine, de 0 à 2015, **calculée par jour de la semaine** |
| tranche | bloc de simulation, de quelques heures à quelques jours |
| ancrage | repartir, à chaque tranche, du niveau **lu au capteur** |
| saturation | part des pas où le niveau du réservoir est collé à 0 ou à 4 m |
| aliasing | deux paramètres colinéaires que les données ne peuvent pas séparer |
| aux bornes | groupe dont le coefficient s'est arrêté sur une contrainte : un diagnostic |
| zone C | les 92 jonctions derrière le réservoir — là où sont tous les compteurs |
| zone A+B | les 690 autres, alimentées directement, sans aucune mesure par nœud |
