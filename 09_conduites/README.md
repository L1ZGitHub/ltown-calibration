# 09 · Les conduites coupées en deux : le prototype, en grand, sur le vrai 2018

Chaque conduite de L-Town est coupée en son milieu ; la fuite est posée sur ce milieu. Un réseau de
neurones sur le graphe des conduites (blocs de type GPS écrits à la main : GINE local + attention
globale, sur une fenêtre d'une heure) note chaque conduite « fuit / ne fuit pas ». Appris sur des
épisodes simulés de 6 h (763 fabriqués, dont 640 d'apprentissage), il est noté sur toute l'année
2018 réelle et comparé au modèle 08c.

## Ce qu'il y a dans le dossier

Trois carnets, à lire dans l'ordre. Le modèle présenté est C (débits + note de fuite) ; B (note
seule), le 08c et le hasard servent de comparaison.

| | |
|---|---|
| `notebook/1_le_modele.ipynb` | les entrées, la cible (débits q1, q2 de chaque moitié, fuite = q1 − q2), le modèle, les données |
| `notebook/2_le_modele_suit_il_la_cible.ipynb` | sur des épisodes de test simulés : débits et bilan contre la vérité, top-k, par taille de fuite |
| `notebook/3_les_cartes_et_les_limites.ipynb` | carte de fuite et carte des débits, le vrai 2018, d'où viennent les erreurs, cartes brutes ou lissées, tour 2 |
| `notebook/annexe_tout_refaire.ipynb` | tout refaire : le jeu, les entrées de 2018, l'apprentissage, les notes |
| `modeles/` | les quatre modèles appris au tour 1 (`B`, `C`, `B_sans_residu_dual`, `B_instantane`) |
| `sorties/apprentissage/` | tour 1 : historiques, test simulé, cartes 2018 de B, C et du 08c, tableau des notes |
| `sorties/tour2/` | tour 2 : tableaux des notes (r1, r2, 08c enrichi), historiques, test simulé |
| `sorties/rejeu/` | les modèles passés à chaque pas des épisodes de test (ce que lisent les carnets 2 et 3) |
| `sorties/exemples/` | deux épisodes d'exemple (avec et sans fuite) |
| `conduites/` | le code |

## Lancer

Les carnets 1 à 3 se lancent tels quels : tout ce qu'ils lisent est livré. Il faut les mesures
de BattLeDIM (voir [`../data/README.md`](../data/README.md)) **et le fichier des fuites de 2018**,
`2018_Leakages.csv`, dans le même dossier : c'est lui qui sert à noter les cartes.

```bash
pip install -r ../02_calibration_modele/requirements.txt -r requirements.txt
cd notebook && jupyter lab
```

## Tout refaire

Le carnet `annexe_tout_refaire`, avec `MODE = "essai"` (une dizaine de minutes, vérifie que tout tourne) puis
`MODE = "complet"`. Les mêmes étapes en ligne de commande, depuis ce dossier :

```bash
python -m conduites.grand_jeu --sortie generes/complet/jeu --processus 8
python -m conduites.entrees2018_brutes --sortie generes/complet/annee --processus 8
python -m conduites.apprendre_gpu --jeux generes/complet/jeu --annee generes/complet/annee \
    --sortie generes/complet/apprentissage --variantes B,C --minutes 105
cp sorties/apprentissage/cartes_08c_2018.npz generes/complet/apprentissage/
python -m conduites.comparer2018 --sortie generes/complet/apprentissage
```

Ordres de grandeur : un épisode ≈ 20 s par cœur (800 prévus) ; un jour de 2018 ≈ 35 s par cœur ;
l'apprentissage est réglé en minutes (`--minutes`) et un GPU est conseillé.

Avec les réglages stricts du prototype, un épisode ou un jour de 2018 où EPANET ne converge pas est
abandonné (listé dans les manifestes). Cela dépend du processeur : sur x86, environ 1 épisode sur 8
et aucun jour de 2018 ; sur un serveur ARM, beaucoup plus
(3 jours de 2018 sur 4 à l'essai ; un Mac M1 à M4 est sans doute dans le même cas, non vérifié).
Préférer une machine x86 pour tout refaire.

## Résultat (une graine)

Cartes brutes (la note telle que le modèle la donne), toute l'année 2018, top-5 à 300 m :

| carte | toutes les heures | par fuite | hors zone C |
|---|---|---|---|
| C, note de fuite | **0,33** | 0,55 | 0,24 |
| C, bilan q1 − q2 | 0,16 | 0,39 | 0,16 |
| B, note de fuite | 0,23 | **0,56** | **0,26** |
| 08c présence | 0,25 | 0,40 | 0,13 |
| hasard | 0,15 | 0,20 | 0,15 |

Cartes lissées sur 300 m (la règle écrite avant le tour 2) :

| carte | empreinte 20 % | top-5 à 300 m |
|---|---|---|
| prototype B | **0,64** | **0,32** |
| prototype C | 0,57 | 0,22 |
| 08c présence | 0,38 | 0,21 |
| B sans « pression − modèle calibré » ni dual | 0,25 | 0,10 |
| B qui ne voit qu'un pas | 0,56 | 0,26 |
| hasard | 0,15 | 0,15 |

Le gain vient presque entièrement des entrées résidu de pression + dual ; la fenêtre d'une heure
ajoute un peu. Hors zone C, le top-5 de B tombe à 0,06 (hasard 0,15) : la zone C occupe la tête de la
carte presque toute l'année. Le lissage efface les pics nets : pour C, il fait reculer 8 fuites sur
14 et n'avance guère que p257, la fuite de la zone C qui coule 98 % des heures (carnet 3).

Tour 2 (carnet 3) : choisir l'étape sur le classement de validation, ajouter la conservation des
débits, apprendre sur un jeu plus varié (v2) ou donner le résidu et le dual au 08c ne bat pas B selon
la règle écrite avant (mieux sur l'empreinte et le top-5, toutes heures et hors zone C). Le 08c enrichi
progresse en empreinte (0,38 → 0,49).
