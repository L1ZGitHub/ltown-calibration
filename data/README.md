# Où déposer les mesures

Ce dossier attend le jeu de données L-Town, diffusé publiquement avec le modèle EPANET du
réseau et deux années de mesures. Il n'est pas redistribué ici. Placez dans `data/raw/` les
fichiers suivants :

```
data/raw/
├── L-TOWN.inp                    # modèle EPANET, paramètres nominaux
├── dataset_configuration.yaml    # emplacement des capteurs
├── 2018_SCADA_Pressures.csv      # 33 capteurs de pression, m
├── 2018_SCADA_Flows.csv          # 3 débitmètres (p227, p235, PUMP_1), m³/h
├── 2018_SCADA_Levels.csv         # niveau du réservoir T1, m
├── 2018_SCADA_Demands.csv        # 82 compteurs communicants, L/h
└── 2019_*.csv                    # idem pour la seconde année (facultatif)
```

Si les fichiers vivent ailleurs, inutile de les copier : pointez la variable d'environnement
`LTOWN_DIR` vers leur dossier.

```bash
export LTOWN_DIR=/chemin/vers/les/donnees
```

## Ce qu'il faut savoir sur ces fichiers

* Les CSV sont au format européen : séparateur `;`, **virgule décimale**. `pandas.read_csv` sans
  `sep=";", decimal=","` lit des chaînes de caractères sans prévenir.
* Les compteurs sont en **L/h**, les débitmètres en **m³/h**. Le facteur 1 000 entre les deux est
  la première chose à vérifier quand un bilan de masse ne tombe pas juste.
* Le modèle `L-TOWN.inp` est écrit en unités CMH (m³/h) alors que `wntr` expose tout en SI
  (m³/s). Le module `calibration.reseau` fait la conversion à un seul endroit.
* Les pressions sont arrondies à deux décimales. Le pas de temps est de 5 minutes partout, soit
  105 120 pas pour une année de 365 jours.

## Les lots de scénarios produits par le carnet 6

`data/generes/<nom du lot>/` — ignoré par git, à régénérer en relançant le carnet.

```
lot_janvier/
├── index.csv                 une ligne par scénario : conduite, dates, débit visé et réel, volume
├── temoin/
│   └── pressions.parquet     la même fenêtre sans aucune fuite
├── scenario_0001/
│   ├── pressions.parquet     33 colonnes (les capteurs officiels), index horodaté au pas de 5 min
│   └── verite.json           conduite, nœud percé, dates, débit réellement soutiré, volume perdu
└── ...
```

Les pressions sont **arrondies au centimètre**, comme les mesures réelles de L-Town : un détecteur
entraîné sur des flottants libres apprendrait un signal que les capteurs ne publient pas.

Le débit porté par `verite.json` est celui qui a **réellement** été soutiré par l'émetteur, et non
la consigne demandée : une fuite fait baisser la pression qui la nourrit, donc les deux diffèrent,
d'autant plus que la fuite est grosse.
