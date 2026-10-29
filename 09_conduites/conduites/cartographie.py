"""Page HTML autonome pour comparer quatre cartes horaires sur le meme reseau."""
from html import escape
import json
from pathlib import Path

import numpy as np


def ecrire_cartes(chemin: Path, wn, conduites: list[str], heures: np.ndarray,
                  bruts: dict, lisses: dict, masques: dict, labels: np.ndarray,
                  rapport: dict) -> None:
    geometrie = []
    for p in conduites:
        lien = wn.get_link(p)
        a, b = (wn.get_node(n).coordinates for n in (lien.start_node_name, lien.end_node_name))
        geometrie.append([*a, *b])
    xy = np.array(geometrie, dtype=float)
    xmin, ymin = np.min(xy[:, [0, 2]]), np.min(xy[:, [1, 3]])
    xmax, ymax = np.max(xy[:, [0, 2]]), np.max(xy[:, [1, 3]])
    facteur = min(940 / (xmax - xmin), 630 / (ymax - ymin))
    xy[:, [0, 2]] = 30 + (xy[:, [0, 2]] - xmin) * facteur
    xy[:, [1, 3]] = 660 - (xy[:, [1, 3]] - ymin) * facteur
    lignes, fonds, cercles = [], [], []
    for i, (x1, y1, x2, y2) in enumerate(xy):
        attributs = f'x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" y2="{y2:.2f}"'
        lignes.append(f'<line class="pipe" data-i="{i}" {attributs}><title>'
                      f'{escape(conduites[i])}</title></line>')
        fonds.append(f'<line class="zone" {attributs}/>')
        cercles.append(f'<circle class="truth" cx="{(x1+x2)/2:.2f}" cy="{(y1+y2)/2:.2f}" '
                       f'r="5"><title>{escape(conduites[i])}</title></circle>')
    donnees = {"heures": heures.astype(str).tolist(), "noms": conduites,
               "labels": [np.flatnonzero(y).tolist() for y in labels],
               "champs": {}, "masques": {
                   k: [np.flatnonzero(m).tolist() for m in v] for k, v in masques.items()}}
    panneaux = []
    titres = {"A_bilan": "A : bilan Q1-Q2 positif (m3/h)",
              "B_logits": "B : logits de suspicion",
              "C_bilan": "C : bilan Q1-Q2 positif (m3/h)",
              "C_logits": "C : logits de suspicion"}
    tableau = []
    for nom in bruts:
        lo, hi = np.quantile(np.concatenate((bruts[nom].ravel(), lisses[nom].ravel())), [.01, .99])
        amplitude = max(float(hi - lo), 1e-12)
        donnees["champs"][nom] = {}
        for mode, v in (("brut", bruts[nom]), ("lisse", lisses[nom])):
            donnees["champs"][nom][mode] = {
                "couleurs": np.round(np.clip((v - lo) / amplitude, 0, 1) * 255).astype(int).tolist(),
                "valeurs": np.round(v, 4).tolist()}
        panneaux.append(
            f'<section id="{nom}"><h2>{escape(titres[nom])}</h2>'
            f'<p>Echelle fixe : {lo:.3g} a {hi:.3g} (quantiles 1-99 %, couleurs saturees au-dela).</p>'
            f'<svg role="img" aria-label="{escape(titres[nom])}" viewBox="0 0 1000 700">'
            '<g aria-hidden="true">'
            + "".join(fonds + lignes + cercles) + '</g></svg><p class="status"></p></section>')
        modes = rapport["comparaisons"][nom]
        b, l = (modes[m]["5%"] for m in ("brut", "lisse"))
        tableau.append(f'<tr><td>{escape(nom)}</td><td>{b["rappel_300m"]:.1%}</td>'
                       f'<td>{l["rappel_300m"]:.1%}</td>'
                       f'<td>{l["controle_carte_saine_rappel_300m"]:.1%}</td>'
                       f'<td>{l["fraction_reseau_a_300m"]["moyenne"]:.1%}</td>'
                       f'<td>{l["stabilite_jaccard_longueur"]:.1%}</td></tr>')
    donnees_json = json.dumps(donnees, separators=(",", ":"), allow_nan=False).replace("<", "\\u003c")
    page = """<!doctype html>
<html lang="fr"><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Cartes A/B/C - six jours 2018</title>
<style>
body{font:15px system-ui,sans-serif;margin:20px;background:#f3f5f8;color:#182334}
h1{font-size:25px}h2{font-size:17px;margin:6px 0}p{max-width:1150px}
.controls{position:sticky;top:0;background:#fff;padding:12px;z-index:1;border:1px solid #b9c5d4;
display:flex;gap:14px;align-items:center;flex-wrap:wrap}
.grid{display:grid;grid-template-columns:repeat(2,minmax(320px,1fr));gap:12px;margin-top:12px}
section{background:#fff;border:1px solid #d6dde8;padding:10px}section p{font-size:12px;margin:4px}
svg{width:100%;max-height:520px;background:#fff}.pipe{stroke-width:1.5}
.zone{stroke:#111;stroke-width:4;display:none}.truth{fill:none;stroke:#00b8b0;stroke-width:2.5;display:none}
table{border-collapse:collapse;background:white}td,th{padding:7px 12px;border:1px solid #ccd4de}
@media(max-width:750px){.grid{grid-template-columns:1fr}}
.gradient{display:inline-block;width:120px;height:12px;background:linear-gradient(to right,hsl(240 85% 48%),hsl(120 85% 48%),hsl(0 85% 48%))}
</style>
<h1>Cartes de suspicion A / B / C - exploration 2018</h1>
<p>Six jours calendaires fixes, 24 fenetres d'une heure par jour, finissant a :55.
Pas d'entrainement ni de nouvelle simulation. C conserve ses deux tetes separees.
Le bilan positif de A/C est un indice derive, pas une estimation de fuite validee.
Les logits ne sont pas des probabilites.</p>
<div class="controls">
<label>Jour <select id="jour">__DATES__</select></label>
<label>Heure <input id="heure" type="range" min="0" max="23" value="12"></label>
<strong id="horodatage"></strong>
<label>Champ <select id="mode"><option value="lisse">Lisse a 300 m</option>
<option value="brut">Brut</option></select></label>
<label><input id="zones" type="checkbox" checked> Budget 5 % (bord noir)</label>
<label><input id="verite" type="checkbox" checked> Fuites publiees (cercles turquoise)</label>
</div>
<p><span class="gradient"></span> Bleu : indice faible ; rouge : indice eleve.
Echelle fixe dans le temps pour chaque methode, commune au brut et au lisse ;
les couleurs ne comparent pas les amplitudes entre modeles. Survol : conduite et valeur.
Une zone est imposee meme pendant le jour sain : ce n'est pas une alerte.</p>
<div class="grid">__PANNEAUX__</div>
<h2>Couverture moyenne a 300 m, budget maximal de 5 % du reseau</h2>
<p>Moyenne de la fraction des fuites actives couvertes par heure positive, pas un taux
d'evenements detectes. Controle sain : carte du 1er janvier figee pour tous les jours.
Une stabilite elevee peut etre un biais spatial permanent, pas un avantage.</p>
<table><thead><tr><th>Champ</th><th>Brut</th><th>Lisse</th>
<th>Carte saine figee (lisse)</th><th>Reseau couvert a 300 m (lisse)</th>
<th>Stabilite horaire (lisse)</th></tr></thead>
<tbody>__TABLEAU__</tbody></table>
<p>Hasard sous le meme budget : __HASARD__. Conduites entieres selectionnees sous budget :
les longueurs effectives sont indiquees par carte, sans pretendre une egalite exacte.
Le hasard disperse les conduites : sa couverture ne constitue pas une comparaison
a compacite ou empreinte spatiale egale. La fraction de reseau a 300 m mesure
cette empreinte en longueur de conduites, pas en surface geographique.
Le lissage est une moyenne ponderee par longueur, normalisee localement, avec un noyau
exp(-distance/300) tronque a 300 m. La distance suit les conduites, pas la carte.</p>
<p>Limites : 2018 retrospectif deja utilise pour calibration ; six jours seulement ;
aucun seuil regle, aucune politique d'intervention, aucune superiorite etablie sur le
pipeline physique. Les fuites publiees servent uniquement a l'evaluation et aux marqueurs,
jamais au calcul des cartes.</p>
<script>
const d=__DATA__;
const longueurs=__LONGUEURS__;
const el=id=>document.getElementById(id);
const refs=Object.keys(d.champs).map(n=>({n,root:el(n),
  pipes:[...el(n).querySelectorAll('.pipe')],zones:[...el(n).querySelectorAll('.zone')],
  truth:[...el(n).querySelectorAll('.truth')]}));
function update(){
 const t=Number(el('jour').value)*24+Number(el('heure').value),mode=el('mode').value;
 el('horodatage').textContent=d.heures[t].replace('T',' ')+' h';
 const actifs=new Set(d.labels[t]);
 for(const r of refs){
  const champ=d.champs[r.n][mode],selection=new Set(d.masques[r.n+'_'+mode][t]);
  r.pipes.forEach((p,i)=>{
   p.style.stroke=`hsl(${240*(1-champ.couleurs[t][i]/255)} 85% 48%)`;
   p.querySelector('title').textContent=d.noms[i]+': '+champ.valeurs[t][i];
   r.zones[i].style.display=el('zones').checked&&selection.has(i)?'block':'none';
   r.truth[i].style.display=el('verite').checked&&actifs.has(i)?'block':'none';
  });
  const metres=[...selection].reduce((a,i)=>a+longueurs[i],0);
  r.root.querySelector('.status').textContent=selection.size+' conduites, '+metres.toFixed(0)
   +' m suspects ; '+actifs.size+' fuites actives publiees.';
 }
}
['jour','heure','mode','zones','verite'].forEach(id=>el(id).addEventListener('input',update));
el('jour').value='3';update();
</script></html>
"""
    dates = "".join(f'<option value="{i}">{escape(date)}</option>'
                    for i, date in enumerate(rapport["dates_fixees_avant_inference"]))
    remplacements = {
        "__DATES__": dates, "__PANNEAUX__": "".join(panneaux), "__TABLEAU__": "".join(tableau),
        "__HASARD__": f'{rapport["hasard"]["5%"]["rappel_300m"]:.1%}',
        "__DATA__": donnees_json,
        "__LONGUEURS__": json.dumps([wn.get_link(p).length for p in conduites])}
    for cle, valeur in remplacements.items():
        page = page.replace(cle, valeur)
    chemin.write_text(page)
