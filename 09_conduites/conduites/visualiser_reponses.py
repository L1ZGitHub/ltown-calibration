"""Exports statiques des diagnostics temporels sauvegardes, sans navigateur ni inference."""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def exporter(dossier: Path) -> None:
    rapport = json.loads((dossier / "rapport.json").read_text())
    if rapport["etat"] != "complet":
        raise ValueError("Diagnostic temporel incomplet")
    noms = list(rapport["synthese"])
    etiquettes = {"A_bilan": "A / Q", "B_logits": "B", "C_bilan": "C / Q",
                  "C_logits": "C / logits", "dual_signatures": "Dual / signatures"}
    couleurs = dict(zip(noms, ("#3b6fb6", "#e58b23", "#40a36b", "#a35bb8", "#202833")))
    with (dossier / "couvertures.csv").open("w", newline="") as fichier:
        writer = csv.writer(fichier)
        writer.writerow(("conduite", "controle", "genre", "champ", "mode", "empreinte",
                         "couverture_avant", "couverture_apres", "gain",
                         "debit_cible_avant_m3h", "debit_cible_apres_m3h"))
        for cas in rapport["cas"]:
            for nom, diag in cas["diagnostics"].items():
                for mode, budgets in diag["couverture"].items():
                    for fraction, stats in budgets.items():
                        writer.writerow((cas["conduite"], cas["controle"], cas["genre"],
                                         nom, mode, fraction, stats["avant"], stats["apres"],
                                         stats["gain_apres_moins_avant"],
                                         cas["debit_cible_avant_m3h"], cas["debit_cible_apres_m3h"]))
    with np.load(dossier / "diagnostics.npz", allow_pickle=False) as d:
        figure, axes = plt.subplots(3, 2, figsize=(14, 11), sharex=True, sharey=True)
        for axe, debut in zip(axes.flat, range(0, len(rapport["cas"]), 2)):
            cas = rapport["cas"][debut]
            for numero, style in ((debut, "-"), (debut + 1, "--")):
                h = d[f"cas{numero:02d}_heures"]
                ancre = np.datetime64(rapport["cas"][numero]["ancre"], "h")
                x = (h - ancre) / np.timedelta64(1, "h")
                for nom in noms:
                    y = d[f"cas{numero:02d}_{nom}_variation_couverture_cible"][1]
                    lisse = pd.Series(y).rolling(6, min_periods=6).mean().to_numpy()
                    axe.plot(x, lisse, style, color=couleurs[nom], linewidth=1.5,
                             alpha=1. if style == "-" else .45,
                             label=etiquettes[nom] if numero == 0 else None)
            axe.axvline(0, color="#666666", linewidth=.8)
            axe.set_title(f'{cas["conduite"]} - {cas["genre"]} - '
                          f'{cas["debit_cible_apres_m3h"]:.1f} m3/h apres')
            axe.set_ylim(-.03, 1.03)
            axe.grid(alpha=.2)
            axe.set_xlabel("Heures relatives au debut / pseudo-debut")
            axe.set_ylabel("Couverture attendue de la cible")
        handles, labels = axes.flat[0].get_legend_handles_labels()
        figure.legend(handles, labels, loc="lower center", ncol=5, bbox_to_anchor=(.5, .015))
        figure.suptitle(
            "Cartes de variation : empreinte spatiale 20 %\n"
            "Trait plein : fuite ; pointille : controle sans nouveau depart ; moyenne mobile 6 h",
            fontsize=13)
        figure.tight_layout(rect=(0., .055, 1., .94))
        figure.savefig(dossier / "couverture_variations_20pct.png", dpi=140)
        plt.close(figure)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dossier", type=Path, required=True)
    exporter(parser.parse_args().dossier)
