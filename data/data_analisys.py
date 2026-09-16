"""
Analisi esplorativa di un CSV di point cloud (sonar/LiDAR)
=============================================================

Uso (dalla root del progetto):
    python data/data_analisys.py

Dipendenze:
    pip install pandas numpy matplotlib
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --- Imposta qui i parametri, invece che da riga di comando ---
PERCORSO_CSV = "data/sonar/ShallowShorelineSurveyPoints.csv"   # percorso al file CSV da analizzare
OUTPUT_PREFIX = "output/analisi"                   # prefisso per il file del grafico salvato


def analizza(percorso_csv: str, output_prefix: str = "analisi"):
    df = pd.read_csv(percorso_csv)

    print("=" * 60)
    print(f"File: {percorso_csv}")
    print("=" * 60)

    # --- Info di base ---
    print(f"\nNumero di righe (punti): {len(df)}")
    print(f"Colonne trovate: {list(df.columns)}")
    print(f"\nPrime righe:\n{df.head()}")
    print(f"\nStatistiche descrittive:\n{df.describe()}")
    print(f"\nValori mancanti per colonna:\n{df.isna().sum()}")

    # Individua le colonne x,y,z in modo case-insensitive.
    # Riconosce sia CSV "semplici" (x,y,z) sia export reali stile SonarView/Omniscan
    # (easting/northing/altitude locali, già in metri e già relativi a un'origine locale).
    colonne_lower = {c.lower(): c for c in df.columns}

    col_x = colonne_lower.get("x") or colonne_lower.get("easting (local m)")
    col_z = colonne_lower.get("z") or colonne_lower.get("northing (local m)")
    col_y = colonne_lower.get("y") or colonne_lower.get("altitude (m)")
    col_intensity = (
        colonne_lower.get("intensity")
        or colonne_lower.get("intensita")
        or colonne_lower.get("power (db)")
    )

    if not (col_x and col_y and col_z):
        print("\n[ATTENZIONE] Non ho trovato colonne x,y,z (o easting/northing/altitude) riconoscibili.")
        print("Controlla i nomi delle colonne sopra e adatta lo script se necessario.")
        return

    print(f"\nUso come coordinate: X={col_x}, Y (verticale)={col_y}, Z={col_z}, intensità={col_intensity}")

    # --- Analisi della colonna 'classification', se presente ---
    col_classification = colonne_lower.get("classification")
    if col_classification:
        print(f"\nConteggio punti per valore di '{col_classification}':")
        conteggi = df[col_classification].value_counts().sort_index()
        for valore, conteggio in conteggi.items():
            percentuale = 100 * conteggio / len(df)
            print(f"  classe {valore}: {conteggio} punti ({percentuale:.1f}%)")

        # Statistiche incrociate: per ogni classe, media di power(dB) e altitude,
        # utile per intuire quale classe è "fondale" e quale "colonna d'acqua"
        colonne_confronto = [c for c in [col_intensity, col_y] if c]
        if colonne_confronto:
            print(f"\nMedia di {colonne_confronto} per classe (indizio su cosa rappresenta ogni classe):")
            print(df.groupby(col_classification)[colonne_confronto].mean())

    x, y, z = df[col_x].to_numpy(), df[col_y].to_numpy(), df[col_z].to_numpy()

    # --- Estensione spaziale (bounding box) ---
    print(f"\nEstensione spaziale:")
    print(f"  X: [{x.min():.3f}, {x.max():.3f}]  (range: {x.max()-x.min():.3f})")
    print(f"  Y: [{y.min():.3f}, {y.max():.3f}]  (range: {y.max()-y.min():.3f})")
    print(f"  Z: [{z.min():.3f}, {z.max():.3f}]  (range: {z.max()-z.min():.3f})")

    # --- Densità approssimativa (punti per unità di area, sul piano X-Z) ---
    area_xz = (x.max() - x.min()) * (z.max() - z.min())
    if area_xz > 0:
        print(f"\nDensità approssimativa: {len(df) / area_xz:.1f} punti per unità^2 (piano X-Z)")

    # --- Figura riassuntiva con più pannelli ---
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"Analisi esplorativa: {percorso_csv}", fontsize=14)

    # Vista dall'alto (X-Z, tipica per un fondale con Y=verticale)
    sc = axes[0, 0].scatter(x, z, s=1, c=y, cmap="viridis")
    axes[0, 0].set_title("Vista dall'alto (colore = altezza Y)")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Z")
    axes[0, 0].set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=axes[0, 0], label="Y")

    # Vista dall'alto colorata per intensità/power -- utile per capire DOVE lungo
    # il percorso ci sono zone di eco anomalo (rumore/multipath), indipendentemente
    # dalla profondità, e correlarle con eventuali artefatti nella mesh ricostruita
    if col_intensity:
        sc2 = axes[0, 1].scatter(x, z, s=1, c=df[col_intensity].to_numpy(), cmap="plasma")
        axes[0, 1].set_title(f"Vista dall'alto (colore = {col_intensity})")
        axes[0, 1].set_xlabel("X")
        axes[0, 1].set_ylabel("Z")
        axes[0, 1].set_aspect("equal", adjustable="box")
        plt.colorbar(sc2, ax=axes[0, 1], label=col_intensity)
    else:
        axes[0, 1].axis("off")
        axes[0, 1].set_title("(colonna intensità non trovata)")

    # Vista laterale (X-Y)
    axes[0, 2].scatter(x, y, s=1, alpha=0.5)
    axes[0, 2].set_title("Vista laterale (X-Y)")
    axes[0, 2].set_xlabel("X")
    axes[0, 2].set_ylabel("Y")

    # Istogramma altezze (Y)
    axes[0, 3].hist(y, bins=60, color="steelblue")
    axes[0, 3].set_title("Distribuzione altezze (Y)")
    axes[0, 3].set_xlabel("Y")
    axes[0, 3].set_ylabel("Conteggio punti")

    # Istogramma X
    axes[1, 0].hist(x, bins=60, color="darkorange")
    axes[1, 0].set_title("Distribuzione coordinata X")

    # Istogramma Z
    axes[1, 1].hist(z, bins=60, color="seagreen")
    axes[1, 1].set_title("Distribuzione coordinata Z")

    # Intensità, se presente; altrimenti densità locale (istogramma 2D)
    if col_intensity:
        axes[1, 2].hist(df[col_intensity].to_numpy(), bins=60, color="purple")
        axes[1, 2].set_title(f"Distribuzione {col_intensity}")
    else:
        h = axes[1, 2].hist2d(x, z, bins=80, cmap="inferno")
        axes[1, 2].set_title("Densità punti (vista dall'alto)")
        plt.colorbar(h[3], ax=axes[1, 2], label="punti per cella")

    # Densità punti (vista dall'alto) -- sempre utile, indipendentemente dall'intensità
    h2 = axes[1, 3].hist2d(x, z, bins=80, cmap="inferno")
    axes[1, 3].set_title("Densità punti (vista dall'alto)")
    axes[1, 3].set_aspect("equal", adjustable="box")
    plt.colorbar(h2[3], ax=axes[1, 3], label="punti per cella")

    plt.tight_layout()
    output_path = f"{output_prefix}_riepilogo.png"
    plt.savefig(output_path, dpi=150)
    print(f"\nGrafico riassuntivo salvato in: {output_path}")


if __name__ == "__main__":
    analizza(PERCORSO_CSV, OUTPUT_PREFIX)