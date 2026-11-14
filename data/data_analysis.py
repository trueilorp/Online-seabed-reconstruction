"""Exploratory analysis of a sonar point-cloud CSV (set CSV_PATH below): statistics, classes, extent and a
summary figure with top/side views and histograms.

Usage (from the repository root):
    python data/data_analysis.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

CSV_PATH = "data/sonar/Terrain_unity.csv"
OUTPUT_PREFIX = "output/analysis"


def analyse(csv_path: str, output_prefix: str = "analysis"):
    df = pd.read_csv(csv_path)

    print("=" * 60)
    print(f"File: {csv_path}")
    print("=" * 60)
    print(f"\nNumber of rows (points): {len(df)}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst rows:\n{df.head()}")
    print(f"\nDescriptive statistics:\n{df.describe()}")
    print(f"\nMissing values per column:\n{df.isna().sum()}")

    # plain x,y,z CSVs and SonarView/Omniscan exports (local easting/northing/altitude, in metres)
    lower = {c.lower(): c for c in df.columns}
    col_x = lower.get("x") or lower.get("easting (local m)")
    col_z = lower.get("z") or lower.get("northing (local m)")
    col_y = lower.get("y") or lower.get("altitude (m)")
    col_intensity = lower.get("intensity") or lower.get("intensita") or lower.get("power (db)")

    if not (col_x and col_y and col_z):
        print("\n[WARNING] No recognisable x,y,z (or easting/northing/altitude) columns.")
        print("Check the column names above and adapt the script if needed.")
        return

    print(f"\nCoordinates: X={col_x}, Y (vertical)={col_y}, Z={col_z}, intensity={col_intensity}")

    col_classification = lower.get("classification")
    if col_classification:
        print(f"\nPoints per value of '{col_classification}':")
        counts = df[col_classification].value_counts().sort_index()
        for value, count in counts.items():
            print(f"  class {value}: {count} points ({100 * count / len(df):.1f}%)")
        # mean power and altitude per class: hints at which class is seabed and which is water column
        compare_columns = [c for c in [col_intensity, col_y] if c]
        if compare_columns:
            print(f"\nMean of {compare_columns} per class:")
            print(df.groupby(col_classification)[compare_columns].mean())

    x, y, z = df[col_x].to_numpy(), df[col_y].to_numpy(), df[col_z].to_numpy()

    print("\nSpatial extent:")
    print(f"  X: [{x.min():.3f}, {x.max():.3f}]  (range: {x.max()-x.min():.3f})")
    print(f"  Y: [{y.min():.3f}, {y.max():.3f}]  (range: {y.max()-y.min():.3f})")
    print(f"  Z: [{z.min():.3f}, {z.max():.3f}]  (range: {z.max()-z.min():.3f})")

    area_xz = (x.max() - x.min()) * (z.max() - z.min())
    if area_xz > 0:
        print(f"\nApproximate density: {len(df) / area_xz:.1f} points per unit^2 (X-Z plane)")

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle(f"{csv_path}", fontsize=14)

    sc = axes[0, 0].scatter(x, z, s=1, c=y, cmap="viridis")
    axes[0, 0].set_title("Top view (colour = height Y)")
    axes[0, 0].set_xlabel("X")
    axes[0, 0].set_ylabel("Z")
    axes[0, 0].set_aspect("equal", adjustable="box")
    plt.colorbar(sc, ax=axes[0, 0], label="Y")

    if col_intensity:
        sc2 = axes[0, 1].scatter(x, z, s=1, c=df[col_intensity].to_numpy(), cmap="plasma")
        axes[0, 1].set_title(f"Top view (colour = {col_intensity})")
        axes[0, 1].set_xlabel("X")
        axes[0, 1].set_ylabel("Z")
        axes[0, 1].set_aspect("equal", adjustable="box")
        plt.colorbar(sc2, ax=axes[0, 1], label=col_intensity)
    else:
        axes[0, 1].axis("off")
        axes[0, 1].set_title("(no intensity column)")

    axes[0, 2].scatter(x, y, s=1, alpha=0.5)
    axes[0, 2].set_title("Side view (X-Y)")
    axes[0, 2].set_xlabel("X")
    axes[0, 2].set_ylabel("Y")

    axes[0, 3].hist(y, bins=60, color="steelblue")
    axes[0, 3].set_title("Height distribution (Y)")
    axes[0, 3].set_xlabel("Y")
    axes[0, 3].set_ylabel("Point count")

    axes[1, 0].hist(x, bins=60, color="darkorange")
    axes[1, 0].set_title("X distribution")

    axes[1, 1].hist(z, bins=60, color="seagreen")
    axes[1, 1].set_title("Z distribution")

    if col_intensity:
        axes[1, 2].hist(df[col_intensity].to_numpy(), bins=60, color="purple")
        axes[1, 2].set_title(f"{col_intensity} distribution")
    else:
        h = axes[1, 2].hist2d(x, z, bins=80, cmap="inferno")
        axes[1, 2].set_title("Point density (top view)")
        plt.colorbar(h[3], ax=axes[1, 2], label="points per cell")

    h2 = axes[1, 3].hist2d(x, z, bins=80, cmap="inferno")
    axes[1, 3].set_title("Point density (top view)")
    axes[1, 3].set_aspect("equal", adjustable="box")
    plt.colorbar(h2[3], ax=axes[1, 3], label="points per cell")

    plt.tight_layout()
    output_path = os.path.join(output_prefix, f"{os.path.basename(CSV_PATH)}_analysis.png")
    plt.savefig(output_path, dpi=150)
    print(f"\nSummary figure saved to: {output_path}")


if __name__ == "__main__":
    analyse(CSV_PATH, OUTPUT_PREFIX)
