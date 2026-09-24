import numpy as np
import pandas as pd
import open3d as o3d
import matplotlib.pyplot as plt

# --- Imposta qui i percorsi ---
CSV_PATH = "data/sonar/Terrain_unity.csv"
OBJ_PATH = "data/sonar/Terrain_ground_truth.obj"
OUTPUT_PNG = "output/confronto_unity_ground_truth.png"


def carica_point_cloud_csv(percorso: str):
    df = pd.read_csv(percorso)
    colonne_lower = {c.lower(): c for c in df.columns}

    col_x = colonne_lower.get("x")
    col_y = colonne_lower.get("y")
    col_z = colonne_lower.get("z")

    if not (col_x and col_y and col_z):
        raise ValueError(f"Non trovo colonne x,y,z nel CSV. Colonne presenti: {list(df.columns)}")

    punti = df[[col_x, col_y, col_z]].to_numpy(dtype=np.float64)
    return punti, df


def stampa_bbox(nome: str, min_xyz, max_xyz):
    print(f"\n{nome}:")
    print(f"  X: [{min_xyz[0]:.3f}, {max_xyz[0]:.3f}]  (estensione: {max_xyz[0]-min_xyz[0]:.3f})")
    print(f"  Y: [{min_xyz[1]:.3f}, {max_xyz[1]:.3f}]  (estensione: {max_xyz[1]-min_xyz[1]:.3f})")
    print(f"  Z: [{min_xyz[2]:.3f}, {max_xyz[2]:.3f}]  (estensione: {max_xyz[2]-min_xyz[2]:.3f})")


def main():
    print("=" * 60)
    print("Point cloud (CSV Unity)")
    print("=" * 60)
    punti, df = carica_point_cloud_csv(CSV_PATH)
    print(f"Numero di punti: {len(punti)}")
    if len(punti) < 5000:
        print("[ATTENZIONE] Pochi punti per una ricostruzione Poisson sensata "
              "(di solito servono almeno alcune migliaia-decine di migliaia). "
              "Valuta di registrare altri scan in Unity prima di procedere.")

    min_pc = punti.min(axis=0)
    max_pc = punti.max(axis=0)
    stampa_bbox("Bounding box point cloud", min_pc, max_pc)

    print("\n" + "=" * 60)
    print("Ground truth (Terrain .obj)")
    print("=" * 60)
    mesh = o3d.io.read_triangle_mesh(OBJ_PATH)
    n_vertici = len(mesh.vertices)
    n_triangoli = len(mesh.triangles)
    print(f"Vertici: {n_vertici}")
    print(f"Triangoli: {n_triangoli}")

    if n_vertici == 0:
        print("[ATTENZIONE] Il file .obj non contiene vertici -- controlla il percorso "
              "o l'esportazione da Unity.")
        return

    vertici_mesh = np.asarray(mesh.vertices)
    min_gt = vertici_mesh.min(axis=0)
    max_gt = vertici_mesh.max(axis=0)
    stampa_bbox("Bounding box ground truth", min_gt, max_gt)

    # --- Controllo di allineamento/copertura ---
    print("\n" + "=" * 60)
    print("Controllo allineamento e copertura")
    print("=" * 60)

    # Sovrapposizione dei bounding box sul piano orizzontale (X, Z)
    overlap_min_x = max(min_pc[0], min_gt[0])
    overlap_max_x = min(max_pc[0], max_gt[0])
    overlap_min_z = max(min_pc[2], min_gt[2])
    overlap_max_z = min(max_pc[2], max_gt[2])

    if overlap_max_x <= overlap_min_x or overlap_max_z <= overlap_min_z:
        print("[ATTENZIONE] NESSUNA sovrapposizione tra la point cloud e il ground truth "
              "sul piano orizzontale (X,Z)! Molto probabile un disallineamento di "
              "coordinate (sistemi di riferimento diversi, o transform sbagliato).")
    else:
        area_pc = (max_pc[0] - min_pc[0]) * (max_pc[2] - min_pc[2])
        area_gt = (max_gt[0] - min_gt[0]) * (max_gt[2] - min_gt[2])
        area_overlap = (overlap_max_x - overlap_min_x) * (overlap_max_z - overlap_min_z)
        pct_gt_coperta = 100 * area_overlap / area_gt if area_gt > 0 else 0
        pct_pc_dentro_gt = 100 * area_overlap / area_pc if area_pc > 0 else 0

        print(f"Area bounding box point cloud (X-Z): {area_pc:.2f}")
        print(f"Area bounding box ground truth (X-Z): {area_gt:.2f}")
        print(f"Area di sovrapposizione: {area_overlap:.2f}")
        print(f"  -> {pct_gt_coperta:.1f}% dell'area del Terrain e' coperta dalla point cloud")
        print(f"  -> {pct_pc_dentro_gt:.1f}% della point cloud cade dentro l'area del Terrain")

        if pct_pc_dentro_gt < 50:
            print("[ATTENZIONE] Meno della meta' della point cloud cade dentro l'area del "
                  "Terrain -- controlla l'allineamento delle coordinate prima di fidarti "
                  "del confronto con ground truth.")

    # Controllo separato sull'asse verticale (Y = altezza)
    if min_pc[1] > max_gt[1] or max_pc[1] < min_gt[1]:
        print("[ATTENZIONE] Il range di altezza (Y) della point cloud non si sovrappone "
              "affatto con quello del Terrain -- sospetto disallineamento verticale "
              "(es. altezza del sensore non scorporata, o origine diversa).")

    # --- Grafico vista dall'alto ---
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.scatter(punti[:, 0], punti[:, 2], s=2, alpha=0.5, label="Point cloud (Unity)", color="steelblue")

    # Contorno del ground truth (bounding box, vista dall'alto)
    rettangolo_x = [min_gt[0], max_gt[0], max_gt[0], min_gt[0], min_gt[0]]
    rettangolo_z = [min_gt[2], min_gt[2], max_gt[2], max_gt[2], min_gt[2]]
    ax.plot(rettangolo_x, rettangolo_z, color="darkorange", linewidth=2,
            label="Estensione Terrain (ground truth)")

    ax.set_xlabel("X")
    ax.set_ylabel("Z")
    ax.set_title("Vista dall'alto: point cloud vs area Terrain")
    ax.set_aspect("equal", adjustable="box")
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_PNG, dpi=150)
    print(f"\nGrafico salvato in: {OUTPUT_PNG}")

if __name__ == "__main__":
    main()