"""
Pipeline classica di ricostruzione 3D da point cloud (sonar/LiDAR)
====================================================================

Input:  un file CSV con colonne x,y,z (,intensity opzionale) prodotto da
        PointCloudRecorder.cs nel simulatore Unity.
Output: una mesh .ply/.obj ricostruita, più eventuali metriche di
        confronto con un ground truth (se disponibile).

Dipendenze:
    pip install open3d numpy pandas

Uso:
    python pipeline_classica_ricostruzione.py --input scan_001.csv --metodo poisson
    python pipeline_classica_ricostruzione.py --input scan_001.csv --metodo bpa
"""

import numpy as np
import open3d as o3d
import pandas as pd

import os
import time

# --- Imposta qui i parametri, invece che da riga di comando ---
INPUT_CSV = "../dati/sonar/OS3D_Max_Range_Test.csv"      # percorso al CSV da ricostruire
METODO = "poisson"                          # "poisson" oppure "bpa"
OUTPUT_MESH = "../output/mesh/mesh_ricostruita.ply"        # percorso file mesh di output
GROUND_TRUTH_MESH = None                    # (opzionale) percorso a mesh .obj/.ply di ground truth, o None
ESCLUDI_CLASSI = [2]                        # valori di 'classification' da escludere (provvisorio); [] per non escludere nulla
ALTITUDE_MIN = None                         # tieni solo punti con altitude >= questo valore (None = nessun limite)
ALTITUDE_MAX = None                         # tieni solo punti con altitude <= questo valore (None = nessun limite)
NB_NEIGHBORS = 20                           # filtro outlier: quanti vicini considerare per punto
STD_RATIO = 2.0                             # filtro outlier: soglia in deviazioni standard (più basso = filtro più aggressivo)
ESCLUDI_ZONA_X = None                       # tuple (min, max) per escludere una zona rettangolare X (es. sosta barca)
ESCLUDI_ZONA_Z = None                       # tuple (min, max) per escludere una zona rettangolare Z (es. sosta barca)
# Esempio per escludere la zona di sosta identificata su OS3D_Max_Range_Test.csv:
#   ESCLUDI_ZONA_X = (-420, -380)
#   ESCLUDI_ZONA_Z = (5590, 5650)
# Esempio per isolare solo la zona costiera/poco profonda del transetto (dai dati OS3D_Max_Range_Test):
#   ALTITUDE_MIN = -50   # tiene solo i punti più vicini alla superficie/costa
#   ALTITUDE_MAX = None
VOXEL_SIZE = 0.5                            # dimensione voxel per downsampling in metri (fisso, invece che stimato automaticamente)
METRICHE_CSV = "../output/metriche_ricostruzione.csv" # file dove accumulare le metriche di ogni run (in append)
CALCOLA_AUTOINTERSEZIONI = False            # True = controllo completo (lento su mesh grandi); False = salta, per test rapidi
MAX_PUNTI_FEDELTA = 50_000                  # numero massimo di punti usati per il calcolo di fedeltà (raycasting); più basso = più veloce
VISUALIZZA = False                          # apre una finestra 3D col risultato a fine script


# ---------------------------------------------------------------------------
# 1. CARICAMENTO E PULIZIA
# ---------------------------------------------------------------------------

def carica_point_cloud(percorso_csv: str, classi_da_escludere: list = None,
                        altitude_min: float = None, altitude_max: float = None,
                        escludi_zona_x: tuple = None, escludi_zona_z: tuple = None) -> o3d.geometry.PointCloud:
    """
    Legge il CSV e crea una PointCloud Open3D.

    Riconosce sia CSV "semplici" (colonne x,y,z[,intensity]) sia export reali
    SonarView/Omniscan (easting/northing/altitude locali + power (dB) + classification).

    classi_da_escludere: lista di valori della colonna 'classification' da rimuovere
        prima della ricostruzione (es. [2] per escludere provvisoriamente la classe
        che sembra rumore/water column, in base all'analisi fatta su OS3D_Max_Range_Test.csv).
        Nessun effetto se il CSV non ha una colonna 'classification'.

    altitude_min / altitude_max: se impostati, tiene solo i punti con altitude
        (colonna y/altitude) in quel range -- utile per isolare una zona specifica
        (es. solo la parte costiera/poco profonda) per iterare più velocemente e
        diagnosticare problemi localizzati. None = nessun filtro su quel lato.

    escludi_zona_x / escludi_zona_z: tuple (min, max) opzionali. Se ENTRAMBE sono
        impostate, rimuove i punti che cadono DENTRO quel rettangolo X e Z insieme
        -- utile per escludere zone di acquisizione anomale (es. dove la barca si
        è fermata e ha accumulato densità/rumore anomali in un punto, come il caso
        osservato su OS3D_Max_Range_Test.csv intorno a X~[-420,-380], Z~[5590,5650]).
    """
    df = pd.read_csv(percorso_csv)
    colonne_lower = {c.lower(): c for c in df.columns}

    col_x = colonne_lower.get("x") or colonne_lower.get("easting (local m)")
    col_z = colonne_lower.get("z") or colonne_lower.get("northing (local m)")
    col_y = colonne_lower.get("y") or colonne_lower.get("altitude (m)")
    col_intensity = (
        colonne_lower.get("intensity")
        or colonne_lower.get("intensita")
        or colonne_lower.get("power (db)")
    )
    col_classification = colonne_lower.get("classification")

    if not (col_x and col_y and col_z):
        raise ValueError(
            f"Non trovo colonne x,y,z (o easting/northing/altitude). Colonne nel CSV: {list(df.columns)}"
        )

    # Filtro per classificazione, se richiesto e se la colonna esiste
    if classi_da_escludere and col_classification:
        n_prima = len(df)
        df = df[~df[col_classification].isin(classi_da_escludere)]
        print(f"[1/5] Filtrate {n_prima - len(df)} righe con classification in {classi_da_escludere} "
              f"(provvisorio, da confermare) -> {len(df)} righe rimaste")

    # Filtro per range di altitudine, se richiesto (per isolare una zona/profondità specifica)
    if altitude_min is not None or altitude_max is not None:
        n_prima = len(df)
        if altitude_min is not None:
            df = df[df[col_y] >= altitude_min]
        if altitude_max is not None:
            df = df[df[col_y] <= altitude_max]
        print(f"[1/5] Filtrate {n_prima - len(df)} righe fuori dal range altitude "
              f"[{altitude_min}, {altitude_max}] -> {len(df)} righe rimaste")

    # Filtro per escludere una zona rettangolare X/Z (es. sosta della barca)
    if escludi_zona_x is not None and escludi_zona_z is not None:
        n_prima = len(df)
        dentro_zona = (
            (df[col_x] >= escludi_zona_x[0]) & (df[col_x] <= escludi_zona_x[1]) &
            (df[col_z] >= escludi_zona_z[0]) & (df[col_z] <= escludi_zona_z[1])
        )
        df = df[~dentro_zona]
        print(f"[1/5] Escluse {n_prima - len(df)} righe nella zona X{escludi_zona_x} x Z{escludi_zona_z} "
              f"(es. sosta barca) -> {len(df)} righe rimaste")

    punti = df[[col_x, col_y, col_z]].to_numpy(dtype=np.float64)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(punti)

    if col_intensity:
        intens = df[col_intensity].to_numpy(dtype=np.float64)
        intens_norm = (intens - intens.min()) / (intens.max() - intens.min() + 1e-9)
        colori = np.stack([intens_norm, intens_norm, 1 - intens_norm], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colori)
        print(f"  Colorazione mesh basata su: {col_intensity}")
    else:
        # Nessuna colonna di intensità disponibile (es. Surveyor 240): senza colore
        # assegnato Open3D userebbe nero di default, rendendo la mesh invisibile a
        # qualsiasi illuminazione. Coloriamo per altezza (Y) come fallback sensato.
        altezze = punti[:, 1]
        altezze_norm = (altezze - altezze.min()) / (altezze.max() - altezze.min() + 1e-9)
        # blu (basso) -> giallo (alto), stessa logica cromatica di viridis semplificata
        colori = np.stack([altezze_norm, altezze_norm, 1 - altezze_norm], axis=1)
        pcd.colors = o3d.utility.Vector3dVector(colori)
        print(f"  Colonna intensità non trovata: colorazione mesh basata sull'altezza (Y) come fallback")

    print(f"[1/5] Caricati {len(pcd.points)} punti da {percorso_csv} (X={col_x}, Y={col_y}, Z={col_z})")
    return pcd


def pulisci_point_cloud(pcd: o3d.geometry.PointCloud,
                         voxel_size=None,
                         nb_neighbors: int = 20,
                         std_ratio: float = 2.0) -> o3d.geometry.PointCloud:
    """
    Downsampling + rimozione outlier statistici.

    voxel_size: dimensione della griglia per il downsampling (in metri).
                Se None, viene stimato automaticamente dalla spaziatura media
                reale tra i punti (utile per dataset reali con densità molto
                diversa dal sintetico, es. Omniscan con pochi punti/m²).
    nb_neighbors / std_ratio: parametri del filtro outlier statistico. Un punto è
                scartato se la sua distanza media dai vicini è troppo lontana
                dalla media generale (std_ratio deviazioni standard).
    """
    if voxel_size is None:
        distanze_vicini = pcd.compute_nearest_neighbor_distance()
        spaziatura_media = np.mean(distanze_vicini)
        voxel_size = spaziatura_media * 0.75  # un po' più fine della spaziatura media
        print(f"[2/5] voxel_size stimato automaticamente: {voxel_size:.4f} "
              f"(spaziatura media punti: {spaziatura_media:.4f})")

    pcd_ds = pcd.voxel_down_sample(voxel_size=voxel_size)
    print(f"[2/5] Dopo downsampling (voxel={voxel_size}): {len(pcd_ds.points)} punti")

    pcd_clean, indici_validi = pcd_ds.remove_statistical_outlier(
        nb_neighbors=nb_neighbors, std_ratio=std_ratio
    )
    n_rimossi = len(pcd_ds.points) - len(pcd_clean.points)
    print(f"[2/5] Rimossi {n_rimossi} outlier -> {len(pcd_clean.points)} punti rimasti")

    return pcd_clean


# ---------------------------------------------------------------------------
# 2. STIMA DELLE NORMALI
# ---------------------------------------------------------------------------

def stima_normali(pcd: o3d.geometry.PointCloud, raggio_ricerca: float = 0.1,
                   max_vicini: int = 30) -> o3d.geometry.PointCloud:
    """
    Stima la normale di ogni punto guardando i vicini in un certo raggio.
    Fondamentale sia per Poisson che per una buona visualizzazione.
    """
    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=raggio_ricerca, max_nn=max_vicini)
    )
    # Orienta le normali in modo consistente (tutte "verso l'alto" rispetto al sensore).
    # Se il fondale è visto principalmente dall'alto, questo è un'euristica ragionevole:
    pcd.orient_normals_to_align_with_direction(orientation_reference=np.array([0.0, 1.0, 0.0]))
    print("[3/5] Normali stimate e orientate")
    return pcd


# ---------------------------------------------------------------------------
# 3. RICOSTRUZIONE DELLA SUPERFICIE
# ---------------------------------------------------------------------------

def ricostruisci_poisson(pcd: o3d.geometry.PointCloud, profondita: int = 9,
                          rimuovi_bassa_densita: float = 0.02) -> o3d.geometry.TriangleMesh:
    """
    Poisson Surface Reconstruction: superficie continua e chiusa.

    profondita: risoluzione dell'octree interno (8-10 è un buon range per iniziare;
                più alto = più dettaglio ma più lento e più rumore).
    rimuovi_bassa_densita: Poisson tende a "inventare" superficie anche dove i dati
                sono scarsi. Questo parametro rimuove i vertici nel percentile più
                basso di densità (0.02 = rimuove il 2% più scarso).
    """
    mesh, densita = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=profondita
    )
    densita = np.asarray(densita)
    soglia = np.quantile(densita, rimuovi_bassa_densita)
    vertici_da_rimuovere = densita < soglia
    mesh.remove_vertices_by_mask(vertici_da_rimuovere)

    print(f"[4/5] Mesh Poisson: {len(mesh.vertices)} vertici, {len(mesh.triangles)} triangoli")
    return mesh


def ricostruisci_ball_pivoting(pcd: o3d.geometry.PointCloud,
                                raggi: list = None) -> o3d.geometry.TriangleMesh:
    """
    Ball Pivoting Algorithm: più fedele ai dati originali di Poisson,
    ma può lasciare buchi dove la densità di punti è bassa.

    raggi: lista di raggi (in metri) da provare in sequenza. Open3D prova
           più raggi per "chiudere" più buchi possibili. Se non li fornisci,
           ne stimiamo alcuni a partire dalla spaziatura media dei punti.
    """
    if raggi is None:
        distanze = pcd.compute_nearest_neighbor_distance()
        dist_media = np.mean(distanze)
        raggi = [dist_media * f for f in (1.5, 2.0, 3.0, 4.0)]
        print(f"[4/5] Raggi BPA stimati automaticamente: {[round(r, 4) for r in raggi]}")

    mesh = o3d.geometry.TriangleMesh.create_from_point_cloud_ball_pivoting(
        pcd, o3d.utility.DoubleVector(raggi)
    )
    print(f"[4/5] Mesh Ball Pivoting: {len(mesh.vertices)} vertici, {len(mesh.triangles)} triangoli")
    return mesh


# ---------------------------------------------------------------------------
# 4. PULIZIA MESH
# ---------------------------------------------------------------------------

def pulisci_mesh(mesh: o3d.geometry.TriangleMesh,
                  dimensione_min_componente: int = 100) -> o3d.geometry.TriangleMesh:
    """Rimuove triangoli/vertici degeneri e componenti isolate piccole (rumore)."""
    mesh.remove_degenerate_triangles()
    mesh.remove_duplicated_triangles()
    mesh.remove_duplicated_vertices()
    mesh.remove_non_manifold_edges()

    componenti, conteggio_per_componente, _ = mesh.cluster_connected_triangles()
    componenti = np.asarray(componenti)
    conteggio_per_componente = np.asarray(conteggio_per_componente)

    triangoli_da_rimuovere = conteggio_per_componente[componenti] < dimensione_min_componente
    mesh.remove_triangles_by_mask(triangoli_da_rimuovere)
    mesh.remove_unreferenced_vertices()

    print(f"[5/5] Mesh pulita: {len(mesh.vertices)} vertici, {len(mesh.triangles)} triangoli")
    return mesh


# ---------------------------------------------------------------------------
# 5. METRICHE DI VALUTAZIONE (senza bisogno di ground truth esterno)
# ---------------------------------------------------------------------------

def valuta_mesh(mesh: o3d.geometry.TriangleMesh, pcd_originale: o3d.geometry.PointCloud,
                 tempo_calcolo: float = None, nome_metodo: str = "",
                 max_punti_fedelta: int = 50_000,
                 calcola_autointersezioni: bool = True) -> dict:
    """
    Calcola metriche di qualità per una mesh, utilizzabili anche senza un
    ground truth esterno:
      - fedeltà ai dati: distanza reale (raycasting) tra i punti misurati e
        la superficie della mesh -> quanto la mesh rappresenta bene i dati
      - qualità intrinseca: watertight, manifold, triangoli auto-intersecanti,
        area di superficie, regolarità dei triangoli

    max_punti_fedelta: su dataset grandi (es. dati reali con centinaia di
        migliaia di punti), il raycasting su TUTTI i punti è lento e statisticamente
        inutile: un campione casuale dà una stima equivalente in una frazione
        del tempo. Se i punti sono meno di questo valore, li usa tutti.
    calcola_autointersezioni: il controllo più lento in assoluto su mesh grandi
        (centinaia di migliaia di triangoli). Disattivalo durante i test rapidi
        sui parametri, riattivalo solo per il run definitivo.
    """
    metriche = {"metodo": nome_metodo}

    # --- Fedeltà ai dati: distanza punto-superficie via raycasting (con subsampling se serve) ---
    punti_originali = np.asarray(pcd_originale.points)
    if len(punti_originali) > max_punti_fedelta:
        indici = np.random.choice(len(punti_originali), max_punti_fedelta, replace=False)
        punti_per_fedelta = punti_originali[indici]
        print(f"  (fedeltà calcolata su un campione di {max_punti_fedelta} punti "
              f"su {len(punti_originali)} totali, per velocità)")
    else:
        punti_per_fedelta = punti_originali

    mesh_t = o3d.t.geometry.TriangleMesh.from_legacy(mesh)
    scene = o3d.t.geometry.RaycastingScene()
    scene.add_triangles(mesh_t)

    punti_query = o3d.core.Tensor(punti_per_fedelta, dtype=o3d.core.Dtype.Float32)
    distanze = scene.compute_distance(punti_query).numpy()

    metriche["fedelta_distanza_media"] = float(np.mean(distanze))
    metriche["fedelta_distanza_rms"] = float(np.sqrt(np.mean(distanze ** 2)))
    metriche["fedelta_distanza_max"] = float(np.max(distanze))

    # --- Qualità intrinseca della mesh ---
    metriche["n_vertici"] = len(mesh.vertices)
    metriche["n_triangoli"] = len(mesh.triangles)
    metriche["watertight"] = mesh.is_watertight()
    metriche["edge_manifold"] = mesh.is_edge_manifold()
    metriche["vertex_manifold"] = mesh.is_vertex_manifold()

    if calcola_autointersezioni:
        triangoli_autointersecanti = mesh.get_self_intersecting_triangles()
        metriche["n_triangoli_autointersecanti"] = len(triangoli_autointersecanti)
        metriche["percentuale_autointersecanti"] = (
            100 * len(triangoli_autointersecanti) / max(len(mesh.triangles), 1)
        )
    else:
        metriche["n_triangoli_autointersecanti"] = None
        metriche["percentuale_autointersecanti"] = None

    metriche["area_superficie"] = mesh.get_surface_area()

    # Regolarità: area di ciascun triangolo, guardiamo dispersione (std/media = coeff. variazione)
    vertici = np.asarray(mesh.vertices)
    triangoli_np = np.asarray(mesh.triangles)
    a = vertici[triangoli_np[:, 0]]
    b = vertici[triangoli_np[:, 1]]
    c = vertici[triangoli_np[:, 2]]
    aree_triangoli = 0.5 * np.linalg.norm(np.cross(b - a, c - a), axis=1)
    metriche["area_triangolo_media"] = float(np.mean(aree_triangoli))
    metriche["coeff_variazione_aree"] = float(np.std(aree_triangoli) / (np.mean(aree_triangoli) + 1e-12))

    if tempo_calcolo is not None:
        metriche["tempo_calcolo_secondi"] = tempo_calcolo

    return metriche


def stampa_metriche(metriche: dict):
    print(f"\n{'=' * 50}")
    print(f"Metriche — {metriche.get('metodo', '')}")
    print(f"{'=' * 50}")
    print(f"Fedeltà ai dati originali (raycasting):")
    print(f"  distanza media : {metriche['fedelta_distanza_media']:.5f} m")
    print(f"  distanza RMS   : {metriche['fedelta_distanza_rms']:.5f} m")
    print(f"  distanza max   : {metriche['fedelta_distanza_max']:.5f} m")
    print(f"Qualità intrinseca:")
    print(f"  vertici/triangoli   : {metriche['n_vertici']} / {metriche['n_triangoli']}")
    print(f"  watertight          : {metriche['watertight']}")
    print(f"  edge/vertex manifold: {metriche['edge_manifold']} / {metriche['vertex_manifold']}")
    if metriche["n_triangoli_autointersecanti"] is not None:
        print(f"  triangoli auto-intersecanti: {metriche['n_triangoli_autointersecanti']} "
              f"({metriche['percentuale_autointersecanti']:.2f}%)")
    else:
        print(f"  triangoli auto-intersecanti: (non calcolato, CALCOLA_AUTOINTERSEZIONI=False)")
    print(f"  area superficie     : {metriche['area_superficie']:.3f}")
    print(f"  area triangolo media: {metriche['area_triangolo_media']:.6f} "
          f"(coeff. variazione: {metriche['coeff_variazione_aree']:.2f})")
    if "tempo_calcolo_secondi" in metriche:
        print(f"  tempo di calcolo    : {metriche['tempo_calcolo_secondi']:.2f} s")


def salva_metriche_csv(metriche: dict, percorso_csv: str = "../output/metriche_ricostruzione.csv",
                        input_csv: str = None, voxel_size=None, escludi_classi: list = None):
    """
    Salva le metriche in un CSV, aggiungendo una riga ad ogni esecuzione
    (senza cancellare i run precedenti) -- utile per confrontare Poisson vs
    BPA, o run diversi, nel tempo.

    input_csv, voxel_size, escludi_classi: parametri espliciti registrati nella
        riga di metriche, per non dipendere dalle variabili globali del modulo
        (necessario per riusare questa funzione da altre pipeline, es. quella
        neurale NKSR, con la propria configurazione).
    """
    import datetime

    riga = dict(metriche)
    riga["timestamp"] = datetime.datetime.now().isoformat(timespec="seconds")
    riga["input_csv"] = input_csv
    riga["voxel_size"] = voxel_size if voxel_size is not None else "auto"
    riga["escludi_classi"] = str(escludi_classi)

    df_riga = pd.DataFrame([riga])
    file_esiste = os.path.isfile(percorso_csv)
    df_riga.to_csv(percorso_csv, mode="a", header=not file_esiste, index=False)
    print(f"Metriche aggiunte a: {percorso_csv}")


# ---------------------------------------------------------------------------
# 6. CONFRONTO CON GROUND TRUTH (opzionale)
# ---------------------------------------------------------------------------

def confronta_con_ground_truth(mesh: o3d.geometry.TriangleMesh,
                                percorso_gt_mesh: str) -> dict:
    """
    Calcola metriche di errore tra la mesh ricostruita e una mesh di ground
    truth esportata dal Terrain di Unity (es. tramite un .obj esportato).

    Ritorna un dizionario con distanza media, RMS e Hausdorff (max).
    """
    gt_mesh = o3d.io.read_triangle_mesh(percorso_gt_mesh)
    gt_pcd = gt_mesh.sample_points_uniformly(number_of_points=100_000)
    ricostruita_pcd = mesh.sample_points_uniformly(number_of_points=100_000)

    distanze = np.asarray(ricostruita_pcd.compute_point_cloud_distance(gt_pcd))

    metriche = {
        "distanza_media": float(np.mean(distanze)),
        "distanza_rms": float(np.sqrt(np.mean(distanze ** 2))),
        "distanza_hausdorff_approx": float(np.max(distanze)),
    }
    print("Metriche di confronto con ground truth:")
    for k, v in metriche.items():
        print(f"  {k}: {v:.4f}")
    return metriche


# ---------------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------------

def main():
    pcd = carica_point_cloud(INPUT_CSV, classi_da_escludere=ESCLUDI_CLASSI,
                              altitude_min=ALTITUDE_MIN, altitude_max=ALTITUDE_MAX,
                              escludi_zona_x=ESCLUDI_ZONA_X, escludi_zona_z=ESCLUDI_ZONA_Z)
    pcd = pulisci_point_cloud(pcd, voxel_size=VOXEL_SIZE, nb_neighbors=NB_NEIGHBORS, std_ratio=STD_RATIO)
    pcd = stima_normali(pcd)

    inizio = time.time()
    if METODO == "poisson":
        mesh = ricostruisci_poisson(pcd)
    else:
        mesh = ricostruisci_ball_pivoting(pcd)
    tempo_calcolo = time.time() - inizio

    mesh = pulisci_mesh(mesh)
    mesh.compute_vertex_normals()

    # Aggiunge automaticamente il nome del metodo al file di output
    # (es. "mesh_ricostruita.ply" -> "mesh_ricostruita_poisson.ply")
    base, estensione = os.path.splitext(OUTPUT_MESH)
    output_mesh_con_metodo = f"{base}_{METODO}{estensione}"

    o3d.io.write_triangle_mesh(output_mesh_con_metodo, mesh)
    print(f"\nMesh salvata in: {output_mesh_con_metodo}")

    metriche = valuta_mesh(mesh, pcd, tempo_calcolo=tempo_calcolo, nome_metodo=METODO,
                            max_punti_fedelta=MAX_PUNTI_FEDELTA,
                            calcola_autointersezioni=CALCOLA_AUTOINTERSEZIONI)
    stampa_metriche(metriche)
    salva_metriche_csv(metriche, percorso_csv=METRICHE_CSV,
                        input_csv=INPUT_CSV, voxel_size=VOXEL_SIZE, escludi_classi=ESCLUDI_CLASSI)

    if GROUND_TRUTH_MESH:
        confronta_con_ground_truth(mesh, GROUND_TRUTH_MESH)

    if VISUALIZZA:
        o3d.visualization.draw_geometries([mesh], mesh_show_back_face=True)


if __name__ == "__main__":
    main()