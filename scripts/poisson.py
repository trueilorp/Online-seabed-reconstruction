"""
Pipeline classica di ricostruzione 3D da point cloud (sonar) con Poisson
============================================================================

Poisson Surface Reconstruction (Open3D) stima una funzione implicita a
partire dalle normali della point cloud e ne estrae una superficie
continua e chiusa. A differenza di Point2Mesh/PPSurf, gira interamente
nell'ambiente conda "ricostruzione3d" (Open3D), senza sottoprocessi.

I parametri di questa pipeline sono in config/config.yaml (sezione
poisson); l'esecuzione avviene tramite main.py nella root del progetto.
"""

import numpy as np
import open3d as o3d


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
    mesh, densita = poisson_grezzo(pcd, profondita)
    return taglia_bassa_densita(mesh, densita, rimuovi_bassa_densita)


def poisson_grezzo(pcd: o3d.geometry.PointCloud, profondita: int = 9):
    """Mesh Poisson prima del taglio, con la densita' per vertice (quanti punti di input
    sostengono quella zona: bassa dove Poisson ha "inventato" superficie)."""
    mesh, densita = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
        pcd, depth=profondita
    )
    return mesh, np.asarray(densita)


def taglia_bassa_densita(mesh: o3d.geometry.TriangleMesh, densita: np.ndarray,
                          rimuovi_bassa_densita: float) -> o3d.geometry.TriangleMesh:
    """Rimuove i vertici sotto il quantile rimuovi_bassa_densita delle densita' (su una copia)."""
    mesh = o3d.geometry.TriangleMesh(mesh)
    soglia = np.quantile(densita, rimuovi_bassa_densita)
    mesh.remove_vertices_by_mask(densita < soglia)

    print(f"[4/5] Mesh Poisson: {len(mesh.vertices)} vertici, {len(mesh.triangles)} triangoli")
    return mesh


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
