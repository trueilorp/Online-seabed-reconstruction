"""
Genera un video orbitale attorno a una mesh (per presentazioni)
====================================================================

La camera orbita sia orizzontalmente che verticalmente, un giro completo
(360°) su entrambi gli assi contemporaneamente -- parte da un'angolazione
leggermente dall'alto (utile per mostrare il fondale ricostruito in modo
leggibile, non "di taglio") e torna alla stessa inclinazione iniziale a
fine giro, dato che una rotazione di 360° è periodica.

Uso:
    Imposta i parametri qui sotto e lancia:
    python orbit_render.py

Dipendenze:
    pip install open3d
    ffmpeg installato e nel PATH (per comporre i frame in un .mp4)
    -- su Windows: scarica da ffmpeg.org, aggiungi la cartella bin/ al PATH
    -- su WSL/Ubuntu: sudo apt install ffmpeg
"""

import os
import subprocess

import open3d as o3d

# --- Imposta qui i parametri ---
PERCORSO_MESH = "output/mesh/mesh_poisson.ply"
CARTELLA_FRAME = "frame_orbita"
VIDEO_OUTPUT = "orbita_poisson.mp4"
N_FRAME = 240                 # più frame = orbita più fluida, ma più lento da generare
FPS = 30                      # frame al secondo nel video finale (240 frame / 30 fps = 8 secondi di orbita)
ANGOLO_DALL_ALTO = 55.0       # gradi di inclinazione verso il basso ad inizio giro (0 = vista orizzontale, 90 = dall'alto) --
                               # piu' alto di prima cosi' il passaggio dall'alto durante il giro verticale (gia'
                               # presente, verificato che la rotazione attraversa davvero lo zenit) sia piu' marcato
GIRI_ORIZZONTALI = 2          # numero di giri completi (360°) sull'asse orizzontale distribuiti sui frame
GIRI_VERTICALI = 2            # numero di giri completi (360°) sull'asse verticale, in sincrono con quello orizzontale
LARGHEZZA_FINESTRA = 1280
ALTEZZA_FINESTRA = 720
SFONDO_BIANCO = True          # True per sfondo bianco (meglio per slide), False per sfondo scuro default


def genera_frame_orbita():
    os.makedirs(CARTELLA_FRAME, exist_ok=True)

    mesh = o3d.io.read_triangle_mesh(PERCORSO_MESH)
    if len(mesh.vertices) == 0:
        print(f"[ATTENZIONE] '{PERCORSO_MESH}' non contiene vertici, o non è stato trovato.")
        return False

    mesh.compute_vertex_normals()

    vis = o3d.visualization.Visualizer()
    vis.create_window(width=LARGHEZZA_FINESTRA, height=ALTEZZA_FINESTRA, visible=True)
    vis.add_geometry(mesh)

    render_opt = vis.get_render_option()
    render_opt.mesh_show_back_face = True
    render_opt.mesh_color_option = o3d.visualization.MeshColorOption.Color
    if SFONDO_BIANCO:
        render_opt.background_color = [1, 1, 1]

    ctrl = vis.get_view_control()

    # Inclina la vista verso il basso una volta sola, all'inizio (angolazione "dall'alto")
    # Nota: rotate() lavora in unità pixel, non gradi esatti -- il fattore di conversione
    # è empirico (circa 3-4 unità per grado con le impostazioni di default di Open3D).
    # Se l'inclinazione finale non sembra corretta, aggiusta FATTORE_PIXEL_PER_GRADO.
    FATTORE_PIXEL_PER_GRADO = 3.5
    ctrl.rotate(0.0, ANGOLO_DALL_ALTO * FATTORE_PIXEL_PER_GRADO)
    vis.poll_events()
    vis.update_renderer()

    # Rotazione orizzontale e verticale distribuite sui frame, con un numero di giri
    # diverso sui due assi (le due avvengono simultaneamente: la telecamera fa una
    # "capriola a nodo" attorno alla mesh invece di girarle solo intorno mantenendo
    # l'inclinazione fissa, coprendo molte piu' angolazioni nello stesso video)
    rotazione_orizzontale_per_frame = 360.0 * GIRI_ORIZZONTALI * FATTORE_PIXEL_PER_GRADO / N_FRAME
    rotazione_verticale_per_frame = 360.0 * GIRI_VERTICALI * FATTORE_PIXEL_PER_GRADO / N_FRAME

    print(f"Generazione di {N_FRAME} frame in corso...")
    for i in range(N_FRAME):
        ctrl.rotate(rotazione_orizzontale_per_frame, rotazione_verticale_per_frame)
        vis.poll_events()
        vis.update_renderer()
        percorso_frame = os.path.join(CARTELLA_FRAME, f"frame_{i:04d}.png")
        vis.capture_screen_image(percorso_frame, do_render=True)
        if (i + 1) % 20 == 0:
            print(f"  {i + 1}/{N_FRAME} frame generati")

    vis.destroy_window()
    print(f"Frame salvati in: {CARTELLA_FRAME}/")
    return True


def componi_video():
    """Usa ffmpeg per comporre i frame PNG in un video mp4."""
    comando = [
        "ffmpeg", "-y",
        "-framerate", str(FPS),
        "-i", os.path.join(CARTELLA_FRAME, "frame_%04d.png"),
        "-c:v", "libx264",
        "-pix_fmt", "yuv420p",
        VIDEO_OUTPUT,
    ]
    print(f"\nComposizione video con ffmpeg...")
    try:
        subprocess.run(comando, check=True)
        print(f"Video salvato in: {VIDEO_OUTPUT}")
    except FileNotFoundError:
        print("[ATTENZIONE] ffmpeg non trovato nel PATH. Installa ffmpeg e riprova,")
        print(f"oppure componi manualmente i frame in {CARTELLA_FRAME}/ con un altro strumento.")
    except subprocess.CalledProcessError as e:
        print(f"[ERRORE] ffmpeg ha restituito un errore: {e}")


if __name__ == "__main__":
    if genera_frame_orbita():
        componi_video()