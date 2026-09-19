# Progetto Barca — Ricostruzione 3D del fondale da sonar

Ricerca su ricostruzione 3D del fondale/infrastrutture critiche (cavi, pipe,
piloni) tramite sonar multibeam (Cerulean Omniscan 3D), in collaborazione con
un'azienda bellunese che produce robot subacquei. Supervisione: Daniele.

## Struttura del progetto

```
.
├── main.py                              # entry point: legge config/config.yaml e lancia la pipeline scelta
├── config/
│   └── config.yaml                      # percorsi + parametri condivisi e specifici per algoritmo
├── data/
│   ├── data_analisys.py                 # analisi esplorativa di un CSV point cloud
│   └── sonar/                           # CSV point cloud reali (Omniscan 3D, Surveyor 240)
├── utils/
│   └── mesh.py                          # funzioni condivise: caricamento/pulizia point cloud, normali, metriche
├── scripts/
│   ├── poisson.py                       # ricostruzione classica: Poisson Surface Reconstruction (Open3D)
│   ├── point2mesh.py                    # ricostruzione neurale: Point2Mesh (self-prior, sottoprocesso)
│   ├── ppsurf.py                        # ricostruzione neurale: PPSurf (patch-based, sottoprocesso)
│   └── sa_convonet.py                   # ricostruzione neurale: SA-ConvONet (pointcloud_crop, sottoprocesso)
├── visualization/
│   ├── visualize_mesh.py                # visualizzatore Open3D per file .ply
│   └── orbite_mesh.py                   # genera video orbitale attorno a una mesh (per slide)
├── output/
│   ├── mesh/                            # mesh .ply generate dalle pipeline
│   ├── screenshot_mesh/                 # screenshot presi con visualize_mesh.py
│   └── metriche_ricostruzione.csv       # log di tutte le metriche, un run per riga
├── helpers/                             # repository di terzi clonati (Point2Mesh, Manifold, PPSurf - non tracciati in git)
├── neural_methods/                      # repository neurali piu' recenti (SA-ConvONet, if-net - non tracciati in git)
├── setup_ambiente.sh                    # da "source"-are a inizio sessione: fix visualizzatore Open3D su WSLg + conda activate
└── presentazione_fase1.tex              # slide Beamer con i risultati della Fase 1
```

Ambiente: WSL2 (Ubuntu 22.04) su Windows, GPU NVIDIA GTX 1650 (4GB VRAM).
Ambienti Python gestiti con conda, separati per progetto:
- `ricostruzione3d` — Open3D, numpy, pandas, matplotlib (pipeline classica + analisi)
- `point2mesh` — ambiente dedicato per l'esperimento Point2Mesh (concluso)
- `ppsurf` — PyTorch + CUDA 11.7-12.1, PyTorch-Lightning, PyG (pipeline PPSurf)
- `sa_conet` — Python 3.6, PyTorch 1.4.0 + CUDA 10.1, torch_scatter (pipeline SA-ConvONet)

## Fase 1 — Confronto ricostruzione classica vs neurale (conclusa)

**Dataset**: `OS3D_Max_Range_Test.csv` — transetto reale Omniscan 3D, 519.501
punti. Una zona di sosta della barca (X∈[-420,-380], Z∈[5590,5650]) è stata
identificata (densità anomala fino a 8000 punti/cella, causa di un artefatto
"a bolle" nella ricostruzione) ed esclusa dal preprocessing.

**Metodi confrontati**, tutti sullo stesso dataset pulito:

| Metrica | Poisson | Point2Mesh (1500 it.) | PPSurf |
|---|---|---|---|
| Distanza media (fedeltà) | **0.51 m** | 1.83 m | 1.43 m |
| Distanza RMS | **0.80 m** | 2.97 m | 2.09 m |
| Distanza massima | **12.93 m** | 30.75 m | 23.37 m |
| Vertici / Triangoli | 234.231 / 469.477 | 1.432 / 3.000 | 13.354 / 26.916 |
| Tempo di calcolo | **5.7 s** | 1006.3 s | 105.6 s |

**Conclusioni**:
- **Poisson** è il metodo di riferimento: fedeltà nettamente migliore, tempi minimi.
- **Point2Mesh** non è adatto: parte da un convex hull (pensato per oggetti
  compatti genus-0), mentre il dato sonar è una superficie aperta e allungata
  ("a nastro", pattern di scansione multibeam). Più iterazioni peggiorano la
  fedeltà (non è un problema di tuning, ma di mismatch strutturale).
- **PPSurf** (patch-based, ricostruisce porzione per porzione) va nettamente
  meglio di Point2Mesh ma resta dietro Poisson — plausibilmente perché il
  modello pre-addestrato non ha mai visto rumore/pattern tipici del sonar
  batimetrico in training (da qui l'idea del fine-tuning, Fase 2).

Slide di sintesi: `presentazione_fase1.tex` (inviate a Daniele).

### Metodi scartati durante l'esplorazione (e perché)

| Metodo | Motivo dello scarto |
|---|---|
| NKSR | Wheel precompilati scaduti, richiede compilazione da sorgente con CUDA 12.8 completo; memory-hungry, probabile OOM su 4GB VRAM |
| ConvONet | Inizialmente scartato per dipendenze fossili (PyTorch 1.4, CUDA 10.1, torch-scatter) — poi ripreso e integrato con successo come **SA-ConvONet**, vedi sezione dedicata sotto |
| DeepSDF | Richiede training su molte forme diverse (shape prior) o ottimizzazione per-scena (costo alla Point2Mesh) |
| NeuS, 3DGS subacqueo | Richiedono immagini RGB multi-vista con pose, non point cloud — fuori scope (Daniele ha chiesto di restare sul sonar) |
| PointNet++/DGCNN | Feature extractor per classificazione/segmentazione, non producono mesh — non sono tool di ricostruzione completi |
| Diffusion-based (WALA, MAKE-A-SHAPE) | Immaturi su input sparsi/rumorosi (failure mode noto, "Meltdown" 2025), addestrati su oggetti non scene outdoor |
| if-net | Ambiente installato e funzionante (`neural_methods/if-net`), ma non è un metodo per-scena come Poisson/Point2Mesh/PPSurf: richiede training su un dataset di mesh note (ShapeNet, 800GB) per imparare un prior di completamento forma. I pesi pretrained coprono categorie ShapeNet generiche (sedie, auto, aerei...) e non generalizzano al dominio sonar/legname. Non testato per ora: avrebbe senso solo con un dataset proprio di mesh di riferimento su cui allenarlo da zero |

### Aggiornamento post-Fase 1: SA-ConvONet integrato

[SA-ConvONet](https://github.com/tangjiapeng/SA-ConvONet) (`neural_methods/SA-ConvONet`)
parte da un modello Convolutional Occupancy Network pre-addestrato e ne
ottimizza i pesi in modo sign-agnostic per-scena a test-time; per ora
usiamo solo l'inferenza col modello pre-addestrato (senza l'ottimizzazione
sign-agnostic, che resta un passo successivo eventuale). A differenza delle
varianti ConvONet "stanza"/ShapeNet (bounding box fisso, coordinate
normalizzate in un cubo unitario — lo stesso motivo per cui era stato
scartato insieme a NKSR/ConvONet in Fase 1), usiamo la variante
**`pointcloud_crop`**: encoder a griglia 3D con finestra scorrevole, pensata
per scene reali di grandi dimensioni in unità metriche vere — strutturalmente
adatta al nostro transetto sonar (superficie aperta, estesa centinaia di
metri), non solo a "stanze" delimitate.

| Metrica | Poisson (riferimento) | SA-ConvONet (`pointcloud_crop/room_grid64`) |
|---|---|---|
| Distanza media (fedeltà) | 0.50 m | 1.76 m |
| Distanza RMS | 0.79 m | 2.44 m |
| Distanza massima | 16.2 m | 15.4 m |
| Vertici / Triangoli | 234.317 / 469.584 | 19.820 / 38.432 |
| Tempo di calcolo | 11.3 s | 115.1 s |

Risultato in linea con Point2Mesh/PPSurf in Fase 1: Poisson resta il
riferimento per fedeltà e velocità, ma SA-ConvONet produce una ricostruzione
reale e ragionevole (non un artefatto), leggermente meglio di Point2Mesh e
vicina a PPSurf. Integrato nella pipeline allo stesso modo degli altri
metodi neurali: `scripts/sa_convonet.py`, sezione `sa_convonet` in
`config/config.yaml`, `algorithm: "sa_convonet"` per selezionarlo.

## Risposta di Daniele (dopo Fase 1)

> Il fatto che PPSurf e i metodi neurali vadano peggio può dipendere da tanti
> fattori, in primis i pochi dati e la diversità rispetto al dataset di
> training. Non mi sposterei sui dati RGB, atteniamoci ai dati che può darci
> il sonar.

Passi richiesti, in ordine:
1. Testare la ricostruzione offline in Unity
2. Fine-tuning di PPSurf (o metodo neurale allo stato dell'arte)
3. Valutazione online al variare del numero di punti

Visione a lungo termine: task di mapping autonomo (RL) dove la barca si
muove per minimizzare l'incertezza batimetrica. Potenzialmente pubblicabile.
Riferimenti forniti da Daniele:
- Mercado, Weng, Maki — *Adaptive Informative Path Planning for Lightweight
  AUV in Coastal Benthic Environments* (IEEE UT 2025) — RL/PPO
- Chavez-Galaviz, Bloss, Mahmoudian — *Uncertainty-guided informative path
  planning for ecological monitoring using ASVs under Dubins motion
  constraints* (Frontiers Robot. AI 2026) — GP+RRT*, non RL

## Fase 2 — In corso

1. **Ricostruzione offline in Unity**: generare dataset sintetico con
   `PointCloudRecorder.cs` (ground truth nota dal Terrain) per calcolare
   finalmente un errore assoluto (non solo fedeltà ai punti misurati).
2. **Fine-tuning PPSurf**: incognita principale è la fattibilità con 4GB
   VRAM e la necessità del codice di training (non solo inferenza, già
   testata in Fase 1). Riferimento concettuale alternativo: famiglia di
   metodi neurali *sonar-native* (NeuSIS/Qadri et al. ICRA 2023, NeuRSS) che
   modellano esplicitamente la propagazione acustica — non testati, ma
   pensati per forward-looking sonar (FLS), non per multibeam batimetrico
   come l'Omniscan; adattarli richiederebbe reimplementare il renderer
   differenziabile per la geometria del vostro sensore.
3. **Fedeltà vs densità di punti**: base per derivare una funzione
   `incertezza_mesh ≈ f(densità_locale)` calibrata sul vostro metodo di
   ricostruzione (vantaggio rispetto a usare la varianza di un GP generico
   scollegato dalla pipeline reale).

## Fase 3 — Mapping attivo con RL (visione, non ancora iniziata)

Architettura a due binari, per non perdere l'obiettivo di avere sempre una
**mesh 3D vera** come output finale:

- **Binario navigazione**: segnale di incertezza leggero (densità locale di
  punti, o GP grossolano aggiornato periodicamente) guida l'agente in tempo
  reale — reward basato su riduzione dell'incertezza batimetrica.
- **Binario output**: Poisson/PPSurf continuano a generare la mesh reale, a
  checkpoint o a fine missione.

Gap identificato rispetto ai due paper di riferimento: nessuno dei due fa
"RL + incertezza derivata dal proprio metodo di ricostruzione + mesh 3D vera
da sonar multibeam reale" — Mercado et al. non ricostruisce geometria (solo
detection/coverage) e usa un'incertezza euristica; Chavez-Galaviz et al.
ricostruisce con GP standard una mappa 2.5D, non una mesh 3D da point cloud
multibeam reale.

**Punto aperto da chiarire con Daniele**: l'agente RL dovrà ragionare su una
mesh 3D o su una mappa di profondità 2.5D stile GP? La seconda è più
semplice da integrare con un reward basato su incertezza; la prima è più
fedele alla complessità geometrica reale del fondale.

## Note tecniche utili (da non riperdere tempo a riscoprire)

- **Formato dati Omniscan 3D (via SonarView)**: CSV con `easting/northing
  (local m)`, `altitude (m)`, `power (dB)`, `classification`, `ping number`.
  Nessuna colonna RGB — il sensore è puramente acustico (il "RGB LED
  Indicator" nella scheda tecnica è una spia di stato, non un sensore ottico).
- **Colonna `classification`**: valori osservati 1/2/5. Ipotesi (da
  confermare in SonarView, opzione "Water Column Points"): la classe 2 è
  rumore/colonna d'acqua — esclusa di default nella pipeline classica.
- **PPSurf**: richiede `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`,
  `gen_resolution_global 129` (metà risoluzione) e `rec_batch_size 10000`
  (1/5 del default) per stare in 4GB VRAM in fase di marching cubes. Fix
  applicato al codice vendorizzato (`source/base/mesh.py`) per compatibilità
  con trimesh 5.1.0 — vedi commento nel file.
- **SA-ConvONet / torch_scatter — conflitto ABI C++**: installare
  `torch-scatter` nell'env `sa_conet` (anche dalla wheel precompilata
  ufficiale, `pip install torch-scatter==2.0.4 -f
  https://data.pyg.org/whl/torch-1.4.0+cu101.html`) va in crash all'avvio
  (`OSError: undefined symbol: ...parseSchemaOrName...`) se PyTorch è
  installata dal canale conda `pytorch`: quella build usa la ABI C++11 nuova
  di libstdc++, mentre ogni build di torch_scatter per questa versione è
  compilata per la ABI vecchia della PyTorch ufficiale pip. Fix: reinstallare
  PyTorch nello stesso env dalla wheel ufficiale pip invece che da conda —
  `pip install --force-reinstall --no-deps
  https://download.pytorch.org/whl/cu101/torch-1.4.0-cp36-cp36m-linux_x86_64.whl`
  (il tag di versione `+cu101` non esiste nell'indice pip per questa
  versione: va installata per URL diretto, non con `pip install torch==...`).
- **SA-ConvONet `pointcloud_crop` — scala della scena**: il numero di crop
  nella generazione a finestre scorrevoli scala con l'estensione reale della
  point cloud normalizzata divisa per `unit_size * query_vol_size`. I default
  del repo (tarati su stanze Matterport, 10-20m) sul nostro transetto
  (~450m) generano ~2200 crop (2+ ore, mai finito nel test). Alzati a
  `unit_size: 0.005` (da 0.002) in `config/config.yaml` → ~140 crop, ~2 min
  (mesh più grossolana, da affinare). La mesh generata è inoltre in
  coordinate normalizzate (centrate/scalate da `make_cropscene_dataset2.py`)
  e va riportata in metri reali applicando l'inverso (`loc`/`scale` salvati
  nel `pointcloud.npz` normalizzato) — gestito in `scripts/sa_convonet.py`.
- **Visualizzatore Open3D su WSLg**: `vis.create_window()` fallisce
  silenziosamente (ritorna `False`) e `vis.get_render_option()` ritorna
  `None` (`AttributeError` a valle) perché GLFW tenta Wayland nativo, non
  supportato dal setup grafico di questa macchina. Fix: `export
  WAYLAND_DISPLAY=` (stringa vuota, non `unset` — GLFW controlla solo se la
  variabile è definita) forza il fallback a XWayland. Automatizzato in
  `setup_ambiente.sh` (da lanciare con `source` a inizio sessione).
- **WSL**: attenzione allo spazio disco reale su C: (il file `.vhdx` cresce
  ma non si restringe da solo — `wsl --shutdown` + `Optimize-VHD` per
  compattarlo se necessario).