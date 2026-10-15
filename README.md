# Progetto Barca — Ricostruzione 3D del fondale da sonar

Ricerca su ricostruzione 3D del fondale/infrastrutture critiche (cavi, pipe,
piloni) tramite sonar multibeam (Cerulean Omniscan 3D), in collaborazione con
un'azienda bellunese che produce robot subacquei. Supervisione: Daniele.

## Struttura del progetto

```
.
├── reconstruct_single_dataset.py        # entry point: ricostruisce il dataset in config/config.yaml con l'algoritmo scelto
├── evaluate.py                          # benchmark dei metodi sui terreni sintetici vs ground truth (sezione evaluation del config)
├── benchmark/
│   ├── tempo_poisson.sh                 # tempi per fase (+ RAM/VRAM, chamfer/F-score) di Poisson
│   ├── tempo_ppsurf.sh                  # idem PPSurf: avvio librerie vs "pps.py rec" completo
│   ├── tempo_sa_convonet.sh             # idem SA-ConvONet: avvio, preparazione dataset, generate.py
│   ├── sweep_densita_poisson.py         # sweep di poisson.rimuovi_bassa_densita sui terreni sintetici (F-score/Chamfer)
│   └── comune.sh, prepara_input.py, ... # funzioni condivise; risultati in output/benchmark/
├── config/
│   └── config.yaml                      # percorsi + parametri condivisi e specifici per algoritmo
├── data/
│   ├── data_analisys.py                 # analisi esplorativa di un CSV point cloud
│   ├── sonar/                           # CSV point cloud reali (Omniscan 3D, Surveyor 240) e sintetici (Unity)
│   └── synthetic/                       # terreni sintetici train/val/test: <id>_scan.csv + <id>_ground_truth.obj
├── utils/
│   ├── mesh.py                          # funzioni condivise: caricamento/pulizia point cloud, normali, metriche
│   ├── pipeline.py                      # pipeline condivisa: config, caricamento/pulizia, esecutori, sliding window
│   ├── generate_seabed_heightmaps.py    # genera i fondali sintetici (heightmap + tubi/piloni)
│   ├── simulate_multibeam_scans.py      # simula le scansioni multibeam e le ground truth chiuse
│   └── make_saconvonet_dataset.py       # terreni sintetici -> dataset di fine-tuning SA-ConvONet
├── scripts/
│   ├── poisson.py                       # ricostruzione classica: Poisson Surface Reconstruction (Open3D)
│   ├── ppsurf.py                        # ricostruzione neurale: PPSurf (patch-based, sottoprocesso)
│   ├── sa_convonet.py                   # ricostruzione neurale: SA-ConvONet (pointcloud_crop, sottoprocesso)
│   └── confronta_mesh_ground_truth.py   # accuratezza/completezza/copertura + heatmap/sezione/istogrammi vs ground truth
├── visualization/
│   ├── visualize_mesh.py                # visualizzatore Open3D per file .ply
│   └── orbite_mesh.py                   # genera video orbitale attorno a una mesh (per slide)
├── output/
│   ├── mesh/                            # mesh .ply generate dalle pipeline
│   ├── comparison/<dataset>/            # heatmap, profili di sezione, istogrammi, tabelle accuratezza/completezza
│   ├── screenshot_mesh/                 # screenshot presi con visualize_mesh.py
│   └── metriche_ricostruzione.csv       # log di tutte le metriche, un run per riga
├── helpers/                             # repository di terzi clonati (Point2Mesh, Manifold, PPSurf - non tracciati in git)
├── neural_methods/                      # repository neurali piu' recenti (SA-ConvONet, if-net - non tracciati in git)
├── setup_ambiente.sh                    # da "source"-are a inizio sessione: fix visualizzatore Open3D su WSLg + conda activate
└── presentazione_fase1.tex              # slide Beamer con i risultati della Fase 1
```

**Simulatore Unity** (progetto separato, Windows): script C# rilevanti —
`PointCloudRecorder.cs` (registra la point cloud del sensore in CSV, tasto
R o registrazione continua) e `ExportTerrainToObj.cs` (Tools → Export
Terrain to OBJ, esporta il Terrain scolpito come mesh ground truth). I CSV
e i `.obj` risultanti vanno copiati manualmente in `data/sonar/`.

Ambiente: WSL2 (Ubuntu 22.04) su Windows, GPU NVIDIA GTX 1650 (4GB VRAM).
Ambienti Python gestiti con conda, separati per progetto:
- `ricostruzione3d` — Open3D, numpy, pandas, matplotlib (pipeline classica + analisi)
- `point2mesh` — ambiente dedicato per l'esperimento Point2Mesh (concluso; codice rimosso dalla pipeline, recuperabile dalla history git)
- `ppsurf` — PyTorch + CUDA 11.7-12.1, PyTorch-Lightning, PyG (pipeline PPSurf)
- `sa_conet` — Python 3.6, PyTorch 1.4.0 + CUDA 10.1, torch_scatter (pipeline SA-ConvONet)

## Fase 1 — Confronto ricostruzione classica vs neurale su dati sonar reali (conclusa)

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
  Per questo `scripts/point2mesh.py` e la sezione `point2mesh` del config
  sono stati rimossi (restano nella history git).
- **PPSurf** (patch-based, ricostruisce porzione per porzione) va nettamente
  meglio di Point2Mesh ma resta dietro Poisson — plausibilmente perché il
  modello pre-addestrato non ha mai visto rumore/pattern tipici del sonar
  batimetrico in training (da qui l'idea del fine-tuning, ancora da fare).

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

### SA-ConvONet integrato (post-Fase 1)

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

Risultato iniziale su dati sonar reali (`OS3D_Max_Range_Test.csv`, config
`room_grid64` di default): fedeltà 1.76 m, in linea con Point2Mesh/PPSurf —
peggiore di Poisson, coerente con l'ipotesi di domain gap. Integrato nella
pipeline allo stesso modo degli altri metodi neurali: `scripts/sa_convonet.py`,
sezione `sa_convonet` in `config/config.yaml`, `algorithm: "sa_convonet"`.

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

## Fase 2 — Validazione con ground truth Unity (conclusa)

### Il bug del sensore, e perché era importante trovarlo

I primi dataset sintetici Unity mostravano uno strato di punti "fantasma"
concentrato alla profondità di montaggio del sensore (~Y=-0.5), separato
nettamente dal vero fondale. Causa: un bug in `IRaycastHitsToPointsJob`
(libreria UnitySensors, `RaycastLiDARSensor`) — ogni raggio che **non
colpisce nulla** genera comunque un punto a `distance=0`, cioè esattamente
nella posizione del sensore. Layer Mask e Min Range del sensore non
risolvono il problema (non è un hit sbagliato, è un miss gestito male a
monte). **Fix** in `PointCloudRecorder.cs`: scarta i punti con
`intensity == 0` (il segnale che la libreria stessa fornisce per i "miss"),
prima di scriverli su CSV. Verificato che il fix non cambia le conclusioni
sostanziali (confronto prima/dopo) — solo le pulisce.

### Due metriche, non una: accuratezza vs completezza

Scoperta metodologica chiave (`scripts/confronta_mesh_ground_truth.py`): la
fedeltà "ai punti misurati" (l'unica disponibile in Fase 1, senza ground
truth) può essere **fuorviante** o persino **invertita** rispetto
all'errore vero. Con un ground truth disponibile, vanno sempre riportate
due direzioni distinte:

- **Accuratezza** (ricostruzione → verità): quanto sono precisi i punti
  *dove la mesh esiste*. Non vede i buchi — un metodo può "barare"
  tagliando via le zone incerte.
- **Completezza** (verità → ricostruzione): quanto del vero terreno è
  effettivamente coperto. Penalizza i buchi pesantemente.
- **Copertura %**: frazione di ground truth con una ricostruzione entro
  una soglia (es. 0.5 m) — il numero più diretto per capire il trade-off.

Un metodo iper-tarato per l'accuratezza (Poisson con soglia di densità
molto alta) può risultare *peggiore* in completezza di un metodo che lascia
meno buchi ma è localmente meno preciso — le due metriche vanno sempre
lette insieme, mai una sola.

### Metriche di superficie: Chamfer e F-score, contro il solo fondale

`evaluate.py` (e gli script di `benchmark/`, che usano la stessa funzione
`metriche_superficie`) riportano anche metriche superficie-superficie, con
distanza esatta punto-superficie su `n_campioni` punti uniformi per lato:

- **chamfer** = (distanza media ricostruzione→fondale + fondale→ricostruzione) / 2, in metri
- **precisione** = frazione di punti della ricostruzione entro `tau` dal fondale
- **richiamo** = frazione di punti del fondale entro `tau` dalla ricostruzione (= copertura)
- **fscore** = media armonica di precisione e richiamo

**Perché "solo fondale" e non la GT intera.** La ground truth sintetica non
è solo il fondale: è un **blocco solido chiuso** (fondale + pareti verticali
+ fondo piatto). Pareti e fondo servono solo a chiudere la mesh, il sonar non
li vede mai e nessun metodo dovrebbe ricostruirli. `top_surface`
(`utils/mesh.py`) li stacca, tenendo solo il fondale.

```
        ~~~~~~~~~~~~~~~~~~~~~~~~~~      fondale (heightmap): l'unica cosa che il sonar vede
        |\                      /|
 parete | \  tenda spuria      / | parete
        |                        |
        +------------------------+      fondo piatto (quota minima)
```

I metodi impliciti tendono a chiudere la superficie: Poisson crea "tende" che
scendono dai bordi o una bolla che si richiude sotto (il filtro di bassa
densità ne toglie solo una parte), PPSurf e SA-ConvONet possono generare
superfici spurie verticali o sotto il fondale (vedi il guscio doppio più
sotto). Una superficie spuria che passa vicino alle pareti o al fondo della GT:

- **contro la GT intera** ha distanza ≈ 0 → l'errore viene **premiato** come corretto;
- **contro il solo fondale** è lontana → viene **penalizzata**, com'è giusto.

Per questo completezza, copertura, chamfer e F-score usano il solo fondale
(altrimenti, nel verso opposto, pareti e fondo mai ricostruiti conterebbero
come mancanti). Fa eccezione `accuratezza_*`, che per ora resta misurata
contro la **GT intera** (per restare confrontabile con i CSV già salvati):
in caso di dubbio leggere `precisione`/`chamfer`, che questo errore non lo
possono premiare. In pratica le due misure spesso coincidono: il problema
c'è solo quando la superficie spuria passa vicino a bordi o base del
terreno; se la scansione non arriva ai bordi, le tende cadono nel vuoto e
sono penalizzate in entrambi i casi.

### Risultati su `Terrain_unity.csv` (dati puliti, post-fix, parametri tarati)

| Metodo | Accuratezza media | Completezza media | Hausdorff (completezza) | Tempo |
|---|---|---|---|---|
| Poisson (`rimuovi_bassa_densita=0.35`) | 0.46 m | **0.20 m** | **1.2 m** | **4.6 s** |
| SA-ConvONet (`unit_size=0.005`) | 0.37 m | 0.22 m | 1.8 m | 63.0 s |
| PPSurf (default) | **0.10 m** | 0.64 m | 4.6 m | 226.7 s |

Pattern opposto tra le due colonne: PPSurf vince nettamente in accuratezza
(mesh liscia, aderente ai punti campionati) ma perde nettamente in
completezza (buchi/estrapolazioni sbagliate pesano di più sul vero
terreno) — coerente con l'ipotesi di domain gap della Fase 1. Poisson ben
tarato resta il miglior compromesso complessivo, e di gran lunga il più
veloce.

### Tuning Poisson: `rimuovi_bassa_densita`

Curva a 4 punti su `Terrain_unity.csv` (GT completezza media): 0.02→0.868,
0.05→0.588, 0.15→0.436, **0.35→0.199** (nessun plateau raggiunto). Il
compromesso: area di superficie coperta scende da 6.298 a 1.943 m² (più
buchi, ma "onesti" — dove Poisson ricostruisce, il profilo di sezione
segue bene il ground truth). Valori oltre 0.35 non ancora esplorati.

**Aggiornamento: soglia rifatta a 0.05 dopo il fix delle normali.** La
curva sopra era falsata: le normali venivano stimate con un raggio fisso di
0.1 m, ma dopo il downsampling a voxel 0.5 m nessun punto aveva vicini
entro quel raggio e tutte le normali restavano al default (0,0,1). Ora
`stima_normali` (`utils/mesh.py`) usa i 30 vicini più prossimi (default
Open3D), indipendente dalla densità, e le normali servono solo a Poisson
(PPSurf e SA-ConvONet non le usano: niente stima per loro, a SA-ConvONet
arrivano normali fittizie solo perché il dataloader si aspetta la chiave).
Con le normali corrette, `benchmark/sweep_densita_poisson.py` (6 terreni
sintetici dello split val, 1 per famiglia, F-score e Chamfer contro il
fondale, `tau` = 0.5 m):

| Quantile | 0.0 | 0.02 | 0.03 | **0.05** | 0.07 | 0.1 | 0.2 | 0.35 |
|---|---|---|---|---|---|---|---|---|
| F-score medio | 0.903 | 0.976 | 0.990 | **0.998** | 0.995 | 0.984 | 0.936 | 0.820 |
| Chamfer medio | 0.34 m | 0.08 m | 0.06 m | **0.05 m** | 0.05 m | 0.07 m | 0.13 m | 0.30 m |

Il massimo è netto a 0.05, che diventa il default in `config/config.yaml`.
I risultati Poisson nella tabella su `Terrain_unity.csv` qui sopra sono
precedenti al fix, quindi con la vecchia soglia 0.35.

### Tuning SA-ConvONet: `unit_size`/`query_vol_size`

Curva a campana, testata su `Terrain_unity.csv`:

| unit_size | GT completezza media | Note |
|---|---|---|
| 0.002 | 3.38 m (pessimo) | Overfitting locale + **allucinazioni**: la rete genera forme da oggetti indoor (blocchi/mobili squadrati) dove mancano dati — confermato visivamente con screenshot, sparito a 0.01 |
| **0.005** | **0.26 m (migliore)** | Punto di equilibrio |
| 0.01 | 0.30 m | ~2× più veloce (34s vs 65s), leggermente peggiore, mesh più frammentata (buchi più ampi ma onesti, niente allucinazioni) |

Su dati sonar reali (`OS3D_Max_Range_Test.csv`), SA-ConvONet **non
replica** il vantaggio visto su Unity (fedeltà 1.76 m, peggiore di Poisson
e comparabile a PPSurf) — il gap di dominio non si risolve con la sola
finestratura, rafforza la necessità di un vero fine-tuning. (Questi risultati
sono senza ottimizzazione test-time: la pipeline usava `generate.py`, cioè
ConvONet pre-addestrato. L'ottimizzazione è stata provata dopo, vedi sotto.)

### SA-ConvONet con ottimizzazione online (test-time): scartata, infattibile

La "SA" di SA-ConvONet è l'ottimizzazione *sign-agnostic* a test-time: prima
di estrarre la mesh, la rete viene riottimizzata sulla point cloud della
singola scena (`generate_optim_largescene.py`, senza ground truth). È
integrata come opzione, **disattivata di default**:
`sa_convonet.ottimizzazione_online: true` in `config/config.yaml` (nei CSV il
metodo compare come `sa_convonet_online`). Due accorgimenti nel wrapper
(`scripts/sa_convonet.py`): una sola soglia di estrazione (lo script genera
una mesh completa per soglia a ogni decadimento del learning rate) e il
checkpoint passato come file locale della cache di torch (con un URL lo
script originale va in errore).

**Perché è infattibile per il nostro obiettivo** (ricostruzione in pochi
secondi, finestra per finestra):

- **Memoria**: con i parametri degli autori (`batch_size: 3` crop per passo)
  va in *CUDA out of memory* sui 4 GB della GTX 1650 — a differenza
  dell'inferenza, qui si allena l'encoder (UNet3D su crop di ~88³ voxel).
  Entra solo con `batch_size: 1`, a ~3.9 GB su 4.
- **Tempo**: ~2.35 s per passo (GTX 1650, batch 1). Con i default della repo
  (2000 passi) sono ~78 min di ottimizzazione + 4 generazioni complete,
  **~1.5 h per terreno**. Anche fermandosi a ~600 passi, dove il paper
  (appendice B) indica che i risultati si stabilizzano, servono ~25 min. Il
  paper non riporta tempi assoluti, ma indica esplicitamente la lentezza
  dell'inferenza come limite del metodo (sez. 8).
- **Budget reale**: con lo sliding window (300 ping, sovrapposizione 60,
  6 ping/s) arriva una finestra nuova ogni ~40 s. L'ottimizzazione online è
  fuori budget di circa due ordini di grandezza.

**Risultato della prova** (`flat_sand_015`, split test, solo 100 passi di
ottimizzazione, ~375 s), confrontato con le altre mesh dello stesso terreno.
"Superfici per raggio" conta quante volte un raggio verticale attraversa la
mesh (griglia 60×60): un fondale corretto ne ha una sola.

| Metodo | Accuratezza | Completezza | Copertura | Superfici per raggio | Spessore tra le due superfici |
|---|---|---|---|---|---|
| Poisson | **0.22 m** | 0.61 m | 77% | 0: 57%, 1: 37%, 2: 5% | — |
| PPSurf | 0.75 m | 0.31 m | **97%** | 2: 98% | 1.05 m |
| SA-ConvONet | 0.29 m | 0.56 m | 60% | 0: 60%, 2: 40% | 0.56 m |
| SA-ConvONet online | 0.50 m | **0.30 m** | 87% | 2: 98% | 1.22 m |

L'ottimizzazione online aumenta copertura e completezza, ma peggiora
l'accuratezza perché la mesh diventa un **guscio doppio** spesso ~1.2 m
attorno ai punti (strato alto ~0.3 m sopra il fondale vero, strato basso
~0.8 m sotto): la loss sign-agnostic non distingue sopra da sotto, quindi
mette superficie su entrambi i lati. Lo stesso guscio doppio si vede in
PPSurf (98% dei raggi, ~1 m): la sua copertura alta va letta con questo in
mente. Le distanze medie (accuratezza/completezza) non penalizzano
esplicitamente una superficie doppia — da tenere presente nei confronti; la
`precisione` (e quindi l'F-score), aggiunta dopo, la penalizza quando lo
strato spurio dista dal fondale più di `tau` (es. lo strato basso a ~0.8 m
con `tau` = 0.5 m).

**Decisione**: niente adattamento online. L'adattamento al dominio sonar va
fatto *offline*, con il fine-tuning supervisionato sui terreni sintetici;
in ricostruzione resta un solo passaggio in avanti della rete
(`generate.py`), da confrontare con Poisson sullo stesso budget per finestra.

## Prossimi passi

**Fine-tuning (Passo 2 di Daniele)**: in corso su SA-ConvONet.
Dataset di training generato dai terreni sintetici con
`utils/make_saconvonet_dataset.py` (input = scansioni simulate, occupancy =
"sotto il fondale" dalla ground truth, stessa normalizzazione
dell'inferenza); training con `neural_methods/SA-ConvONet/fine_tuning.py` e
`configs/seabed/finetune_crop.yaml`. Il dry-run sulla GTX 1650 entra in
memoria (~2 GB, ~1 s/iterazione, solo decoder). Prima di un run lungo: run
di prova breve per verificare che la loss di validazione scenda.

**Sliding window / valutazione online (Passo 3 di Daniele)**: implementato
(sezione `sliding_window` in `config/config.yaml`, vale sia per
`reconstruct_single_dataset.py` sia per `evaluate.py`): il CSV viene letto a
blocchi, ricostruito a finestre di ping e i dati usati vengono scartati.
Prossimo passo: misurare il tempo per finestra di ciascun metodo rispetto
al budget (~40 s con i valori di default; strumenti pronti in `benchmark/`,
che separano il costo fisso di avvio librerie/CUDA da quello della
ricostruzione vera e propria, es. `bash benchmark/tempo_ppsurf.sh
<scan.csv> [ripetizioni]`) e derivare una funzione
`qualità_ricostruzione ≈ f(densità_locale, dimensione_finestra)` calibrata
sui metodi già integrati, invece di usare la varianza di un GP generico
scollegato dalla pipeline reale. Per studiare la dipendenza dalla densità,
downsampling e rimozione degli outlier si possono disattivare dal config
(`preprocessing.downsampling.abilitato`, `preprocessing.rimozione_outlier.abilitato`).
`evaluate.py --limit N` prende i primi N terreni di **ogni famiglia**, non N
in totale.

Riferimento concettuale alternativo per il fine-tuning, non ancora
esplorato: famiglia di metodi neurali *sonar-native* (NeuSIS/Qadri et al.
ICRA 2023, NeuRSS) che modellano esplicitamente la propagazione acustica —
pensati per forward-looking sonar (FLS), non per multibeam batimetrico
come l'Omniscan; adattarli richiederebbe reimplementare il renderer
differenziabile per la geometria del nostro sensore.

## Fase 3 — Mapping attivo con RL (visione, non ancora iniziata)

Architettura a due binari, per non perdere l'obiettivo di avere sempre una
**mesh 3D vera** come output finale:

- **Binario navigazione**: segnale di incertezza leggero (densità locale di
  punti/sliding window, o GP grossolano aggiornato periodicamente) guida
  l'agente in tempo reale — reward basato su riduzione dell'incertezza
  batimetrica.
- **Binario output**: Poisson/PPSurf/SA-ConvONet continuano a generare la
  mesh reale, a checkpoint o a fine missione.

Gap identificato rispetto ai due paper di riferimento: nessuno dei due fa
"RL + incertezza derivata dal proprio metodo di ricostruzione + mesh 3D
vera da sonar multibeam reale" — Mercado et al. non ricostruisce geometria
(solo detection/coverage) e usa un'incertezza euristica; Chavez-Galaviz et
al. ricostruisce con GP standard una mappa 2.5D, non una mesh 3D da point
cloud multibeam reale.

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
- **UnitySensors, bug raycast miss → punti fantasma**: vedi Fase 2 sopra.
  Qualunque nuovo sensore/setup Unity va verificato per lo stesso pattern
  (controllare `intensity == 0` nei dati grezzi) prima di fidarsi.
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
  (~450m) generano ~2200 crop (2+ ore, mai finito nel test). Valore in uso:
  `unit_size: 0.005` (vedi curva di tuning sopra) → ~140 crop, ~1 min. La
  mesh generata è inoltre in coordinate normalizzate (centrate/scalate da
  `make_cropscene_dataset2.py`) e va riportata in metri reali applicando
  l'inverso (`loc`/`scale` salvati nel `pointcloud.npz` normalizzato) —
  gestito in `scripts/sa_convonet.py`.
- **Visualizzatore Open3D su WSLg**: `vis.create_window()` fallisce
  silenziosamente (ritorna `False`) e `vis.get_render_option()` ritorna
  `None` (`AttributeError` a valle) perché GLFW tenta Wayland nativo, non
  supportato dal setup grafico di questa macchina. Fix: `export
  WAYLAND_DISPLAY=` (stringa vuota, non `unset` — GLFW controlla solo se la
  variabile è definita) forza il fallback a XWayland. Automatizzato in
  `setup_ambiente.sh` (da lanciare con `source` a inizio sessione).
- **matplotlib headless (WSL senza display)**: qualunque script che importa
  `matplotlib.pyplot` va preceduto da `matplotlib.use("Agg")` **prima**
  dell'import di `pyplot`, altrimenti puo' scegliere un backend interattivo
  (es. QtAgg) che crasha (`Aborted (core dumped)`) al primo utilizzo, anche
  solo salvando un file — non basta rimuovere `plt.show()`.
- **WSL**: attenzione allo spazio disco reale su C: (il file `.vhdx` cresce
  ma non si restringe da solo — `wsl --shutdown` + `Optimize-VHD`/`diskpart
  compact vdisk` per compattarlo se necessario).