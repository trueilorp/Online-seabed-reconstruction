# Analisi: esperimenti, scelte dei parametri e metodi scartati

Complemento del [README](README.md): qui c'è il *perché* delle scelte fatte nella
pipeline. Il README descrive cosa fa il progetto e i risultati finali.

## Indice

1. [Dimensione della finestra e buchi di PPSurf](#dimensione-della-finestra-e-buchi-di-ppsurf)
2. [La cartella benchmark: esperimenti fatti e risultati](#la-cartella-benchmark-esperimenti-fatti-e-risultati)
3. [Perché PPSurf è configurato così](#perché-ppsurf-è-configurato-così)
4. [Metodi scartati](#metodi-scartati)

## Dimensione della finestra e buchi di PPSurf

Con finestre da 300 ping (~50 m, run sul portatile,
`PPSURF_2026-11-10_23-40-13_test+val_sw`, 2 terreni per famiglia) PPSurf + SW
lasciava molti buchi sui fondali lisci. Dimezzando la finestra i buchi
spariscono:

| Richiamo τ=0.5 (PPSurf + SW) | flat_sand | mixed | ridges | rocky | sand_ripples | slope |
|---|---|---|---|---|---|---|
| finestre da 300 ping | 0.799 | 0.946 | 0.919 | 0.986 | 0.845 | 0.858 |
| finestre da 150 ping | **0.996** | **0.998** | **0.997** | **0.999** | **0.996** | **0.984** |

Altre prove fatte su `flat_sand_012` e altri terreni `*_012` prima di
scegliere:

- **Griglia 193 invece di 129** (finestre da 300 ping): richiamo +2.5/+5
  punti su `mixed` e `ridges`, tempo ×2.3, e `flat_sand_012` in timeout. La
  cella di marching cubes non era la causa principale.
- **Sovrapposizione 30 invece di 60** (finestre da 300 ping): nessun effetto
  (richiamo 0.839 → 0.843).
- **Finestre da 150 ping**: richiamo τ=0.2 0.70 → 0.98, p95 fondale→mesh
  1.22 → 0.16 m. Negli screenshot restano pochi buchi piccoli, ~2% della
  superficie.

Il limite di PPSurf dipende quindi dall'estensione della scena che riceve,
non dalla griglia: è lo stesso effetto visto con le porzioni da 25 m nel
[benchmark (test 5)](#5-ppsurf-a-porzioni-tile_ppsurfpy).

## La cartella benchmark: esperimenti fatti e risultati

Gli script di `benchmark/` sono esperimenti mirati, fatti per rispondere a
domande precise su costi e qualità. Si lanciano tutti dalla root del
progetto e scrivono in `benchmark/benchmark_output/`, una cartella ignorata da
git: i risultati sono riportati qui sotto. Salvo dove indicato, le metriche
sono le stesse di `evaluate.py` (contro il fondale, τ = 0.5 m).

| Script | Domanda | Output |
|---|---|---|
| `tempo_poisson.sh`, `tempo_ppsurf.sh` | Quanto costa ogni fase di ciascun metodo? | `tempi.csv`, `poisson/`, `ppsurf/` |
| `sweep_densita_poisson.py` | Qual è la soglia giusta per il taglio per densità di Poisson? | `sweep_densita_poisson.csv` |
| `strato_singolo_ppsurf.py` | Quanto pesa il guscio doppio di PPSurf? | `strato_singolo_ppsurf.csv`, `ppsurf_strato_singolo/` |
| `crop_ppsurf.py` | Il limite di PPSurf è la griglia o il modello? | `crop_ppsurf*.csv`, `ppsurf_crop/` |
| `tile_ppsurf.py` | PPSurf a porzioni arriva al livello di Poisson? | `tile_ppsurf.csv`, `ppsurf_tile/` |
| `comune.sh`, `prepara_input.py`, `poisson_fasi.py`, `metriche_superficie.py` | funzioni di supporto per gli script dei tempi | – |

### 1. Tempi per fase (`tempo_*.sh`)

```bash
bash benchmark/tempo_poisson.sh data/synthetic/val/flat_sand_012/flat_sand_012_scan.csv 3
bash benchmark/tempo_ppsurf.sh  data/synthetic/val/flat_sand_012/flat_sand_012_scan.csv 3
```

`prepara_input.py` esegue il preprocessing una volta sola, fuori dal
cronometro. Poi ogni fase del metodo viene cronometrata con picco di RAM e
VRAM. Per PPSurf, `avvio_librerie` misura il solo costo fisso di avvio
(Python, PyTorch, CUDA) e `rec` la ricostruzione completa. Su `flat_sand_012`
(77 702 punti dopo il preprocessing), GTX 1650 da 4 GB, 3 ripetizioni:

| Metodo | Fase | Tempo | RAM picco | VRAM picco |
|---|---|---|---|---|
| Poisson | ricostruzione (octree depth 9) | 5–12 s | ~470 MB | – |
| Poisson | processo completo (avvio + normali + ricostruzione + pulizia) | 9–16 s | ~470 MB | – |
| PPSurf | avvio librerie + CUDA | 4–10 s | ~690 MB | – |
| PPSurf | `pps.py rec` completo | ~170 s | ~1.6 GB | ~1.9 GB |

PPSurf costa circa 15–20 volte Poisson, e il suo tempo dipende dalla griglia
di marching cubes, non dal numero di punti. In `tempi.csv` ci sono anche
Chamfer e F-score di queste run, ma sono precedenti al fix delle normali di
Poisson: per la qualità valgono le tabelle seguenti.

### 2. Soglia del taglio per densità di Poisson (`sweep_densita_poisson.py`)

Poisson viene eseguito una volta per terreno. Poi la mesh grezza viene
tagliata a diversi quantili di densità e ognuno viene misurato. Prova su
6 terreni val, 1 per famiglia:

| Quantile | 0.0 | 0.02 | 0.03 | **0.05** | 0.07 | 0.1 | 0.2 | 0.35 |
|---|---|---|---|---|---|---|---|---|
| F-score medio | 0.903 | 0.976 | 0.990 | **0.998** | 0.995 | 0.984 | 0.936 | 0.820 |
| Chamfer medio | 0.34 m | 0.08 m | 0.06 m | **0.05 m** | 0.05 m | 0.07 m | 0.13 m | 0.30 m |

Il massimo a 0.05 è netto ed è il valore in config. Senza taglio restano le
"tende" spurie ai bordi; tagliando troppo si aprono buchi.

Prima di questo sweep le normali erano stimate con un raggio fisso di 0.1 m.
Dopo il voxel da 0.5 m nessun punto aveva vicini entro quel raggio, quindi
tutte le normali restavano (0,0,1). Ora `stima_normali` usa i 30 vicini più
prossimi.

### 3. Il guscio doppio di PPSurf (`strato_singolo_ppsurf.py`)

PPSurf, pre-addestrato su oggetti chiusi, avvolge il fondale aperto in un
solido sottile: quasi ogni raggio verticale attraversa **due** superfici, una
sopra e una sotto il fondale vero, distanti circa 1 m. Lo script lavora sulle
mesh già salvate da `evaluate.py`, senza GPU. Su una griglia XZ di 0.25 m
prende il primo hit dall'alto (strato alto), il primo dal basso (strato basso)
e il punto medio fra i due. Media sui 6 terreni `*_012`:

| Variante | Chamfer | Precisione | Richiamo | F-score |
|---|---|---|---|---|
| PPSurf originale | 0.344 m | 0.67 | 0.96 | 0.79 |
| strato alto | 0.548 m | 0.52 | 0.52 | 0.52 |
| strato basso | 0.374 m | 0.80 | 0.75 | 0.77 |
| strato medio | 0.262 m | 0.91 | 0.86 | 0.88 |
| Poisson | 0.048 m | 1.00 | 1.00 | 1.00 |

Il guscio spiega una parte dell'errore, non tutto: anche il punto medio resta
più di 5 volte peggio di Poisson. Usare l'input senza downsampling (250k
punti) non cambia nulla: Chamfer 0.42 contro 0.46 m su `flat_sand_012`.

### 4. Griglia o modello? (`crop_ppsurf.py`)

Con `gen_resolution_global: 129` su un terreno da 100 m le celle di marching
cubes sono di ~0.8 m, contro ~0.2 m di Poisson. Lo script rilancia PPSurf su
un ritaglio di 25×25 m, dove con la stessa griglia le celle diventano ~0.2 m.
Poi misura tutte le varianti sul quadrato interno di 20 m. Chamfer in metri:

| Terreno | Completo | Completo, strato medio | Ritaglio | **Ritaglio, strato medio** | Poisson |
|---|---|---|---|---|---|
| flat_sand_012 | 0.454 | 0.301 | 0.113 | **0.035** | 0.039 |
| ridges_012 | 0.302 | 0.325 | 0.107 | **0.054** | 0.031 |
| rocky_012 | 0.292 | 0.326 | 0.111 | **0.093** | 0.047 |
| **Media** | 0.349 | 0.317 | 0.110 | **0.061** | 0.039 |

Sul ritaglio il guscio scende da 0.9–1.2 m a 0.25–0.34 m, e con lo strato
medio PPSurf arriva al livello di Poisson. Il ritaglio però riduce insieme la
cella e l'estensione della scena, quindi questo test da solo non dice quale
delle due conta.

**Il test che separa i due effetti**: griglia più fine sull'intero terreno
(GTX TITAN X da 12 GB, split val, 1 terreno per famiglia):

| `gen_resolution_global` | Cella su 100 m | Chamfer | F-score | Tempo/terreno |
|---|---|---|---|---|
| 129 | ~0.8 m | 0.360 m | 0.779 | ~46 s |
| 257 (default del paper) | ~0.4 m | 0.363 m | 0.782 | ~187 s |
| 519 | ~0.2 m | 0.354 m | 0.786 | ~918 s |

Risoluzione ×4, tempo ×20, nessun miglioramento: le differenze sono nel
rumore tra run (~0.015 m). A 519 il guscio resta di 0.8–0.9 m. **Lo spessore
del guscio è circa l'1% del lato della scena** (100 m → ~0.9 m, 25 m →
~0.3 m), qualunque sia la griglia. Questo è coerente con la normalizzazione
di PPSurf, che porta ogni point cloud nel cubo unitario: il guscio ha uno
spessore fisso in coordinate normalizzate. Il limite è quindi nel campo
predetto dalla rete, non nella discretizzazione.

### 5. PPSurf a porzioni (`tile_ppsurf.py`)

Applica la conclusione precedente a tutto il terreno. Il terreno viene diviso
in quadrati da 25 m (nucleo di 20 m + 2.5 m di margine per lato, scartato
nell'unione). Tutti i quadrati passano in un solo sottoprocesso PPSurf
(modalità dataset, modello caricato una volta) e vengono riuniti in due modi:
`tile_originale` (mesh ritagliate sui nuclei) e `tile_medio` (strato medio su
una griglia globale). Griglia 129 per quadrato, 25 quadrati per terreno,
terreni `*_012`, TITAN X:

| Variante | Chamfer | Precisione | Richiamo | F-score | Spessore del guscio |
|---|---|---|---|---|---|
| PPSurf sull'intero terreno | 0.344 m | 0.67 | 0.96 | 0.79 | ~1.0 m |
| PPSurf sull'intero terreno, strato medio | 0.262 m | 0.91 | 0.86 | 0.88 | ~1.0 m |
| **a porzioni, originale** | **0.121 m** | 0.98 | 0.99 | **0.99** | 0.28 m |
| **a porzioni, strato medio** | **0.086 m** | 0.99 | 0.99 | **0.99** | 0.28 m |
| Poisson | 0.048 m | 1.00 | 1.00 | 1.00 | – |

A τ = 0.5 m PPSurf a porzioni è alla pari con Poisson. Il Chamfer resta circa
1.8 volte quello di Poisson, ma tutto lo scarto sta sotto la soglia. Lo strato
medio aggiunge poco ed essendo 2.5D cancella tubi e piloni: `tile_originale`
è la variante da tenere.

**Il costo resta l'ostacolo**: ~17 min per terreno (~41 s per quadrato,
fissati dalla griglia) contro ~7 s di Poisson, circa 150 volte tanto.

## Perché PPSurf è configurato così

- **Perché PPSurf e non un altro metodo neurale.** È patch-based: ricostruisce
  porzione per porzione invece di deformare una forma globale, quindi si adatta
  a superfici aperte e allungate come una scansione multibeam. Gli altri
  candidati sono stati scartati (vedi [Metodi scartati](#metodi-scartati)).
  SA-ConvONet, l'ultimo rimasto in gara, andava peggio di PPSurf (F-score ~0.59
  contro ~0.70 su `flat_sand_012`). La sua variante con adattamento per scena
  era infattibile: ~1.5 h per terreno, e va in OOM su 4 GB.
- **`gen_resolution_global: 129`** (default 257): su 4 GB di VRAM il default va
  in OOM. Sulla TITAN X da 12 GB, 257 e 519 non migliorano la qualità (test 4)
  e costano 4 e 20 volte di più. Non c'è motivo di alzarla.
- **`rec_batch_size: 10000`** (default 50000): serve solo a stare nei 4 GB
  durante il marching cubes. Non cambia il risultato. Sulla TITAN X si usa
  50000 (`--set`), con ~8.6 GB di VRAM.
- **`gen_subsample_manifold_iter: 3` e `gen_refine_iter: 3`** (default 10 e
  10): circa un terzo del tempo in meno, a qualità invariata o leggermente
  migliore.
  - Il primo parametro è la *test-time augmentation*: la parte globale della
    rete viene ripetuta su sottoinsiemi dei punti, finché ogni punto è stato
    usato tante volte quanto il valore impostato.
  - Il secondo è il numero di iterazioni con cui PPSurf raffina le intersezioni
    di marching cubes.

  Prove sui 6 terreni `*_012`, senza SW, GTX 1650 (con `evaluate.py --set`):

  | TTA / refine | Tempo per terreno | F τ=0.1 | F τ=0.2 | F τ=0.5 |
  |---|---|---|---|---|
  | 10 / 10 (default) | 176 s | 0.151 | 0.315 | 0.801 |
  | 5 / 10 | 167 s | – | – | 0.808 |
  | 3 / 10 | 164 s | – | – | 0.818 |
  | 10 / 5 | 133 s | 0.151 | 0.314 | 0.806 |
  | 10 / 3 | 119 s | 0.149 | 0.313 | 0.801 |
  | **3 / 3** | **112 s** | **0.169** | **0.351** | **0.817** |

  La leva vera è il raffinamento: posizionare meglio la superficie dentro la
  cella non serve, se la superficie è comunque spostata di 20–50 cm per il
  guscio doppio. La test-time augmentation pesa poco sul tempo, e ridurla ha
  migliorato un po' la qualità due volte su due. Un'ipotesi: mediando meno, il
  guscio è meno regolare e tocca più spesso il fondale vero. Non è verificata,
  e la prova è su soli 6 terreni.

  I circa 110 s che restano sono il costo dell'inferenza della rete sui punti
  della griglia vicini alla superficie. PPSurf non valuta tutto il cubo: usa un
  marching cubes a *region growing* che parte dai punti di input (paper,
  sez. 3.3).
- **Stesso preprocessing di Poisson** (voxel 0.5 m + outlier): togliere il
  downsampling non cambia la qualità e allunga i tempi (test 3).
  `max_punti: 250000` è solo un limite di sicurezza.
- **Niente strato medio nella pipeline**: migliora un po' il Chamfer ma è
  2.5D, quindi cancella proprio le strutture (tubi, piloni) dove un metodo
  neurale potrebbe battere Poisson.
- **Scene piccole invece di griglie fini**: poiché il guscio scala con
  l'estensione della scena (test 4 e 5), la leva giusta è ricostruire scene più
  piccole. A porzioni funziona ma costa troppo. La sliding window produce
  scene piccole per costruzione, e ne eredita il beneficio: con finestre da 150
  ping F τ=0.2 da 0.14–0.54 a 0.80–0.86 (vedi
  [Risultati](README.md#risultati-poisson-e-ppsurf-con-e-senza-sw) e
  [Dimensione della finestra](#dimensione-della-finestra-e-buchi-di-ppsurf)).
- **Prossimi passi possibili**: tenere il modello caricato tra una finestra e
  l'altra della sliding window (5–10 s risparmiati a finestra), oppure un
  fine-tuning su superfici aperte, che attacca la causa del guscio. Gli autori
  stessi definiscono i tempi di ricostruzione *"non-interactive"* (paper,
  sez. 4.3).

## Metodi scartati

| Metodo | Motivo |
|---|---|
| Point2Mesh | Parte da un convex hull, pensato per forme compatte di genere 0. Sul nastro di scansione più iterazioni peggiorano la fedeltà. 1000 s per un risultato 3.5 volte peggio di Poisson. |
| SA-ConvONet | Sui sintetici era peggio di PPSurf. L'ottimizzazione per scena (la parte "SA") richiede ~1.5 h per terreno e va in OOM su 4 GB. Dipendenze ferme a PyTorch 1.4 / CUDA 10.1. Rimosso dal codice, recuperabile dalla history git. |
| NKSR | Wheel non più disponibili, va compilato con CUDA 12.8; probabile OOM su 4 GB. |
| DeepSDF, if-net | Richiedono un training su un dataset di forme (prior di forma); i pesi disponibili sono su ShapeNet. |
| NeuS, 3DGS | Richiedono immagini RGB multi-vista, mentre il progetto resta sul sonar. |
| PointNet++, DGCNN | Estrattori di feature, non producono mesh. |
| Metodi a diffusione (WALA, Make-A-Shape) | Immaturi su input sparsi e rumorosi, addestrati su oggetti e non su scene. |
