# Progetto Barca — Ricostruzione 3D del fondale da sonar multibeam

Ricostruzione 3D del fondale e di infrastrutture sommerse (tubi, piloni) da
point cloud di sonar multibeam (Cerulean Omniscan 3D), in collaborazione con
un'azienda che produce robot subacquei. Supervisione: Daniele.

Oggi il progetto confronta due metodi, **Poisson** (classico) e **PPSurf**
(neurale, pre-addestrato), su terreni sintetici con ground truth nota. Ogni
metodo gira sia sull'intera scansione sia a **finestre di ping**
(*sliding window*, SW), cioè come se i dati arrivassero dal sonar durante la
navigazione.

La storia completa delle fasi precedenti (dati reali, Unity, Point2Mesh,
SA-ConvONet) è in [README_old.md](README_old.md). Le istruzioni di
installazione sono in [SETUP.md](SETUP.md).

## Indice

1. [Struttura del progetto](#struttura-del-progetto)
2. [Come funziona](#come-funziona)
3. [Uso](#uso)
4. [Risultati: Poisson e PPSurf, con e senza SW](#risultati-poisson-e-ppsurf-con-e-senza-sw)
5. [La cartella benchmark: esperimenti fatti e risultati](#la-cartella-benchmark-esperimenti-fatti-e-risultati)
6. [Perché PPSurf è configurato così](#perché-ppsurf-è-configurato-così)
7. [Metodi scartati](#metodi-scartati)
8. [Note tecniche](#note-tecniche)

## Struttura del progetto

```
.
├── evaluate.py                     # ENTRY POINT: ricostruisce e valuta i terreni sintetici
├── config/config.yaml              # parametri di preprocessing, sliding window, metodi e valutazione
├── scripts/
│   ├── poisson.py                  # Poisson Surface Reconstruction (Open3D) + taglio per densità + pulizia
│   └── ppsurf.py                   # PPSurf lanciato come sottoprocesso nell'ambiente conda "pps"
├── utils/
│   ├── pipeline.py                 # caricamento config, preprocessing, esecutori dei metodi, sliding window
│   ├── mesh.py                     # lettura CSV, pulizia point cloud, normali, top_surface (fondale della GT)
│   ├── generate_seabed_heightmaps.py  # genera i fondali sintetici (6 famiglie, con tubi e piloni)
│   ├── simulate_multibeam_scans.py    # simula la scansione multibeam -> <id>_scan.csv + <id>_ground_truth.obj
│   ├── mesh_comparison.py          # (legacy) grafici di confronto sul dataset Unity Terrain_unity
│   └── comparison_ground_truth_obj_and_unity_csv.py  # (legacy) allineamento CSV Unity / OBJ
├── benchmark/                      # esperimenti di diagnosi e tempi (vedi sezione dedicata)
│   └── benchmark_output/           # i loro risultati (CSV, mesh; ignorata da git)
├── visualization/
│   ├── visualize_mesh.py           # visualizzatore Open3D di una mesh .ply (con screenshot)
│   └── orbite_mesh.py              # video orbitale attorno a una mesh (per slide)
├── data/
│   ├── data_analisys.py            # analisi esplorativa di un CSV di point cloud
│   ├── sonar/                      # dati reali Omniscan 3D e scansioni Unity
│   └── synthetic/                  # terreni sintetici: train/ val/ test/ test+val/
├── README_old.md                   # README storico (fasi precedenti)
├── output/                         # tutto ciò che producono le run (ignorata da git)
│   └── evaluation/<data_ora>_<split>[_sw]/  # una cartella per run: config.txt, CSV, meshes/, distanze/
├── neural_methods/ppsurf/          # repo PPSurf clonato (non tracciato)
├── env_setup.sh                    # "source env_setup.sh": fix WSLg + attiva l'ambiente conda
├── requirements.txt                # dipendenze dell'ambiente "ricostruzione3d"
└── SETUP.md                        # installazione su una macchina nuova
```

## Come funziona

### 1. Dati sintetici

I dati reali non hanno una ground truth, quindi la valutazione si fa su
fondali sintetici generati in due passi:

1. `utils/generate_seabed_heightmaps.py` genera heightmap di circa 100×100 m in
   **6 famiglie**: `flat_sand`, `sand_ripples`, `ridges`, `rocky`, `slope` e
   `mixed`. Con `--with-structures` aggiunge tubi e piloni. Lo split è fatto
   per terreno: train 72, val 18, test 12.
2. `utils/simulate_multibeam_scans.py` simula il sonar. La piattaforma percorre
   una serpentina; a ogni ping un ventaglio di raggi viene tracciato sulla mesh,
   con rumore sul range, dropout e outlier. I parametri sono stimati dal dataset
   reale `OS3D_Max_Range_Test.csv` (~6 ping/s, ~276 punti per ping). Per ogni
   terreno produce:
   - `<id>_scan.csv`: la point cloud (`x,y,z,intensity,ping_number,elapsed_s`, con Y verticale);
   - `<id>_ground_truth.obj`: la mesh vera, chiusa (fondale + pareti + fondo).

`data/synthetic/test+val/` raccoglie i 30 terreni di test e val, 5 per famiglia.
È l'insieme su cui girano i test attuali.

### 2. Pipeline di ricostruzione (`utils/pipeline.py`)

```
<id>_scan.csv ──► preprocessing ──► metodo ──────────────────────────────► ritaglio ──► mesh
                  voxel 0.5 m        Poisson: normali (30 vicini) → depth 9   via i triangoli
                  + outlier          → taglio del 5% a densità più bassa      a > 1 m dai punti
                  statistici         PPSurf: sottoprocesso pps.py, griglia 129
```

- **Preprocessing** (sezione `preprocessing`): downsampling su voxel da 0.5 m e
  rimozione statistica degli outlier (20 vicini, 2 std). È uguale per tutti i
  metodi.
- **Poisson** (`scripts/poisson.py`): stima delle normali, ricostruzione,
  rimozione dei vertici a bassa densità, dove Poisson "inventa" superficie
  lontano dai punti. Le mesh vengono salvate senza colori.
- **PPSurf** (`scripts/ppsurf.py`): la point cloud viene scritta su disco e
  ricostruita da `neural_methods/ppsurf/pps.py` nell'ambiente conda `pps`. La
  mesh viene riletta in metri.
- **Ritaglio** (sezione `postprocessing`): vengono tolti i triangoli il cui
  baricentro dista più di 1 m (`distanza_max_punti`) dal punto più vicino della
  point cloud pulita. Serve soprattutto alla sliding window (vedi sotto), ma si
  applica anche senza, così i due modi restano confrontabili. La soglia deve
  restare sopra τ (0.5 m): con una soglia ≤ τ la precisione sarebbe garantita
  dal ritaglio stesso.

### 3. Sliding window (sezione `sliding_window`)

Simula l'uso online: il CSV viene letto a blocchi di righe come se arrivasse
dal sonar, e i punti vengono raggruppati in finestre di **150 ping** (circa
25 m di rotta) che si sovrappongono di 30 ping. Con finestre da 300 ping
(~50 m) PPSurf lasciava molti buchi sui fondali lisci: vedi
[Risultati](#risultati-poisson-e-ppsurf-con-e-senza-sw).

Ogni finestra viene pulita e ricostruita da sola. Della sua mesh si tengono
solo i triangoli della zona "propria" della finestra: la sovrapposizione è
divisa a metà con la finestra successiva. Poi la mesh viene ritagliata sulla
distanza dai punti **della finestra** e unita alle altre. In memoria resta
solo la finestra corrente.

Il ritaglio sulla distanza è indispensabile per Poisson. Una finestra è una
striscia aperta su tutti i lati (~26 m di larghezza), e Poisson la chiude con
una **cupola nell'acqua**, circa 6 m sopra il fondale. Il taglio per densità
al 5% non la toglie: su `flat_sand_012`, senza ritaglio, era spuria il 36%
della superficie (F-score 0.78, p95 ricostruzione→fondale 12.5 m). Con il
ritaglio a 1 m Poisson + SW torna al livello di Poisson sull'intero terreno.
Questi i valori di prova, a τ = 0.5 m, con le finestre da 300 ping usate
allora:

| Terreno | Modo | Chamfer | Precisione | F-score | p95 rec→GT |
|---|---|---|---|---|---|
| flat_sand_012 | intero | 0.050 m | 0.997 | 0.999 | 0.11 m |
| flat_sand_012 | SW, senza ritaglio | 1.133 m | 0.642 | 0.782 | 12.4 m |
| flat_sand_012 | SW, ritaglio 1 m | 0.047 m | 0.988 | 0.994 | 0.15 m |
| rocky_012 | intero | 0.047 m | 1.000 | 1.000 | 0.12 m |
| rocky_012 | SW, senza ritaglio | 0.151 m | 0.905 | 0.950 | 1.19 m |
| rocky_012 | SW, ritaglio 1 m | 0.053 m | 0.986 | 0.993 | 0.22 m |

**Limiti noti della finestratura**:

- Le finestre seguono il tempo, non lo spazio. Le passate adiacenti della
  serpentina si sovrappongono del 30% (strisce di ~26 m ogni ~18 m) e la
  stessa zona viene ricostruita da due finestre lontane nel tempo. Con la SW
  circa metà delle colonne verticali ha quindi **due superfici**, molto vicine
  fra loro: le metriche non ne risentono, la mesh finale però sì.
  Deduplicarle richiederebbe di assegnare ogni zona del fondale alla prima
  finestra che la copre.
- Circa una finestra su tre contiene una virata (passa da una passata alla
  successiva) e ha quindi forma a U invece che a striscia. Finora non ha
  dato problemi.

Si attiva con `--sliding_window on` (o `abilitato: true` nel config). Le mesh
e le distanze prendono il suffisso `_sw`.

**Tempo reale.** Per ogni finestra vengono salvati in `finestre.csv`:

| Campo | Definizione |
|---|---|
| `latenza_s` | da quando la finestra è completa (arriva il suo ultimo ping) a quando la sua mesh è unita al resto: filtri, pulizia, ricostruzione (compreso l'avvio del metodo) e ritaglio |
| `tempo_metodo_s` | solo la ricostruzione, come misurata dal metodo |
| `acquisizione_s` | durata della finestra, dal primo all'ultimo ping (colonna `elapsed_s` della scansione) |
| `budget_s` | tempo tra il completamento della finestra precedente e di questa, cioè il tempo disponibile prima che arrivi la successiva. Con 150 ping e 30 di sovrapposizione arrivano 120 ping nuovi, ~20 s a 6 ping/s |
| `rtf` | *real-time factor* = `latenza_s / budget_s` |

Con **RTF < 1** il metodo tiene il passo con il sonar; con RTF > 1 le finestre
si accumulano. Per ogni terreno il CSV principale riporta anche
`latenza_mediana_s`, `latenza_max_s`, `rtf_mediano`, `rtf_max` e
`finestre_in_tempo` (la frazione di finestre con RTF ≤ 1).

### 4. Valutazione (`evaluate.py`)

Per ogni terreno e metodo vengono campionati **100 000 punti** uniformi sulla
mesh ricostruita e altrettanti sul fondale della ground truth. Poi si misura
la distanza esatta punto-superficie in entrambe le direzioni:

| Metrica | Definizione | Cosa misura |
|---|---|---|
| **precisione** | % di punti della ricostruzione entro τ dal fondale | superficie spuria (guscio doppio, "tende" ai bordi) |
| **richiamo** | % di punti del fondale entro τ dalla ricostruzione | buchi |
| **F-score** | media armonica di precisione e richiamo | sintesi a soglia τ |
| **Chamfer** | (media ricostruzione→fondale + media fondale→ricostruzione) / 2, in metri | errore medio, senza soglia |
| **mediana, IQR, p95** | per ciascuna delle due direzioni (`rec2gt_*`, `gt2rec_*`) | distribuzione degli errori: la mediana è l'errore tipico, l'IQR (p75 − p25) la dispersione, il p95 la coda |

La media, e quindi il Chamfer, è dominata dalle code: pochi metri di
superficie spuria lontana pesano quanto moltissimi punti corretti. Mediana e
p95 separano i due casi, cioè "è tutto un po' impreciso" da "è quasi tutto
giusto, tranne una parte molto sbagliata".

Le metriche si calcolano contro il **solo fondale**, non contro la GT intera.
La GT è un blocco chiuso: le pareti e il fondo servono solo a chiuderla e il
sonar non li vede mai. `top_surface` (`utils/mesh.py`) li stacca. Contro la GT
intera, una superficie spuria che scende lungo le pareti risulterebbe
"corretta"; contro il solo fondale viene penalizzata, com'è giusto.

τ = 0.5 m (pari al voxel del preprocessing). Le colonne `accuratezza_*` del
CSV, invece, sono misurate contro la GT intera e restano solo per confronto
con i CSV vecchi.

**Output.** Ogni lancio di `evaluate.py` crea una cartella nuova,
`output/evaluation/<data_ora>_<split>[_sw]/` (per esempio
`2026-11-10_11-04-59_test+val_sw/`), con tutto ciò che riguarda quella run e
solo quella:

- `config.txt`: la config effettivamente usata, con le opzioni da riga di
  comando già applicate, e il comando lanciato;
- `metriche_per_terreno.csv`: una riga per (terreno, metodo);
- `metriche_per_famiglia.csv`: medie per famiglia;
- `finestre.csv` (solo con SW): una riga per finestra, con latenza e RTF;
- `meshes/<id>_<metodo>[_sw].ply`;
- `distanze/<id>_<metodo>[_sw]_{rec2gt,gt2rec}.npy`: i due vettori di
  distanza grezzi (float32, 100k valori). `rec2gt` va dalla ricostruzione al
  fondale, `gt2rec` dal fondale alla ricostruzione.

**Altre soglie τ senza ricostruire.** Dalle distanze grezze di una run si
ricalcolano Chamfer, precisione, richiamo e F-score per qualsiasi τ:

```bash
python evaluate.py --ricalcola output/evaluation/<run> --tau 0.5 0.2 0.1
```

Il comando scrive nella cartella della run `metriche_tau.csv` (una riga per
terreno, metodo e τ) e `metriche_tau_per_famiglia.csv` (medie), e stampa le
medie per metodo.

## Uso

Dalla root del progetto, con l'ambiente `ricostruzione3d` attivo
(`source env_setup.sh`):

```bash
# test completi su test+val (30 terreni): le quattro configurazioni
python evaluate.py --split test+val --methods poisson ppsurf --sliding_window off
python evaluate.py --split test+val --methods poisson ppsurf --sliding_window on

# prova rapida: 1 terreno per famiglia, un solo metodo
python evaluate.py --split val --limit 1 --methods poisson

# metriche a τ diversi su una run già fatta (nessuna ricostruzione)
python evaluate.py --ricalcola output/evaluation/<run> --tau 0.5 0.2

# guardare una mesh
python visualization/visualize_mesh.py      # percorso impostato in testa al file
```

`--limit N` prende i primi N terreni **di ogni famiglia**, non N in totale.
Senza argomenti `evaluate.py` usa i default della sezione `evaluation` del
config (`test+val`, tutti i terreni, Poisson e PPSurf).

## Risultati: Poisson e PPSurf, con e senza SW

**Domande**

1. Quanto si perde ricostruendo a finestre, come servirà online, rispetto
   all'intera scansione?
2. Ciascun metodo tiene il passo con il sonar, cioè ha RTF < 1?

**Setup**: split `test+val` (30 terreni, 5 per famiglia), finestre da 150
ping con 30 di sovrapposizione (~25 m di rotta, ~20 s di budget), ritaglio a
1 m dai punti, config di default (PPSurf: griglia 129, TTA/refine 3/3) con
`ppsurf.rec_batch_size=50000`, che cambia solo velocità e memoria. Macchina:
PC di Verona, GTX TITAN X 12 GB. τ in metri; i valori a τ = 0.1 e 0.2 sono
ricalcolati con `--ricalcola`. Media ± std tra i 5 terreni della famiglia (std
solo su Chamfer e F τ=0.5; le altre sono ≤ 0.03, si trovano nei CSV). In
grassetto il migliore fra con e senza SW.

| Run | Cartella |
|---|---|
| Poisson | `2026-11-13_00-17-36_test+val` |
| PPSurf | `2026-11-13_00-21-04_test+val` |
| Poisson + SW | `2026-11-13_00-44-19_test+val_sw` |
| PPSurf + SW | `2026-11-13_01-11-45_test+val_sw` |

### Poisson: effetto della sliding window

| Fondale | SW | Chamfer ↓ | Prec. τ=0.1 | Rich. τ=0.1 | F τ=0.1 | Prec. τ=0.2 | Rich. τ=0.2 | F τ=0.2 | Prec. τ=0.5 | Rich. τ=0.5 | F τ=0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flat_sand | No | 0.050 ± 0.001 | **0.929** | 0.953 | 0.941 | **0.982** | **0.999** | **0.990** | **0.993** | **1.000** | **0.997** ± 0.002 |
| | Sì | **0.042** ± 0.001 | 0.922 | **0.970** | **0.946** | 0.974 | **0.999** | 0.987 | 0.990 | **1.000** | 0.995 ± 0.000 |
| mixed | No | 0.045 ± 0.001 | **0.936** | 0.929 | **0.933** | **0.997** | 0.993 | **0.995** | **1.000** | 0.999 | **0.999** ± 0.000 |
| | Sì | **0.044** ± 0.002 | 0.899 | **0.957** | 0.927 | 0.968 | **0.996** | 0.982 | 0.990 | **1.000** | 0.995 ± 0.001 |
| ridges | No | **0.034** ± 0.001 | **0.974** | 0.970 | **0.972** | **0.999** | 0.996 | **0.998** | **1.000** | 0.999 | **0.999** ± 0.001 |
| | Sì | 0.037 ± 0.001 | 0.925 | **0.980** | 0.952 | 0.971 | **0.998** | 0.985 | 0.990 | **1.000** | 0.995 ± 0.001 |
| rocky | No | **0.045** ± 0.002 | **0.932** | 0.923 | **0.927** | **0.994** | 0.992 | **0.993** | **0.999** | 0.999 | **0.999** ± 0.001 |
| | Sì | 0.046 ± 0.004 | 0.891 | **0.954** | 0.921 | 0.967 | **0.997** | 0.982 | 0.990 | **1.000** | 0.995 ± 0.001 |
| sand_ripples | No | 0.049 ± 0.002 | **0.931** | 0.948 | 0.939 | **0.990** | 0.999 | **0.994** | **0.998** | **1.000** | **0.999** ± 0.001 |
| | Sì | **0.043** ± 0.000 | 0.917 | **0.965** | **0.940** | 0.977 | **1.000** | 0.988 | 0.990 | **1.000** | 0.995 ± 0.000 |
| slope | No | 0.052 ± 0.010 | **0.950** | 0.949 | **0.949** | **0.995** | 0.987 | **0.991** | **0.999** | 0.990 | **0.995** ± 0.006 |
| | Sì | **0.043** ± 0.005 | 0.928 | **0.966** | 0.947 | 0.978 | **0.992** | 0.985 | 0.991 | **0.995** | 0.993 ± 0.003 |

### PPSurf: effetto della sliding window

| Fondale | SW | Chamfer ↓ | Prec. τ=0.1 | Rich. τ=0.1 | F τ=0.1 | Prec. τ=0.2 | Rich. τ=0.2 | F τ=0.2 | Prec. τ=0.5 | Rich. τ=0.5 | F τ=0.5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| flat_sand | No | 0.367 ± 0.019 | 0.027 | 0.051 | 0.036 | 0.107 | 0.200 | 0.140 | 0.678 | 0.961 | 0.792 ± 0.053 |
| | Sì | **0.111** ± 0.003 | **0.399** | **0.787** | **0.530** | **0.750** | **0.971** | **0.846** | **0.979** | **0.996** | **0.987** ± 0.005 |
| mixed | No | 0.282 ± 0.006 | 0.176 | 0.300 | 0.221 | 0.351 | 0.589 | 0.439 | 0.725 | 0.952 | 0.823 ± 0.009 |
| | Sì | **0.118** ± 0.002 | **0.369** | **0.722** | **0.488** | **0.721** | **0.969** | **0.827** | **0.976** | **0.998** | **0.987** ± 0.003 |
| ridges | No | 0.259 ± 0.007 | 0.254 | 0.439 | 0.322 | 0.428 | 0.724 | 0.538 | 0.697 | 0.971 | 0.811 ± 0.008 |
| | Sì | **0.115** ± 0.002 | **0.390** | **0.717** | **0.505** | **0.774** | **0.967** | **0.860** | **0.973** | **0.997** | **0.985** ± 0.006 |
| rocky | No | 0.271 ± 0.007 | 0.202 | 0.350 | 0.256 | 0.368 | 0.631 | 0.465 | 0.704 | 0.973 | 0.817 ± 0.008 |
| | Sì | **0.119** ± 0.003 | **0.357** | **0.712** | **0.476** | **0.675** | **0.972** | **0.797** | **0.982** | **0.999** | **0.991** ± 0.004 |
| sand_ripples | No | 0.326 ± 0.013 | 0.051 | 0.089 | 0.065 | 0.175 | 0.296 | 0.220 | 0.821 | 0.917 | 0.864 ± 0.035 |
| | Sì | **0.114** ± 0.002 | **0.363** | **0.716** | **0.481** | **0.751** | **0.960** | **0.843** | **0.989** | **0.996** | **0.992** ± 0.002 |
| slope | No | 0.329 ± 0.031 | 0.079 | 0.143 | 0.102 | 0.216 | 0.381 | 0.276 | 0.732 | 0.947 | 0.825 ± 0.033 |
| | Sì | **0.128** ± 0.017 | **0.347** | **0.679** | **0.459** | **0.753** | **0.941** | **0.837** | **0.986** | **0.984** | **0.985** ± 0.012 |

### Dimensione della finestra e buchi di PPSurf

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
benchmark (test 5).

### Tempi

Poisson e PPSurf senza SW: tempo di ricostruzione per terreno. Con SW: budget
~20 s per finestra (120 ping nuovi a 6 ping/s), RTF = latenza / budget.
Intervalli dalla famiglia migliore alla peggiore.

| Metodo | Tempo per terreno | Latenza mediana | Latenza max | RTF mediano | RTF max | Finestre in tempo | VRAM |
|---|---|---|---|---|---|---|---|
| Poisson | 2.2–4.3 s | – | – | – | – | – | – |
| PPSurf | 38–43 s | – | – | – | – | – | 8.7 GB |
| Poisson + SW | 42–47 s | 1.6–1.7 s | 2.5–2.7 s | 0.08–0.09 | 0.12–0.14 | 100% | – |
| PPSurf + SW | 950–1140 s | 36–41 s | 47–57 s | 1.8–2.1 | 2.6–3.2 | 0% | 8.6 GB |

La VRAM di PPSurf è alta per via di `rec_batch_size=50000`; con 10000 (il
default del config, per i 4 GB del portatile) scende a ~1.9 GB.

### Discussione

**Poisson**
- La SW non costa qualità: Chamfer 0.04–0.05 m e F τ=0.5 ≥ 0.99 in entrambi i
  modi, std ≤ 0.01.
- Con la SW la mesh è più fine, perché la profondità dell'octree è relativa
  alla finestra: il richiamo a τ=0.1 sale (0.92–0.97 → 0.95–0.98) e il
  Chamfer migliora su 4 famiglie su 6.
- La precisione scende dell'1–4% a τ ≤ 0.2: resta un po' di superficie spuria
  entro 1 m, cioè resti della cupola vicino ai punti e doppi strati dove le
  passate si sovrappongono.
- È ampiamente in tempo reale (RTF 0.08, tutte le finestre in tempo):
  **pronto per l'uso online**.

**PPSurf**
- La SW migliora tutto, su tutte le famiglie: Chamfer da 0.26–0.37 a
  0.11–0.13 m (−55/−70%), F τ=0.5 da 0.79–0.86 a 0.985–0.992, F τ=0.2 da
  0.14–0.54 a 0.80–0.86.
- A τ=0.5 è quasi alla pari con Poisson (F 0.985–0.992 contro 0.993–0.995).
- **Sotto i 20 cm resta lontano da Poisson**: F τ=0.1 0.46–0.53 contro
  0.92–0.95. Il richiamo è buono (~0.97 a τ=0.2), la precisione no (~0.75 a
  τ=0.2, ~0.38 a τ=0.1): è il guscio doppio, due strati a ~10–15 cm sopra e
  sotto il fondale.
- **Non è in tempo reale**, nemmeno sulla TITAN X: RTF mediano ~1.9, nessuna
  finestra in tempo. Ogni chiamata costa ~35–40 s indipendentemente dalla
  dimensione della finestra (lo fissano griglia e TTA), e finestre più piccole
  dimezzano il budget.

**Prossimi passi**
1. PPSurf + SW, precisione a τ piccoli: tenere un solo strato del guscio
   (strato medio, come in `strato_singolo_ppsurf.py`).
2. PPSurf + SW, tempo: griglia più grossa sulle finestre piccole (97: cella
   ~0.26 m), meno TTA, o più finestre per chiamata con il modello caricato
   una volta.
3. Poisson + SW: deduplicare le zone dove le passate si sovrappongono.

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
  durante il marching cubes. Non cambia il risultato.
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
  scene piccole per costruzione, e ne eredita il beneficio: F τ=0.2 da
  0.08–0.52 a 0.70–0.77 (vedi [Risultati](#risultati-poisson-e-ppsurf-con-e-senza-sw)).
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

## Note tecniche

- **PPSurf su 4 GB di VRAM**: servono
  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, griglia 129 e batch
  10000 (già impostati da `scripts/ppsurf.py` e dal config). Per trimesh ≥ 5
  va applicato il fix in `source/base/mesh.py` descritto in [SETUP.md](SETUP.md).
- **GPU Maxwell (TITAN X)**: non aggiornare PyTorch a ≥ 2.8 nell'ambiente
  `pps`.
- **Visualizzatore Open3D su WSLg**: GLFW tenta Wayland e la finestra non
  compare. `visualize_mesh.py` forza X11; per gli altri script basta
  `source env_setup.sh`.
- **matplotlib senza display**: `matplotlib.use("Agg")` va chiamato **prima**
  di importare `pyplot`, altrimenti un backend interattivo può andare in crash
  anche solo salvando un file.
- **Formato Omniscan 3D** (dati reali, via SonarView): CSV con
  `easting/northing`, `altitude`, `power (dB)`, `classification` e
  `ping number`. Il sensore non ha RGB. La classe 2 è probabilmente colonna
  d'acqua. I filtri per i dati reali (classi, quota, zone escluse) sono ancora
  supportati da `utils/pipeline.py`, ma non sono più nel config: vanno
  aggiunti sotto `preprocessing` (`escludi_classi`, `altitude_min`, …) se si
  torna sui dati reali.
- **WSL e spazio disco**: il `.vhdx` cresce ma non si restringe da solo. Per
  compattarlo: `wsl --shutdown`, poi `Optimize-VHD` o `diskpart compact vdisk`.
