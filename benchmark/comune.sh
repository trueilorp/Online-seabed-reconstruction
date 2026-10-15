# Funzioni condivise dagli script benchmark/tempo_*.sh (da includere con "source", non da eseguire).
#
# Argomenti degli script: [csv_input] [ripetizioni]
#   csv_input    CSV della scansione (default: paths.input_csv di config/config.yaml)
#   ripetizioni  quante volte ripetere il metodo (default 1); la prima run puo'
#                includere costi una tantum (download pesi, cache CUDA), le successive no
#
# Ogni fase cronometrata viene stampata e aggiunta a output/benchmark/tempi.csv:
#   data,metodo,dataset,n_punti,ripetizione,fase,secondi,ram_picco_mb,vram_picco_mb,chamfer,precisione,richiamo,fscore,tau
# (vram_picco_mb = picco di memoria GPU usata durante la fase, al netto di quella gia' occupata prima)
# Le metriche di superficie (benchmark/metriche_superficie.py, contro il fondale della ground truth)
# sono compilate solo sulla riga della fase che produce la mesh e solo se la GT esiste.

set -euo pipefail

RADICE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RADICE"

CSV_INPUT="${1:-}"
RIPETIZIONI="${2:-1}"

CONDA_BASE="$(conda info --base)"
# percorso assoluto: le fasi dei metodi neurali girano dentro la cartella del loro repo
FILE_TEMPI="$RADICE/output/benchmark/tempi.csv"
mkdir -p "$RADICE/output/benchmark"
INTESTAZIONE="data,metodo,dataset,n_punti,ripetizione,fase,secondi,ram_picco_mb,vram_picco_mb,chamfer,precisione,richiamo,fscore,tau"
# un file vuoto viene semplicemente riscritto con l'intestazione, senza rinominarlo
[[ -s "$FILE_TEMPI" ]] || rm -f "$FILE_TEMPI"
if [[ -f "$FILE_TEMPI" && "$(head -1 "$FILE_TEMPI")" != "$INTESTAZIONE" ]]; then
	mv "$FILE_TEMPI" "${FILE_TEMPI%.csv}_vecchio_$(date +%Y%m%d_%H%M%S).csv"
	echo "Intestazione di $FILE_TEMPI cambiata: il vecchio file e' stato rinominato" >&2
fi
[[ -f "$FILE_TEMPI" ]] || echo "$INTESTAZIONE" > "$FILE_TEMPI"

# interprete python di un ambiente conda, chiamato direttamente (niente "conda run",
# che aggiungerebbe il proprio overhead ai tempi misurati)
python_env() {
	local bin="$CONDA_BASE/envs/$1/bin/python"
	[[ -x "$bin" ]] || { echo "Ambiente conda '$1' non trovato ($bin)" >&2; exit 1; }
	echo "$bin"
}

PY_BASE="$(python_env ricostruzione3d)"

if [[ -n "$CSV_INPUT" ]]; then
	DATASET="$(basename "$CSV_INPUT" .csv)"
else
	DATASET="$("$PY_BASE" -c "import yaml, os; print(os.path.splitext(os.path.basename(yaml.safe_load(open('config/config.yaml'))['paths']['input_csv']))[0])")"
fi

# prepara_input <metodo>: preprocessing comune (fuori dal cronometro del metodo) e caricamento parametri
prepara_input() {
	local metodo=$1
	DIR_LAVORO="$RADICE/output/benchmark/$metodo/$DATASET"
	rm -rf "$DIR_LAVORO"
	local args=(--metodo "$metodo" --out_dir "$DIR_LAVORO")
	[[ -n "$CSV_INPUT" ]] && args+=(--csv "$CSV_INPUT")
	"$PY_BASE" benchmark/prepara_input.py "${args[@]}"
	DIR_LAVORO="$(cd "$DIR_LAVORO" && pwd)"
	# shellcheck disable=SC1091
	source "$DIR_LAVORO/params.sh"
}

vram_usata_mb() {
	nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0
}

# metriche_superficie: "chamfer,precisione,richiamo,fscore,tau" della mesh $VALUTA_MESH (glob
# accettato, si prende il file piu' recente) contro $GT_MESH; ",,,," se mancano mesh o GT.
# $VALUTA_NPZ opzionale: normalizzazione da invertire (SA-ConvONet).
metriche_superficie() {
	local mesh
	mesh="$(ls -t $VALUTA_MESH 2>/dev/null | head -1 || true)"
	if [[ -z "$GT_MESH" || -z "$mesh" ]]; then
		[[ -z "$mesh" ]] && echo "  (mesh non trovata: $VALUTA_MESH)" >&2
		echo ",,,,"
		return
	fi
	local args=("$mesh" "$GT_MESH")
	[[ -n "${VALUTA_NPZ:-}" ]] && args+=(--npz "$VALUTA_NPZ")
	# solo l'ultima riga: eventuali avvisi di Open3D finiscono anch'essi su stdout
	(cd "$RADICE" && "$PY_BASE" benchmark/metriche_superficie.py "${args[@]}" | tail -1) || echo ",,,,"
}

# cronometra <metodo> <ripetizione> <fase> <comando...>
# esegue il comando mostrandone l'output e registra wall time, picco di RAM e picco di VRAM;
# se VALUTA_MESH e' impostata (es. VALUTA_MESH=... cronometra ...) aggiunge le metriche di superficie
cronometra() {
	local metodo=$1 rip=$2 fase=$3; shift 3
	local f_time f_gpu pid_gpu vram_base vram_picco esito=0
	f_time="$(mktemp)"; f_gpu="$(mktemp)"

	vram_base="$(vram_usata_mb)"
	nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$f_gpu" 2>/dev/null &
	pid_gpu=$!

	echo -e "\n=== [$metodo] rip $rip - $fase ==="
	/usr/bin/time -f "%e %M" -o "$f_time" "$@" || esito=$?

	kill "$pid_gpu" 2>/dev/null || true
	wait "$pid_gpu" 2>/dev/null || true

	if [[ $esito -ne 0 ]]; then
		rm -f "$f_time" "$f_gpu"
		echo "!!! [$metodo] fase '$fase' fallita (codice $esito)" >&2
		exit "$esito"
	fi

	local secondi ram_kb
	read -r secondi ram_kb < <(tail -1 "$f_time")
	vram_picco="$(sort -n "$f_gpu" | tail -1)"
	vram_picco=$(( ${vram_picco:-$vram_base} - vram_base ))
	(( vram_picco < 0 )) && vram_picco=0
	rm -f "$f_time" "$f_gpu"

	printf "  >> %s / %s: %ss, RAM picco %d MB, VRAM picco %d MB\n" \
		"$metodo" "$fase" "$secondi" $((ram_kb / 1024)) "$vram_picco"
	local metriche=",,,,"
	[[ -n "${VALUTA_MESH:-}" ]] && metriche="$(metriche_superficie)"
	echo "$(date -Iseconds),$metodo,$DATASET,$N_PUNTI,$rip,$fase,$secondi,$((ram_kb / 1024)),$vram_picco,$metriche" >> "$FILE_TEMPI"
}

riepilogo() {
	echo -e "\n=== Riepilogo $1 su $DATASET (tutte le run registrate in $FILE_TEMPI) ==="
	grep ",$1,$DATASET," "$FILE_TEMPI" | cut -d, -f5-13 \
		| column -t -s, -N rip,fase,secondi,ram_mb,vram_mb,chamfer,precisione,richiamo,fscore
}
