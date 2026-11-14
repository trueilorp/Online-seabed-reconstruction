# Shared functions of the benchmark/time_*.sh scripts (to be sourced, not executed).
# Arguments: <input_csv> [repetitions]. Each timed phase is printed and appended to
# benchmark/benchmark_output/timing.csv; the surface metrics (benchmark/surface_metrics.py, against the
# ground-truth seabed) are filled only on the phase that produces the mesh, and only if the GT exists.

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

INPUT_CSV="${1:?Usage: bash benchmark/time_<method>.sh <input_csv> [repetitions]}"
REPETITIONS="${2:-1}"

CONDA_BASE="$(conda info --base)"
# absolute path: the neural-method phases run inside their repository folder
TIMING_FILE="$ROOT/benchmark/benchmark_output/timing.csv"
mkdir -p "$ROOT/benchmark/benchmark_output"
HEADER="date,method,dataset,n_points,repetition,phase,seconds,peak_ram_mb,peak_vram_mb,chamfer,precision,recall,fscore,tau"
[[ -s "$TIMING_FILE" ]] || rm -f "$TIMING_FILE"
if [[ -f "$TIMING_FILE" && "$(head -1 "$TIMING_FILE")" != "$HEADER" ]]; then
	mv "$TIMING_FILE" "${TIMING_FILE%.csv}_old_$(date +%Y%m%d_%H%M%S).csv"
	echo "Header of $TIMING_FILE changed: the old file was renamed" >&2
fi
[[ -f "$TIMING_FILE" ]] || echo "$HEADER" > "$TIMING_FILE"

# python of a conda environment, called directly ("conda run" would add its own overhead to the timings)
python_env() {
	local bin="$CONDA_BASE/envs/$1/bin/python"
	[[ -x "$bin" ]] || { echo "Conda environment '$1' not found ($bin)" >&2; exit 1; }
	echo "$bin"
}

PY_BASE="$(python_env ricostruzione3d)"

DATASET="$(basename "$INPUT_CSV" .csv)"

# prepare_input <method>: shared preprocessing (outside the method timer) and parameter loading
prepare_input() {
	local method=$1
	WORK_DIR="$ROOT/benchmark/benchmark_output/$method/$DATASET"
	rm -rf "$WORK_DIR"
	"$PY_BASE" benchmark/prepare_input.py --method "$method" --out_dir "$WORK_DIR" --csv "$INPUT_CSV"
	WORK_DIR="$(cd "$WORK_DIR" && pwd)"
	# shellcheck disable=SC1091
	source "$WORK_DIR/params.sh"
}

used_vram_mb() {
	nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null | head -1 || echo 0
}

# surface_metrics: "chamfer,precision,recall,fscore,tau" of the mesh $EVAL_MESH (glob accepted, newest file)
# against $GT_MESH; ",,,," if the mesh or the GT is missing
surface_metrics() {
	local mesh
	mesh="$(ls -t $EVAL_MESH 2>/dev/null | head -1 || true)"
	if [[ -z "$GT_MESH" || -z "$mesh" ]]; then
		[[ -z "$mesh" ]] && echo "  (mesh not found: $EVAL_MESH)" >&2
		echo ",,,,"
		return
	fi
	# last line only: Open3D warnings also go to stdout
	(cd "$ROOT" && "$PY_BASE" benchmark/surface_metrics.py "$mesh" "$GT_MESH" | tail -1) || echo ",,,,"
}

# time_phase <method> <repetition> <phase> <command...>
# runs the command showing its output and records wall time, peak RAM and peak VRAM;
# with EVAL_MESH set (EVAL_MESH=... time_phase ...) it also appends the surface metrics
time_phase() {
	local method=$1 rep=$2 phase=$3; shift 3
	local f_time f_gpu pid_gpu vram_base peak_vram status=0
	f_time="$(mktemp)"; f_gpu="$(mktemp)"

	vram_base="$(used_vram_mb)"
	nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -lms 200 > "$f_gpu" 2>/dev/null &
	pid_gpu=$!

	echo -e "\n=== [$method] rep $rep - $phase ==="
	/usr/bin/time -f "%e %M" -o "$f_time" "$@" || status=$?

	kill "$pid_gpu" 2>/dev/null || true
	wait "$pid_gpu" 2>/dev/null || true

	if [[ $status -ne 0 ]]; then
		rm -f "$f_time" "$f_gpu"
		echo "!!! [$method] phase '$phase' failed (code $status)" >&2
		exit "$status"
	fi

	local seconds ram_kb
	read -r seconds ram_kb < <(tail -1 "$f_time")
	peak_vram="$(sort -n "$f_gpu" | tail -1)"
	peak_vram=$(( ${peak_vram:-$vram_base} - vram_base ))
	(( peak_vram < 0 )) && peak_vram=0
	rm -f "$f_time" "$f_gpu"

	printf "  >> %s / %s: %ss, peak RAM %d MB, peak VRAM %d MB\n" \
		"$method" "$phase" "$seconds" $((ram_kb / 1024)) "$peak_vram"
	local metrics=",,,,"
	[[ -n "${EVAL_MESH:-}" ]] && metrics="$(surface_metrics)"
	echo "$(date -Iseconds),$method,$DATASET,$N_POINTS,$rep,$phase,$seconds,$((ram_kb / 1024)),$peak_vram,$metrics" >> "$TIMING_FILE"
}

summary() {
	echo -e "\n=== Summary of $1 on $DATASET (all runs recorded in $TIMING_FILE) ==="
	grep ",$1,$DATASET," "$TIMING_FILE" | cut -d, -f5-13 \
		| column -t -s, -N rep,phase,seconds,ram_mb,vram_mb,chamfer,precision,recall,fscore
}
