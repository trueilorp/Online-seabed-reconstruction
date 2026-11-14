#!/usr/bin/env bash
# Times Poisson (Open3D, CPU) on the preprocessed point cloud.
# Usage: bash benchmark/time_poisson.sh <input_csv> [repetitions]
# Phases: full_process (python start-up + open3d import + the phases below);
#         normals / reconstruction / mesh_cleanup, measured inside python (poisson_phases.py).

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

prepare_input poisson

for rep in $(seq 1 "$REPETITIONS"); do
	EVAL_MESH="$WORK_DIR/mesh_poisson.ply" time_phase poisson "$rep" full_process \
		"$PY_BASE" benchmark/poisson_phases.py "$WORK_DIR/input.ply" "$TIMING_FILE" "$DATASET" "$rep" \
		"$WORK_DIR/mesh_poisson.ply"
done

summary poisson
