#!/usr/bin/env bash
# Times PPSurf inference with the pretrained model ("pps.py rec") on the preprocessed point cloud.
# Usage: bash benchmark/time_ppsurf.sh <input_csv> [repetitions]
# Phases: library_startup (python + PyTorch Lightning / PyTorch Geometric imports + CUDA init: a fixed cost
#         per launch); rec (the full command, as run by scripts/ppsurf.py).
# rec - library_startup ~ actual reconstruction time.

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

prepare_input ppsurf
PY_PPS="$(python_env "$CONDA_ENV")"
# without it CUDA runs out of memory on a 4 GB GPU (see scripts/ppsurf.py)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for rep in $(seq 1 "$REPETITIONS"); do
	time_phase ppsurf "$rep" library_startup \
		"$PY_PPS" -c "import torch, pytorch_lightning, torch_geometric; torch.zeros(1).cuda(); torch.cuda.synchronize()"

	rm -rf "$WORK_DIR/results"
	# PPSurf writes the mesh to <out_dir>/<input name>/*.ply
	(cd "$METHOD_DIR" && EVAL_MESH="$WORK_DIR/results/input.ply/*.ply" time_phase ppsurf "$rep" rec \
		"$PY_PPS" pps.py rec "$WORK_DIR/input.ply" "$WORK_DIR/results" \
		--model.init_args.gen_resolution_global "$PPS_RES" \
		--model.init_args.rec_batch_size "$PPS_BATCH")
done

summary ppsurf
