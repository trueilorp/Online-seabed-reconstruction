#!/usr/bin/env bash
# Tempi di PPSurf (inferenza col modello pre-addestrato, "pps.py rec") sulla point cloud preprocessata.
#
# Uso: bash benchmark/tempo_ppsurf.sh [csv_input] [ripetizioni]
#
# Fasi registrate:
#   avvio_librerie  solo avvio di python + import di PyTorch Lightning / PyTorch Geometric + init CUDA:
#                   costo fisso pagato a ogni lancio, indipendente dai punti
#   rec             comando completo, come lo lancia scripts/ppsurf.py (avvio + caricamento pesi +
#                   inferenza + marching cubes + scrittura mesh)
# rec - avvio_librerie ~ tempo effettivo di ricostruzione.

source "$(dirname "${BASH_SOURCE[0]}")/comune.sh"

prepara_input ppsurf
PY_PPS="$(python_env "$CONDA_ENV")"
# senza va in CUDA OutOfMemoryError sulla GPU da 4GB (vedi scripts/ppsurf.py)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for rip in $(seq 1 "$RIPETIZIONI"); do
	cronometra ppsurf "$rip" avvio_librerie \
		"$PY_PPS" -c "import torch, pytorch_lightning, torch_geometric; torch.zeros(1).cuda(); torch.cuda.synchronize()"

	rm -rf "$DIR_LAVORO/results"
	# PPSurf salva la mesh in <out_dir>/<nome input>/*.ply (vedi scripts/ppsurf.py)
	(cd "$METODO_DIR" && VALUTA_MESH="$DIR_LAVORO/results/input.ply/*.ply" cronometra ppsurf "$rip" rec \
		"$PY_PPS" pps.py rec "$DIR_LAVORO/input.ply" "$DIR_LAVORO/results" \
		--model.init_args.gen_resolution_global "$PPS_RES" \
		--model.init_args.rec_batch_size "$PPS_BATCH")
done

riepilogo ppsurf
