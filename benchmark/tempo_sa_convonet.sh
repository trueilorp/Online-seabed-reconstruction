#!/usr/bin/env bash
# Tempi di SA-ConvONet senza ottimizzazione online (generate.py, ConvONet pre-addestrato)
# sulla point cloud preprocessata.
#
# Uso: bash benchmark/tempo_sa_convonet.sh [csv_input] [ripetizioni]
#
# Fasi registrate:
#   avvio_librerie   solo avvio di python + import torch/torch_scatter + init CUDA (costo fisso)
#   prepara_dataset  make_cropscene_dataset2.py (normalizzazione + crop della scena, CPU)
#   generate         generate.py completo, come lo lancia scripts/sa_convonet.py (avvio +
#                    caricamento pesi + generazione a finestre scorrevoli + marching cubes)
# A fine generate.py il repo stampa anche la propria tabella dei tempi: la colonna "mesh"
# e' il tempo della sola generazione, senza avvio e caricamento pesi.
# Il tempo che finisce in tempo_calcolo nella pipeline e' solo quello di generate.

source "$(dirname "${BASH_SOURCE[0]}")/comune.sh"

prepara_input sa_convonet
PY_SAC="$(python_env "$CONDA_ENV")"

for rip in $(seq 1 "$RIPETIZIONI"); do
	cronometra sa_convonet "$rip" avvio_librerie \
		"$PY_SAC" -c "import torch, torch_scatter; torch.zeros(1).cuda(); torch.cuda.synchronize()"

	rm -rf "$DIR_LAVORO/normalize" "$DIR_LAVORO/out"
	(cd "$METODO_DIR" && cronometra sa_convonet "$rip" prepara_dataset \
		"$PY_SAC" scripts/dataset_matterport/make_cropscene_dataset2.py \
		--in_folder "$DIR_LAVORO/raw" --out_folder "$DIR_LAVORO/normalize" --do_norm)

	# mesh in spazio normalizzato: le metriche invertono loc/scale salvati da make_cropscene_dataset2.py
	(cd "$METODO_DIR" && VALUTA_MESH="$DIR_LAVORO/out/*/meshes/*/$NOME_RUN.off" \
		VALUTA_NPZ="$DIR_LAVORO/normalize/large_scenes/$NOME_RUN/pointcloud.npz" \
		cronometra sa_convonet "$rip" generate \
		"$PY_SAC" -u generate.py "$DIR_LAVORO/run_config.yaml")
done

riepilogo sa_convonet
