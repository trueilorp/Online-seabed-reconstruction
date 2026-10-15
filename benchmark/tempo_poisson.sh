#!/usr/bin/env bash
# Tempi di Poisson (Open3D, CPU) sulla point cloud preprocessata.
#
# Uso: bash benchmark/tempo_poisson.sh [csv_input] [ripetizioni]
#
# Fasi registrate:
#   processo_completo  avvio di python + import open3d + tutte le fasi sotto
#   normali / ricostruzione / pulizia_mesh  misurate dentro python (poisson_fasi.py)

source "$(dirname "${BASH_SOURCE[0]}")/comune.sh"

prepara_input poisson

for rip in $(seq 1 "$RIPETIZIONI"); do
	VALUTA_MESH="$DIR_LAVORO/mesh_poisson.ply" cronometra poisson "$rip" processo_completo \
		"$PY_BASE" benchmark/poisson_fasi.py "$DIR_LAVORO/input.ply" "$FILE_TEMPI" "$DATASET" "$rip" \
		"$DIR_LAVORO/mesh_poisson.ply"
done

riepilogo poisson
