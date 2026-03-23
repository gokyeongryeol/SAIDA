#!/bin/bash
set -euo pipefail

GPUS=($(nvidia-smi --query-gpu=index --format=csv,noheader))
NUM_GPUS=${#GPUS[@]}
IDX=0

if [ ${NUM_GPUS} -eq 0 ]; then
    echo "No GPU found"
    exit 1
fi

echo "Using GPUs: ${GPUS[@]}"

for data_name in dataset1 dataset2 dataset3; do
    for shot in 1 5 10; do
        # GPU slot wait
        while (( $(jobs -rp | wc -l) >= NUM_GPUS )); do
            wait -n
        done

        GPU_ID=${GPUS[$((IDX % NUM_GPUS))]}
        IDX=$((IDX + 1))

        echo "[GPU ${GPU_ID}] ${data_name} ${shot}"

        docker run --rm \
          --gpus "device=${GPU_ID}" \
          --shm-size=8G \
          -e SERVER_IP=${SERVER_IP} \
          -v "$(pwd)/NTIRE2026/":/workspace/NTIRE2026/ \
          -v "$(pwd)/submit/":/workspace/submit/ \
          -v "$(pwd)/weights/":/workspace/weights/ \
          --entrypoint python saida eval.py \
          "${data_name}" \
          "${shot}" &

    done
done

wait || { echo "Some jobs failed"; exit 1; }

echo "All jobs finished"