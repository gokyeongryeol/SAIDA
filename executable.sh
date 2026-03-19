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

run_eval() {
    local GPU_ID=$1
    local data_name=$2
    local shot=$3

    docker run --rm \
        --gpus "device=${GPU_ID}" \
        -e SERVER_IP=${SERVER_IP} \
        -v "$(pwd)/NTIRE2026/":/workspace/NTIRE2026/ \
        -v "$(pwd)/submit/":/workspace/submit/ \
        -v "$(pwd)/weights/":/workspace/weights/ \
        --entrypoint python saida /workspace/eval.py \
            "${data_name}" "${shot}"
}

run_test() {
    local GPU_ID=$1
    local data_name=$2
    local shot=$3
    local USE_TTA=$4

    docker run --rm \
        --gpus "device=${GPU_ID}" \
        --shm-size=8G \
        -e DATA_ROOT=/workspace/NTIRE2026/${data_name}/ \
        -v "$(pwd)/NTIRE2026/":/workspace/NTIRE2026/ \
        -v "$(pwd)/submit/":/workspace/submit/ \
        -v "$(pwd)/weights/":/workspace/weights/ \
        --entrypoint bash saida /workspace/mmdet_eval.sh \
            ${data_name} ${shot} ${USE_TTA}
}

for data_name in dataset1 dataset2 dataset3; do
    for shot in 1 5 10; do

        # GPU slot wait
        while (( $(jobs -rp | wc -l) >= NUM_GPUS )); do
            wait -n
        done

        GPU_ID=${GPUS[$((IDX % NUM_GPUS))]}
        IDX=$((IDX + 1))

        echo "[GPU ${GPU_ID}] ${data_name} ${shot}"

        if [[ "$data_name" == "dataset3" && "$shot" -eq 1 ]]; then
            run_eval "$GPU_ID" "$data_name" "$shot" &

        elif [[ "$data_name" == "dataset3" || ( "$data_name" == "dataset1" && "$shot" -eq 10 ) ]]; then
            run_test "$GPU_ID" "$data_name" "$shot" "" &

        else
            run_test "$GPU_ID" "$data_name" "$shot" "--tta" &
        fi

    done
done

wait || { echo "Some jobs failed"; exit 1; }

echo "All jobs finished"