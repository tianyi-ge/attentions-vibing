#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
OUT_DIR="${ROOT_DIR}/benchmarks/profiles"

OPERATOR="${1:-online_softmax_fwd}"
SEQ_Q="${2:-512}"
SEQ_K="${3:-512}"
HEAD_DIM="${4:-128}"
DTYPE="${5:-fp16}"
BATCH_SIZE="${6:-4}"
HEADS="${7:-32}"
WARMUP="${8:-5}"
STEPS="${9:-20}"
KERNEL_NAME="${10:-}"

mkdir -p "${OUT_DIR}"

STEM="${OPERATOR}_b${BATCH_SIZE}_h${HEADS}_sq${SEQ_Q}_sk${SEQ_K}_d${HEAD_DIM}_${DTYPE}"
REPORT_PATH="${OUT_DIR}/${STEM}"
TEXT_PATH="${OUT_DIR}/${STEM}_ncu.txt"

echo "Writing Nsight Compute report to ${REPORT_PATH}.ncu-rep"
echo "Writing text summary to ${TEXT_PATH}"

NCU_ARGS=(
  --target-processes all
  --set speedOfLight
  --launch-skip "${WARMUP}"
  --launch-count 1
  --export "${REPORT_PATH}"
)

if [[ -n "${KERNEL_NAME}" ]]; then
  NCU_ARGS+=(--kernel-name "${KERNEL_NAME}")
fi

ncu \
  "${NCU_ARGS[@]}" \
  python "${ROOT_DIR}/scripts/profile_attention.py" \
    --operator "${OPERATOR}" \
    --batch-size "${BATCH_SIZE}" \
    --heads "${HEADS}" \
    --seq-q "${SEQ_Q}" \
    --seq-k "${SEQ_K}" \
    --head-dim "${HEAD_DIM}" \
    --dtype "${DTYPE}" \
    --warmup "${WARMUP}" \
    --steps "${STEPS}" | tee "${TEXT_PATH}"
