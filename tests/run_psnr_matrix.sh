#!/usr/bin/env bash
#
# PSNR noise-floor matrix for the triraster paper (Table 4).
#
# Four arms -- tile_size {8,16} x ras_bwd {baseline,triton} -- each run REPEATS
# times with an identical configuration. The repeats are the point: 3DGS
# densification thresholds an atomically accumulated gradient norm, so two runs
# of the *same* arm diverge, and the spread between them is the noise floor
# against which any between-arm difference has to be judged. Changing the seed
# between repeats would measure something else and weaken the argument.
#
# data_factor 4 rather than 8: at factor 8 the bicycle scene carries ~28
# Gaussians per pixel, which is over-parameterized to the point that survival of
# any individual Gaussian is chaotically sensitive. Factor 4 is also the setting
# every published number uses.
#
# Usage:
#   bash tests/run_psnr_matrix.sh                       # repeats 1-3, GPUs 0-7
#   REPEATS=2 GPUS="5 6" bash tests/run_psnr_matrix.sh  # smaller sweep
#
# START_REP extends an existing matrix instead of recomputing it. Repeat number
# is part of the result directory name, so adding repeats 4-6 to a matrix that
# already has 1-3 is a matter of skipping the ones on disk:
#
#   START_REP=4 REPEATS=6 bash tests/run_psnr_matrix.sh
#
# The collector globs every tile*_*_r* under RESULT_ROOT, so it will report all
# six once they are there.
#
# The container needs a large /dev/shm for this: eight concurrent trainers each
# run DataLoader workers, and Docker's 64 MB default crashes them. Start it with
# --shm-size=64g, as the multi-GPU section of the README already does.
#
set -uo pipefail

DATA_DIR=${DATA_DIR:-/datasets/bicycle}
DATA_FACTOR=${DATA_FACTOR:-4}
RESULT_ROOT=${RESULT_ROOT:-./results/psnr_matrix}
START_REP=${START_REP:-1}
REPEATS=${REPEATS:-3}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}
REPO=${REPO:-/home/charyang/gsplat-rocm-rdna4}

read -r -a GPU_ARR <<< "$GPUS"
NGPU=${#GPU_ARR[@]}

# Build the run list: arm x repeat.
RUNS=()
for tile in 8 16; do
  for bwd in baseline triton; do
    for rep in $(seq "$START_REP" "$REPEATS"); do
      RUNS+=("${tile}:${bwd}:${rep}")
    done
  done
done
TOTAL=${#RUNS[@]}

if [ "$TOTAL" -eq 0 ]; then
  echo "nothing to do: START_REP=$START_REP > REPEATS=$REPEATS" >&2
  exit 1
fi

echo "=== PSNR matrix ==="
echo "runs=$TOTAL  gpus=$NGPU ($GPUS)  data_factor=$DATA_FACTOR" \
     " repeats=${START_REP}..${REPEATS}"
echo "results -> $RESULT_ROOT"
echo

mkdir -p "$RESULT_ROOT/logs"

# Dispatch in waves of NGPU so no two runs share a device.
i=0
wave=0
while [ $i -lt $TOTAL ]; do
  wave=$((wave + 1))
  echo "--- wave $wave ---"
  pids=()
  for gpu in "${GPU_ARR[@]}"; do
    [ $i -ge $TOTAL ] && break
    IFS=':' read -r tile bwd rep <<< "${RUNS[$i]}"
    tag="tile${tile}_${bwd}_r${rep}"
    log="$RESULT_ROOT/logs/${tag}.log"

    # Idempotent: a run that already produced a validation checkpoint is left
    # alone, so an interrupted sweep can simply be re-issued.
    if compgen -G "$RESULT_ROOT/$tag/stats/val_step*.json" > /dev/null; then
      echo "  gpu $gpu  --  $tag (already done, skipping)"
      i=$((i + 1))
      continue
    fi

    echo "  gpu $gpu  <-  $tag"
    # Only HIP_VISIBLE_DEVICES. Setting ROCR_/CUDA_ as well stacks the filters:
    # the first narrows the device list, the second then indexes into the
    # already-narrowed list and finds nothing.
    HIP_VISIBLE_DEVICES="$gpu" \
    GSPLAT_TILE_SIZE="$tile" \
    GSPLAT_RAS_BWD="$bwd" \
    python "$REPO/tests/run_simple_trainer.py" default \
      --data_dir "$DATA_DIR" \
      --data_factor "$DATA_FACTOR" \
      --result_dir "$RESULT_ROOT/$tag" \
      --disable-viewer \
      > "$log" 2>&1 &
    pids+=("$!:$tag")
    i=$((i + 1))
  done

  for entry in "${pids[@]}"; do
    pid=${entry%%:*}; tag=${entry#*:}
    if wait "$pid"; then
      echo "  ok   $tag"
    else
      echo "  FAIL $tag  (see $RESULT_ROOT/logs/$tag.log)"
    fi
  done
  echo
done

echo "=== all waves done; collecting ==="
python "$REPO/tests/collect_psnr_matrix.py" "$RESULT_ROOT"
