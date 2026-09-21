#!/usr/bin/env bash
set -euo pipefail

# DiT4SR on AID-2.52K: Base, All-LoRA, and Grad-Top8 + noise-aware
# backward bypass. Run this from the Ubuntu star-DTM repository root.
#
# Smoke test:
#   MODE=smoke bash code/run_dit4sr_aid.sh
#
# Formal seed-42 run:
#   MODE=full SEED=42 bash code/run_dit4sr_aid.sh

MODE="${MODE:-smoke}"
SEED="${SEED:-42}"
DATA_ROOT="${DATA_ROOT:-data/aid2520}"
OUTPUT_ROOT="${OUTPUT_ROOT:-outputs/dit4sr_aid}"
MODEL_ID="${MODEL_ID:-acceptee/DiT4SR}"
BASE_MODEL_ID="${BASE_MODEL_ID:-stabilityai/stable-diffusion-3.5-medium}"
VARIANT="${VARIANT:-dit4sr_q}"

case "$MODE" in
  smoke)
    TRAIN_STEPS="${TRAIN_STEPS:-20}"
    TRAIN_MAX_IMAGES="${TRAIN_MAX_IMAGES:-32}"
    EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-20}"
    PROFILE_BATCHES="${PROFILE_BATCHES:-2}"
    FIDELITY_BATCHES="${FIDELITY_BATCHES:-2}"
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-0}"
    ;;
  full)
    TRAIN_STEPS="${TRAIN_STEPS:-1000}"
    TRAIN_MAX_IMAGES="${TRAIN_MAX_IMAGES:-0}"
    EVAL_MAX_IMAGES="${EVAL_MAX_IMAGES:-0}"
    PROFILE_BATCHES="${PROFILE_BATCHES:-5}"
    FIDELITY_BATCHES="${FIDELITY_BATCHES:-5}"
    CHECKPOINT_EVERY="${CHECKPOINT_EVERY:-250}"
    ;;
  *)
    echo "MODE must be smoke or full, got: $MODE" >&2
    exit 2
    ;;
esac

TRAIN_DIR="$DATA_ROOT/train_hr"
VAL_DIR="$DATA_ROOT/val_hr"
RUN_ROOT="$OUTPUT_ROOT/${MODE}_seed${SEED}"
SELECTION_DIR="$RUN_ROOT/selection"
CALIBRATION_DIR="$RUN_ROOT/calibration"
FIDELITY_DIR="$RUN_ROOT/bypass_fidelity"
ALL_DIR="$RUN_ROOT/all_lora"
BYPASS_DIR="$RUN_ROOT/gradtop8_bypass"
EVAL_DIR="$RUN_ROOT/eval"

resolve_script() {
  local name="$1"
  if [[ -f "$name" ]]; then
    printf '%s\n' "$name"
  elif [[ -f "code/$name" ]]; then
    printf '%s\n' "code/$name"
  else
    echo "Cannot find $name in the repository root or code/." >&2
    exit 2
  fi
}

TRAIN_SCRIPT="$(resolve_script train_hf_dit4sr_all_lora_importance.py)"
PROFILE_SCRIPT="$(resolve_script profile_hf_dit4sr_grad.py)"
FIDELITY_SCRIPT="$(resolve_script profile_dit4sr_bypass_gradient_fidelity.py)"
EVAL_SCRIPT="$(resolve_script eval_hf_dit4sr_sr_metrics.py)"

count_images() {
  find "$1" -type f \( -iname '*.png' -o -iname '*.jpg' -o -iname '*.jpeg' -o -iname '*.tif' -o -iname '*.tiff' -o -iname '*.bmp' -o -iname '*.webp' \) | wc -l
}

if [[ ! -d "$TRAIN_DIR" || ! -d "$VAL_DIR" ]]; then
  echo "Missing AID split. Expected $TRAIN_DIR and $VAL_DIR." >&2
  exit 2
fi

TRAIN_COUNT="$(count_images "$TRAIN_DIR")"
VAL_COUNT="$(count_images "$VAL_DIR")"
if [[ "$MODE" == "full" && ("$TRAIN_COUNT" -ne 1680 || "$VAL_COUNT" -ne 420) ]]; then
  echo "Formal run requires AID train=1680 and val=420; found train=$TRAIN_COUNT val=$VAL_COUNT." >&2
  exit 2
fi
echo "AID split: train=$TRAIN_COUNT val=$VAL_COUNT"

mkdir -p "$RUN_ROOT"

COMMON_MODEL_ARGS=(
  --model_id "$MODEL_ID"
  --base_model_id "$BASE_MODEL_ID"
  --variant "$VARIANT"
  --loss_mode official_flow
  --image_size 256
  --max_images "$TRAIN_MAX_IMAGES"
  --sr_scale 4
  --dtype bf16
  --target qv
  --rank 8
  --alpha 16
  --batch_size 1
  --num_workers 0
  --seed "$SEED"
)

NOISE_RATIOS=(0.05 0.2 0.4 0.6 0.8 0.95)

echo "[1/5] AID-specific Grad-Top8 selection"
python3 "$PROFILE_SCRIPT" \
  --model_id "$MODEL_ID" \
  --base_model_id "$BASE_MODEL_ID" \
  --variant "$VARIANT" \
  --loss_mode official_flow \
  --data_dir "$TRAIN_DIR" \
  --output_dir "$SELECTION_DIR" \
  --image_size 256 \
  --max_images "$TRAIN_MAX_IMAGES" \
  --sr_scale 4 \
  --dtype bf16 \
  --target qv \
  --rank 8 \
  --alpha 16 \
  --topk_blocks 8 \
  --probe_batches 20 \
  --batch_size 1 \
  --num_workers 0 \
  --seed "$SEED"

SELECTION_FILE="$SELECTION_DIR/hf_dit4sr_grad_metadata.json"

echo "[2/5] Multi-noise calibration for backward-bypass scoring"
python3 "$TRAIN_SCRIPT" \
  "${COMMON_MODEL_ARGS[@]}" \
  --data_dir "$TRAIN_DIR" \
  --output_dir "$CALIBRATION_DIR" \
  --lora_selection all \
  --train_steps 1 \
  --profile_steps 0 \
  --profile_batches "$PROFILE_BATCHES" \
  --profile_noise_ratios "${NOISE_RATIOS[@]}" \
  --train_noise_ratios "${NOISE_RATIOS[@]}" \
  --lr 1e-5 \
  --grad_clip 1.0 \
  --profile_seed 2026 \
  --reset_seed_after_lora_injection \
  --checkpoint_every 0 \
  --log_every 1

IMPORTANCE_FILE="$CALIBRATION_DIR/lora_importance_evolution.csv"

echo "[3/5] Gradient-fidelity audit and automatic noise-aware bypass schedule"
python3 "$FIDELITY_SCRIPT" \
  --model_id "$MODEL_ID" \
  --base_model_id "$BASE_MODEL_ID" \
  --variant "$VARIANT" \
  --data_dir "$TRAIN_DIR" \
  --output_dir "$FIDELITY_DIR" \
  --selection_file "$SELECTION_FILE" \
  --importance_csv "$IMPORTANCE_FILE" \
  --importance_step 0 \
  --image_size 256 \
  --max_images "$TRAIN_MAX_IMAGES" \
  --probe_batches "$FIDELITY_BATCHES" \
  --noise_ratios "${NOISE_RATIOS[@]}" \
  --bypass_budgets 0 2 4 6 8 \
  --blockskip_min_run 2 \
  --blockskip_max_run 4 \
  --blockskip_max_runs 3 \
  --selection_criterion descent \
  --min_cosine 0.80 \
  --min_descent_retention 0.50 \
  --target qv \
  --rank 8 \
  --alpha 16 \
  --batch_size 1 \
  --num_workers 0 \
  --seed "$SEED" \
  --eval_seed 4242 \
  --dtype bf16

SCHEDULE_FILE="$FIDELITY_DIR/recommended_bypass_schedule.txt"
read -r -a BYPASS_SCHEDULE < "$SCHEDULE_FILE"
if [[ "${#BYPASS_SCHEDULE[@]}" -eq 0 ]]; then
  echo "The fidelity audit produced an empty bypass schedule." >&2
  exit 2
fi
HAS_NONZERO_BYPASS=0
for entry in "${BYPASS_SCHEDULE[@]}"; do
  if [[ "${entry##*:}" -gt 0 ]]; then
    HAS_NONZERO_BYPASS=1
    break
  fi
done
if [[ "$HAS_NONZERO_BYPASS" -ne 1 ]]; then
  echo "The safety audit rejected bypass at every noise ratio; not forcing an unsafe policy." >&2
  exit 3
fi
echo "Selected bypass schedule: ${BYPASS_SCHEDULE[*]}"

echo "[4/5] Controlled training: All-LoRA and Grad-Top8 + backward bypass"
python3 "$TRAIN_SCRIPT" \
  "${COMMON_MODEL_ARGS[@]}" \
  --data_dir "$TRAIN_DIR" \
  --output_dir "$ALL_DIR" \
  --lora_selection all \
  --train_steps "$TRAIN_STEPS" \
  --disable_profiling \
  --train_noise_ratios "${NOISE_RATIOS[@]}" \
  --lr 1e-5 \
  --grad_clip 1.0 \
  --profile_seed 2026 \
  --reset_seed_after_lora_injection \
  --checkpoint_every "$CHECKPOINT_EVERY" \
  --log_every 10

python3 "$TRAIN_SCRIPT" \
  "${COMMON_MODEL_ARGS[@]}" \
  --data_dir "$TRAIN_DIR" \
  --output_dir "$BYPASS_DIR" \
  --lora_selection metadata \
  --lora_block_budget 8 \
  --lora_selection_file "$SELECTION_FILE" \
  --blockskip_importance_csv "$IMPORTANCE_FILE" \
  --blockskip_importance_step 0 \
  --protect_selected_lora_blocks \
  --blockskip_schedule "${BYPASS_SCHEDULE[@]}" \
  --blockskip_min_run 2 \
  --blockskip_max_run 4 \
  --blockskip_max_runs 3 \
  --residual_execution single_pass \
  --residual_cache_device cpu \
  --residual_cache_dtype fp32 \
  --train_steps "$TRAIN_STEPS" \
  --disable_profiling \
  --train_noise_ratios "${NOISE_RATIOS[@]}" \
  --lr 1e-5 \
  --grad_clip 1.0 \
  --profile_seed 2026 \
  --reset_seed_after_lora_injection \
  --checkpoint_every "$CHECKPOINT_EVERY" \
  --log_every 10

echo "[5/5] Held-out AID evaluation: Base, All-LoRA, and Bypass"
python3 "$EVAL_SCRIPT" \
  --model_id "$MODEL_ID" \
  --base_model_id "$BASE_MODEL_ID" \
  --variant "$VARIANT" \
  --data_dir "$VAL_DIR" \
  --train_dir_for_overlap_check "$TRAIN_DIR" \
  --output_dir "$EVAL_DIR" \
  --adapter All-LoRA-AID-${TRAIN_STEPS}="$ALL_DIR/lora_adapter.pt" \
  --adapter GradTop8-Bypass-AID-${TRAIN_STEPS}="$BYPASS_DIR/lora_adapter.pt" \
  --allow_sparse_adapter \
  --image_size 256 \
  --max_images "$EVAL_MAX_IMAGES" \
  --sr_scale 4 \
  --num_inference_steps 20 \
  --eval_seed 4242 \
  --warmup_images 1 \
  --crop_border 4 \
  --color_fix adain \
  --dtype bf16 \
  --target qv \
  --rank 8 \
  --alpha 16 \
  --save_images

echo
echo "Completed: $RUN_ROOT"
echo "Quality summary: $EVAL_DIR/sr_metrics_summary.csv"
echo "All-LoRA efficiency: $ALL_DIR/summary.csv"
echo "Bypass efficiency: $BYPASS_DIR/summary.csv"
echo "Bypass fidelity: $FIDELITY_DIR/gradient_fidelity_summary.csv"
