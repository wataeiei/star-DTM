#!/usr/bin/env bash

# Complete the missing three-seed training runs for the CVPR core comparison.
# Run from the star-DTM repository root with the `star` conda environment active:
#   bash code/run_cvpr_core_multiseed.sh

ROOT="${ROOT:-/mnt/disk1T/liyijuan/star-DTM}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-train_hf_dit4sr_all_lora_importance.py}"
DATA_DIR="${DATA_DIR:-data/ucmerced/train_hr}"
META="${META:-outputs/dit4sr_gradtop8_selection.json}"
IMPORTANCE="${IMPORTANCE:-outputs/dit4sr_all_lora_1000_saved/lora_importance_evolution.csv}"
TRAIN_STEPS="${TRAIN_STEPS:-1000}"

cd "$ROOT" || exit 1

if [ ! -f "$TRAIN_SCRIPT" ] && [ -f "code/train_hf_dit4sr_all_lora_importance.py" ]; then
  TRAIN_SCRIPT="code/train_hf_dit4sr_all_lora_importance.py"
fi

for required in "$TRAIN_SCRIPT" "$META" "$IMPORTANCE"; do
  if [ ! -f "$required" ]; then
    echo "ERROR: required file not found: $required"
    exit 1
  fi
done

if [ ! -d "$DATA_DIR" ]; then
  echo "ERROR: training directory not found: $DATA_DIR"
  exit 1
fi

is_complete() {
  out="$1"
  if [ ! -f "$out/train_log.csv" ] || [ ! -f "$out/summary.csv" ] || [ ! -f "$out/lora_adapter.pt" ]; then
    return 1
  fi
  last_line=$(tail -n 1 "$out/train_log.csv")
  last_step=${last_line%%,*}
  [ "$last_step" = "$TRAIN_STEPS" ]
}

run_all_lora() {
  seed="$1"
  out="outputs/dit4sr_all_lora_1000_seed${seed}_final"

  if is_complete "$out"; then
    echo "SKIP: All-LoRA seed $seed is already complete: $out"
    return
  fi

  mkdir -p "$out"
  echo "===== All-LoRA seed $seed ====="
  python3 "$TRAIN_SCRIPT" \
    --model_id acceptee/DiT4SR \
    --base_model_id stabilityai/stable-diffusion-3.5-medium \
    --variant dit4sr_q \
    --data_dir "$DATA_DIR" \
    --output_dir "$out" \
    --loss_mode official_flow \
    --image_size 256 \
    --max_images 0 \
    --sr_scale 4 \
    --dtype bf16 \
    --target qv \
    --rank 8 \
    --alpha 16 \
    --lora_selection all \
    --train_steps "$TRAIN_STEPS" \
    --profile_steps 0 "$TRAIN_STEPS" \
    --profile_batches 2 \
    --profile_noise_ratios 0.05 0.2 0.4 0.6 0.8 0.95 \
    --train_noise_ratios 0.05 0.2 0.4 0.6 0.8 0.95 \
    --topk_blocks 8 \
    --batch_size 1 \
    --lr 1e-5 \
    --grad_clip 1.0 \
    --num_workers 0 \
    --seed "$seed" \
    --profile_seed 42 \
    --checkpoint_every 250 \
    --log_every 10

  if ! is_complete "$out"; then
    echo "ERROR: All-LoRA seed $seed did not finish cleanly: $out"
    exit 1
  fi
}

run_gradskip_lora() {
  seed="$1"
  out="outputs/dit4sr_gradtop8_noiseaware_singlepass_1000_seed${seed}_final"

  if is_complete "$out"; then
    echo "SKIP: GradSkip-LoRA seed $seed is already complete: $out"
    return
  fi

  mkdir -p "$out"
  echo "===== GradSkip-LoRA seed $seed ====="
  python3 "$TRAIN_SCRIPT" \
    --model_id acceptee/DiT4SR \
    --base_model_id stabilityai/stable-diffusion-3.5-medium \
    --variant dit4sr_q \
    --data_dir "$DATA_DIR" \
    --output_dir "$out" \
    --loss_mode official_flow \
    --image_size 256 \
    --max_images 0 \
    --sr_scale 4 \
    --dtype bf16 \
    --target qv \
    --rank 8 \
    --alpha 16 \
    --lora_selection metadata \
    --lora_block_budget 8 \
    --lora_selection_file "$META" \
    --blockskip_importance_csv "$IMPORTANCE" \
    --blockskip_importance_step 0 \
    --protect_selected_lora_blocks \
    --blockskip_schedule 0.05:8 0.2:4 0.4:4 0.6:4 0.8:6 0.95:8 \
    --blockskip_min_run 2 \
    --blockskip_max_run 4 \
    --blockskip_max_runs 3 \
    --residual_execution single_pass \
    --train_steps "$TRAIN_STEPS" \
    --profile_steps 0 "$TRAIN_STEPS" \
    --profile_batches 2 \
    --profile_noise_ratios 0.05 0.2 0.4 0.6 0.8 0.95 \
    --train_noise_ratios 0.05 0.2 0.4 0.6 0.8 0.95 \
    --topk_blocks 8 \
    --batch_size 1 \
    --lr 1e-5 \
    --grad_clip 1.0 \
    --num_workers 0 \
    --seed "$seed" \
    --profile_seed 42 \
    --checkpoint_every 250 \
    --log_every 10

  if ! is_complete "$out"; then
    echo "ERROR: GradSkip-LoRA seed $seed did not finish cleanly: $out"
    exit 1
  fi
}

# Run sequentially so Orin timing and memory measurements are not contaminated
# by a second GPU training process.
for seed in 43 44; do
  run_all_lora "$seed"
done

for seed in 43 44; do
  run_gradskip_lora "$seed"
done

echo "All missing CVPR core multi-seed training runs are complete."
echo "Next: python3 code/summarize_cvpr_core_multiseed.py"
