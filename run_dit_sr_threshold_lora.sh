#!/usr/bin/env bash
set -eu

# Run with the existing star environment active. This workflow never installs packages.
ROOT="${DIT_SR_ROOT:-/mnt/disk1T/liyijuan/dit-sr}"
STAR_ROOT="${STAR_DTM_ROOT:-/mnt/disk1T/liyijuan/star-DTM}"
cd "$ROOT"
POLICY="${POLICY_DIR:-outputs/dit_sr_threshold_lora_alpha1_policy}"
OUT="${TRAIN_OUT:-outputs/dit_sr_threshold_lora_alpha1_nobypass_smoke20_seed42}"

case "${1:-policy}" in
  policy)
    python3 build_dit_sr_threshold_lora.py \
      --importance_csv outputs/dit_sr_fullprobe_ucmerced/lora_importance_evolution.csv \
      --source_metadata outputs/dit_sr_fullprobe_ucmerced/metadata.json \
      --importance_step 0 \
      --importance_threshold 1.0 \
      --expected_blocks 58 \
      --output_dir "$POLICY"
    ;;
  smoke)
    SEL="$POLICY/dit_sr_grad_metadata.json"
    K=$(python3 - "$SEL" <<'PY'
import json
import sys
with open(sys.argv[1], encoding="utf-8") as f:
    meta = json.load(f)
for key, expected in {"loss_mode": "official", "target": "qv", "rank": 8, "alpha": 16}.items():
    if meta.get(key) != expected:
        raise SystemExit(f"Selection mismatch: {key}={meta.get(key)!r}, expected {expected!r}")
if len(meta["selected_blocks"]) != meta["topk_blocks"]:
    raise SystemExit("Selection count mismatch")
print(meta["topk_blocks"])
PY
    )
    if [ -e "$OUT" ]; then
      printf 'Output already exists: %s. Set TRAIN_OUT to a new directory.\n' "$OUT" >&2
      exit 1
    fi
    printf 'Threshold-selected K=%s; backward bypass disabled.\n' "$K"
    python3 train_dit_sr_all_lora_importance.py \
      --config_path configs/realsr_DiT.yaml \
      --ckpt_path weights/realsr.pth \
      --autoencoder_ckpt weights/autoencoder_vq_f4.pth \
      --data_dir "$STAR_ROOT/data/ucmerced/train_hr" \
      --output_dir "$OUT" \
      --loss_mode official \
      --image_size 256 \
      --lq_size 64 \
      --target qv \
      --rank 8 \
      --alpha 16 \
      --lora_selection metadata \
      --lora_block_budget "$K" \
      --topk_blocks "$K" \
      --lora_selection_file "$SEL" \
      --blockskip_count 0 \
      --train_steps 20 \
      --profile_steps 0 20 \
      --profile_batches 5 \
      --profile_noise_ratios 0.05 0.2 0.4 0.6 0.8 0.95 \
      --train_noise_ratios 0.05 0.2 0.4 0.6 0.8 0.95 \
      --batch_size 1 \
      --lr 1e-5 \
      --grad_clip 1.0 \
      --max_images 0 \
      --num_workers 0 \
      --seed 42 \
      --profile_seed 42 \
      --log_every 5
    ;;
  *)
    printf 'Usage: bash run_dit_sr_threshold_lora.sh {policy|smoke}\n' >&2
    exit 2
    ;;
esac
