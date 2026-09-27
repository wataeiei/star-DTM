# TSD-SR Threshold bypass

This is the fixed implementation of **Threshold bypass**. It jointly searches
sparse LoRA block count `K` and one global threshold `tau`; it does not use a
fixed bypass count or any contiguity/run constraints.

## Sync first

Copy these local files into `/mnt/disk1T/liyijuan/TSD-SR/` before running:

```text
profile_tsdsr_bypass_costs.py
build_threshold_bypass_candidates.py
train_tsdsr_lora.py
run_tsdsr_threshold_bypass_search.py
summarize_tsdsr_threshold_stage.py
select_threshold_bypass_final.py
eval_tsdsr_lora_sr_metrics.py
```

Keep `adaptive_grad_blockskip.py` beside them. Do not overwrite or delete the
existing All-LoRA run.

## 1. Profile every block once

The old K8 cost file covers only 16 frozen blocks and cannot support a joint K
search. Generate one complete 24-block profile at the fixed image and batch size:

```bash
cd /mnt/disk1T/liyijuan/TSD-SR
conda activate tsdsr310

SD3="/mnt/disk1T/liyijuan/models/stable-diffusion-3-medium-diffusers"
DATA="/mnt/disk1T/liyijuan/star-DTM/data/aid2520/train_hr"
IMP="outputs/tsdsr_aid2520_grad_profile_20_seed2026/lora_importance_evolution.csv"
COST="outputs/tsdsr_aid2520_allblock_cost_profile_v2"

python3 profile_tsdsr_bypass_costs.py \
  --pretrained_model "$SD3" \
  --official_lora_dir checkpoint/tsdsr-mse \
  --teacher_lora_dir checkpoint/teacher \
  --default_embedding_dir dataset/default \
  --null_embedding_dir dataset/null \
  --data_dir "$DATA" \
  --importance_csv "$IMP" \
  --output_dir "$COST" \
  --selected_k 0 \
  --profile_noise_ratio 0.4 \
  --image_size 256 \
  --sr_scale 4 \
  --rank 8 \
  --alpha 16 \
  --reg_rank 16 \
  --batch_size 1 \
  --warmup 3 \
  --repeats 10 \
  --seed 2026 \
  --dtype fp16
```

`selected_k=0` is deliberate: it profiles all 24 candidate blocks. The builder
will automatically protect any block with fallback, forward/loss mismatch, or
invalid saved FLOPs.

## 2. Build joint K/tau candidates

```bash
POLICY="outputs/tsdsr_aid2520_threshold_bypass_joint_v2"

python3 build_threshold_bypass_candidates.py \
  --importance_csv "$IMP" \
  --cost_csv "$COST/block_backward_costs.csv" \
  --output_dir "$POLICY" \
  --k_values 3 5 8 12 24 \
  --threshold_metric importance \
  --max_threshold_candidates 9

column -s, -t "$POLICY/k_candidates.csv"
column -s, -t "$POLICY/threshold_candidate_manifest.csv"
cat "$POLICY/metadata.json"
```

For every K and condition, the implementation recomputes importance over only
the safe frozen set `E_K`. A block is bypassed exactly when its normalized score
is no larger than the same global `tau`.

## 3. Run the 20-step stage serially

First inspect the generated plan:

```bash
SEARCH20="outputs/tsdsr_aid2520_threshold_bypass_search20_v2"

python3 run_tsdsr_threshold_bypass_search.py \
  --candidate_manifest "$POLICY/threshold_candidate_manifest.csv" \
  --output_dir "$SEARCH20" \
  --train_script train_tsdsr_lora.py \
  --pretrained_model "$SD3" \
  --official_lora_dir checkpoint/tsdsr-mse \
  --teacher_lora_dir checkpoint/teacher \
  --default_embedding_dir dataset/default \
  --null_embedding_dir dataset/null \
  --data_dir "$DATA" \
  --train_steps 20 \
  --image_size 256 \
  --sr_scale 4 \
  --rank 8 \
  --alpha 16 \
  --reg_rank 16 \
  --batch_size 1 \
  --lr 1e-5 \
  --reg_lr 1e-6 \
  --dtype fp16 \
  --seed 42 \
  --include_all_lora \
  --dry_run

column -s, -t "$SEARCH20/run_plan.csv" | less -S
```

Remove only `--dry_run` to execute. Every run is serial; no GPU concurrency is
used. The trainer fails immediately on fallback, non-finite loss, or a forward
difference above zero. Policy lookup and controller configuration are included
inside `mean_train_step_time_s`.

## 4. Timing/correctness shortlist

```bash
STAGE20="outputs/tsdsr_aid2520_threshold_bypass_stage20_summary_v2"

python3 summarize_tsdsr_threshold_stage.py \
  --stage_results "$SEARCH20/stage_results.csv" \
  --output_dir "$STAGE20" \
  --min_speedup_vs_controller_pct 2 \
  --forward_diff_tolerance 0 \
  --top_per_k 2

column -s, -t "$STAGE20/timing_comparison.csv"
column -s, -t "$STAGE20/shortlist.csv"
cat "$STAGE20/stage_report.json"
```

Repeat the shortlisted candidate IDs at 100 steps, then evaluate validation
quality. Advance quality-feasible candidates to 250 and 1000 steps. The final
selection is made with `select_threshold_bypass_final.py`; its input must contain
Base, All-LoRA, Native-K, and candidate metrics plus measured step times.

The test split remains untouched until K and tau are fixed on validation. The
final configuration is then repeated with seeds 42, 43, and 44.
