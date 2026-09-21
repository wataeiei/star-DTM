# DiT4SR on AID: Base, All-LoRA, and Bypass

This experiment uses the fixed AID-2.52K split:

- `train_hr`: 1,680 images for calibration and adaptation.
- `val_hr`: 420 held-out images for model selection and reporting.
- `test_hr`: 420 images kept untouched until the configuration is frozen.

The three reported methods are:

1. `Base-DiT4SR`: the official `acceptee/DiT4SR` checkpoint, without training.
2. `All-LoRA`: Q/V LoRA, rank 8, on all 24 DiT4SR blocks.
3. `GradTop8-Bypass`: AID-calibrated Grad-Top8 Q/V LoRA plus an automatically
   selected noise-aware backward-bypass schedule. Selected LoRA blocks are
   protected from bypass.

Bypass is active only during training. At inference, the third method is the
adapter learned with bypass; it does not skip forward blocks.

## 1. Copy/update the scripts on the Ubuntu training machine

The runner accepts the Python programs either in the repository root or in
`code/`. At minimum it needs:

```text
train_hf_dit4sr_all_lora_importance.py
profile_hf_dit4sr_grad.py
profile_dit4sr_bypass_gradient_fidelity.py
eval_hf_dit4sr_sr_metrics.py
adaptive_grad_blockskip.py
```

## 2. Run the smoke test first

```bash
cd /mnt/disk1T/liyijuan/star-DTM
conda activate star

MODE=smoke SEED=42 bash code/run_dit4sr_aid.sh
```

The smoke test uses 32 training images, 20 validation images, and 20 training
steps. Check:

```bash
cat outputs/dit4sr_aid/smoke_seed42/bypass_fidelity/recommended_bypass_schedule.txt
column -s, -t outputs/dit4sr_aid/smoke_seed42/all_lora/summary.csv
column -s, -t outputs/dit4sr_aid/smoke_seed42/gradtop8_bypass/summary.csv
column -s, -t outputs/dit4sr_aid/smoke_seed42/eval/sr_metrics_summary.csv
```

Before a formal run, require:

- finite loss in both training logs;
- `fallback_blocks == 0` in the Bypass log;
- at least one non-zero entry in the recommended bypass schedule;
- exactly 20 evaluated images for every method;
- Base, All-LoRA, and Bypass evaluated with the same `eval_seed=4242`.

If the audit returns a zero-only schedule, do not force bypass. It means the
current gradient-safety gate rejects bypass on AID under the configured
thresholds.

## 3. Run the formal validation experiment

```bash
MODE=full SEED=42 bash code/run_dit4sr_aid.sh
```

This uses all 1,680 AID training images, trains both adapters for 1,000 steps,
and evaluates all 420 validation images. The main outputs are:

```text
outputs/dit4sr_aid/full_seed42/eval/sr_metrics_summary.csv
outputs/dit4sr_aid/full_seed42/eval/sr_metrics_per_image.csv
outputs/dit4sr_aid/full_seed42/all_lora/summary.csv
outputs/dit4sr_aid/full_seed42/gradtop8_bypass/summary.csv
outputs/dit4sr_aid/full_seed42/bypass_fidelity/gradient_fidelity_summary.csv
```

The runner deliberately performs a separate calibration stage. Formal
All-LoRA and Bypass timing therefore excludes the common gradient-profiling
cost. Report calibration time separately if end-to-end adaptation cost is
needed.

## 4. Evaluate IQA and AID scene semantics

The SR evaluator saves generated images under the evaluation directory. In the
IQA environment, run:

```bash
cd /mnt/disk1T/liyijuan/star-DTM
conda activate iqa

EVAL=outputs/dit4sr_aid/full_seed42/eval

python3 eval_sr_iqa.py \
  --eval_manifest data/aid2520/val_eval_manifest.csv \
  --hr_dir data/aid2520/val_hr \
  --source Base-DiT4SR="$EVAL/images/Base-DiT4SR" \
  --source All-LoRA-AID-1000="$EVAL/images/All-LoRA-AID-1000" \
  --source GradTop8-Bypass-AID-1000="$EVAL/images/GradTop8-Bypass-AID-1000" \
  --output_dir outputs/dit4sr_aid/full_seed42/iqa \
  --metrics lpips musiq maniqa clipiqa liqe \
  --device cuda
```

Then evaluate scene classification with the already trained AID classifier:

```bash
conda activate star

EVAL=outputs/dit4sr_aid/full_seed42/eval

python3 ucmerced_semantic_classifier.py eval \
  --checkpoint outputs/aid2520_resnet18_classifier/best_classifier.pt \
  --eval_manifest data/aid2520/val_eval_manifest.csv \
  --source Base-DiT4SR="$EVAL/images/Base-DiT4SR" \
  --source All-LoRA-AID-1000="$EVAL/images/All-LoRA-AID-1000" \
  --source GradTop8-Bypass-AID-1000="$EVAL/images/GradTop8-Bypass-AID-1000" \
  --output_dir outputs/dit4sr_aid/full_seed42/semantic \
  --batch_size 64 \
  --dtype bf16 \
  --num_workers 2
```

Do not evaluate `test_hr` until the selection budget, bypass thresholds,
training steps, and early-stopping choice have been frozen on `val_hr`.

## 5. Follow-up seeds

Run seeds 43 and 44 only after seed 42 passes the full quality and efficiency
checks:

```bash
MODE=full SEED=43 bash code/run_dit4sr_aid.sh
MODE=full SEED=44 bash code/run_dit4sr_aid.sh
```

Always compare methods within the same seed. Report mean and sample standard
deviation across the three training seeds.
