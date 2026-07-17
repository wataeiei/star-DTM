# Sandwich-LoRA On-board SR Experiment

This folder contains the experiment script:

```bash
python3 onboard_sandwich_lora_sr.py --help
```

## 1. Install dependencies on Jetson

Install PyTorch/torchvision using the NVIDIA Jetson PyTorch wheel that matches
your JetPack version. Then install the Python packages:

```bash
pip3 install diffusers transformers accelerate safetensors pillow tqdm scikit-image datasets
```

Log in if the model or dataset download requires Hugging Face credentials:

```bash
huggingface-cli login
```

## 2. Prepare UC Merced

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode prepare_ucmerced \
  --data_root data/ucmerced \
  --hr_size 256 \
  --train_ratio 0.8 \
  --seed 42
```

## 3. Communication feasibility

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode comm \
  --update_sizes_mb 1200 80 30 15 7.5 \
  --uplink_mbps 0.128 0.5 1 5 10 \
  --contact_window_s 600 \
  --eta 0.7 \
  --output_csv outputs/comm_feasibility.csv
```

## 4. Train Sandwich-LoRA

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode train \
  --train_dir data/ucmerced/train_hr \
  --output_dir outputs/lora_sandwich_r8 \
  --hr_size 256 \
  --lr_size 64 \
  --rank 8 \
  --alpha 16 \
  --target qv \
  --lora_scope shallow_deep \
  --train_steps 200 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 1e-4 \
  --power_w 30 \
  --full_model_size_mb 1200
```

Start resource logging in another terminal before training:

```bash
mkdir -p outputs
tegrastats --interval 1000 | tee outputs/tegrastats_sandwich_train.log
```

After training, parse the real resource log:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode parse_tegrastats \
  --tegrastats_log outputs/tegrastats_sandwich_train.log \
  --tegrastats_interval_s 1 \
  --output_csv outputs/tegrastats_sandwich_train.csv
```

## 5. Evaluate Base and LoRA

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode eval \
  --val_dir data/ucmerced/val_hr \
  --output_dir outputs/eval_base \
  --hr_size 256 \
  --lr_size 64 \
  --eval_max_images 20 \
  --num_inference_steps 25
```

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode eval \
  --val_dir data/ucmerced/val_hr \
  --lora_dir outputs/lora_sandwich_r8 \
  --output_dir outputs/eval_sandwich_r8 \
  --hr_size 256 \
  --lr_size 64 \
  --eval_max_images 20 \
  --num_inference_steps 25 \
  --base_summary_csv outputs/eval_base/eval_summary.csv \
  --train_summary_csv outputs/lora_sandwich_r8/summary.csv
```

The extra summary fields include throughput, trainable parameter ratio,
adapter compression ratio, adapter upload time, PSNR/SSIM gains, and PSNR gain
per MB/Wh when the corresponding training or base summaries are provided.

## 6. Ablations

Change only `--lora_scope` and `--output_dir`:

- `--lora_scope shallow` for Shallow-LoRA. The script automatically selects the
  first down block that actually contains target attention layers in the current
  diffusers UNet.
- `--lora_scope last2_up` for Last2-Up-LoRA
- `--lora_scope topk --topk_blocks 8 --topk_policy balanced` for Top-K Block LoRA
- `--lora_scope all` for All-LoRA

For formal results, prefer increasing `--eval_max_images` or setting it to `0`
to evaluate the full validation split.

Top-K Block LoRA selects K Transformer blocks that contain the target attention
projections, then applies LoRA only to those blocks. The selected block names are
stored in `lora_metadata.json`, so evaluation reloads exactly the same placement.

Example Top-8 run:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode train \
  --train_method lora \
  --train_dir data/ucmerced/train_hr \
  --output_dir outputs/lora_top8_balanced_r8_lr1e5_1000_fp32_gpu \
  --hr_size 256 \
  --lr_size 64 \
  --rank 8 \
  --alpha 16 \
  --target qv \
  --lora_scope topk \
  --topk_blocks 8 \
  --topk_policy balanced \
  --train_steps 1000 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 1e-5 \
  --grad_clip 1.0 \
  --power_w 30 \
  --full_model_size_mb 1200 \
  --no_fp16
```

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode eval \
  --val_dir data/ucmerced/val_hr \
  --lora_dir outputs/lora_top8_balanced_r8_lr1e5_1000_fp32_gpu \
  --output_dir outputs/eval_top8_balanced_r8_lr1e5_1000_fp32_gpu_full \
  --hr_size 256 \
  --lr_size 64 \
  --eval_max_images 0 \
  --num_inference_steps 25 \
  --base_summary_csv outputs/eval_base_gpu_full/eval_summary.csv \
  --train_summary_csv outputs/lora_top8_balanced_r8_lr1e5_1000_fp32_gpu/summary.csv
```

## 7. Summarize Base vs Sandwich-LoRA

After Base evaluation, Sandwich-LoRA training, and Sandwich-LoRA evaluation are
finished, generate a compact comparison table and report:

```bash
python3 summarize_base_sandwich.py \
  --base_eval_summary outputs/eval_base/eval_summary.csv \
  --sandwich_eval_summary outputs/eval_sandwich_r8/eval_summary.csv \
  --sandwich_train_summary outputs/lora_sandwich_r8/summary.csv \
  --output_csv outputs/base_vs_sandwich_summary.csv \
  --output_md outputs/base_vs_sandwich_report.md
```

Outputs:

- `outputs/base_vs_sandwich_summary.csv`
- `outputs/base_vs_sandwich_report.md`

Training memory metrics are written automatically:

- `summary.csv`: start/end/peak CUDA allocated and reserved memory
- `train_log.csv`: per-step current and peak CUDA memory
- If training loss becomes `nan`, rerun a small check with `--no_fp16`; non-finite steps are logged in `train_log.csv`.

Example stable debug run:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode train \
  --train_dir data/ucmerced/train_hr \
  --output_dir outputs/lora_sandwich_debug_fp32 \
  --hr_size 256 \
  --lr_size 64 \
  --rank 8 \
  --alpha 16 \
  --target qv \
  --lora_scope shallow_deep \
  --train_steps 20 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 1e-5 \
  --grad_clip 1.0 \
  --no_fp16
```

## 8. Inspect a LoRA Adapter

If LoRA evaluation is identical to Base, first check whether the saved adapter is
nonzero:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode inspect_lora \
  --lora_dir outputs/lora_sandwich_r8_gpu \
  --output_dir outputs/inspect_lora_sandwich_r8_gpu
```

For a deeper check that loads the model and verifies module-name matching:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode inspect_lora \
  --lora_dir outputs/lora_sandwich_r8_gpu \
  --output_dir outputs/inspect_lora_sandwich_r8_gpu \
  --inspect_load_model
```

Outputs:

- `lora_inspect_summary.csv`
- `lora_inspect_tensors.csv`

## 9. Full UNet Fine-tuning Baseline

Use this to compare full fine-tuning with Sandwich-LoRA. First run a short
debug job to check memory:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode train \
  --train_method full_unet \
  --train_dir data/ucmerced/train_hr \
  --output_dir outputs/full_unet_lr1e5_20_fp32_gpu \
  --hr_size 256 \
  --lr_size 64 \
  --train_steps 20 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 1e-5 \
  --grad_clip 1.0 \
  --power_w 30 \
  --full_model_size_mb 1200 \
  --no_fp16
```

If it fits, run the full baseline:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode train \
  --train_method full_unet \
  --train_dir data/ucmerced/train_hr \
  --output_dir outputs/full_unet_lr1e5_1000_fp32_gpu \
  --hr_size 256 \
  --lr_size 64 \
  --train_steps 1000 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 1e-5 \
  --grad_clip 1.0 \
  --power_w 30 \
  --full_model_size_mb 1200 \
  --no_fp16
```

Evaluate the full fine-tuned UNet:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode eval \
  --val_dir data/ucmerced/val_hr \
  --unet_dir outputs/full_unet_lr1e5_1000_fp32_gpu/unet \
  --output_dir outputs/eval_full_unet_lr1e5_1000_fp32_gpu_full \
  --hr_size 256 \
  --lr_size 64 \
  --eval_max_images 0 \
  --num_inference_steps 25 \
  --base_summary_csv outputs/eval_base_gpu_full/eval_summary.csv \
  --train_summary_csv outputs/full_unet_lr1e5_1000_fp32_gpu/summary.csv
```

Generate a three-way summary. The summarizer accepts repeatable
`--extra_method name,eval_summary_csv,train_summary_csv` entries, so new
methods can be added without editing the script:

```bash
python3 summarize_base_sandwich.py \
  --base_eval_summary outputs/eval_base_gpu_full/eval_summary.csv \
  --extra_method Sandwich-LoRA,outputs/eval_sandwich_r8_lr1e5_1000_fp32_gpu_full/eval_summary.csv,outputs/lora_sandwich_r8_lr1e5_1000_fp32_gpu/summary.csv \
  --extra_method Full-UNet,outputs/eval_full_unet_lr1e5_1000_fp32_gpu_full/eval_summary.csv,outputs/full_unet_lr1e5_1000_fp32_gpu/summary.csv \
  --output_csv outputs/base_vs_sandwich_vs_full_summary.csv \
  --output_md outputs/base_vs_sandwich_vs_full_report.md
```

## 10. FP32 Training + FP16 LoRA Saving

To keep stable FP32 training but save a smaller FP16 adapter, add:

```bash
--no_fp16 \
--save_lora_dtype fp16
```

Example:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode train \
  --train_method lora \
  --train_dir data/ucmerced/train_hr \
  --output_dir outputs/lora_sandwich_r8_lr1e5_1000_fp32_savefp16_gpu \
  --hr_size 256 \
  --lr_size 64 \
  --rank 8 \
  --alpha 16 \
  --target qv \
  --lora_scope shallow_deep \
  --train_steps 1000 \
  --batch_size 1 \
  --grad_accum 4 \
  --lr 1e-5 \
  --grad_clip 1.0 \
  --power_w 30 \
  --full_model_size_mb 1200 \
  --no_fp16 \
  --save_lora_dtype fp16
```

To convert an existing FP32 adapter without retraining:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode quantize_lora \
  --lora_dir outputs/lora_sandwich_r8_lr1e5_1000_fp32_gpu \
  --output_dir outputs/lora_sandwich_r8_lr1e5_1000_fp32_gpu_savefp16 \
  --save_lora_dtype fp16
```

Evaluate the converted adapter in the same way:

```bash
python3 onboard_sandwich_lora_sr.py \
  --mode eval \
  --val_dir data/ucmerced/val_hr \
  --lora_dir outputs/lora_sandwich_r8_lr1e5_1000_fp32_gpu_savefp16 \
  --output_dir outputs/eval_sandwich_r8_lr1e5_1000_fp32_gpu_savefp16_full \
  --hr_size 256 \
  --lr_size 64 \
  --eval_max_images 0 \
  --num_inference_steps 25 \
  --base_summary_csv outputs/eval_base_gpu_full/eval_summary.csv \
  --train_summary_csv outputs/lora_sandwich_r8_lr1e5_1000_fp32_gpu_savefp16/quantize_lora_summary.csv
```

Then include it in the comparison table:

```bash
python3 summarize_base_sandwich.py \
  --base_eval_summary outputs/eval_base_gpu_full/eval_summary.csv \
  --extra_method Sandwich-LoRA-FP32,outputs/eval_sandwich_r8_lr1e5_1000_fp32_gpu_full/eval_summary.csv,outputs/lora_sandwich_r8_lr1e5_1000_fp32_gpu/summary.csv \
  --extra_method Sandwich-LoRA-FP16-Adapter,outputs/eval_sandwich_r8_lr1e5_1000_fp32_gpu_savefp16_full/eval_summary.csv,outputs/lora_sandwich_r8_lr1e5_1000_fp32_gpu_savefp16/quantize_lora_summary.csv \
  --extra_method Full-UNet,outputs/eval_full_unet_lr1e5_1000_fp32_gpu_full/eval_summary.csv,outputs/full_unet_lr1e5_1000_fp32_gpu/summary.csv \
  --output_csv outputs/base_vs_sandwich_fp32_fp16_vs_full_summary.csv \
  --output_md outputs/base_vs_sandwich_fp32_fp16_vs_full_report.md
```

After the All-LoRA experiment finishes, add it as one more method:

```bash
python3 summarize_base_sandwich.py \
  --base_eval_summary outputs/eval_base_gpu_full/eval_summary.csv \
  --extra_method Sandwich-LoRA-FP32,outputs/eval_sandwich_r8_lr1e5_1000_fp32_gpu_full/eval_summary.csv,outputs/lora_sandwich_r8_lr1e5_1000_fp32_gpu/summary.csv \
  --extra_method Sandwich-LoRA-FP16-Adapter,outputs/eval_sandwich_r8_lr1e5_1000_fp32_gpu_savefp16_full/eval_summary.csv,outputs/lora_sandwich_r8_lr1e5_1000_fp32_gpu_savefp16/quantize_lora_summary.csv \
  --extra_method All-LoRA,outputs/eval_all_r8_lr1e5_1000_fp32_gpu_full/eval_summary.csv,outputs/lora_all_r8_lr1e5_1000_fp32_gpu/summary.csv \
  --extra_method Full-UNet,outputs/eval_full_unet_lr1e5_1000_fp32_gpu_full/eval_summary.csv,outputs/full_unet_lr1e5_1000_fp32_gpu/summary.csv \
  --output_csv outputs/base_vs_sandwich_vs_all_vs_full_summary.csv \
  --output_md outputs/base_vs_sandwich_vs_all_vs_full_report.md
```
