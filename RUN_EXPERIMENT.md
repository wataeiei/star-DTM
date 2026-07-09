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

- `--lora_scope shallow` for Shallow-LoRA
- `--lora_scope last2_up` for Last2-Up-LoRA
- `--lora_scope all` for All-LoRA

For formal results, prefer increasing `--eval_max_images` or setting it to `0`
to evaluate the full validation split.
