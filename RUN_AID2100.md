# AID-2.1K data preparation

Run all commands from the `star-DTM` repository root.

## 1. Expected source layout

Extract the complete AID dataset without resizing or cropping it. Point
`AID_SOURCE` at the directory whose immediate child directories are the 30
scene classes.

```text
AID_SOURCE/
  airport/
  bareland/
  baseballfield/
  ...
```

## 2. Build the fixed 2,100-image split

```bash
cd /mnt/disk1T/liyijuan/star-DTM
conda activate star

AID_SOURCE="/mnt/disk1T/liyijuan/datasets/AID"
AID_OUT="data/aid2100"

python3 build_aid_2100_split.py \
  --source_dir "$AID_SOURCE" \
  --output_dir "$AID_OUT" \
  --seed 2026 \
  --expected_classes 30 \
  --samples_per_class 70 \
  --train_per_class 56 \
  --materialize hardlink
```

If `build_aid_2100_split.py` is kept in the local `code` directory rather than
the repository root, invoke it as `python3 code/build_aid_2100_split.py`.

The split is fixed independently of training seeds 42, 43, and 44:

- 56 images per class for SR adaptation: 1,680 total.
- 14 images per class for evaluation: 420 total.
- No selected train/validation pair has identical RGB pixels.

Hard links consume no additional image storage on the same filesystem. If the
source and output are on different filesystems, the script falls back to a
normal copy and records that fact in the summary.

## 3. Audit the result

```bash
find data/aid2100/train_hr -type f | wc -l
find data/aid2100/val_hr -type f | wc -l

cat data/aid2100/split_manifest_summary.json

python3 - <<'EOF'
import pandas as pd

root = "data/aid2100"
train = pd.read_csv(f"{root}/train_manifest.csv")
val = pd.read_csv(f"{root}/val_eval_manifest.csv")

print("train images:", len(train))
print("val images:", len(val))
print("train classes:", train["class_name"].nunique())
print("val classes:", val["class_name"].nunique())
print("cross-split pixel overlap:", len(set(train.pixel_sha256) & set(val.pixel_sha256)))

print("\nTrain count per class:")
print(train.groupby("class_name").size().to_string())
print("\nValidation count per class:")
print(val.groupby("class_name").size().to_string())
EOF
```

Expected output:

```text
train images: 1680
val images: 420
train classes: 30
val classes: 30
cross-split pixel overlap: 0
```

## 4. Train the AID semantic evaluator

The existing classifier accepts the generated generic manifest even though its
historical filename contains `ucmerced`.

```bash
python3 ucmerced_semantic_classifier.py train \
  --train_manifest data/aid2100/train_manifest.csv \
  --output_dir outputs/aid2100_resnet18_classifier \
  --epochs 40 \
  --batch_size 32 \
  --image_size 224 \
  --monitor_fraction 0.1 \
  --freeze_backbone_epochs 2 \
  --lr 1e-4 \
  --weight_decay 1e-4 \
  --label_smoothing 0.1 \
  --dtype bf16 \
  --num_workers 2 \
  --seed 2026
```

Evaluate the HR upper bound before using this classifier for generated images:

```bash
python3 ucmerced_semantic_classifier.py eval \
  --checkpoint outputs/aid2100_resnet18_classifier/best_classifier.pt \
  --eval_manifest data/aid2100/val_eval_manifest.csv \
  --source HR=data/aid2100/val_hr \
  --output_dir outputs/aid2100_resnet18_classifier_hr_eval \
  --batch_size 64 \
  --dtype bf16 \
  --num_workers 2

cat outputs/aid2100_resnet18_classifier_hr_eval/semantic_metrics_summary.csv
```

Use the same `data/aid2100` split for every SR method and every training seed.
Do not regenerate it with seed 42, 43, or 44.
