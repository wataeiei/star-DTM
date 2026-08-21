#!/usr/bin/env python3
"""Train a torchvision-free UC Merced classifier and score SR outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
RESNET18_WEIGHTS_URL = "https://download.pytorch.org/models/resnet18-f37072fd.pth"
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp"}


def write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"Refusing to write an empty CSV: {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path, include_excluded: bool = False) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise SystemExit(f"Empty manifest: {path}")
    required = {"filename", "relative_path", "class_id", "class_name"}
    missing = required - set(rows[0])
    if missing:
        raise SystemExit(f"Manifest is missing columns: {sorted(missing)}")
    if not include_excluded and "exclude_from_eval" in rows[0]:
        rows = [row for row in rows if row["exclude_from_eval"].lower() != "true"]
    return rows


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def pil_to_normalized_tensor(image: Image.Image, size: int, train: bool) -> torch.Tensor:
    image = image.convert("RGB")
    resize_size = max(size + 32, size)
    image = image.resize((resize_size, resize_size), Image.Resampling.BICUBIC)
    if train:
        left = random.randint(0, resize_size - size)
        top = random.randint(0, resize_size - size)
        image = image.crop((left, top, left + size, top + size))
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        if random.random() < 0.5:
            image = image.transpose(Image.Transpose.FLIP_TOP_BOTTOM)
        angle = random.choice((0, 90, 180, 270))
        if angle:
            image = image.rotate(angle)
    else:
        offset = (resize_size - size) // 2
        image = image.crop((offset, offset, offset + size, offset + size))
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    return (tensor - IMAGENET_MEAN) / IMAGENET_STD


class ManifestDataset(Dataset):
    def __init__(self, rows: list[dict], train: bool, image_size: int):
        self.rows = rows
        self.train = train
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int):
        row = self.rows[index]
        path = Path(row["resolved_path"] if "resolved_path" in row else row["relative_path"])
        with Image.open(path) as image:
            tensor = pil_to_normalized_tensor(image, self.image_size, self.train)
        return tensor, int(row["class_id"]), row["filename"]


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(
            inplanes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = None
        if stride != 1 or inplanes != planes:
            self.downsample = nn.Sequential(
                nn.Conv2d(inplanes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu(out + identity)


class ResNet18(nn.Module):
    def __init__(self, num_classes: int = 1000):
        super().__init__()
        self.inplanes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, planes: int, blocks: int, stride: int) -> nn.Sequential:
        layers = [BasicBlock(self.inplanes, planes, stride)]
        self.inplanes = planes
        layers.extend(BasicBlock(self.inplanes, planes) for _ in range(1, blocks))
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


def load_imagenet_resnet18(num_classes: int, weights_path: str) -> ResNet18:
    model = ResNet18(num_classes=1000)
    if weights_path:
        state = torch.load(weights_path, map_location="cpu", weights_only=True)
    else:
        state = torch.hub.load_state_dict_from_url(
            RESNET18_WEIGHTS_URL, map_location="cpu", check_hash=True, progress=True
        )
    if "state_dict" in state:
        state = state["state_dict"]
    state = {key.removeprefix("module."): value for key, value in state.items()}
    model.load_state_dict(state, strict=True)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def stratified_monitor_split(
    rows: list[dict], fraction: float, seed: int
) -> tuple[list[dict], list[dict]]:
    by_class: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        by_class[int(row["class_id"])].append(row)
    rng = random.Random(seed)
    train_rows, monitor_rows = [], []
    for class_rows in by_class.values():
        rng.shuffle(class_rows)
        count = max(1, round(len(class_rows) * fraction))
        count = min(count, len(class_rows) - 1)
        monitor_rows.extend(class_rows[:count])
        train_rows.extend(class_rows[count:])
    rng.shuffle(train_rows)
    rng.shuffle(monitor_rows)
    return train_rows, monitor_rows


def confusion_metrics(confusion: torch.Tensor) -> dict[str, float]:
    matrix = confusion.to(torch.float64)
    total = matrix.sum().item()
    accuracy = matrix.diag().sum().item() / max(total, 1.0)
    f1_values = []
    recalls = []
    for index in range(matrix.shape[0]):
        tp = matrix[index, index].item()
        fp = matrix[:, index].sum().item() - tp
        fn = matrix[index, :].sum().item() - tp
        precision = tp / max(tp + fp, 1.0)
        recall = tp / max(tp + fn, 1.0)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        recalls.append(recall)
    return {
        "top1_accuracy": accuracy,
        "macro_f1": sum(f1_values) / len(f1_values),
        "balanced_accuracy": sum(recalls) / len(recalls),
    }


def amp_settings(dtype: str, device: torch.device):
    enabled = device.type == "cuda" and dtype != "fp32"
    amp_dtype = torch.bfloat16 if dtype == "bf16" else torch.float16
    return enabled, amp_dtype


@torch.no_grad()
def evaluate_loader(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    num_classes: int,
    dtype: str,
) -> tuple[dict[str, float], torch.Tensor, list[dict]]:
    model.eval()
    confusion = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    rows = []
    amp_enabled, amp_dtype = amp_settings(dtype, device)
    started = time.perf_counter()
    for images, labels, filenames in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        with torch.autocast(device_type=device.type, enabled=amp_enabled, dtype=amp_dtype):
            logits = model(images)
        probabilities = logits.softmax(dim=1)
        confidence, predictions = probabilities.max(dim=1)
        for truth, prediction, score, filename in zip(
            labels.cpu(), predictions.cpu(), confidence.cpu(), filenames
        ):
            confusion[int(truth), int(prediction)] += 1
            rows.append(
                {
                    "filename": filename,
                    "class_id": int(truth),
                    "prediction_id": int(prediction),
                    "confidence": float(score),
                    "correct": int(truth) == int(prediction),
                }
            )
    metrics = confusion_metrics(confusion)
    metrics["num_images"] = int(confusion.sum())
    metrics["elapsed_s"] = time.perf_counter() - started
    metrics["images_per_second"] = metrics["num_images"] / max(metrics["elapsed_s"], 1e-9)
    return metrics, confusion, rows


def make_loader(
    rows: list[dict], train: bool, args: argparse.Namespace, shuffle: bool
) -> DataLoader:
    dataset = ManifestDataset(rows, train=train, image_size=args.image_size)
    return DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=shuffle,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=train,
    )


def train(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    rows = read_manifest(Path(args.train_manifest), include_excluded=True)
    class_map = {int(row["class_id"]): row["class_name"] for row in rows}
    class_names = [class_map[index] for index in sorted(class_map)]
    if sorted(class_map) != list(range(len(class_names))):
        raise SystemExit("class_id values must be contiguous and zero-based")
    train_rows, monitor_rows = stratified_monitor_split(
        rows, args.monitor_fraction, args.seed
    )
    print(
        f"classifier train={len(train_rows)} monitor={len(monitor_rows)} "
        f"classes={len(class_names)} device={device}"
    )

    train_loader = make_loader(train_rows, True, args, True)
    monitor_loader = make_loader(monitor_rows, False, args, False)
    model = load_imagenet_resnet18(len(class_names), args.pretrained_weights).to(device)
    if args.freeze_backbone_epochs > 0:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")

    counts = Counter(int(row["class_id"]) for row in train_rows)
    class_weights = torch.tensor(
        [len(train_rows) / (len(class_names) * counts[index]) for index in range(len(class_names))],
        dtype=torch.float32,
        device=device,
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = torch.amp.GradScaler(
        "cuda", enabled=device.type == "cuda" and args.dtype == "fp16"
    )
    amp_enabled, amp_dtype = amp_settings(args.dtype, device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history = []
    best_macro_f1 = -1.0
    best_path = output_dir / "best_classifier.pt"

    for epoch in range(1, args.epochs + 1):
        if epoch == args.freeze_backbone_epochs + 1:
            for parameter in model.parameters():
                parameter.requires_grad = True
        model.train()
        loss_sum = 0.0
        correct = 0
        seen = 0
        started = time.perf_counter()
        for images, labels, _filenames in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type, enabled=amp_enabled, dtype=amp_dtype
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_size = labels.numel()
            loss_sum += float(loss.detach()) * batch_size
            correct += int((logits.argmax(dim=1) == labels).sum())
            seen += batch_size
        scheduler.step()
        monitor, _confusion, _rows = evaluate_loader(
            model, monitor_loader, device, len(class_names), args.dtype
        )
        epoch_row = {
            "epoch": epoch,
            "train_loss": loss_sum / max(seen, 1),
            "train_top1_accuracy": correct / max(seen, 1),
            "monitor_top1_accuracy": monitor["top1_accuracy"],
            "monitor_macro_f1": monitor["macro_f1"],
            "monitor_balanced_accuracy": monitor["balanced_accuracy"],
            "lr": optimizer.param_groups[0]["lr"],
            "epoch_time_s": time.perf_counter() - started,
        }
        history.append(epoch_row)
        print(
            f"epoch {epoch:03d}/{args.epochs} loss={epoch_row['train_loss']:.4f} "
            f"train_acc={epoch_row['train_top1_accuracy']:.4f} "
            f"monitor_acc={monitor['top1_accuracy']:.4f} "
            f"macro_f1={monitor['macro_f1']:.4f}"
        )
        if monitor["macro_f1"] > best_macro_f1:
            best_macro_f1 = monitor["macro_f1"]
            torch.save(
                {
                    "model": model.state_dict(),
                    "class_names": class_names,
                    "image_size": args.image_size,
                    "epoch": epoch,
                    "monitor_metrics": monitor,
                    "train_args": vars(args),
                },
                best_path,
            )

    write_csv(output_dir / "classifier_train_log.csv", history)
    metadata = {
        "checkpoint": str(best_path),
        "best_monitor_macro_f1": best_macro_f1,
        "num_train_images": len(train_rows),
        "num_monitor_images": len(monitor_rows),
        "class_names": class_names,
        "train_class_counts": dict(sorted(Counter(row["class_name"] for row in train_rows).items())),
        "monitor_class_counts": dict(sorted(Counter(row["class_name"] for row in monitor_rows).items())),
    }
    (output_dir / "classifier_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(f"Wrote {best_path}")


def parse_source(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("--source must use LABEL=IMAGE_DIR")
    label, path = value.split("=", 1)
    if not label or not path:
        raise argparse.ArgumentTypeError("--source must use LABEL=IMAGE_DIR")
    return label, Path(path)


def safe_name(value: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("_") or "method"


def resolve_source_rows(manifest_rows: list[dict], source_dir: Path) -> list[dict]:
    if not source_dir.is_dir():
        raise SystemExit(f"Image source directory not found: {source_dir}")
    image_paths = sorted(
        path for path in source_dir.rglob("*") if path.suffix.lower() in IMAGE_SUFFIXES
    )
    by_name: dict[str, Path] = {}
    for path in image_paths:
        by_name[path.name] = path
        match = re.match(r"^\d+_(.+)$", path.name)
        if match:
            by_name[match.group(1)] = path
    resolved = []
    missing = []
    for row in manifest_rows:
        path = by_name.get(row["filename"])
        if path is None:
            missing.append(row["filename"])
            continue
        item = dict(row)
        item["resolved_path"] = str(path)
        resolved.append(item)
    if missing:
        raise SystemExit(
            f"{source_dir} is missing {len(missing)} manifest images, examples: {missing[:5]}"
        )
    return resolved


def write_confusion(path: Path, confusion: torch.Tensor, class_names: list[str]) -> None:
    rows = []
    for index, class_name in enumerate(class_names):
        row = {"true_class": class_name}
        row.update(
            {f"pred_{name}": int(confusion[index, pred]) for pred, name in enumerate(class_names)}
        )
        rows.append(row)
    write_csv(path, rows)


def evaluate_sources(args: argparse.Namespace) -> None:
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    class_names = checkpoint["class_names"]
    model = ResNet18(num_classes=len(class_names))
    model.load_state_dict(checkpoint["model"], strict=True)
    model.to(device)
    manifest_rows = read_manifest(Path(args.eval_manifest))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    summaries = []
    all_rows = []

    for label, source_dir in args.source:
        rows = resolve_source_rows(manifest_rows, source_dir)
        loader = make_loader(rows, False, args, False)
        metrics, confusion, predictions = evaluate_loader(
            model, loader, device, len(class_names), args.dtype
        )
        for row in predictions:
            row["method"] = label
            row["class_name"] = class_names[row["class_id"]]
            row["prediction_name"] = class_names[row["prediction_id"]]
        all_rows.extend(predictions)
        summaries.append({"method": label, **metrics})
        write_confusion(
            output_dir / f"confusion_{safe_name(label)}.csv", confusion, class_names
        )
        print(
            f"{label:28s} n={metrics['num_images']} "
            f"top1={metrics['top1_accuracy']:.4f} "
            f"macro_f1={metrics['macro_f1']:.4f} "
            f"balanced_acc={metrics['balanced_accuracy']:.4f}"
        )

    write_csv(output_dir / "semantic_metrics_summary.csv", summaries)
    write_csv(output_dir / "semantic_metrics_per_image.csv", all_rows)
    print(f"Wrote results to {output_dir}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="mode", required=True)

    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--train_manifest", required=True)
    train_parser.add_argument("--output_dir", required=True)
    train_parser.add_argument("--pretrained_weights", default="")
    train_parser.add_argument("--epochs", type=int, default=30)
    train_parser.add_argument("--batch_size", type=int, default=32)
    train_parser.add_argument("--image_size", type=int, default=224)
    train_parser.add_argument("--monitor_fraction", type=float, default=0.1)
    train_parser.add_argument("--freeze_backbone_epochs", type=int, default=2)
    train_parser.add_argument("--lr", type=float, default=1e-4)
    train_parser.add_argument("--weight_decay", type=float, default=1e-4)
    train_parser.add_argument("--label_smoothing", type=float, default=0.1)
    train_parser.add_argument("--grad_clip", type=float, default=1.0)
    train_parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    train_parser.add_argument("--num_workers", type=int, default=2)
    train_parser.add_argument("--seed", type=int, default=2026)
    train_parser.add_argument("--cpu", action="store_true")

    eval_parser = subparsers.add_parser("eval")
    eval_parser.add_argument("--checkpoint", required=True)
    eval_parser.add_argument("--eval_manifest", required=True)
    eval_parser.add_argument("--source", action="append", type=parse_source, required=True)
    eval_parser.add_argument("--output_dir", required=True)
    eval_parser.add_argument("--batch_size", type=int, default=64)
    eval_parser.add_argument("--image_size", type=int, default=224)
    eval_parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="bf16")
    eval_parser.add_argument("--num_workers", type=int, default=2)
    eval_parser.add_argument("--seed", type=int, default=2026)
    eval_parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode == "train":
        train(args)
    else:
        evaluate_sources(args)


if __name__ == "__main__":
    main()
