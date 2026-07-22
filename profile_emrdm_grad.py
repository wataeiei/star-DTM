#!/usr/bin/env python3
"""Gradient profiling for the official EMRDM repository.

Copy this file into the EMRDM repository root, then run it there.

The script instantiates EMRDM from its YAML config, optionally loads a checkpoint,
inserts temporary probe LoRA modules into attention Linear layers, computes
gradient norms on paired cloudy/clear calibration images, and writes CSV output
compatible with plot_cross_model_grad_scores.py.
"""

from __future__ import annotations

import argparse
import csv
import importlib
import json
import math
import random
import re
import sys
import types
from pathlib import Path
from typing import Iterable

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
NATTEN_BACKEND = "auto"
NATTEN_NATIVE_NA2D = None
NATTEN_WARNED_FALLBACK = False


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def instantiate_from_config(config):
    install_runtime_stubs()
    target = config.get("target")
    if not target:
        raise ValueError("Config section has no target field.")
    module_name, cls_name = target.rsplit(".", 1)
    cls = getattr(importlib.import_module(module_name), cls_name)
    params = config.get("params", {})
    return cls(**params)


def install_runtime_stubs() -> None:
    """Install lightweight stubs for optional training/evaluation dependencies."""
    install_lightning_stub()
    install_torchvision_stub()
    install_metric_stubs()
    install_logging_stubs()
    install_natten_compat()


class LightningModuleStub(nn.Module):
    def log(self, *args, **kwargs):
        return None

    def log_dict(self, *args, **kwargs):
        return None

    def save_hyperparameters(self, *args, **kwargs):
        return None


def install_lightning_stub() -> None:
    """Avoid importing Lightning's torchmetrics/torchvision stack.

    EMRDM modules subclass pl.LightningModule, but this profiler only needs
    ordinary nn.Module behavior and direct forward/shared_step calls. On Jetson,
    Lightning imports torchmetrics, which imports an incompatible torchvision.
    """
    if "pytorch_lightning" in sys.modules:
        return
    pl = types.ModuleType("pytorch_lightning")
    callbacks = types.ModuleType("pytorch_lightning.callbacks")
    loggers = types.ModuleType("pytorch_lightning.loggers")
    utilities = types.ModuleType("pytorch_lightning.utilities")
    rank_zero = types.ModuleType("pytorch_lightning.utilities.rank_zero")

    class Callback:
        pass

    class ModelCheckpoint(Callback):
        def __init__(self, *args, **kwargs):
            pass

    class LearningRateMonitor(Callback):
        def __init__(self, *args, **kwargs):
            pass

    class Trainer:
        pass

    class WandbLogger:
        def __init__(self, *args, **kwargs):
            pass

        def log_metrics(self, *args, **kwargs):
            return None

        def log_image(self, *args, **kwargs):
            return None

    def rank_zero_only(fn=None, *args, **kwargs):
        if fn is None:
            return lambda inner: inner
        return fn

    pl.LightningModule = LightningModuleStub
    pl.Callback = Callback
    pl.Trainer = Trainer
    pl.seed_everything = lambda *args, **kwargs: None
    callbacks.Callback = Callback
    callbacks.ModelCheckpoint = ModelCheckpoint
    callbacks.LearningRateMonitor = LearningRateMonitor
    loggers.WandbLogger = WandbLogger
    rank_zero.rank_zero_only = rank_zero_only
    rank_zero.rank_zero_info = lambda *args, **kwargs: None
    rank_zero.rank_zero_warn = lambda *args, **kwargs: None
    utilities.rank_zero = rank_zero
    sys.modules["pytorch_lightning"] = pl
    sys.modules["pytorch_lightning.callbacks"] = callbacks
    sys.modules["pytorch_lightning.loggers"] = loggers
    sys.modules["pytorch_lightning.utilities"] = utilities
    sys.modules["pytorch_lightning.utilities.rank_zero"] = rank_zero


def install_torchvision_stub() -> None:
    """Avoid importing an incompatible torchvision build on Jetson."""
    if "torchvision" in sys.modules:
        return

    torchvision = types.ModuleType("torchvision")
    transforms = types.ModuleType("torchvision.transforms")
    transforms_v2 = types.ModuleType("torchvision.transforms.v2")
    utils = types.ModuleType("torchvision.utils")
    models = types.ModuleType("torchvision.models")
    datasets = types.ModuleType("torchvision.datasets")
    ops = types.ModuleType("torchvision.ops")
    ops_misc = types.ModuleType("torchvision.ops.misc")

    class IdentityTransform:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, x):
            return x

    class Compose:
        def __init__(self, transforms_list):
            self.transforms = list(transforms_list)

        def __call__(self, x):
            for transform in self.transforms:
                x = transform(x)
            return x

    class ImageFolder:
        def __init__(self, *args, **kwargs):
            raise RuntimeError("torchvision.datasets.ImageFolder is unavailable in this profiling environment.")

    transforms_v2.Compose = Compose
    transforms_v2.Normalize = IdentityTransform
    transforms_v2.ToImage = IdentityTransform
    transforms_v2.ToDtype = IdentityTransform
    transforms_v2.Resize = IdentityTransform
    transforms.Compose = Compose
    transforms.Normalize = IdentityTransform
    transforms.Resize = IdentityTransform
    transforms.ToTensor = IdentityTransform
    transforms.CenterCrop = IdentityTransform
    transforms.v2 = transforms_v2
    utils.make_grid = lambda tensor, *args, **kwargs: tensor
    datasets.ImageFolder = ImageFolder
    ops_misc.FrozenBatchNorm2d = nn.BatchNorm2d
    ops.misc = ops_misc
    for model_name in ("alexnet", "vgg16", "squeezenet1_1"):
        setattr(models, model_name, lambda *args, **kwargs: nn.Identity())
    torchvision.transforms = transforms
    torchvision.utils = utils
    torchvision.models = models
    torchvision.datasets = datasets
    torchvision.ops = ops

    sys.modules["torchvision"] = torchvision
    sys.modules["torchvision.transforms"] = transforms
    sys.modules["torchvision.transforms.v2"] = transforms_v2
    sys.modules["torchvision.utils"] = utils
    sys.modules["torchvision.models"] = models
    sys.modules["torchvision.datasets"] = datasets
    sys.modules["torchvision.ops"] = ops
    sys.modules["torchvision.ops.misc"] = ops_misc


def install_metric_stubs() -> None:
    """Stub optional image metrics that are irrelevant for grad profiling."""
    if "lpips" not in sys.modules:
        lpips = types.ModuleType("lpips")

        class LPIPS(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def forward(self, x, y, *args, **kwargs):
                return torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)

        lpips.LPIPS = LPIPS
        sys.modules["lpips"] = lpips
    if "torchmetrics" not in sys.modules:
        torchmetrics = types.ModuleType("torchmetrics")
        functional = types.ModuleType("torchmetrics.functional")

        class Metric(nn.Module):
            def __init__(self, *args, **kwargs):
                super().__init__()

            def update(self, *args, **kwargs):
                return None

            def compute(self):
                return torch.tensor(0.0)

        torchmetrics.Metric = Metric
        torchmetrics.functional = functional
        sys.modules["torchmetrics"] = torchmetrics
        sys.modules["torchmetrics.functional"] = functional


def install_logging_stubs() -> None:
    if "wandb" not in sys.modules:
        wandb = types.ModuleType("wandb")
        wandb.init = lambda *args, **kwargs: None
        wandb.log = lambda *args, **kwargs: None
        wandb.Image = lambda x, *args, **kwargs: x
        wandb.finish = lambda *args, **kwargs: None
        sys.modules["wandb"] = wandb


def install_natten_compat() -> None:
    """Patch NATTEN API differences with a pure PyTorch fallback.

    EMRDM was written against an older NATTEN API. Recent NATTEN releases moved
    or removed legacy functions such as na2d_qk/na2d_av. For profiling we prefer
    a slower but stable implementation over chasing CUDA extension variants.
    """
    global NATTEN_NATIVE_NA2D
    try:
        import natten
    except ImportError:
        natten = types.ModuleType("natten")
        sys.modules["natten"] = natten
    if not hasattr(natten, "functional"):
        natten.functional = types.ModuleType("natten.functional")
        sys.modules["natten.functional"] = natten.functional
    NATTEN_NATIVE_NA2D = getattr(natten, "na2d", None)
    if NATTEN_BACKEND == "torch" or NATTEN_NATIVE_NA2D is None:
        natten.functional.na2d = torch_na2d
    else:
        natten.functional.na2d = native_or_torch_na2d
    natten.has_fused_na = lambda: True


def native_or_torch_na2d(query, key, value, kernel_size, scale=None, **kwargs):
    """Use native NATTEN when possible, with a pure PyTorch fallback."""
    global NATTEN_WARNED_FALLBACK
    if NATTEN_NATIVE_NA2D is not None and NATTEN_BACKEND in {"auto", "native"}:
        q = query.contiguous()
        k = key.contiguous()
        v = value.contiguous()
        attempts = (
            lambda: NATTEN_NATIVE_NA2D(q, k, v, kernel_size=kernel_size, dilation=1, scale=scale),
            lambda: NATTEN_NATIVE_NA2D(q, k, v, kernel_size, dilation=1, scale=scale),
            lambda: NATTEN_NATIVE_NA2D(q, k, v, kernel_size=kernel_size, scale=scale),
            lambda: NATTEN_NATIVE_NA2D(q, k, v, kernel_size, scale=scale),
            lambda: NATTEN_NATIVE_NA2D(q, k, v, kernel_size),
        )
        last_error = None
        for attempt in attempts:
            try:
                out = attempt()
                if isinstance(out, (tuple, list)):
                    out = out[0]
                return out
            except Exception as exc:
                last_error = exc
                if NATTEN_BACKEND == "native" and not isinstance(exc, TypeError):
                    raise
                continue
        if NATTEN_BACKEND == "native":
            raise last_error if last_error is not None else RuntimeError("Native NATTEN na2d failed.")
        if not NATTEN_WARNED_FALLBACK:
            print(f"Warning: native NATTEN na2d failed ({last_error}); falling back to torch_na2d.")
            NATTEN_WARNED_FALLBACK = True
    return torch_na2d(query, key, value, kernel_size, scale=scale, **kwargs)


def torch_na2d(query, key, value, kernel_size, scale=None, **kwargs):
    """Pure PyTorch 2D neighborhood attention.

    Expected layout follows the fused EMRDM branch:
      query/key/value: [batch, height, width, heads, head_dim]
      output:          [batch, height, width, heads, head_dim]
    """
    if isinstance(kernel_size, int):
        kh = kw = kernel_size
    else:
        kh, kw = int(kernel_size[0]), int(kernel_size[1])
    if kh % 2 == 0 or kw % 2 == 0:
        raise ValueError(f"Only odd neighborhood kernel sizes are supported, got {kernel_size}")
    bsz, height, width, heads, dim = query.shape
    scale = float(scale) if scale is not None else dim ** -0.5

    def neighborhoods(x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 3, 4, 1, 2).reshape(bsz * heads, dim, height, width)
        x = F.pad(x, (kw // 2, kw // 2, kh // 2, kh // 2))
        patches = F.unfold(x, kernel_size=(kh, kw))
        patches = patches.view(bsz, heads, dim, kh * kw, height, width)
        return patches.permute(0, 4, 5, 1, 3, 2)

    k_neigh = neighborhoods(key)
    v_neigh = neighborhoods(value)
    scores = (query.unsqueeze(-2).float() * k_neigh.float()).sum(dim=-1) * scale
    attn = torch.softmax(scores, dim=-1).to(dtype=value.dtype)
    out = (attn.unsqueeze(-1) * v_neigh).sum(dim=-2)
    return out


def image_to_tensor(path: Path, image_size: int) -> torch.Tensor:
    image = Image.open(path).convert("RGB").resize((image_size, image_size), Image.BICUBIC)
    x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
    x = x.view(image_size, image_size, 3).permute(2, 0, 1).float() / 255.0
    return x * 2.0 - 1.0


def pad_or_trim_channels(x: torch.Tensor, channels: int) -> torch.Tensor:
    if x.shape[1] == channels:
        return x
    if x.shape[1] > channels:
        return x[:, :channels]
    pad = torch.zeros(
        x.shape[0],
        channels - x.shape[1],
        x.shape[2],
        x.shape[3],
        device=x.device,
        dtype=x.dtype,
    )
    return torch.cat([x, pad], dim=1)


class PairedImageDataset(Dataset):
    def __init__(
        self,
        cloudy_dir: str | Path,
        clear_dir: str | Path,
        image_size: int,
        max_images: int = 0,
    ) -> None:
        cloudy_dir = Path(cloudy_dir)
        clear_dir = Path(clear_dir)
        cloudy = [p for p in cloudy_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS]
        clear_by_stem = {p.stem: p for p in clear_dir.rglob("*") if p.suffix.lower() in IMAGE_EXTS}
        pairs = []
        for cpath in sorted(cloudy):
            gpath = clear_by_stem.get(cpath.stem)
            if gpath is not None:
                pairs.append((cpath, gpath))
        if not pairs:
            raise FileNotFoundError(
                f"No matched cloudy/clear pairs found.\ncloudy_dir={cloudy_dir}\nclear_dir={clear_dir}\n"
                "Pairs are matched by filename stem."
            )
        if max_images > 0:
            pairs = pairs[:max_images]
        self.pairs = pairs
        self.image_size = image_size

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        cloudy, clear = self.pairs[idx]
        return {
            "cond_image": image_to_tensor(cloudy, self.image_size),
            "label": image_to_tensor(clear, self.image_size),
            "cloudy_path": str(cloudy),
            "clear_path": str(clear),
        }


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int, alpha: int, up_init_scale: float = 1e-4) -> None:
        super().__init__()
        self.base = base
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.lora_down = nn.Linear(base.in_features, rank, bias=False)
        self.lora_up = nn.Linear(rank, base.out_features, bias=False)
        self.lora_down.to(device=base.weight.device, dtype=torch.float32)
        self.lora_up.to(device=base.weight.device, dtype=torch.float32)
        nn.init.kaiming_uniform_(self.lora_down.weight, a=math.sqrt(5))
        if up_init_scale > 0:
            nn.init.normal_(self.lora_up.weight, mean=0.0, std=up_init_scale)
        else:
            nn.init.zeros_(self.lora_up.weight)
        for param in self.base.parameters():
            param.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        base_out = self.base(x)
        lora_out = self.lora_up(self.lora_down(x.float())) * self.scale
        return base_out + lora_out.to(dtype=base_out.dtype)


def split_parent_name(root: nn.Module, dotted_name: str) -> tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    parent = root
    for part in parts[:-1]:
        parent = parent[int(part)] if part.isdigit() else getattr(parent, part)
    return parent, parts[-1]


def target_match(name: str, target: str) -> bool:
    leaf = name.split(".")[-1].lower()
    full = name.lower()
    if target == "q":
        return leaf in {"q", "to_q", "q_proj", "wq"}
    if target == "v":
        return leaf in {"v", "to_v", "v_proj", "wv"}
    if target == "qv":
        return leaf in {"q", "v", "to_q", "to_v", "q_proj", "v_proj", "wq", "wv", "qkv_proj"}
    if target == "qkv":
        return leaf in {
            "qkv",
            "qkv_proj",
            "to_qkv",
            "q",
            "k",
            "v",
            "to_q",
            "to_k",
            "to_v",
            "q_proj",
            "k_proj",
            "v_proj",
            "wq",
            "wk",
            "wv",
        }
    if target == "attention_linear":
        return any(key in full for key in ("attn", "attention", "self_attn")) and leaf not in {"norm", "dropout"}
    if target == "all_linear":
        return True
    raise SystemExit(f"Unknown target={target}")


def block_key(name: str, block_regex: str = "") -> str:
    if block_regex:
        m = re.search(block_regex, name)
        if m:
            return m.group(1) if m.groups() else m.group(0)

    parts = name.split(".")
    if "down_levels" in parts:
        idx = parts.index("down_levels")
        if idx + 2 < len(parts):
            return f"down_levels.{parts[idx + 1]}.{parts[idx + 2]}"
    if "up_levels" in parts:
        idx = parts.index("up_levels")
        if idx + 2 < len(parts):
            return f"up_levels.{parts[idx + 1]}.{parts[idx + 2]}"
    if "mid_level" in parts:
        idx = parts.index("mid_level")
        if idx + 1 < len(parts):
            return f"mid_level.{parts[idx + 1]}"

    for marker in ("levels", "layers", "stages", "blocks", "transformer_blocks"):
        if marker in parts:
            idx = parts.index(marker)
            if idx + 1 < len(parts):
                if marker == "blocks" and idx > 0 and parts[idx - 1].isdigit():
                    return ".".join(parts[max(idx - 2, 0) : idx + 2])
                return f"{marker}.{parts[idx + 1]}"

    nums = re.findall(r"\.(\d+)(?=\.)", "." + name + ".")
    if nums:
        return f"block.{nums[0]}"
    return ""


def natural_key(text: str) -> list:
    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", text)]


def iter_lora_modules(root: nn.Module) -> Iterable[tuple[str, LoRALinear]]:
    for name, module in root.named_modules():
        if isinstance(module, LoRALinear):
            yield name, module


def inject_lora(root: nn.Module, target: str, rank: int, alpha: int, block_regex: str, up_init_scale: float) -> list[str]:
    replacements = []
    for name, module in root.named_modules():
        if isinstance(module, nn.Linear) and block_key(name, block_regex) and target_match(name, target):
            replacements.append((name, module))
    if not replacements:
        raise SystemExit("No target Linear modules found. Run --inspect_only and consider --target attention_linear.")
    for name, module in replacements:
        parent, child_name = split_parent_name(root, name)
        setattr(parent, child_name, LoRALinear(module, rank=rank, alpha=alpha, up_init_scale=up_init_scale))
    return [name for name, _ in replacements]


def lora_grad_norm(module: LoRALinear) -> float:
    total = 0.0
    for param in (module.lora_down.weight, module.lora_up.weight):
        if param.grad is not None:
            total += float(param.grad.detach().float().pow(2).sum().cpu())
    return math.sqrt(total)


def lora_part_grad_norms(module: LoRALinear) -> tuple[float, float]:
    def norm(param: torch.Tensor) -> float:
        if param.grad is None:
            return 0.0
        return math.sqrt(float(param.grad.detach().float().pow(2).sum().cpu()))

    return norm(module.lora_down.weight), norm(module.lora_up.weight)


def is_profile_layer(name: str, module: nn.Module) -> bool:
    cls = module.__class__.__name__
    if cls not in {"NeighborhoodTransformerLayer", "GlobalTransformerLayer", "ShiftedWindowTransformerLayer"}:
        return False
    return any(marker in name for marker in ("down_levels", "up_levels", "mid_level"))


class ActivationGradRecorder:
    def __init__(self, model: nn.Module, block_regex: str = "") -> None:
        self.records: dict[str, dict] = {}
        self.saved_outputs: list[tuple[str, torch.Tensor]] = []
        self.handles = []
        for name, module in model.named_modules():
            if not is_profile_layer(name, module):
                continue
            bkey = block_key(name, block_regex) or name
            self.records.setdefault(
                bkey,
                {
                    "grad_norm": 0.0,
                    "activation_elements": 0,
                    "module_count": 0,
                    "requires_grad_forwards": 0,
                },
            )
            self.records[bkey]["module_count"] += 1
            self.handles.append(module.register_forward_hook(self._forward_hook(bkey)))

    def _forward_hook(self, bkey: str):
        def hook(module, inputs, output):
            tensor = output[0] if isinstance(output, (tuple, list)) else output
            if torch.is_tensor(tensor):
                self.records[bkey]["activation_elements"] += int(tensor.numel())
                if tensor.requires_grad:
                    tensor.retain_grad()
                    self.saved_outputs.append((bkey, tensor))
                    self.records[bkey]["requires_grad_forwards"] += 1

        return hook

    def collect_and_clear(self) -> None:
        for bkey, tensor in self.saved_outputs:
            if tensor.grad is None:
                continue
            self.records[bkey]["grad_norm"] += math.sqrt(float(tensor.grad.detach().float().pow(2).sum().cpu()))
        self.saved_outputs.clear()

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        self.saved_outputs.clear()


def load_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    try:
        from omegaconf import OmegaConf
    except ImportError as exc:
        raise SystemExit("Missing package: omegaconf\nInstall with: pip3 install omegaconf") from exc

    sys.path.insert(0, str(Path.cwd()))
    config = OmegaConf.load(args.config_path)
    if args.ckpt_path:
        config.model.params.ckpt_path = args.ckpt_path
    model = instantiate_from_config(config.model)
    if not hasattr(model, "global_step"):
        model.global_step = 0
    in_features = getattr(getattr(getattr(model.model.diffusion_model, "patch_in", None), "proj", None), "in_features", 0)
    if in_features and args.image_channels <= 0:
        args.image_channels = max(1, int(in_features) // 2)
        print(f"Auto-set --image_channels {args.image_channels} from patch_in in_features={in_features}")
    model.to(device)
    model.eval()
    return model


def inspect_model(model: nn.Module, args: argparse.Namespace) -> None:
    print("== Transformer/attention-like modules ==")
    for name, module in model.named_modules():
        cls = module.__class__.__name__
        if any(key in cls.lower() for key in ("transformer", "attention", "attn", "block", "layer")):
            print(name, cls)
    print("\n== Candidate Linear modules ==")
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            bkey = block_key(name, args.block_regex)
            mark = "*" if bkey and target_match(name, args.target) else " "
            print(f"{mark} {name} [{module.in_features}->{module.out_features}] block={bkey}")


def write_csv(path: str | Path, rows: list[dict]) -> None:
    path = Path(path)
    ensure_dir(path.parent)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def profile(args: argparse.Namespace) -> None:
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = load_model(args, device)

    if args.inspect_only:
        inspect_model(model, args)
        return

    for param in model.parameters():
        param.requires_grad_(False)
    injected = []
    recorder = None
    if args.importance_mode == "lora":
        injected = inject_lora(model, args.target, args.rank, args.alpha, args.block_regex, args.lora_up_init_scale)
    else:
        recorder = ActivationGradRecorder(model, args.block_regex)
        if not recorder.records:
            raise SystemExit("No EMRDM transformer layers found for activation-gradient profiling.")
    model.train()

    dataset = PairedImageDataset(args.cloudy_dir, args.clear_dir, args.image_size, args.max_images)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_iter = iter(loader)

    total_loss = 0.0
    valid = 0
    model.zero_grad(set_to_none=True)
    for idx in range(1, args.probe_batches + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        batch_t = {
            "label": pad_or_trim_channels(batch["label"].to(device), args.image_channels),
            "cond_image": pad_or_trim_channels(batch["cond_image"].to(device), args.image_channels),
            "global_step": 0,
        }
        if args.importance_mode == "activation":
            batch_t["label"].requires_grad_(True)
            batch_t["cond_image"].requires_grad_(True)
        if args.input_noise_std > 0:
            batch_t["cond_image"] = batch_t["cond_image"] + torch.randn_like(batch_t["cond_image"]) * args.input_noise_std
            if args.importance_mode == "activation":
                batch_t["cond_image"].requires_grad_(True)

        loss, _ = model.shared_step(batch_t)
        if not torch.isfinite(loss):
            print(f"probe batch {idx}/{args.probe_batches} skipped: non-finite loss")
            model.zero_grad(set_to_none=True)
            continue
        loss.backward()
        if args.importance_mode == "activation":
            assert recorder is not None
            recorder.collect_and_clear()
        valid += 1
        total_loss += float(loss.detach().cpu())
        print(f"probe batch {idx:03d}/{args.probe_batches} loss={float(loss.detach().cpu()):.6f}")

    if valid == 0:
        raise SystemExit("No valid probe batches.")

    by_block: dict[str, dict] = {}
    if args.importance_mode == "lora":
        for name, module in iter_lora_modules(model):
            bkey = block_key(name, args.block_regex)
            if not bkey:
                continue
            row = by_block.setdefault(bkey, {"grad_norm": 0.0, "lora_param_count": 0, "module_count": 0})
            down_grad, up_grad = lora_part_grad_norms(module)
            row["grad_norm"] += lora_grad_norm(module)
            row["down_grad_norm"] = row.get("down_grad_norm", 0.0) + down_grad
            row["up_grad_norm"] = row.get("up_grad_norm", 0.0) + up_grad
            row["lora_param_count"] += module.lora_down.weight.numel() + module.lora_up.weight.numel()
            row["module_count"] += 1
    else:
        assert recorder is not None
        by_block = recorder.records
        recorder.close()

    blocks = sorted(by_block, key=natural_key)
    total_blocks = max(len(blocks), 1)
    rows = []
    for block_index, block in enumerate(blocks):
        row = by_block[block]
        p_count = int(row.get("lora_param_count") or row.get("activation_elements") or 1)
        norm_score = row["grad_norm"] / math.sqrt(max(p_count, 1))
        bp_cost = total_blocks - block_index
        selection = norm_score / (p_count + args.compute_lambda * bp_cost) if args.compute_lambda > 0 else norm_score
        rows.append(
            {
                "block": block,
                "block_index": block_index,
                "grad_norm": row["grad_norm"],
                "down_grad_norm": row.get("down_grad_norm", 0.0),
                "up_grad_norm": row.get("up_grad_norm", 0.0),
                "lora_param_count": p_count,
                "activation_elements": int(row.get("activation_elements", 0)),
                "module_count": int(row["module_count"]),
                "requires_grad_forwards": int(row.get("requires_grad_forwards", 0)),
                "normalized_grad_score": norm_score,
                "bp_cost": bp_cost,
                "compute_lambda": args.compute_lambda,
                "selection_score": selection,
                "probe_batches": valid,
                "mean_probe_loss": total_loss / valid,
                "selected": False,
            }
        )

    selected = set(r["block"] for r in sorted(rows, key=lambda r: r["selection_score"], reverse=True)[: args.topk_blocks])
    for row in rows:
        row["selected"] = row["block"] in selected

    out_dir = ensure_dir(args.output_dir)
    write_csv(out_dir / "emrdm_grad_scores.csv", rows)
    metadata = {
        "config_path": args.config_path,
        "ckpt_path": args.ckpt_path,
        "cloudy_dir": args.cloudy_dir,
        "clear_dir": args.clear_dir,
        "target": args.target,
        "importance_mode": args.importance_mode,
        "rank": args.rank,
        "alpha": args.alpha,
        "lora_up_init_scale": args.lora_up_init_scale,
        "topk_blocks": args.topk_blocks,
        "probe_batches": args.probe_batches,
        "seed": args.seed,
        "selected_blocks": sorted(selected, key=natural_key),
        "injected_module_count": len(injected),
    }
    (out_dir / "emrdm_grad_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"Wrote {out_dir / 'emrdm_grad_scores.csv'}")
    print(f"Wrote {out_dir / 'emrdm_grad_metadata.json'}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config_path", default="configs/example_training/cuhk.yaml")
    parser.add_argument("--ckpt_path", default="", help="Optional .ckpt checkpoint path")
    parser.add_argument("--cloudy_dir", required=True)
    parser.add_argument("--clear_dir", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--image_size", type=int, default=256)
    parser.add_argument("--image_channels", type=int, default=0, help="0 auto-infers half of patch_in input channels")
    parser.add_argument("--max_images", type=int, default=0)
    parser.add_argument("--target", default="qkv", choices=["q", "v", "qv", "qkv", "attention_linear", "all_linear"])
    parser.add_argument(
        "--importance_mode",
        default="activation",
        choices=["activation", "lora"],
        help="activation records block output-gradient sensitivity; lora records probe LoRA parameter gradients.",
    )
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--lora_up_init_scale", type=float, default=1e-4)
    parser.add_argument("--topk_blocks", type=int, default=8)
    parser.add_argument("--probe_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input_noise_std", type=float, default=0.0)
    parser.add_argument("--compute_lambda", type=float, default=0.0)
    parser.add_argument("--block_regex", default="")
    parser.add_argument("--neighborhood_backend", default="auto", choices=["auto", "native", "torch"])
    parser.add_argument("--inspect_only", action="store_true")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    global NATTEN_BACKEND
    args = build_parser().parse_args()
    NATTEN_BACKEND = args.neighborhood_backend
    profile(args)


if __name__ == "__main__":
    main()
