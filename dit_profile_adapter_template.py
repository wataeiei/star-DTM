"""Adapter template for profile_dit_grad_topk.py.

Copy this file into the DiT-SR or DiT4SR repository root, rename it, and fill in
the three required functions:

  build_model(args)
  build_dataloader(args)
  compute_loss(model, batch, args, device)

The profiling script imports this module with:

  --adapter_module dit_profile_adapter_ditsr
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}


class PairedOrSyntheticSRDataset(Dataset):
    """Simple SR dataset.

    If your dataset has HR images only, this creates LR by bicubic downsampling.
    If your DiT-SR/DiT4SR repo already has its own dataset class, replace this
    with the official dataset builder.
    """

    def __init__(self, root: str | Path, hr_size: int, lr_size: int) -> None:
        self.root = Path(root)
        self.paths = sorted([p for p in self.root.rglob("*") if p.suffix.lower() in IMAGE_EXTS])
        if not self.paths:
            raise SystemExit(f"No images found in {self.root}")
        self.hr_size = hr_size
        self.lr_size = lr_size

    def __len__(self) -> int:
        return len(self.paths)

    def _to_tensor(self, image: Image.Image, size: int) -> torch.Tensor:
        image = image.convert("RGB").resize((size, size), Image.BICUBIC)
        x = torch.frombuffer(bytearray(image.tobytes()), dtype=torch.uint8)
        x = x.view(size, size, 3).permute(2, 0, 1).float() / 255.0
        return x

    def __getitem__(self, idx: int) -> dict:
        image = Image.open(self.paths[idx]).convert("RGB")
        hr = self._to_tensor(image, self.hr_size)
        lr_img = image.resize((self.lr_size, self.lr_size), Image.BICUBIC)
        lr = self._to_tensor(lr_img, self.lr_size)
        lr_up = F.interpolate(lr.unsqueeze(0), size=(self.hr_size, self.hr_size), mode="bicubic", align_corners=False)[0]
        return {"lr": lr, "lr_up": lr_up, "hr": hr, "path": str(self.paths[idx])}


def build_model(args):
    """Build and load the DiT-SR / DiT4SR model.

    Replace this function with the official model construction code.
    Examples of what usually belongs here:

      from basicsr.utils.options import parse_options
      from basicsr.models import build_model

    or:

      from models.dit_sr import DiTSR
      model = DiTSR(...)
      state = torch.load(args.checkpoint, map_location="cpu")
      model.load_state_dict(state["params"], strict=False)
      return model
    """

    raise NotImplementedError(
        "Fill build_model(args) for the target DiT-SR/DiT4SR repository."
    )


def build_dataloader(args):
    dataset = PairedOrSyntheticSRDataset(args.data_dir, args.hr_size, args.lr_size)
    return DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)


def compute_loss(model, batch: dict, args, device: torch.device):
    """Compute calibration loss for one batch.

    Replace the forward call below according to the target repo.

    Common possibilities:

      pred = model(batch["lr"].to(device))
      loss = F.l1_loss(pred, batch["hr"].to(device))

    or diffusion-style:

      loss = model.training_losses(...)

    The only requirement is returning a scalar torch.Tensor.
    """

    lr = batch["lr"].to(device)
    lr_up = batch["lr_up"].to(device)
    hr = batch["hr"].to(device)

    # Placeholder. Many SR models take LR and output SR directly.
    # Change this to the official DiT-SR/DiT4SR forward.
    try:
        pred = model(lr)
    except Exception:
        pred = model(lr_up)

    if isinstance(pred, dict):
        pred = pred.get("sr") or pred.get("output") or pred.get("pred")
    if isinstance(pred, (tuple, list)):
        pred = pred[0]
    if pred.shape[-2:] != hr.shape[-2:]:
        pred = F.interpolate(pred, size=hr.shape[-2:], mode="bicubic", align_corners=False)
    return F.l1_loss(pred.float(), hr.float())
