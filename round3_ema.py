# Round 3 runner. Augmentations below this marker are pasted from the notebook cell
# so config F matches the ResNet50 reference trajectory. EMA updates shadow weights only.

import argparse
import copy
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import accuracy_score
from torch.utils.data import DataLoader, TensorDataset
from torchvision import models
from torchvision.transforms import _functional_tensor as _ft

DATA_DIR = Path("./data")
SEED = 42
DEVICE = "cuda"
NUM_CLASSES = 20
RN_IMG = 224
RN_CACHE = 256
RN_EPOCHS = 15
RN_MEAN = (0.485, 0.456, 0.406)
RN_STD = (0.229, 0.224, 0.225)
EMA_DECAY = 0.999
IMG_SIZE = 256
BATCH = 64
CORRUPT_SEED = 1234567
CORRUPT_PATH = Path("outputs") / "val_corrupted.pt"
EXP_PATH = Path("outputs") / "experiments.csv"
CKPT_DIR = Path("outputs") / "checkpoints"
LOG_PATH = Path("outputs") / "logs" / "round3.log"
STATUS_PATH = Path("outputs") / "logs" / "round3_status.json"
N_TEST = 8299
REF_CLEAN_MAX = {
    42: {"epoch": 11, "clean": 0.949928, "corrupt": 0.944206},
    7: {"epoch": 13, "clean": 0.959943, "corrupt": 0.947067},
}
TTA_REF = (0.964235, 0.952790)
F_SCORE_REF = (0.959943 + 0.947067) / 2.0


def seed_everything(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def say(msg):
    print(msg, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def image_file(folder, image_id):
    name = image_id if str(image_id).endswith(".jpg") else f"{image_id}.jpg"
    return DATA_DIR / folder / name


# --- pasted notebook helpers (config F) ---
def load_uint8(table, folder, size=RN_CACHE):
    """One pass: RGB uint8 tensors (N, 3, 256, 256) kept in RAM."""
    ids = table.id.tolist()
    images = torch.empty((len(ids), 3, size, size), dtype=torch.uint8)
    for i, image_id in enumerate(ids):
        with Image.open(image_file(folder, image_id)) as img:
            im = img.convert("RGB")
            if im.size != (size, size):
                im = im.resize((size, size), Image.BILINEAR)
            hwc = np.array(im, dtype=np.uint8, copy=True)
            images[i].copy_(torch.from_numpy(hwc).permute(2, 0, 1))
    return images


def _labels_of(table):
    if "label" in table.columns:
        return torch.tensor(table.label.to_numpy(), dtype=torch.long)
    return torch.full((len(table),), -1, dtype=torch.long)


def _blend(img, other, ratio):
    return (ratio * img + (1.0 - ratio) * other).clamp_(0.0, 1.0)


def _brightness(img, factor):
    return (img * factor).clamp_(0.0, 1.0)


def _contrast(img, factor):
    r, g, b = img.unbind(dim=1)
    gray = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(1)
    return _blend(img, gray.mean(dim=(-2, -1), keepdim=True), factor)


def _saturation(img, factor):
    r, g, b = img.unbind(dim=1)
    gray = (0.2989 * r + 0.587 * g + 0.114 * b).unsqueeze(1)
    return _blend(img, gray, factor)


def _hue(img, factor):
    hsv = _ft._rgb2hsv(img)
    h, s, v = hsv.unbind(dim=-3)
    h = torch.remainder(h + factor.reshape(-1, 1, 1), 1.0)
    return _ft._hsv2rgb(torch.stack((h, s, v), dim=-3)).clamp_(0.0, 1.0)


def _color_jitter(x):
    n = x.shape[0]
    b = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
    c = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
    s = torch.empty(n, 1, 1, 1, device=x.device).uniform_(0.8, 1.2)
    h = torch.empty(n, 1, 1, 1, device=x.device).uniform_(-0.02, 0.02)
    order = torch.argsort(torch.rand(n, 4, device=x.device), dim=1)
    ops = (_brightness, _contrast, _saturation, _hue)
    factors = (b, c, s, h)
    for step in range(4):
        for op_i, (fn, fac) in enumerate(zip(ops, factors)):
            mask = order[:, step] == op_i
            if mask.any():
                x[mask] = fn(x[mask], fac[mask])
    return x


def _gaussian_blur(x, p=0.5):
    n, channels, height, width = x.shape
    mask = torch.rand(n, device=x.device) < p
    if not mask.any():
        return x
    idx = mask.nonzero(as_tuple=False).squeeze(1)
    sigma = torch.empty(idx.numel(), device=x.device).uniform_(0.1, 2.0)
    coords = torch.linspace(-2.0, 2.0, 5, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-0.5 * (coords / sigma[:, None]).pow(2))
    kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)
    kernel = (kernel_1d[:, :, None] * kernel_1d[:, None, :]).repeat_interleave(channels, dim=0)
    kernel = kernel[:, None, :, :].contiguous()
    y = torch.nn.functional.pad(x[idx], (2, 2, 2, 2), mode="reflect")
    y = y.reshape(1, idx.numel() * channels, height + 4, width + 4)
    y = torch.nn.functional.conv2d(y, kernel, groups=idx.numel() * channels)
    x[idx] = y.reshape(idx.numel(), channels, height, width).clamp_(0.0, 1.0)
    return x


def _downscale(x, p=0.5, factors=(2, 4)):
    n, _, height, width = x.shape
    apply = torch.rand(n, device=x.device) < p
    if not apply.any():
        return x
    choice = torch.randint(0, len(factors), (n,), device=x.device)
    for factor_i, factor in enumerate(factors):
        mask = apply & (choice == factor_i)
        if not mask.any():
            continue
        small = (max(8, height // factor), max(8, width // factor))
        y = torch.nn.functional.interpolate(
            x[mask], size=small, mode="bilinear", align_corners=False, antialias=True
        )
        x[mask] = torch.nn.functional.interpolate(y, size=(height, width), mode="bilinear", align_corners=False)
    return x


def _gaussian_blur_wide(x, p, sigma_lo, sigma_hi):
    """Per-image Gaussian blur. Kernel 15 covers sigma up to 2.5."""
    n, channels, height, width = x.shape
    mask = torch.rand(n, device=x.device) < p
    if not mask.any():
        return x
    idx = mask.nonzero(as_tuple=False).flatten()
    sigma = torch.empty(idx.numel(), device=x.device).uniform_(sigma_lo, sigma_hi)
    radius = 7
    coords = torch.arange(-radius, radius + 1, device=x.device, dtype=x.dtype)
    kernel_1d = torch.exp(-0.5 * (coords / sigma[:, None]).pow(2))
    kernel_1d = kernel_1d / kernel_1d.sum(dim=1, keepdim=True)
    y = torch.nn.functional.pad(x[idx], (radius, radius, radius, radius), mode="reflect")
    y = y.reshape(1, idx.numel() * channels, height + 2 * radius, width + 2 * radius)
    kh = kernel_1d[:, None, None, :].repeat_interleave(channels, dim=0).contiguous()
    y = torch.nn.functional.conv2d(y, kh, groups=idx.numel() * channels)
    kv = kernel_1d[:, None, :, None].repeat_interleave(channels, dim=0).contiguous()
    y = torch.nn.functional.conv2d(y, kv, groups=idx.numel() * channels)
    x[idx] = y.reshape(idx.numel(), channels, height, width).clamp_(0.0, 1.0)
    return x


def _downscale_uniform(x, p, lo=2.0, hi=4.0):
    """Per-image downscale-upscale. Factor is uniform on {2, 2.5, 3, 3.5, 4}; antialias on the way down."""
    n, _, height, width = x.shape
    apply = torch.rand(n, device=x.device) < p
    if not apply.any():
        return x
    bucket = torch.randint(0, 5, (n,), device=x.device)
    for bid in range(5):
        mask = apply & (bucket == bid)
        if not mask.any():
            continue
        factor = lo + bid * (hi - lo) / 4.0
        small = (max(8, int(round(height / factor))), max(8, int(round(width / factor))))
        y = torch.nn.functional.interpolate(
            x[mask], size=small, mode="bilinear", align_corners=False, antialias=True
        )
        x[mask] = torch.nn.functional.interpolate(y, size=(height, width), mode="bilinear", align_corners=False)
    return x


def _random_resized_crop(x, scale=(0.7, 1.0), ratio=(0.75, 4.0 / 3.0)):
    """Per-image RandomResizedCrop back to the current spatial size."""
    n, _, height, width = x.shape
    area = float(height * width)
    target = torch.empty(n, device=x.device).uniform_(scale[0], scale[1]) * area
    log_ratio = torch.log(torch.tensor(ratio, device=x.device, dtype=x.dtype))
    aspect = torch.empty(n, device=x.device).uniform_(float(log_ratio[0]), float(log_ratio[1])).exp()
    crop_w = torch.sqrt(target * aspect).clamp(1.0, width)
    crop_h = torch.sqrt(target / aspect).clamp(1.0, height)
    top = torch.rand(n, device=x.device) * (height - crop_h)
    left = torch.rand(n, device=x.device) * (width - crop_w)
    ys = torch.linspace(0.0, 1.0, height, device=x.device, dtype=x.dtype)
    xs = torch.linspace(0.0, 1.0, width, device=x.device, dtype=x.dtype)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    src_y = top[:, None, None] + grid_y[None] * (crop_h[:, None, None] - 1.0)
    src_x = left[:, None, None] + grid_x[None] * (crop_w[:, None, None] - 1.0)
    grid = torch.stack((src_x / (width - 1) * 2.0 - 1.0, src_y / (height - 1) * 2.0 - 1.0), dim=-1)
    return torch.nn.functional.grid_sample(x, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _jpeg_or_noise(x, p=0.2):
    """Per-image JPEG-like quantisation/blocking, or Gaussian noise."""
    n, _, height, width = x.shape
    apply = torch.rand(n, device=x.device) < p
    if not apply.any():
        return x
    use_noise = torch.rand(n, device=x.device) < 0.5
    noise_m = apply & use_noise
    jpeg_m = apply & ~use_noise
    if noise_m.any():
        std = torch.empty(int(noise_m.sum()), 1, 1, 1, device=x.device).uniform_(0.02, 0.08)
        x[noise_m] = (x[noise_m] + torch.randn_like(x[noise_m]) * std).clamp_(0.0, 1.0)
    if jpeg_m.any():
        y = x[jpeg_m].contiguous()
        levels = torch.empty(y.shape[0], 1, 1, 1, device=x.device).uniform_(8.0, 28.0)
        y = torch.round(y * levels).div(levels)
        pooled = torch.nn.functional.avg_pool2d(y, kernel_size=8, stride=8)
        up = torch.nn.functional.interpolate(pooled, size=(height, width), mode="nearest")
        strength = torch.empty(y.shape[0], 1, 1, 1, device=x.device).uniform_(0.15, 0.45)
        x[jpeg_m] = (y * (1.0 - strength) + up * strength).clamp_(0.0, 1.0)
    return x


def gpu_augment(images, cfg="A", img_size=None):
    """Per-image GPU augmentations, then ImageNet norm.

    A is the current pipeline. B adds blur, downscale and invert. C adds JPEG/noise and RandomResizedCrop.
    E and F keep A, then a lighter pass: blur p=0.2 (sigma 0.5-2.5), downscale-upscale p=0.2
    with antialias on the way down, JPEG/noise p=0.1, invert p=0.03. F stays at 256 (no resize to 224).
    """
    if cfg not in ("A", "B", "C", "E", "F"):
        raise ValueError(cfg)
    if img_size is None:
        img_size = RN_CACHE if cfg == "F" else RN_IMG
    x = images.float().mul_(1.0 / 255.0)
    if cfg == "C":
        x = _random_resized_crop(x, scale=(0.7, 1.0))
    n = x.shape[0]
    flip_h = torch.rand(n, device=x.device) < 0.5
    flip_v = torch.rand(n, device=x.device) < 0.5
    if flip_h.any():
        x[flip_h] = x[flip_h].flip(-1)
    if flip_v.any():
        x[flip_v] = x[flip_v].flip(-2)
    turns = torch.randint(0, 4, (n,), device=x.device)
    for k in (1, 2, 3):
        mask = turns == k
        if mask.any():
            x[mask] = torch.rot90(x[mask], k, dims=(-2, -1))
    x = _color_jitter(x)
    x = _gaussian_blur(x, p=0.5)
    invert = torch.rand(n, device=x.device) < 0.1
    if invert.any():
        x[invert] = 1.0 - x[invert]
    x = _downscale(x, p=0.5, factors=(2, 4))
    if cfg in ("B", "C"):
        x = _gaussian_blur_wide(x, p=0.4, sigma_lo=0.5, sigma_hi=2.5)
        x = _downscale_uniform(x, p=0.4, lo=2.0, hi=4.0)
        invert_b = torch.rand(n, device=x.device) < 0.05
        if invert_b.any():
            x[invert_b] = 1.0 - x[invert_b]
    if cfg == "C":
        x = _jpeg_or_noise(x, p=0.2)
    if cfg in ("E", "F"):
        x = _gaussian_blur_wide(x, p=0.2, sigma_lo=0.5, sigma_hi=2.5)
        x = _downscale_uniform(x, p=0.2, lo=2.0, hi=4.0)
        x = _jpeg_or_noise(x, p=0.1)
        invert_e = torch.rand(n, device=x.device) < 0.03
        if invert_e.any():
            x[invert_e] = 1.0 - x[invert_e]
    if x.shape[-2] != img_size or x.shape[-1] != img_size:
        x = torch.nn.functional.interpolate(
            x, size=(img_size, img_size), mode="bilinear", align_corners=False, antialias=True
        )
    x = (x - _RN_MEAN) / _RN_STD
    return x.contiguous(memory_format=torch.channels_last)

def gpu_eval(images, img_size=None):
    spatial = RN_IMG if img_size is None else img_size
    x = images.float().mul_(1.0 / 255.0)
    if x.shape[-2] != spatial or x.shape[-1] != spatial:
        x = torch.nn.functional.interpolate(
            x, size=(spatial, spatial), mode="bilinear", align_corners=False, antialias=True
        )
    x = (x - _RN_MEAN) / _RN_STD
    return x.contiguous(memory_format=torch.channels_last)

def freeze_running_bn(model):
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and not any(p.requires_grad for p in module.parameters()):
            module.eval()


def build_backbone(arch):
    if arch == "resnet50":
        backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    elif arch == "resnet101":
        backbone = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    else:
        raise ValueError(arch)
    for param in backbone.parameters():
        param.requires_grad = False
    for block in (backbone.layer3, backbone.layer4):
        for param in block.parameters():
            param.requires_grad = True
    num_features = backbone.fc.in_features
    assert num_features == 2048
    backbone.fc = nn.Identity()
    classifier = nn.Sequential(
        nn.Linear(num_features, 512),
        nn.LeakyReLU(),
        nn.Dropout(0.3),
        nn.Linear(512, NUM_CLASSES),
    )
    model = nn.Sequential(backbone, classifier).to(DEVICE, memory_format=torch.channels_last)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    assert trainable
    assert all(
        name.startswith("0.layer3") or name.startswith("0.layer4") or name.startswith("1.")
        for name in trainable
    )
    assert any(name.startswith("0.layer3") for name in trainable)
    assert any(name.startswith("0.layer4") for name in trainable)
    return model


def make_loader(dataset, shuffle, seed, batch):
    kwargs = dict(batch_size=batch, shuffle=shuffle, num_workers=0, pin_memory=True)
    if shuffle:
        kwargs["generator"] = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, **kwargs)


def clone_shadow(model):
    cpu_rng = torch.get_rng_state()
    cuda_rng = torch.cuda.get_rng_state_all()
    shadow = copy.deepcopy(model)
    if not torch.equal(cpu_rng, torch.get_rng_state()):
        raise RuntimeError("deepcopy changed CPU RNG")
    if any(not torch.equal(a, b) for a, b in zip(cuda_rng, torch.cuda.get_rng_state_all())):
        raise RuntimeError("deepcopy changed CUDA RNG")
    shadow.eval()
    for param in shadow.parameters():
        param.requires_grad_(False)
    live_names = [name for name, _ in model.named_modules()]
    shadow_names = [name for name, _ in shadow.named_modules()]
    assert live_names == shadow_names
    return shadow


@torch.no_grad()
def ema_update(shadow, model, decay=EMA_DECAY):
    for src, dst in zip(model.parameters(), shadow.parameters()):
        dst.mul_(decay).add_(src.detach(), alpha=1.0 - decay)
    for module, shadow_module in zip(model.modules(), shadow.modules()):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            shadow_module.running_mean.copy_(module.running_mean)
            shadow_module.running_var.copy_(module.running_var)
            shadow_module.num_batches_tracked.copy_(module.num_batches_tracked)


@torch.no_grad()
def accuracy_of(model, loader):
    model.eval()
    correct = 0
    total = 0
    pred_chunks = []
    true_chunks = []
    for images, labels in loader:
        images = gpu_eval(images.to(DEVICE, non_blocking=True), img_size=IMG_SIZE)
        labels = labels.to(DEVICE, non_blocking=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(images)
        pred = logits.argmax(dim=1)
        correct += (pred == labels).sum().item()
        total += labels.shape[0]
        pred_chunks.append(pred.cpu())
        true_chunks.append(labels.cpu())
    y_true = torch.cat(true_chunks).numpy()
    y_pred = torch.cat(pred_chunks).numpy()
    counted = correct / total
    sklearn_value = float(accuracy_score(y_true, y_pred))
    assert abs(counted - sklearn_value) < 1e-12
    return counted, sklearn_value


@torch.no_grad()
def predict_tta(model, loader):
    model.eval()
    pred, true = [], []
    for images, labels in loader:
        base = gpu_eval(images.to(DEVICE, non_blocking=True), img_size=IMG_SIZE)
        probs = None
        for turns in range(4):
            view = base if turns == 0 else torch.rot90(base, turns, dims=(-2, -1))
            for flipped in (view, view.flip(-1)):
                batch = flipped.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch)
                piece = torch.softmax(logits.float(), dim=1)
                probs = piece if probs is None else probs + piece
        pred.append(probs.argmax(dim=1).cpu())
        true.append(labels)
    y_true = torch.cat(true).numpy()
    y_pred = torch.cat(pred).numpy()
    return float(accuracy_score(y_true, y_pred))


def optimizer_of(model):
    optimizer = torch.optim.AdamW(
        [
            {"params": model[0].layer3.parameters(), "lr": 5e-5},
            {"params": model[0].layer4.parameters(), "lr": 1e-4},
            {"params": model[1].parameters(), "lr": 1e-3},
        ],
        weight_decay=1e-4,
    )
    assert [group["lr"] for group in optimizer.param_groups] == [5e-5, 1e-4, 1e-3]
    return optimizer


def pack_row(epoch, clean, corrupt):
    return {
        "epoch": int(epoch),
        "clean": float(clean),
        "corrupt": float(corrupt),
        "score": (float(clean) + float(corrupt)) / 2.0,
    }


def consider(best, row):
    if best is None or (row["score"], row["corrupt"], row["clean"]) > (
        best["score"],
        best["corrupt"],
        best["clean"],
    ):
        return row
    return best


def train_one(model, shadow, train_loader, val_loader, corrupt_loader, arch, seed, accum_steps, ref_epoch5):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optimizer_of(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=RN_EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best_plain = None
    best_ema = None
    best_plain_state = None
    best_ema_state = None
    clean_max = None
    clean_max_state = None
    stopped = False
    stop_reason = ""
    for epoch in range(1, RN_EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        freeze_running_bn(model)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        n_seen = 0
        total_batches = len(train_loader)
        for images, labels in train_loader:
            images = gpu_augment(images.to(DEVICE, non_blocking=True), cfg="F", img_size=IMG_SIZE)
            labels = labels.to(DEVICE, non_blocking=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                loss = criterion(logits, labels)
                if accum_steps != 1:
                    loss = loss / accum_steps
            scaler.scale(loss).backward()
            pending += 1
            n_seen += 1
            if pending == accum_steps or n_seen == total_batches:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                pending = 0
                ema_update(shadow, model)
        clean, _ = accuracy_of(model, val_loader)
        corrupt, _ = accuracy_of(model, corrupt_loader)
        ema_clean, _ = accuracy_of(shadow, val_loader)
        ema_corrupt, _ = accuracy_of(shadow, corrupt_loader)
        scheduler.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        plain_row = pack_row(epoch, clean, corrupt)
        ema_row = pack_row(epoch, ema_clean, ema_corrupt)
        history.append({"plain": plain_row, "ema": ema_row, "sec": elapsed, "peak_mb": peak_mb})
        if best_plain is None or (plain_row["score"], plain_row["corrupt"], plain_row["clean"]) > (
            best_plain["score"],
            best_plain["corrupt"],
            best_plain["clean"],
        ):
            best_plain = plain_row
            best_plain_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if best_ema is None or (ema_row["score"], ema_row["corrupt"], ema_row["clean"]) > (
            best_ema["score"],
            best_ema["corrupt"],
            best_ema["clean"],
        ):
            best_ema = ema_row
            best_ema_state = {k: v.detach().cpu().clone() for k, v in shadow.state_dict().items()}
        if clean_max is None or clean > clean_max["clean"]:
            clean_max = plain_row
            clean_max_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        say(
            f"{arch} seed {seed} | epoch {epoch:2d} | plain {clean:.4f}/{corrupt:.4f} | "
            f"ema {ema_clean:.4f}/{ema_corrupt:.4f} | {elapsed:.1f}s | vram {peak_mb:.0f}MB"
        )
        if epoch == 5 and ref_epoch5 is not None:
            gap_clean = clean - ref_epoch5["clean"]
            gap_corrupt = corrupt - ref_epoch5["corrupt"]
            say(
                f"gate epoch 5 | resnet101 plain {clean:.4f}/{corrupt:.4f} | "
                f"resnet50 seed 7 plain {ref_epoch5['clean']:.4f}/{ref_epoch5['corrupt']:.4f} | "
                f"gap {gap_clean:+.4f}/{gap_corrupt:+.4f}"
            )
            if gap_clean < -0.02 or gap_corrupt < -0.02:
                stopped = True
                stop_reason = (
                    f"epoch 5 plain {clean:.4f}/{corrupt:.4f} vs "
                    f"F resnet50 seed 7 {ref_epoch5['clean']:.4f}/{ref_epoch5['corrupt']:.4f}"
                )
                say(f"STOP | {stop_reason}")
                break
    return {
        "history": history,
        "best_plain": best_plain,
        "best_ema": best_ema,
        "best_plain_state": best_plain_state,
        "best_ema_state": best_ema_state,
        "clean_max": clean_max,
        "clean_max_state": clean_max_state,
        "final_plain": history[-1]["plain"],
        "final_ema": history[-1]["ema"],
        "mean_sec": float(np.mean([row["sec"] for row in history])),
        "peak_mb": float(max(row["peak_mb"] for row in history)),
        "epoch1_sec": history[0]["sec"],
        "epoch1_peak": history[0]["peak_mb"],
        "stopped": stopped,
        "stop_reason": stop_reason,
        "epochs_ran": history[-1]["plain"]["epoch"],
    }


def probe_batch(arch, batch):
    model = None
    shadow = None
    try:
        model = build_backbone(arch)
        shadow = clone_shadow(model)
        model.train()
        freeze_running_bn(model)
        optimizer = optimizer_of(model)
        scaler = torch.amp.GradScaler("cuda")
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        images = torch.randint(0, 256, (batch, 3, RN_CACHE, RN_CACHE), dtype=torch.uint8)
        labels = torch.randint(0, NUM_CLASSES, (batch,), dtype=torch.long)
        for _ in range(2):
            batch_x = gpu_augment(images.to(DEVICE), cfg="F", img_size=IMG_SIZE)
            batch_y = labels.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(batch_x), batch_y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ema_update(shadow, model)
        torch.cuda.synchronize()
    finally:
        del model, shadow
        torch.cuda.empty_cache()


def choose_batch(arch):
    if arch == "resnet50":
        return 64, 1
    for batch in (64, 32, 16):
        try:
            probe_batch(arch, batch)
        except torch.cuda.OutOfMemoryError:
            say(f"probe oom | {arch} | batch {batch}")
            torch.cuda.empty_cache()
            continue
        except RuntimeError as exc:
            if "out of memory" not in str(exc).lower():
                raise
            say(f"probe oom | {arch} | batch {batch}")
            torch.cuda.empty_cache()
            continue
        accum = 64 // batch
        say(f"probe ok | {arch} | batch {batch} | accum {accum}")
        return batch, accum
    raise RuntimeError(f"OOM for {arch}")


def append_exp(row):
    with EXP_PATH.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fields = list(reader.fieldnames)
        rows = list(reader)
    for key in row:
        if key not in fields:
            fields.append(key)
    rows = [
        item
        for item in rows
        if not (
            item.get("pipeline") == str(row["pipeline"])
            and item.get("model") == str(row["model"])
            and item.get("aug") == str(row["aug"])
            and str(item.get("seed")) == str(row["seed"])
            and item.get("variant", "") == str(row.get("variant", ""))
        )
    ]
    written = [{key: item.get(key, "") for key in fields} for item in rows]
    written.append({key: row.get(key, "") for key in fields})
    tmp = EXP_PATH.with_suffix(".csv.tmp")
    with tmp.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(written)
    tmp.replace(EXP_PATH)


def exp_row(arch, seed, variant, best, final, sec, peak_mb, batch, accum, epochs_ran, sklearn_clean, sklearn_corrupt):
    return {
        "model": arch,
        "weights": "IMAGENET1K_V1",
        "unfrozen": "layer3+layer4+classifier",
        "img_size": IMG_SIZE,
        "epochs": epochs_ran,
        "batch_size": batch,
        "optimizer": "AdamW",
        "lr_head": 1e-3,
        "lr_layer4": 1e-4,
        "label_smoothing": 0.1,
        "best_val_acc": round(best["clean"], 6),
        "best_epoch": best["epoch"],
        "sklearn_val_acc": round(sklearn_clean, 6),
        "mean_epoch_sec": round(sec, 2),
        "seed": seed,
        "pipeline": "round3",
        "aug": "F",
        "val_corrupt_acc": round(best["corrupt"], 6),
        "sklearn_val_corrupt": round(sklearn_corrupt, 6),
        "lr_layer3": 5e-5,
        "accum_steps": accum,
        "peak_vram_mb": round(peak_mb, 1),
        "base_aug": "F",
        "variant": variant,
        "ema_decay": EMA_DECAY if variant == "ema" else "",
        "final_epoch": final["epoch"],
        "final_val_acc": round(final["clean"], 6),
        "final_corrupt": round(final["corrupt"], 6),
        "best_score": round(best["score"], 6),
    }


def write_status(payload):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def fmt_pair(row):
    return f"e{row['epoch']} {row['clean']:.4f}/{row['corrupt']:.4f}"


def upsert_state(bullets):
    path = Path("docs/STATE.md")
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("- Раунд 3") and not line.startswith("- Следующий шаг:")
    ]
    lines.extend(bullets)
    lines.append("- Следующий шаг: сравнение с MLP/CNN (таблица + графики) и полная confusion matrix на val")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tta_pair(model, val_loader, corrupt_loader):
    t0 = time.perf_counter()
    clean = predict_tta(model, val_loader)
    corrupt = predict_tta(model, corrupt_loader)
    torch.cuda.synchronize()
    return clean, corrupt, time.perf_counter() - t0


def load_arch(arch, state):
    model = build_backbone(arch)
    model.load_state_dict(state)
    model.eval()
    return model


def sklearn_loaded(model, loader):
    _, sklearn_value = accuracy_of(model, loader)
    return sklearn_value


def run_fit(arch, seed, train_u8, train_y, val_u8, val_y, corrupt_u8, batch, accum, ref_epoch5):
    say(f"run {arch} seed {seed} | batch {batch} | accum {accum} | ema {EMA_DECAY}")
    seed_everything(seed)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_backbone(arch)
    shadow = clone_shadow(model)
    train_loader = make_loader(TensorDataset(train_u8, train_y), True, seed, batch)
    val_loader = make_loader(TensorDataset(val_u8, val_y), False, seed, batch)
    corrupt_loader = make_loader(TensorDataset(corrupt_u8, val_y), False, seed, batch)
    result = train_one(
        model, shadow, train_loader, val_loader, corrupt_loader, arch, seed, accum, ref_epoch5
    )
    result["batch"] = batch
    result["accum"] = accum
    result["arch"] = arch
    result["seed"] = seed
    del model, shadow
    torch.cuda.empty_cache()
    return result


def remember(best_bank, result):
    for variant, best_key, state_key in (
        ("plain", "best_plain", "best_plain_state"),
        ("ema", "best_ema", "best_ema_state"),
    ):
        current = best_bank[result["arch"]].get(variant)
        candidate = {
            "arch": result["arch"],
            "seed": result["seed"],
            "best": result[best_key],
            "final": result["final_plain"] if variant == "plain" else result["final_ema"],
            "state": result[state_key],
            "mean_sec": result["mean_sec"],
            "peak_mb": result["peak_mb"],
            "batch": result["batch"],
            "accum": result["accum"],
            "epochs_ran": result["epochs_ran"],
        }
        if current is None or candidate["best"]["score"] > current["best"]["score"]:
            best_bank[result["arch"]][variant] = candidate


KEPT_PREV = {}


def save_kept(best_bank):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    saved = []
    for arch, variants in best_bank.items():
        for variant, item in variants.items():
            if item is None:
                continue
            path = CKPT_DIR / f"{arch}_F_seed{item['seed']}_{variant}.pt"
            torch.save(
                {
                    "epoch": item["best"]["epoch"],
                    "val_acc": item["best"]["clean"],
                    "val_corrupt": item["best"]["corrupt"],
                    "score": item["best"]["score"],
                    "aug": "F",
                    "seed": item["seed"],
                    "img_size": IMG_SIZE,
                    "variant": variant,
                    "arch": arch,
                    "ema_decay": EMA_DECAY if variant == "ema" else None,
                    "state_dict": item["state"],
                },
                path,
            )
            prev = KEPT_PREV.get((arch, variant))
            if prev is not None and prev != path and prev.exists():
                prev.unlink()
                say(f"ckpt drop superseded | {prev.as_posix()}")
            KEPT_PREV[(arch, variant)] = path
            saved.append(path.as_posix())
            say(f"ckpt | {path.as_posix()} | {fmt_pair(item['best'])}")
    return saved


def log_run(result, val_u8, val_y, corrupt_u8):
    for variant, best_key, final_key, state_key in (
        ("plain", "best_plain", "final_plain", "best_plain_state"),
        ("ema", "best_ema", "final_ema", "best_ema_state"),
    ):
        checker = load_arch(result["arch"], result[state_key])
        v_loader = make_loader(TensorDataset(val_u8, val_y), False, result["seed"], result["batch"])
        c_loader = make_loader(TensorDataset(corrupt_u8, val_y), False, result["seed"], result["batch"])
        sk_clean = sklearn_loaded(checker, v_loader)
        sk_corrupt = sklearn_loaded(checker, c_loader)
        best = result[best_key]
        assert abs(sk_clean - best["clean"]) < 1e-6, (variant, sk_clean, best["clean"])
        assert abs(sk_corrupt - best["corrupt"]) < 1e-6, (variant, sk_corrupt, best["corrupt"])
        append_exp(
            exp_row(
                result["arch"],
                result["seed"],
                variant,
                best,
                result[final_key],
                result["mean_sec"],
                result["peak_mb"],
                result["batch"],
                result["accum"],
                result["epochs_ran"],
                sk_clean,
                sk_corrupt,
            )
        )
        del checker
        torch.cuda.empty_cache()


def smoke(train_u8, train_y, val_u8, val_y):
    seed_everything(42)
    model = build_backbone("resnet50")
    shadow = clone_shadow(model)
    loader = make_loader(TensorDataset(train_u8[:128], train_y[:128]), True, 42, 64)
    val_loader = make_loader(TensorDataset(val_u8[:64], val_y[:64]), False, 42, 64)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optimizer_of(model)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    freeze_running_bn(model)
    for step, (images, labels) in enumerate(loader):
        images = gpu_augment(images.to(DEVICE), cfg="F", img_size=IMG_SIZE)
        labels = labels.to(DEVICE)
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(images), labels)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        ema_update(shadow, model)
        after = model.state_dict()
        for key, value in before.items():
            assert torch.equal(value, after[key]), key
        if step + 1 == 2:
            break
    clean, sk = accuracy_of(model, val_loader)
    ema_clean, ema_sk = accuracy_of(shadow, val_loader)
    assert abs(clean - sk) < 1e-12 and abs(ema_clean - ema_sk) < 1e-12
    say(f"smoke ok | plain {clean:.4f} | ema {ema_clean:.4f}")
    del model, shadow
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not args.smoke:
        LOG_PATH.write_text("", encoding="utf-8")
    assert torch.cuda.is_available(), "CUDA is not available"
    seed_everything(SEED)
    assert torch.backends.cudnn.deterministic is True
    assert torch.backends.cudnn.benchmark is False
    global _RN_MEAN, _RN_STD
    _RN_MEAN = torch.tensor(RN_MEAN, device=DEVICE).view(1, 3, 1, 1)
    _RN_STD = torch.tensor(RN_STD, device=DEVICE).view(1, 3, 1, 1)

    classes = pd.read_csv(DATA_DIR / "classes.csv").sort_values("label").class_name.tolist()
    assert len(classes) == NUM_CLASSES
    train_table = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    val_table = pd.read_csv(DATA_DIR / "val.csv", dtype={"id": str})
    assert set(train_table.label.unique()) == set(range(NUM_CLASSES))
    assert set(val_table.label.unique()) == set(range(NUM_CLASSES))

    t0 = time.perf_counter()
    train_u8 = load_uint8(train_table, "train")
    val_u8 = load_uint8(val_table, "val")
    cache_sec = time.perf_counter() - t0
    train_y = _labels_of(train_table)
    val_y = _labels_of(val_table)
    assert train_u8.shape[1:] == (3, RN_CACHE, RN_CACHE)
    assert val_u8.shape == (len(val_table), 3, RN_CACHE, RN_CACHE)
    blob = torch.load(CORRUPT_PATH, map_location="cpu", weights_only=False)
    assert blob["seed"] == CORRUPT_SEED and blob["images"].shape == tuple(val_u8.shape)
    assert torch.equal(blob["labels"], val_y)
    corrupt_u8 = blob["images"]
    say(
        f"cache train+val | n {len(train_table)}/{len(val_table)} | {cache_sec:.1f}s | "
        f"corrupt n={blob['stats']['n']} | test not loaded"
    )
    if args.smoke:
        smoke(train_u8, train_y, val_u8, val_y)
        return

    val_loader = make_loader(TensorDataset(val_u8, val_y), False, 7, BATCH)
    corrupt_loader = make_loader(TensorDataset(corrupt_u8, val_y), False, 7, BATCH)
    ckpt = torch.load(CKPT_DIR / "resnet50_F_seed7.pt", map_location="cpu", weights_only=False)
    ref_model = load_arch("resnet50", ckpt["state_dict"])
    tta_clean, tta_corrupt, tta_sec = tta_pair(ref_model, val_loader, corrupt_loader)
    say(
        f"pipeline check saved ckpt | tta {tta_clean:.4f}/{tta_corrupt:.4f} | "
        f"val tta {tta_sec:.1f}s | ref {TTA_REF[0]:.4f}/{TTA_REF[1]:.4f}"
    )
    if round(tta_clean, 4) != 0.9642 or round(tta_corrupt, 4) != 0.9528:
        raise SystemExit("pipeline TTA check failed on saved resnet50_F_seed7.pt")
    est_test = tta_sec * N_TEST / (2 * len(val_table))
    say(f"scale check | resnet50 test tta estimate {est_test:.1f}s | published 75.1s")
    del ref_model
    torch.cuda.empty_cache()
    say("prefetch | resnet101 IMAGENET1K_V1")
    _prefetched = models.resnet101(weights=models.ResNet101_Weights.IMAGENET1K_V1)
    del _prefetched

    runs = []
    best_bank = {"resnet50": {}, "resnet101": {}}
    batch50, accum50 = 64, 1
    run42 = run_fit("resnet50", 42, train_u8, train_y, val_u8, val_y, corrupt_u8, batch50, accum50, None)
    runs.append(run42)
    check_reference(run42)
    log_run(run42, val_u8, val_y, corrupt_u8)
    remember(best_bank, run42)
    save_kept(best_bank)
    run7 = run_fit("resnet50", 7, train_u8, train_y, val_u8, val_y, corrupt_u8, batch50, accum50, None)
    runs.append(run7)
    check_reference(run7)
    log_run(run7, val_u8, val_y, corrupt_u8)
    remember(best_bank, run7)
    save_kept(best_bank)
    retrained = load_arch("resnet50", run7["clean_max_state"])
    tta_clean2, tta_corrupt2, _ = tta_pair(retrained, val_loader, corrupt_loader)
    say(f"pipeline check retrained seed7 plain clean-max | tta {tta_clean2:.4f}/{tta_corrupt2:.4f}")
    if round(tta_clean2, 4) != 0.9642 or round(tta_corrupt2, 4) != 0.9528:
        upsert_state(
            [
                "- Раунд 3: retrained F seed 7 TTA не совпал с 0.9642/0.9528, ResNet101 не запускался. test не использовался. cudnn deterministic=True, benchmark=False.",
                f"- Раунд 3 FAIL TTA retrained seed 7 clean-max: {tta_clean2:.4f}/{tta_corrupt2:.4f}.",
            ]
        )
        raise SystemExit("retrained F seed7 TTA does not match 0.9642/0.9528")
    del retrained
    torch.cuda.empty_cache()

    ref_epoch5 = run7["history"][4]["plain"]
    assert ref_epoch5["epoch"] == 5
    batch101, accum101 = choose_batch("resnet101")
    run101 = run_fit(
        "resnet101", 7, train_u8, train_y, val_u8, val_y, corrupt_u8, batch101, accum101, ref_epoch5
    )
    runs.append(run101)
    log_run(run101, val_u8, val_y, corrupt_u8)
    remember(best_bank, run101)
    save_kept(best_bank)

    f_score = max(
        run42["best_plain"]["score"],
        run42["best_ema"]["score"],
        run7["best_plain"]["score"],
        run7["best_ema"]["score"],
    )
    score101 = max(run101["best_plain"]["score"], run101["best_ema"]["score"])
    say(
        f"decision | best F mean {f_score:.4f} | resnet101 best-mean {score101:.4f} | "
        f"published plain {F_SCORE_REF:.4f} | need +0.005"
    )
    if (not run101["stopped"]) and score101 >= f_score + 0.005:
        say("decision | run resnet101 seed 42")
        run101_42 = run_fit(
            "resnet101", 42, train_u8, train_y, val_u8, val_y, corrupt_u8, batch101, accum101, None
        )
        runs.append(run101_42)
        log_run(run101_42, val_u8, val_y, corrupt_u8)
        remember(best_bank, run101_42)
        save_kept(best_bank)
    elif run101["stopped"]:
        say("decision | skip resnet101 seed 42 | stopped at epoch 5")
    else:
        say("decision | skip resnet101 seed 42 | gain < 0.005")

    tta_notes = []
    for arch in ("resnet50", "resnet101"):
        pool = [item for key, item in best_bank[arch].items() if item is not None]
        if not pool:
            continue
        plain_best = best_bank[arch].get("plain")
        ema_best = best_bank[arch].get("ema")
        winner = plain_best
        variant = "plain"
        if ema_best is not None and (
            plain_best is None or ema_best["best"]["score"] > plain_best["best"]["score"]
        ):
            winner = ema_best
            variant = "ema"
        model = load_arch(arch, winner["state"])
        arch_val = make_loader(TensorDataset(val_u8, val_y), False, winner["seed"], winner["batch"])
        arch_corrupt = make_loader(TensorDataset(corrupt_u8, val_y), False, winner["seed"], winner["batch"])
        clean, corrupt, sec = tta_pair(model, arch_val, arch_corrupt)
        say(
            f"tta best {arch} seed {winner['seed']} {variant} | {clean:.4f}/{corrupt:.4f} | "
            f"val tta {sec:.1f}s | no-tta {winner['best']['clean']:.4f}/{winner['best']['corrupt']:.4f}"
        )
        infer_test = sec * N_TEST / (2 * len(val_table))
        tta_notes.append(
            {
                "arch": arch,
                "seed": winner["seed"],
                "variant": variant,
                "clean": clean,
                "corrupt": corrupt,
                "val_sec": sec,
                "test_sec": infer_test,
                "mean_epoch": winner["mean_sec"],
            }
        )
        append_exp(
            {
                "model": arch,
                "weights": "IMAGENET1K_V1",
                "unfrozen": "layer3+layer4+classifier",
                "img_size": IMG_SIZE,
                "epochs": "",
                "batch_size": winner["batch"],
                "optimizer": "AdamW",
                "lr_head": 1e-3,
                "lr_layer4": 1e-4,
                "label_smoothing": 0.1,
                "best_val_acc": round(clean, 6),
                "best_epoch": "",
                "sklearn_val_acc": round(clean, 6),
                "mean_epoch_sec": "",
                "seed": winner["seed"],
                "pipeline": "round3_tta",
                "aug": "F",
                "val_corrupt_acc": round(corrupt, 6),
                "sklearn_val_corrupt": round(corrupt, 6),
                "lr_layer3": 5e-5,
                "accum_steps": "",
                "peak_vram_mb": "",
                "base_aug": "F",
                "variant": variant,
                "ema_decay": EMA_DECAY if variant == "ema" else "",
                "final_epoch": "",
                "final_val_acc": "",
                "final_corrupt": "",
                "best_score": "",
            }
        )
        del model
        torch.cuda.empty_cache()

    saved = save_kept(best_bank)

    say("table | model | seed | variant | best | final | epoch_s | peak_MB")
    table_bits = []
    for result in runs:
        for variant, best_key, final_key in (
            ("plain", "best_plain", "final_plain"),
            ("ema", "best_ema", "final_ema"),
        ):
            line = (
                f"table | {result['arch']} | {result['seed']} | {variant} | "
                f"{fmt_pair(result[best_key])} | {fmt_pair(result[final_key])} | "
                f"{result['mean_sec']:.1f} | {result['peak_mb']:.0f}"
            )
            say(line)
            table_bits.append(line)

    note101 = next(item for item in tta_notes if item["arch"] == "resnet101") if any(
        item["arch"] == "resnet101" for item in tta_notes
    ) else None
    epoch_for_estimate = run101["mean_sec"]
    cache_full = cache_sec * (len(train_table) + len(val_table) + N_TEST) / (len(train_table) + len(val_table))
    if note101 is not None:
        test_sec = note101["test_sec"]
    else:
        test_sec = tta_sec * N_TEST / (2 * len(val_table))
    pipeline_sec = cache_full + 15 * epoch_for_estimate + test_sec
    say(
        f"pipeline resnet101 | cache {cache_full:.0f}s | 15 epochs {15 * epoch_for_estimate:.0f}s | "
        f"test tta {test_sec:.0f}s | total {pipeline_sec / 60:.1f} min"
    )
    write_status(
        {
            "runs": [
                {
                    "arch": result["arch"],
                    "seed": result["seed"],
                    "stopped": result["stopped"],
                    "stop_reason": result["stop_reason"],
                    "mean_sec": result["mean_sec"],
                    "peak_mb": result["peak_mb"],
                    "epoch1_sec": result["epoch1_sec"],
                    "epoch1_peak": result["epoch1_peak"],
                    "plain_best": result["best_plain"],
                    "plain_final": result["final_plain"],
                    "ema_best": result["best_ema"],
                    "ema_final": result["final_ema"],
                    "clean_max": result["clean_max"],
                }
                for result in runs
            ],
            "tta": tta_notes,
            "pipeline_min": pipeline_sec / 60,
            "saved": saved,
        }
    )
    bullets = ["- Раунд 3: F, EMA decay 0.999, BN-буферы копируются, decay не входит в оптимизатор. test не использовался. cudnn deterministic=True, benchmark=False."]
    for result in runs:
        bullets.append(
            f"- Раунд 3 {result['arch']} seed {result['seed']}: plain лучшая {fmt_pair(result['best_plain'])}, "
            f"финал {fmt_pair(result['final_plain'])}; ema лучшая {fmt_pair(result['best_ema'])}, "
            f"финал {fmt_pair(result['final_ema'])}. Эпоха {result['mean_sec']:.1f} с, пик {result['peak_mb']:.0f} MB"
            + (f". STOP: {result['stop_reason']}" if result["stopped"] else ".")
        )
    for note in tta_notes:
        bullets.append(
            f"- Раунд 3 TTA 8 {note['arch']} seed {note['seed']} {note['variant']}: "
            f"{note['clean']:.4f}/{note['corrupt']:.4f}."
        )
    bullets.append(
        f"- Раунд 3 pipeline ResNet101: кэш ~{cache_full:.0f} с + 15 эпох {15 * epoch_for_estimate:.0f} с + "
        f"test TTA ~{test_sec:.0f} с = {pipeline_sec / 60:.1f} мин. Чекпоинты: {', '.join(saved)}."
    )
    upsert_state(bullets)
    say("ROUND3_DONE")


def check_reference(result):
    if result["arch"] != "resnet50":
        return
    ref = REF_CLEAN_MAX[result["seed"]]
    got = result["clean_max"]
    say(
        f"ref clean-max seed {result['seed']} | got {fmt_pair(got)} | "
        f"expect e{ref['epoch']} {ref['clean']:.4f}/{ref['corrupt']:.4f}"
    )
    if got["epoch"] != ref["epoch"] or abs(got["clean"] - ref["clean"]) > 5e-4 or abs(got["corrupt"] - ref["corrupt"]) > 5e-4:
        upsert_state(
            [
                "- Раунд 3: plain-траектория F разошлась с эталоном, следующие прогоны остановлены. test не использовался. cudnn deterministic=True, benchmark=False.",
                (
                    f"- Раунд 3 FAIL {result['arch']} seed {result['seed']}: clean-max {fmt_pair(got)}, "
                    f"эталон e{ref['epoch']} {ref['clean']:.4f}/{ref['corrupt']:.4f}."
                ),
            ]
        )
        raise SystemExit(f"plain trajectory diverged from F seed {result['seed']}")


if __name__ == "__main__":
    main()
