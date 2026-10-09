"""ResNet50 config F. Round 4 runs from this module; config F augmentations match the reference runner."""

import argparse
import copy
import csv
import json
import os
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

DATA_DIR = Path(os.environ.get("CONTEST_DATA", "./data"))
DEVICE = "cuda"
NUM_CLASSES = 20
RN_IMG = 224
RN_CACHE = 256
IMG_SIZE = 256
BATCH = 64
RN_MEAN = (0.485, 0.456, 0.406)
RN_STD = (0.229, 0.224, 0.225)
CORRUPT_SEED = 1234567
CORRUPT_PATH = Path("outputs") / "val_corrupted.pt"
EXP_PATH = Path("outputs") / "experiments.csv"
CKPT_DIR = Path("outputs") / "checkpoints"
LOG_PATH = Path("outputs") / "logs" / "round4.log"
STATUS_PATH = Path("outputs") / "logs" / "round4_status.json"
N_VAL = 699
N_TEST = 8299
REF7_CLEAN = 0.9599427753934192
REF7_CORRUPT = 0.9470672389127325
REF7_SCORE = (REF7_CLEAN + REF7_CORRUPT) / 2.0
REF42_CLEAN = 0.949928469241774
REF42_CORRUPT = 0.944206008583691
REF42_SCORE = (REF42_CLEAN + REF42_CORRUPT) / 2.0
REF_E5_CLEAN = 645 / 699
REF_E5_CORRUPT = 631 / 699
TTA_REF_CLEAN = 674 / 699
TTA_REF_CORRUPT = 666 / 699
TTA_SCORE = (TTA_REF_CLEAN + TTA_REF_CORRUPT) / 2.0
# Rounded plain trajectory of F seed 7 (round 3 log). One image is 0.0014, so 1.5e-4 catches a miss.
B_PLAIN = {
    1: (0.8655, 0.8498),
    2: (0.9041, 0.8798),
    3: (0.9185, 0.9041),
    4: (0.9270, 0.9113),
    5: (0.9227, 0.9027),
    6: (0.9428, 0.9242),
    7: (0.9342, 0.9142),
    8: (0.9428, 0.9299),
    9: (0.9528, 0.9356),
    10: (0.9514, 0.9399),
    11: (0.9485, 0.9385),
    12: (0.9571, 0.9385),
    13: (0.9599, 0.9471),
    14: (0.9456, 0.9342),
    15: (0.9528, 0.9413),
}
_RN_MEAN = None
_RN_STD = None


def seed_everything(seed):
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

def build_backbone(weights_name):
    weights = getattr(models.ResNet50_Weights, weights_name)
    backbone = models.resnet50(weights=weights)
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


def optimizer_of(model, lr_layer3, lr_layer4, lr_head):
    optimizer = torch.optim.AdamW(
        [
            {"params": model[0].layer3.parameters(), "lr": lr_layer3},
            {"params": model[0].layer4.parameters(), "lr": lr_layer4},
            {"params": model[1].parameters(), "lr": lr_head},
        ],
        weight_decay=1e-4,
    )
    assert [group["lr"] for group in optimizer.param_groups] == [lr_layer3, lr_layer4, lr_head]
    return optimizer


@torch.no_grad()
def ema_update(shadow, model, decay):
    for src, dst in zip(model.parameters(), shadow.parameters()):
        dst.mul_(decay).add_(src.detach(), alpha=1.0 - decay)
    for module, shadow_module in zip(model.modules(), shadow.modules()):
        if isinstance(module, nn.modules.batchnorm._BatchNorm):
            shadow_module.running_mean.copy_(module.running_mean)
            shadow_module.running_var.copy_(module.running_var)
            shadow_module.num_batches_tracked.copy_(module.num_batches_tracked)


def swa_accumulate(avg, model, count):
    count += 1
    state = model.state_dict()
    if avg is None:
        avg = {key: value.detach().cpu().clone() for key, value in state.items()}
        return avg, count
    for key, value in state.items():
        if torch.is_floating_point(value):
            avg[key].add_(value.detach().cpu() - avg[key], alpha=1.0 / count)
        else:
            avg[key] = value.detach().cpu().clone()
    return avg, count


def cutmix_batch(images, labels, alpha=1.0, p=0.5):
    """Per-batch CutMix. Alpha 1.0 is Uniform(0, 1) on the CUDA generator."""
    if float(torch.rand((), device=images.device)) >= p:
        return labels, None, None
    if alpha != 1.0:
        raise ValueError(alpha)
    lam = float(torch.empty((), device=images.device).uniform_(0.0, 1.0))
    index = torch.randperm(images.shape[0], device=images.device)
    _, _, height, width = images.shape
    cut_rat = (1.0 - lam) ** 0.5
    cut_w = int(width * cut_rat)
    cut_h = int(height * cut_rat)
    cx = int(torch.randint(0, width, (), device=images.device))
    cy = int(torch.randint(0, height, (), device=images.device))
    x1 = max(cx - cut_w // 2, 0)
    y1 = max(cy - cut_h // 2, 0)
    x2 = min(cx + cut_w // 2, width)
    y2 = min(cy + cut_h // 2, height)
    if x2 <= x1 or y2 <= y1:
        return labels, None, None
    images[:, :, y1:y2, x1:x2] = images[index, :, y1:y2, x1:x2]
    lam_adj = 1.0 - (x2 - x1) * (y2 - y1) / float(height * width)
    return labels, labels[index], lam_adj


def pack_row(epoch, clean, corrupt):
    return {
        "epoch": int(epoch),
        "clean": float(clean),
        "corrupt": float(corrupt),
        "score": (float(clean) + float(corrupt)) / 2.0,
    }


def better(row, best):
    if best is None:
        return True
    return (row["score"], row["corrupt"], row["clean"]) > (best["score"], best["corrupt"], best["clean"])


@torch.no_grad()
def accuracy_of(model, loader, img_size=IMG_SIZE):
    model.eval()
    correct = 0
    total = 0
    pred_chunks = []
    true_chunks = []
    for images, labels in loader:
        images = gpu_eval(images.to(DEVICE, non_blocking=True), img_size=img_size)
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
    if abs(counted - sklearn_value) >= 1e-12:
        raise RuntimeError(f"metric mismatch {counted} vs {sklearn_value}")
    return counted, sklearn_value


def _resize_norm(images, img_size, mode):
    x = images.float().mul_(1.0 / 255.0)
    if x.shape[-2] != img_size or x.shape[-1] != img_size:
        x = torch.nn.functional.interpolate(
            x, size=(img_size, img_size), mode=mode, align_corners=False, antialias=True
        )
    x = (x - _RN_MEAN) / _RN_STD
    return x.contiguous(memory_format=torch.channels_last)


@torch.no_grad()
def accuracy_scaled(model, loader, img_size, mode):
    model.eval()
    pred_chunks = []
    true_chunks = []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        if img_size == RN_CACHE and mode == "bilinear":
            batch = gpu_eval(images, img_size=img_size)
        else:
            batch = _resize_norm(images, img_size, mode)
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            logits = model(batch)
        pred_chunks.append(logits.argmax(dim=1).cpu())
        true_chunks.append(labels)
    return float(accuracy_score(torch.cat(true_chunks).numpy(), torch.cat(pred_chunks).numpy()))


@torch.no_grad()
def predict_tta(model, loader, img_size=IMG_SIZE):
    model.eval()
    pred, true = [], []
    for images, labels in loader:
        base = gpu_eval(images.to(DEVICE, non_blocking=True), img_size=img_size)
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
    return float(accuracy_score(torch.cat(true).numpy(), torch.cat(pred).numpy()))


@torch.no_grad()
def predict_multiscale_tta(model, loader, sizes=(256, 288)):
    model.eval()
    pred, true = [], []
    for images, labels in loader:
        images = images.to(DEVICE, non_blocking=True)
        probs = None
        for size in sizes:
            if size == RN_CACHE:
                base = gpu_eval(images, img_size=size)
            else:
                base = _resize_norm(images, size, "bicubic")
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
    return float(accuracy_score(torch.cat(true).numpy(), torch.cat(pred).numpy()))


@torch.no_grad()
def recompute_trainable_bn(model, loader):
    """Cumulative BN stats on the training images, eval preprocessing, frozen blocks left at ImageNet stats."""
    model.train()
    freeze_running_bn(model)
    tracked = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d) and module.training:
            module.reset_running_stats()
            tracked.append((module, module.momentum))
            module.momentum = None
    if not tracked:
        raise RuntimeError("no trainable BatchNorm")
    seen = 0
    for images, _labels in loader:
        batch = gpu_eval(images.to(DEVICE, non_blocking=True), img_size=IMG_SIZE)
        model(batch)
        seen += images.shape[0]
    for module, momentum in tracked:
        module.momentum = momentum
    model.eval()
    return seen


def clone_state(model):
    return {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}


def load_state(model, state):
    model.load_state_dict(state)
    model.eval()
    return model


def train_one(model, shadow, train_loader, val_loader, corrupt_loader, cfg, seed, gate):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optimizer_of(model, cfg["lr_layer3"], cfg["lr_layer4"], cfg["lr_head"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
    scaler = torch.amp.GradScaler("cuda")
    history = []
    best_plain = None
    best_ema = None
    best_plain_state = None
    best_ema_state = None
    swa_avg = None
    swa_count = 0
    stopped = False
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.perf_counter()
        model.train()
        freeze_running_bn(model)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        n_seen = 0
        total_batches = len(train_loader)
        accum_steps = 1
        for images, labels in train_loader:
            images = gpu_augment(images.to(DEVICE, non_blocking=True), cfg="F", img_size=IMG_SIZE)
            labels = labels.to(DEVICE, non_blocking=True)
            mixed_b = None
            lam = None
            if cfg["cutmix_p"]:
                labels, mixed_b, lam = cutmix_batch(images, labels, alpha=1.0, p=cfg["cutmix_p"])
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                logits = model(images)
                if lam is None:
                    loss = criterion(logits, labels)
                else:
                    loss = lam * criterion(logits, labels) + (1.0 - lam) * criterion(logits, mixed_b)
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
                if shadow is not None:
                    ema_update(shadow, model, cfg["ema_decay"])
        clean, _ = accuracy_of(model, val_loader)
        corrupt, _ = accuracy_of(model, corrupt_loader)
        if shadow is not None:
            ema_clean, _ = accuracy_of(shadow, val_loader)
            ema_corrupt, _ = accuracy_of(shadow, corrupt_loader)
        else:
            ema_clean = ema_corrupt = None
        scheduler.step()
        if cfg["swa"] is not None and cfg["swa"][0] <= epoch <= cfg["swa"][1]:
            swa_avg, swa_count = swa_accumulate(swa_avg, model, swa_count)
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        plain_row = pack_row(epoch, clean, corrupt)
        ema_row = None if ema_clean is None else pack_row(epoch, ema_clean, ema_corrupt)
        history.append({"plain": plain_row, "ema": ema_row, "sec": elapsed, "peak_mb": peak_mb})
        if better(plain_row, best_plain):
            best_plain = plain_row
            best_plain_state = clone_state(model)
        if ema_row is not None and better(ema_row, best_ema):
            best_ema = ema_row
            best_ema_state = clone_state(shadow)
        ema_txt = "" if ema_row is None else f" | ema {ema_clean:.4f}/{ema_corrupt:.4f}"
        gap_txt = ""
        if epoch == 5 and gate:
            gap_c = clean - REF_E5_CLEAN
            gap_k = corrupt - REF_E5_CORRUPT
            gap_txt = f" | e5 gap {gap_c:+.4f}/{gap_k:+.4f}"
            if gap_c < -0.02 or gap_k < -0.02:
                stopped = True
        say(
            f"{cfg['name']} seed {seed} | epoch {epoch:2d} | plain {clean:.4f}/{corrupt:.4f}"
            f"{ema_txt} | {elapsed:.1f}s | vram {peak_mb:.0f}MB{gap_txt}"
        )
        if cfg["name"] == "B" and seed == 7:
            ref_c, ref_k = B_PLAIN[epoch]
            if abs(clean - ref_c) > 1.5e-4 or abs(corrupt - ref_k) > 1.5e-4:
                say(
                    f"DIVERGE | B seed 7 epoch {epoch} plain {clean:.4f}/{corrupt:.4f} "
                    f"vs F {ref_c:.4f}/{ref_k:.4f}"
                )
                raise SystemExit(2)
        write_status({"exp": cfg["name"], "seed": seed, "epoch": epoch, "stopped": stopped})
        if stopped:
            say(
                f"STOP | {cfg['name']} seed {seed} epoch 5 plain {clean:.4f}/{corrupt:.4f} "
                f"vs F seed 7 {REF_E5_CLEAN:.4f}/{REF_E5_CORRUPT:.4f}"
            )
            break
    return {
        "history": history,
        "best_plain": best_plain,
        "best_ema": best_ema,
        "best_plain_state": best_plain_state,
        "best_ema_state": best_ema_state,
        "final_plain": history[-1]["plain"],
        "final_ema": history[-1]["ema"],
        "swa_avg": swa_avg,
        "swa_count": swa_count,
        "mean_sec": float(np.mean([row["sec"] for row in history])),
        "peak_mb": float(max(row["peak_mb"] for row in history)),
        "stopped": stopped,
        "epochs_ran": history[-1]["plain"]["epoch"],
    }


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
            and item.get("aug") == str(row.get("aug", ""))
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


def base_row(cfg, seed, variant, best, final, sec, peak_mb, batch, sklearn_clean, sklearn_corrupt):
    return {
        "model": "resnet50",
        "weights": cfg["weights"] if cfg else "IMAGENET1K_V1",
        "unfrozen": "layer3+layer4+classifier",
        "img_size": IMG_SIZE,
        "epochs": "" if best is None else best.get("epochs_ran", ""),
        "batch_size": batch,
        "optimizer": "AdamW",
        "lr_head": "" if cfg is None else cfg["lr_head"],
        "lr_layer4": "" if cfg is None else cfg["lr_layer4"],
        "label_smoothing": 0.1,
        "best_val_acc": round(best["clean"], 6),
        "best_epoch": best["epoch"],
        "sklearn_val_acc": round(sklearn_clean, 6),
        "mean_epoch_sec": "" if sec is None else round(sec, 2),
        "seed": seed,
        "pipeline": "round4",
        "aug": "F",
        "val_corrupt_acc": round(best["corrupt"], 6),
        "sklearn_val_corrupt": round(sklearn_corrupt, 6),
        "lr_layer3": "" if cfg is None else cfg["lr_layer3"],
        "accum_steps": 1,
        "peak_vram_mb": "" if peak_mb is None else round(peak_mb, 1),
        "base_aug": "F",
        "variant": variant,
        "ema_decay": "" if cfg is None or not cfg.get("ema_decay") or variant != "ema" else cfg["ema_decay"],
        "final_epoch": "" if final is None else final["epoch"],
        "final_val_acc": "" if final is None else round(final["clean"], 6),
        "final_corrupt": "" if final is None else round(final["corrupt"], 6),
        "best_score": round(best["score"], 6),
    }


def write_status(payload):
    STATUS_PATH.parent.mkdir(parents=True, exist_ok=True)
    STATUS_PATH.write_text(json.dumps(payload), encoding="utf-8")


def refresh_state(bullets):
    path = Path("docs/STATE.md")
    kept = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("- Раунд 4") and not line.startswith("- Следующий шаг:")
    ]
    kept.extend(bullets)
    kept.append("- Следующий шаг: сравнение с MLP/CNN (таблица + графики) и полная confusion matrix на val")
    path.write_text("\n".join(kept) + "\n", encoding="utf-8")


def pair_txt(row):
    if row is None:
        return "-"
    return f"{row['clean']:.4f}/{row['corrupt']:.4f}"


def is_candidate(row, margin_base):
    return row["score"] >= margin_base + 0.004 - 1e-12


def pipeline_min(cache_sec, mean_sec, epochs, tta_sec):
    test_sec = tta_sec * N_TEST / (2 * N_VAL)
    total = cache_sec + mean_sec * epochs + test_sec
    return total / 60.0, test_sec


def save_winner(name, state, meta):
    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    path = CKPT_DIR / name
    torch.save({"state_dict": state, **meta}, path)
    say(f"saved {path.as_posix()}")
    return path.name


def winner_name(exp, variant, seed):
    names = {
        ("B", "swa"): "resnet50_F_seed{seed}_swa_e11-15.pt",
        ("B", "ema"): "resnet50_F_seed{seed}_ema_d099.pt",
        ("B", "plain"): "resnet50_F_seed{seed}_plain.pt",
        ("C", "plain"): "resnet50_F_seed{seed}_imagenetv2.pt",
        ("D", "plain"): "resnet50_F_seed{seed}_cutmix.pt",
        ("E", "plain"): "resnet50_F_seed{seed}_layer3_1e-4.pt",
        ("F", "plain"): "resnet50_F_seed{seed}_batch32.pt",
    }
    return names[(exp, variant)].format(seed=seed)


def eval_pair(model, val_loader, corrupt_loader):
    clean, sk_c = accuracy_of(model, val_loader)
    corrupt, sk_k = accuracy_of(model, corrupt_loader)
    if abs(clean - sk_c) >= 1e-12 or abs(corrupt - sk_k) >= 1e-12:
        raise RuntimeError("sklearn mismatch")
    return clean, corrupt


def finish_swa(avg, weights_name, train_u8, train_y, val_loader, corrupt_loader):
    model = build_backbone(weights_name)
    model.load_state_dict(avg)
    bn_loader = make_loader(TensorDataset(train_u8, train_y), False, 0, BATCH)
    seen = recompute_trainable_bn(model, bn_loader)
    clean, corrupt = eval_pair(model, val_loader, corrupt_loader)
    say(f"swa bn | train images {seen} | no aug | {clean:.4f}/{corrupt:.4f}")
    state = clone_state(model)
    row = pack_row(15, clean, corrupt)
    del model
    torch.cuda.empty_cache()
    return row, state


def tta_pair(model, val_loader, corrupt_loader):
    t0 = time.perf_counter()
    clean = predict_tta(model, val_loader)
    corrupt = predict_tta(model, corrupt_loader)
    torch.cuda.synchronize()
    return clean, corrupt, time.perf_counter() - t0


def run_train(cfg, seed, data, cache_sec, gate):
    say(
        f"run {cfg['name']} seed {seed} | epochs {cfg['epochs']} | batch {cfg['batch']} | "
        f"lr3 {cfg['lr_layer3']} | weights {cfg['weights']} | ema {cfg['ema_decay']} | "
        f"swa {cfg['swa']} | cutmix {cfg['cutmix_p']}"
    )
    seed_everything(seed)
    if torch.backends.cudnn.deterministic is not True or torch.backends.cudnn.benchmark is not False:
        raise RuntimeError("cudnn flags")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = build_backbone(cfg["weights"])
    shadow = None
    if cfg["ema_decay"]:
        shadow = clone_shadow(model)
    train_u8, train_y, val_u8, val_y, corrupt_u8 = data
    train_loader = make_loader(TensorDataset(train_u8, train_y), True, seed, cfg["batch"])
    val_loader = make_loader(TensorDataset(val_u8, val_y), False, seed, BATCH)
    corrupt_loader = make_loader(TensorDataset(corrupt_u8, val_y), False, seed, BATCH)
    result = train_one(model, shadow, train_loader, val_loader, corrupt_loader, cfg, seed, gate)
    result["cfg"] = cfg
    result["seed"] = seed
    swa_row = None
    swa_state = None
    if result["swa_avg"] is not None and not result["stopped"]:
        if result["swa_count"] != 5:
            raise RuntimeError(f"swa count {result['swa_count']}")
        swa_row, swa_state = finish_swa(result["swa_avg"], cfg["weights"], train_u8, train_y, val_loader, corrupt_loader)
    result["swa"] = swa_row
    result["swa_state"] = swa_state

    variants = [("plain", result["best_plain"], result["final_plain"], result["best_plain_state"])]
    if result["best_ema"] is not None:
        variants.append(("ema", result["best_ema"], result["final_ema"], result["best_ema_state"]))
    if swa_row is not None:
        variants.append(("swa", swa_row, swa_row, swa_state))
    best_variant = max(variants, key=lambda item: (item[1]["score"], item[1]["corrupt"], item[1]["clean"]))
    tta_clean = tta_corrupt = tta_sec = None
    if not result["stopped"]:
        holder = build_backbone(cfg["weights"])
        load_state(holder, best_variant[3])
        tta_clean, tta_corrupt, tta_sec = tta_pair(holder, val_loader, corrupt_loader)
        say(
            f"tta8 {cfg['name']} seed {seed} {best_variant[0]} | {tta_clean:.4f}/{tta_corrupt:.4f} | "
            f"val {tta_sec:.1f}s"
        )
        del holder
        torch.cuda.empty_cache()
    result["best_variant"] = best_variant[0]
    result["tta"] = None if tta_clean is None else pack_row(0, tta_clean, tta_corrupt)
    result["tta_sec"] = tta_sec

    margin = REF7_SCORE if seed == 7 else REF42_SCORE
    winners = []
    for name, best, _final, state in variants:
        if result["stopped"]:
            break
        if is_candidate(best, margin):
            winners.append(name)
            saved = save_winner(
                winner_name(cfg["name"], name, seed),
                state,
                {
                    "epoch": best["epoch"],
                    "val_acc": best["clean"],
                    "val_corrupt": best["corrupt"],
                    "variant": name,
                    "seed": seed,
                    "img_size": IMG_SIZE,
                    "exp": cfg["name"],
                },
            )
            result.setdefault("saved", []).append(saved)
    result["winners"] = winners
    result["candidate"] = bool(winners) and seed == 7 and not result["stopped"]

    for name, best, final, _state in variants:
        row = base_row(
            cfg,
            seed,
            f"{cfg['name']}_{name}",
            best,
            final,
            result["mean_sec"],
            result["peak_mb"],
            cfg["batch"],
            best["clean"],
            best["corrupt"],
        )
        row["epochs"] = result["epochs_ran"]
        row["gate"] = "stop" if result["stopped"] else ""
        append_exp(row)
    if result["tta"] is not None:
        tta_row = base_row(
            cfg,
            seed,
            f"{cfg['name']}_{best_variant[0]}_tta8",
            result["tta"],
            None,
            None,
            None,
            cfg["batch"],
            result["tta"]["clean"],
            result["tta"]["corrupt"],
        )
        tta_row["epochs"] = ""
        tta_row["best_epoch"] = best_variant[1]["epoch"]
        append_exp(tta_row)

    minutes = None
    if tta_sec is not None:
        minutes, test_sec = pipeline_min(cache_sec, result["mean_sec"], cfg["epochs"], tta_sec)
        flag = " | OVER 40 min" if minutes > 40 else ""
        say(
            f"pipeline {cfg['name']} seed {seed} | cache {cache_sec:.1f}s + "
            f"{cfg['epochs']}x{result['mean_sec']:.1f}s + test TTA est {test_sec:.1f}s = {minutes:.1f} min{flag}"
        )
        result["pipeline_min"] = minutes
    say(
        f"summary | {cfg['name']} | {seed} | plain {pair_txt(result['best_plain'])} | "
        f"swa {pair_txt(swa_row)} | ema {pair_txt(result['best_ema'])} | "
        f"best e{result['best_plain']['epoch']} | final e{result['final_plain']['epoch']} "
        f"{pair_txt(result['final_plain'])} | epoch {result['mean_sec']:.1f}s | "
        f"winners {','.join(winners) if winners else '-'}"
    )
    result["best_plain_state"] = None
    result["best_ema_state"] = None
    result["swa_state"] = None
    result["swa_avg"] = None
    del model, shadow
    torch.cuda.empty_cache()
    return result


def run_scale(model, val_loader, corrupt_loader, seed, tag):
    rows = []
    say("scale | clean | corrupt")
    for size in (256, 288, 320):
        mode = "bilinear" if size == 256 else "bicubic"
        clean = accuracy_scaled(model, val_loader, size, mode)
        corrupt = accuracy_scaled(model, corrupt_loader, size, mode)
        say(f"{size} | {clean:.4f} | {corrupt:.4f}")
        row = pack_row(13 if seed == 7 else 11, clean, corrupt)
        rows.append((str(size), row))
        append_exp(
            {
                "model": "resnet50",
                "weights": "IMAGENET1K_V1",
                "unfrozen": "layer3+layer4+classifier",
                "img_size": size,
                "epochs": "",
                "batch_size": BATCH,
                "optimizer": "AdamW",
                "lr_head": 1e-3,
                "lr_layer4": 1e-4,
                "label_smoothing": 0.1,
                "best_val_acc": round(clean, 6),
                "best_epoch": 13 if seed == 7 else 11,
                "sklearn_val_acc": round(clean, 6),
                "mean_epoch_sec": "",
                "seed": seed,
                "pipeline": "round4",
                "aug": "F",
                "val_corrupt_acc": round(corrupt, 6),
                "sklearn_val_corrupt": round(corrupt, 6),
                "lr_layer3": 5e-5,
                "accum_steps": "",
                "peak_vram_mb": "",
                "base_aug": "F",
                "variant": f"scale{size}_{tag}",
                "ema_decay": "",
                "final_epoch": "",
                "final_val_acc": "",
                "final_corrupt": "",
                "best_score": round(row["score"], 6),
                "resize": "identity" if size == 256 else "bicubic_antialias",
            }
        )
    ms_clean = predict_multiscale_tta(model, val_loader)
    ms_corrupt = predict_multiscale_tta(model, corrupt_loader)
    say(f"ms256+288 | {ms_clean:.4f} | {ms_corrupt:.4f}")
    ms = pack_row(13 if seed == 7 else 11, ms_clean, ms_corrupt)
    append_exp(
        {
            "model": "resnet50",
            "weights": "IMAGENET1K_V1",
            "unfrozen": "layer3+layer4+classifier",
            "img_size": "256+288",
            "epochs": "",
            "batch_size": BATCH,
            "optimizer": "AdamW",
            "lr_head": 1e-3,
            "lr_layer4": 1e-4,
            "label_smoothing": 0.1,
            "best_val_acc": round(ms_clean, 6),
            "best_epoch": "",
            "sklearn_val_acc": round(ms_clean, 6),
            "mean_epoch_sec": "",
            "seed": seed,
            "pipeline": "round4",
            "aug": "F",
            "val_corrupt_acc": round(ms_corrupt, 6),
            "sklearn_val_corrupt": round(ms_corrupt, 6),
            "lr_layer3": 5e-5,
            "accum_steps": "",
            "peak_vram_mb": "",
            "base_aug": "F",
            "variant": f"ms_tta_{tag}",
            "ema_decay": "",
            "final_epoch": "",
            "final_val_acc": "",
            "final_corrupt": "",
            "best_score": round(ms["score"], 6),
            "resize": "bicubic_antialias",
        }
    )
    return rows, ms


def load_data():
    classes = pd.read_csv(DATA_DIR / "classes.csv").sort_values("label").class_name.tolist()
    if len(classes) != NUM_CLASSES:
        raise RuntimeError("class count")
    train_table = pd.read_csv(DATA_DIR / "train.csv", dtype={"id": str})
    val_table = pd.read_csv(DATA_DIR / "val.csv", dtype={"id": str})
    if set(train_table.label.unique()) != set(range(NUM_CLASSES)):
        raise RuntimeError("train labels")
    if set(val_table.label.unique()) != set(range(NUM_CLASSES)):
        raise RuntimeError("val labels")
    t0 = time.perf_counter()
    train_u8 = load_uint8(train_table, "train")
    val_u8 = load_uint8(val_table, "val")
    cache_sec = time.perf_counter() - t0
    train_y = _labels_of(train_table)
    val_y = _labels_of(val_table)
    if train_u8.shape[1:] != (3, RN_CACHE, RN_CACHE):
        raise RuntimeError(train_u8.shape)
    if val_u8.shape != (len(val_table), 3, RN_CACHE, RN_CACHE):
        raise RuntimeError(val_u8.shape)
    if len(val_table) != N_VAL:
        raise RuntimeError(len(val_table))
    blob = torch.load(CORRUPT_PATH, map_location="cpu", weights_only=False)
    if blob["seed"] != CORRUPT_SEED or blob["images"].shape != tuple(val_u8.shape):
        raise RuntimeError("corrupt cache")
    if not torch.equal(blob["labels"], val_y):
        raise RuntimeError("corrupt labels")
    say(
        f"cache train+val | n {len(train_table)}/{len(val_table)} | {cache_sec:.1f}s | "
        f"corrupt n={blob['stats']['n']} | test not loaded"
    )
    return (train_u8, train_y, val_u8, val_y, blob["images"]), cache_sec


def bullet_train(result):
    cfg = result["cfg"]
    plain = result["best_plain"]
    final = result["final_plain"]
    swa = result["swa"]
    ema = result["best_ema"]
    stop = " STOP epoch 5." if result["stopped"] else ""
    tta = ""
    if result["tta"] is not None:
        tta = f" TTA8 {result['best_variant']} {pair_txt(result['tta'])}."
    minutes = ""
    if result.get("pipeline_min") is not None:
        minutes = f" Пайплайн {result['pipeline_min']:.1f} мин."
    saved = ""
    if result.get("saved"):
        saved = " Чекпоинт " + ", ".join(result["saved"]) + "."
    return (
        f"- Раунд 4 {cfg['name']} seed {result['seed']}: plain {pair_txt(plain)} e{plain['epoch']} "
        f"(финал e{final['epoch']} {pair_txt(final)}); swa {pair_txt(swa)}; ema {pair_txt(ema)}. "
        f"Эпоха {result['mean_sec']:.1f} с.{tta}{minutes}{saved}{stop} "
        f"test не использовался. cudnn deterministic=True, benchmark=False."
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp", default="all", choices=["all", "A", "B", "C", "D", "E", "F"])
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.exp == "all" and args.seed is None:
        LOG_PATH.write_text("", encoding="utf-8")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    seed_everything(7)
    if torch.backends.cudnn.deterministic is not True or torch.backends.cudnn.benchmark is not False:
        raise SystemExit("cudnn flags")
    global _RN_MEAN, _RN_STD
    _RN_MEAN = torch.tensor(RN_MEAN, device=DEVICE).view(1, 3, 1, 1)
    _RN_STD = torch.tensor(RN_STD, device=DEVICE).view(1, 3, 1, 1)
    say(f"round4 start | {torch.cuda.get_device_name(0)} | exp {args.exp}")
    data, cache_sec = load_data()
    _train_u8, _train_y, val_u8, val_y, corrupt_u8 = data
    val_loader = make_loader(TensorDataset(val_u8, val_y), False, 7, BATCH)
    corrupt_loader = make_loader(TensorDataset(corrupt_u8, val_y), False, 7, BATCH)
    bullets = []
    exp_list = ["A", "B", "C", "D", "E", "F"] if args.exp == "all" else [args.exp]
    for exp in exp_list:
        if exp == "A":
            ckpt = torch.load(CKPT_DIR / "resnet50_F_seed7.pt", map_location="cpu", weights_only=False)
            if abs(ckpt["val_acc"] - REF7_CLEAN) > 1e-12 or abs(ckpt["val_corrupt"] - REF7_CORRUPT) > 1e-12:
                raise SystemExit("checkpoint metadata is not the F seed 7 reference")
            model = build_backbone("IMAGENET1K_V1")
            load_state(model, ckpt["state_dict"])
            clean, corrupt = eval_pair(model, val_loader, corrupt_loader)
            say(f"ckpt check 256 | {clean:.4f}/{corrupt:.4f}")
            if abs(clean - REF7_CLEAN) > 1e-12 or abs(corrupt - REF7_CORRUPT) > 1e-12:
                raise SystemExit("resnet50_F_seed7.pt val does not match 0.9599/0.9471")
            rows, ms = run_scale(model, val_loader, corrupt_loader, 7, "seed7")
            if abs(rows[0][1]["clean"] - REF7_CLEAN) > 1e-12 or abs(rows[0][1]["corrupt"] - REF7_CORRUPT) > 1e-12:
                raise SystemExit("scale 256 drifted from the reference eval")
            tta_c, tta_k, tta_sec = tta_pair(model, val_loader, corrupt_loader)
            say(f"tta8 ckpt seed 7 | {tta_c:.4f}/{tta_k:.4f} | val {tta_sec:.1f}s")
            if abs(tta_c - TTA_REF_CLEAN) > 1.5e-4 or abs(tta_k - TTA_REF_CORRUPT) > 1.5e-4:
                raise SystemExit("TTA8 on resnet50_F_seed7.pt drifted")
            scale_winners = []
            for size, row in rows:
                mark = ""
                if size != "256" and is_candidate(row, REF7_SCORE):
                    scale_winners.append(size)
                    mark = " CANDIDATE"
                say(f"scale decision | {size} | {row['clean']:.4f}/{row['corrupt']:.4f} | score {row['score']:.4f}{mark}")
            ms_mark = ""
            if ms["score"] >= TTA_SCORE + 0.004 - 1e-12:
                scale_winners.append("ms")
                ms_mark = " CANDIDATE"
            say(
                f"scale decision | ms256+288 | {ms['clean']:.4f}/{ms['corrupt']:.4f} | "
                f"score {ms['score']:.4f} vs tta {TTA_SCORE:.4f}{ms_mark}"
            )
            if scale_winners:
                ckpt42 = torch.load(CKPT_DIR / "resnet50_F_seed42.pt", map_location="cpu", weights_only=False)
                model42 = build_backbone("IMAGENET1K_V1")
                load_state(model42, ckpt42["state_dict"])
                rows42, ms42 = run_scale(model42, val_loader, corrupt_loader, 42, "seed42")
                say(
                    f"scale seed42 | ref plain {REF42_CLEAN:.4f}/{REF42_CORRUPT:.4f} | "
                    + " | ".join(f"{size} {row['clean']:.4f}/{row['corrupt']:.4f}" for size, row in rows42)
                    + f" | ms {ms42['clean']:.4f}/{ms42['corrupt']:.4f}"
                )
                del model42
            bits = " ".join(f"{size} {row['clean']:.4f}/{row['corrupt']:.4f}" for size, row in rows)
            bullets.append(
                f"- Раунд 4 A: {bits}; ms256+288 {ms['clean']:.4f}/{ms['corrupt']:.4f}; "
                f"TTA8 {tta_c:.4f}/{tta_k:.4f}. Кандидаты: {','.join(scale_winners) if scale_winners else 'нет'}. "
                f"test не использовался. cudnn deterministic=True, benchmark=False."
            )
            del model
            torch.cuda.empty_cache()
            refresh_state(bullets)
            continue
        cfg = dict(CONFIGS[exp])
        seeds = [args.seed] if args.seed is not None else [7]
        for seed in seeds:
            gate = seed == 7
            try:
                result = run_train(cfg, seed, data, cache_sec, gate)
            except SystemExit:
                bullets.append(
                    f"- Раунд 4 {exp} seed {seed}: plain разошёлся с эталоном F seed 7, следующие варианты не запускались. "
                    "test не использовался. cudnn deterministic=True, benchmark=False."
                )
                refresh_state(bullets)
                raise
            bullets.append(bullet_train(result))
            refresh_state(bullets)
            if args.seed is None and result["candidate"]:
                result42 = run_train(cfg, 42, data, cache_sec, False)
                bullets.append(bullet_train(result42))
                say(
                    f"seed42 compare | {exp} | plain {pair_txt(result42['best_plain'])} | "
                    f"swa {pair_txt(result42['swa'])} | ema {pair_txt(result42['best_ema'])} | "
                    f"ref {REF42_CLEAN:.4f}/{REF42_CORRUPT:.4f}"
                )
                refresh_state(bullets)
    say("round4 done")
    write_status({"exp": "done"})


CONFIGS = {
    "B": {
        "name": "B",
        "epochs": 15,
        "batch": 64,
        "lr_layer3": 5e-5,
        "lr_layer4": 1e-4,
        "lr_head": 1e-3,
        "weights": "IMAGENET1K_V1",
        "ema_decay": 0.99,
        "swa": (11, 15),
        "cutmix_p": 0.0,
    },
    "C": {
        "name": "C",
        "epochs": 15,
        "batch": 64,
        "lr_layer3": 5e-5,
        "lr_layer4": 1e-4,
        "lr_head": 1e-3,
        "weights": "IMAGENET1K_V2",
        "ema_decay": None,
        "swa": None,
        "cutmix_p": 0.0,
    },
    "D": {
        "name": "D",
        "epochs": 20,
        "batch": 64,
        "lr_layer3": 5e-5,
        "lr_layer4": 1e-4,
        "lr_head": 1e-3,
        "weights": "IMAGENET1K_V1",
        "ema_decay": None,
        "swa": None,
        "cutmix_p": 0.5,
    },
    "E": {
        "name": "E",
        "epochs": 25,
        "batch": 64,
        "lr_layer3": 1e-4,
        "lr_layer4": 1e-4,
        "lr_head": 1e-3,
        "weights": "IMAGENET1K_V1",
        "ema_decay": None,
        "swa": None,
        "cutmix_p": 0.0,
    },
    "F": {
        "name": "F",
        "epochs": 15,
        "batch": 32,
        "lr_layer3": 5e-5,
        "lr_layer4": 1e-4,
        "lr_head": 1e-3,
        "weights": "IMAGENET1K_V1",
        "ema_decay": None,
        "swa": None,
        "cutmix_p": 0.0,
    },
}


if __name__ == "__main__":
    main()

