"""TV1: config F on train + clean val. Test is loaded only for TTA after epoch 15."""

import argparse
import re
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset

from src import train_f as tf

N_TEST = 8299
EPOCHS = 15
BATCH = 64
SEED = 7
LR_LAYER3 = 5e-5
LR_LAYER4 = 1e-4
LR_HEAD = 1e-3
CKPT_PATH = Path("outputs/checkpoints/resnet50_F_trainval_seed7.pt")
SUB_PATH = Path("outputs/submission_F_trainval_seed7_tta.csv")
REF_PATH = Path("outputs/submission_F_seed7_tta.csv")
LOG_PATH = Path("outputs/logs/tv1.log")
NOTE = "val в обучении, метрики val недействительны"


def say(msg):
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")
    try:
        print(msg, flush=True)
    except UnicodeEncodeError:
        print(msg.encode("ascii", "backslashreplace").decode(), flush=True)


def check_tables():
    train_table = pd.read_csv(tf.DATA_DIR / "train.csv", dtype={"id": str})
    val_table = pd.read_csv(tf.DATA_DIR / "val.csv", dtype={"id": str})
    if list(train_table.columns) != ["id", "label"] or list(val_table.columns) != ["id", "label"]:
        raise RuntimeError("csv columns")
    n_train = len(train_table)
    n_val = len(val_table)
    overlap = set(train_table.id) & set(val_table.id)
    say(
        f"TV1 check | train {n_train} | val {n_val} | union {n_train + n_val} | "
        f"val unique {val_table.id.nunique()} | overlap {len(overlap)}"
    )
    if n_train != 4999 or n_val != 699:
        raise RuntimeError(f"expected 4999+699, got {n_train}+{n_val}")
    if val_table.id.nunique() != n_val or train_table.id.nunique() != n_train:
        raise RuntimeError("repeated ids")
    if overlap:
        raise RuntimeError("train/val id overlap")
    if set(val_table.label.unique()) != set(range(tf.NUM_CLASSES)):
        raise RuntimeError("val labels")
    if set(train_table.label.unique()) != set(range(tf.NUM_CLASSES)):
        raise RuntimeError("train labels")
    return train_table, val_table


def load_fit_images(train_table, val_table):
    t0 = time.perf_counter()
    train_u8 = tf.load_uint8(train_table, "train")
    val_u8 = tf.load_uint8(val_table, "val")
    cache_sec = time.perf_counter() - t0
    train_y = tf._labels_of(train_table)
    val_y = tf._labels_of(val_table)
    if not torch.equal(val_y, torch.tensor(val_table.label.to_numpy(), dtype=torch.long)):
        raise RuntimeError("val labels not from val.csv")
    if train_u8.shape != (len(train_table), 3, tf.RN_CACHE, tf.RN_CACHE):
        raise RuntimeError(tuple(train_u8.shape))
    if val_u8.shape != (len(val_table), 3, tf.RN_CACHE, tf.RN_CACHE):
        raise RuntimeError(tuple(val_u8.shape))
    images = torch.cat([train_u8, val_u8], dim=0)
    labels = torch.cat([train_y, val_y], dim=0)
    if images.shape[0] != len(train_table) + len(val_table):
        raise RuntimeError("union length")
    say(
        f"TV1 cache | train {len(train_table)} + val {len(val_table)} = {images.shape[0]} | "
        f"{cache_sec:.1f}s | labels from val.csv | test not loaded | corrupt cache not loaded"
    )
    return images, labels, val_u8, val_y, cache_sec


def train_epochs(model, loader):
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = tf.optimizer_of(model, LR_LAYER3, LR_LAYER4, LR_HEAD)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda")
    times = []
    last_loss = None
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        tf.freeze_running_bn(model)
        loss_sum = 0.0
        seen = 0
        for images, labels in loader:
            images = tf.gpu_augment(images.to(tf.DEVICE, non_blocking=True), cfg="F", img_size=tf.IMG_SIZE)
            labels = labels.to(tf.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * labels.shape[0]
            seen += labels.shape[0]
        scheduler.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        last_loss = loss_sum / seen
        times.append(elapsed)
        say(f"TV1 epoch {epoch:02d} | train_loss {last_loss:.4f} | {elapsed:.1f}s")
    return float(sum(times) / len(times)), last_loss


@torch.no_grad()
def tta_labels(model, images):
    model.eval()
    chunks = []
    for start in range(0, images.shape[0], BATCH):
        base = tf.gpu_eval(images[start : start + BATCH].to(tf.DEVICE, non_blocking=True), img_size=tf.IMG_SIZE)
        probs = None
        for turns in range(4):
            view = base if turns == 0 else torch.rot90(base, turns, dims=(-2, -1))
            for flipped in (view, view.flip(-1)):
                batch = flipped.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch)
                piece = torch.softmax(logits.float(), dim=1)
                probs = piece if probs is None else probs + piece
        chunks.append(probs.argmax(dim=1).cpu())
    return torch.cat(chunks).numpy().astype(int)


def write_submission(test_table, labels):
    submission = pd.DataFrame({"id": test_table.id.to_numpy(), "label": labels})
    assert submission.id.tolist() == test_table.id.tolist()
    assert len(submission) == N_TEST and submission.id.is_unique
    assert submission.label.between(0, 19).all()
    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUB_PATH, index=False, lineterminator="\n")
    back = pd.read_csv(SUB_PATH, dtype={"id": str})
    assert list(back.columns) == ["id", "label"] and len(back) == N_TEST
    assert back.id.tolist() == test_table.id.tolist()
    assert back.label.between(0, 19).all()
    assert set(back.label.astype(int).unique()) <= set(range(20))
    return back


def logged_train_stats():
    text = LOG_PATH.read_text(encoding="utf-8")
    epochs = [float(item) for item in re.findall(r"train_loss [0-9.]+ \| ([0-9.]+)s", text)]
    losses = [float(item) for item in re.findall(r"train_loss ([0-9.]+) \|", text)]
    cache = re.search(r"TV1 cache \| .* \| ([0-9.]+)s \|", text)
    if len(epochs) != EPOCHS or cache is None:
        raise RuntimeError("train log is incomplete")
    return float(cache.group(1)), float(sum(epochs) / len(epochs)), losses[-1]


def prepare_model(state):
    tf.seed_everything(SEED)
    if torch.backends.cudnn.deterministic is not True or torch.backends.cudnn.benchmark is not False:
        raise SystemExit("cudnn flags")
    tf._RN_MEAN = torch.tensor(tf.RN_MEAN, device=tf.DEVICE).view(1, 3, 1, 1)
    tf._RN_STD = torch.tensor(tf.RN_STD, device=tf.DEVICE).view(1, 3, 1, 1)
    model = tf.build_backbone("IMAGENET1K_V1")
    tf.load_state(model, state)
    return model


def finish(model, cache_sec, mean_sec, last_loss, train_n, val_n, val_u8, val_y):
    t0 = time.perf_counter()
    val_loader = tf.make_loader(TensorDataset(val_u8, val_y), False, SEED, BATCH)
    clean, sklearn_clean = tf.accuracy_of(model, val_loader)
    if abs(clean - sklearn_clean) >= 1e-12:
        raise RuntimeError("val check mismatch")
    say(f"TV1 clean-val check {clean:.4f} | {NOTE}")
    test_table = pd.read_csv(tf.DATA_DIR / "test.csv", dtype={"id": str})
    assert list(test_table.columns) == ["id"]
    assert len(test_table) == N_TEST and test_table.id.is_unique
    test_u8 = tf.load_uint8(test_table, "test")
    assert tuple(test_u8.shape) == (N_TEST, 3, 256, 256)
    pred = tta_labels(model, test_u8)
    torch.cuda.synchronize()
    tail_sec = time.perf_counter() - t0
    back = write_submission(test_table, pred)
    ref = pd.read_csv(REF_PATH, dtype={"id": str})
    assert ref.id.tolist() == back.id.tolist()
    same = int((ref.label.to_numpy() == back.label.to_numpy()).sum())
    agree = same / N_TEST
    wall = cache_sec + mean_sec * EPOCHS + tail_sec
    peak_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
    say(
        f"TV1 done | epoch {mean_sec:.1f}s | pipeline {wall / 60:.1f} min | "
        f"cache {cache_sec:.1f}s | tail {tail_sec:.1f}s | clean-val check {clean:.4f} | "
        f"agree F seed7 {same}/{N_TEST} {agree:.4f} | vram {peak_mb:.0f}MB"
    )
    tf.append_exp(
        {
            "model": "resnet50",
            "weights": "IMAGENET1K_V1",
            "unfrozen": "layer3+layer4+classifier",
            "img_size": tf.IMG_SIZE,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "optimizer": "AdamW",
            "lr_head": LR_HEAD,
            "lr_layer4": LR_LAYER4,
            "label_smoothing": 0.1,
            "best_val_acc": "",
            "best_epoch": EPOCHS,
            "sklearn_val_acc": "",
            "mean_epoch_sec": round(mean_sec, 2),
            "seed": SEED,
            "pipeline": "TV1",
            "aug": "F",
            "val_corrupt_acc": "",
            "sklearn_val_corrupt": "",
            "lr_layer3": LR_LAYER3,
            "accum_steps": 1,
            "peak_vram_mb": round(peak_mb, 1),
            "base_aug": "F",
            "variant": "trainval_epoch15_tta8",
            "ema_decay": "",
            "final_epoch": EPOCHS,
            "final_val_acc": "",
            "final_corrupt": "",
            "best_score": "",
            "note": NOTE,
            "check_clean_val": round(clean, 6),
            "agree_ref": round(agree, 6),
            "train_loss_last": round(last_loss, 6),
            "pipeline_min": round(wall / 60, 2),
        }
    )


def infer():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    blob = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if blob.get("epoch") != EPOCHS or blob.get("seed") != SEED:
        raise RuntimeError("checkpoint is not epoch 15 seed 7")
    cache_sec, mean_sec, last_loss = logged_train_stats()
    _train_table, val_table = check_tables()
    val_u8 = tf.load_uint8(val_table, "val")
    val_y = tf._labels_of(val_table)
    if not torch.equal(val_y, torch.tensor(val_table.label.to_numpy(), dtype=torch.long)):
        raise RuntimeError("val labels not from val.csv")
    model = prepare_model(blob["state_dict"])
    say("TV1 infer | weights epoch 15 | test loaded for TTA only")
    finish(model, cache_sec, mean_sec, last_loss, blob["train_n"], blob["val_n"], val_u8, val_y)


def main():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    LOG_PATH.write_text("", encoding="utf-8")
    train_table, val_table = check_tables()
    images, labels, val_u8, val_y, cache_sec = load_fit_images(train_table, val_table)
    tf.seed_everything(SEED)
    if torch.backends.cudnn.deterministic is not True or torch.backends.cudnn.benchmark is not False:
        raise SystemExit("cudnn flags")
    tf._RN_MEAN = torch.tensor(tf.RN_MEAN, device=tf.DEVICE).view(1, 3, 1, 1)
    tf._RN_STD = torch.tensor(tf.RN_STD, device=tf.DEVICE).view(1, 3, 1, 1)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = tf.build_backbone("IMAGENET1K_V1")
    loader = tf.make_loader(TensorDataset(images, labels), True, SEED, BATCH)
    say(
        f"TV1 train | n {images.shape[0]} | batch {BATCH} | epochs {EPOCHS} | "
        f"lr3 {LR_LAYER3} | seed {SEED} | aug F | epoch pick none"
    )
    mean_sec, last_loss = train_epochs(model, loader)
    state = tf.clone_state(model)
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": state,
            "epoch": EPOCHS,
            "seed": SEED,
            "img_size": tf.IMG_SIZE,
            "exp": "TV1",
            "note": NOTE,
            "train_n": len(train_table),
            "val_n": len(val_table),
        },
        CKPT_PATH,
    )
    del images, labels
    finish(model, cache_sec, mean_sec, last_loss, len(train_table), len(val_table), val_u8, val_y)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--infer", action="store_true")
    infer() if parser.parse_args().infer else main()
