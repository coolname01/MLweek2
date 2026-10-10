"""Inception v3, seed 7, config F augmentations. Val is evaluation only. Test is loaded only by --ens."""

import argparse
import json
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset
from torchvision import models

from src import train_f as tf

IMG = 299
BATCH = 48
SEED = 7
EPOCHS = 15
LR_6E = 5e-5
LR_7 = 1e-4
LR_HEAD = 1e-3
WD = 1e-2
N_VAL = 699
N_TEST = 8299
F7_EPOCH_SEC = 62.49
F42_EPOCH_SEC = 62.82
F_TTA_TEST_SEC = 75.1
TTA_REF_CLEAN = 674 / 699
TTA_REF_CORRUPT = 666 / 699
CKPT_PATH = Path("outputs/checkpoints/inception_v3_seed7.pt")
LOG_PATH = Path("outputs/logs/iv3.log")
RESULT_PATH = Path("outputs/logs/iv3_result.json")
SUB_PATH = Path("outputs/submission_F7_IV3_tta.csv")
REF_SUB = Path("outputs/submission_ens_F7F42_tta.csv")
UNFROZEN = ("Mixed_6e", "Mixed_7a", "Mixed_7b", "Mixed_7c")


def say(msg):
    print(msg, flush=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")


def mark_cx_excluded():
    tf.append_exp(
        {
            "model": "convnext",
            "pipeline": "CX1",
            "variant": "excluded",
            "seed": "",
            "aug": "",
            "note": "нарушает правила",
        }
    )


def setup():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    tf.seed_everything(SEED)
    if torch.backends.cudnn.deterministic is not True or torch.backends.cudnn.benchmark is not False:
        raise SystemExit("cudnn flags")
    tf._RN_MEAN = torch.tensor(tf.RN_MEAN, device=tf.DEVICE).view(1, 3, 1, 1)
    tf._RN_STD = torch.tensor(tf.RN_STD, device=tf.DEVICE).view(1, 3, 1, 1)


def build_inception():
    weights = models.Inception_V3_Weights.IMAGENET1K_V1
    model = models.inception_v3(weights=weights, aux_logits=True)
    if model.aux_logits is not True or model.AuxLogits is None:
        raise RuntimeError("aux logits were not loaded")
    if model.transform_input is not True:
        raise RuntimeError("transform_input")
    if model.fc.in_features != 2048:
        raise RuntimeError(model.fc.in_features)
    model.aux_logits = False
    model.AuxLogits = None
    model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(2048, 20))
    for param in model.parameters():
        param.requires_grad_(False)
    for name in UNFROZEN:
        for param in getattr(model, name).parameters():
            param.requires_grad_(True)
    for param in model.fc.parameters():
        param.requires_grad_(True)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    allowed = tuple(f"{name}." for name in UNFROZEN) + ("fc.",)
    if not trainable or not all(name.startswith(allowed) for name in trainable):
        raise RuntimeError(trainable[:8])
    for name in UNFROZEN:
        if not any(item.startswith(f"{name}.") for item in trainable):
            raise RuntimeError(name)
    if not any(name.startswith("fc.") for name in trainable):
        raise RuntimeError("head frozen")
    model = model.to(tf.DEVICE, memory_format=torch.channels_last)
    model.train()
    tf.freeze_running_bn(model)
    if model.Conv2d_1a_3x3.bn.training or not model.Mixed_7c.branch1x1.bn.training:
        raise RuntimeError("frozen BN not held in eval")
    if any(name.startswith("AuxLogits") for name in model.state_dict()):
        raise RuntimeError("aux weights still present")
    return model


def optimizer_of(model):
    group_6e = list(model.Mixed_6e.parameters())
    group_7 = (
        list(model.Mixed_7a.parameters())
        + list(model.Mixed_7b.parameters())
        + list(model.Mixed_7c.parameters())
    )
    group_head = list(model.fc.parameters())
    named = group_6e + group_7 + group_head
    if len({id(param) for param in named}) != len(named):
        raise RuntimeError("overlapping param groups")
    trainable = [param for param in model.parameters() if param.requires_grad]
    if {id(param) for param in trainable} != {id(param) for param in named}:
        raise RuntimeError("param groups do not match trainable weights")
    optimizer = torch.optim.AdamW(
        [
            {"params": group_6e, "lr": LR_6E},
            {"params": group_7, "lr": LR_7},
            {"params": group_head, "lr": LR_HEAD},
        ],
        weight_decay=WD,
    )
    if [group["lr"] for group in optimizer.param_groups] != [LR_6E, LR_7, LR_HEAD]:
        raise RuntimeError("lr groups")
    if any(group["weight_decay"] != WD for group in optimizer.param_groups):
        raise RuntimeError("weight decay")
    return optimizer


def load_fit():
    data, cache_sec = tf.load_data()
    train_u8, train_y, val_u8, val_y, corrupt_u8 = data
    train_ids = set(pd.read_csv(tf.DATA_DIR / "train.csv", dtype={"id": str}).id)
    val_ids = set(pd.read_csv(tf.DATA_DIR / "val.csv", dtype={"id": str}).id)
    if train_ids & val_ids:
        raise RuntimeError("val ids overlap train")
    if train_u8.shape[0] != len(train_ids) or val_u8.shape[0] != N_VAL:
        raise RuntimeError("split sizes")
    return train_u8, train_y, val_u8, val_y, corrupt_u8, cache_sec


def smoke(train_u8, train_y):
    loader = tf.make_loader(TensorDataset(train_u8, train_y), True, SEED, BATCH)
    model = build_inception()
    optimizer = optimizer_of(model)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step_times = []
    fwd_times = []
    batches = iter(loader)
    for _ in range(2):
        images, labels = next(batches)
        model.train()
        tf.freeze_running_bn(model)
        optimizer.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        images = tf.gpu_augment(images.to(tf.DEVICE, non_blocking=True), cfg="F", img_size=IMG)
        labels = labels.to(tf.DEVICE, non_blocking=True)
        torch.cuda.synchronize()
        t_fwd = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.float16):
            loss = criterion(model(images), labels)
        torch.cuda.synchronize()
        fwd_times.append(time.perf_counter() - t_fwd)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
    n_train = len(loader)
    n_eval = 2 * ((N_VAL + BATCH - 1) // BATCH)
    iter_s = float(sum(step_times) / len(step_times))
    epoch_est = iter_s * n_train + float(sum(fwd_times) / len(fwd_times)) * n_eval
    peak_gb = torch.cuda.max_memory_reserved() / (1024 ** 3)
    say(
        f"SMOKE iter {iter_s:.3f}s | steps {step_times[0]:.3f}/{step_times[1]:.3f} | "
        f"epoch_est {epoch_est:.1f}s | vram {peak_gb:.2f}GB | train_batches {n_train}"
    )
    if epoch_est > 100.0 or peak_gb > 11.0:
        say("STOP smoke limit")
        raise SystemExit(2)
    del model
    torch.cuda.empty_cache()


def train(train_u8, train_y, val_u8, val_y, corrupt_u8, cache_sec):
    train_loader = tf.make_loader(TensorDataset(train_u8, train_y), True, SEED, BATCH)
    val_loader = tf.make_loader(TensorDataset(val_u8, val_y), False, SEED, BATCH)
    corrupt_loader = tf.make_loader(TensorDataset(corrupt_u8, val_y), False, SEED, BATCH)
    model = build_inception()
    optimizer = optimizer_of(model)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    scaler = torch.amp.GradScaler("cuda")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    best = None
    best_state = None
    history = []
    for epoch in range(1, EPOCHS + 1):
        t0 = time.perf_counter()
        model.train()
        tf.freeze_running_bn(model)
        for images, labels in train_loader:
            images = tf.gpu_augment(images.to(tf.DEVICE, non_blocking=True), cfg="F", img_size=IMG)
            labels = labels.to(tf.DEVICE, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                loss = criterion(model(images), labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        clean, sk_clean = tf.accuracy_of(model, val_loader, img_size=IMG)
        corrupt, sk_corrupt = tf.accuracy_of(model, corrupt_loader, img_size=IMG)
        if abs(clean - sk_clean) >= 1e-12 or abs(corrupt - sk_corrupt) >= 1e-12:
            raise RuntimeError("sklearn mismatch")
        scheduler.step()
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        peak_mb = torch.cuda.max_memory_reserved() / (1024 ** 2)
        row = tf.pack_row(epoch, clean, corrupt)
        history.append({"epoch": epoch, "clean": clean, "corrupt": corrupt, "sec": elapsed, "vram_mb": peak_mb})
        if tf.better(row, best):
            best = row
            best_state = tf.clone_state(model)
        say(
            f"IV3 seed 7 | epoch {epoch:02d} | {clean:.4f}/{corrupt:.4f} | "
            f"{elapsed:.1f}s | vram {peak_mb:.0f}MB"
        )
    tf.load_state(model, best_state)
    t_tta = time.perf_counter()
    tta_clean = tf.predict_tta(model, val_loader, img_size=IMG)
    tta_corrupt = tf.predict_tta(model, corrupt_loader, img_size=IMG)
    torch.cuda.synchronize()
    tta_sec = time.perf_counter() - t_tta
    say(f"IV3 TTA8 e{best['epoch']} | {tta_clean:.4f}/{tta_corrupt:.4f} | val {tta_sec:.1f}s")
    CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "epoch": best["epoch"],
            "seed": SEED,
            "img_size": IMG,
            "val_acc": best["clean"],
            "val_corrupt": best["corrupt"],
            "score": best["score"],
            "arch": "inception_v3",
        },
        CKPT_PATH,
    )
    mean_sec = float(sum(item["sec"] for item in history) / len(history))
    peak_mb = float(max(item["vram_mb"] for item in history))
    final = history[-1]
    payload = {
        "mean_sec": mean_sec,
        "peak_mb": peak_mb,
        "tta_val_sec": tta_sec,
        "cache_sec": cache_sec,
        "best_epoch": best["epoch"],
        "clean": best["clean"],
        "corrupt": best["corrupt"],
        "score": best["score"],
        "tta_clean": tta_clean,
        "tta_corrupt": tta_corrupt,
        "final_epoch": final["epoch"],
        "final_clean": final["clean"],
        "final_corrupt": final["corrupt"],
        "history": history,
    }
    RESULT_PATH.write_text(json.dumps(payload), encoding="utf-8")
    tf.append_exp(
        {
            "model": "inception_v3",
            "weights": "IMAGENET1K_V1",
            "unfrozen": "Mixed_6e+Mixed_7abc+fc",
            "img_size": IMG,
            "epochs": EPOCHS,
            "batch_size": BATCH,
            "optimizer": "AdamW",
            "lr_head": LR_HEAD,
            "lr_layer4": LR_7,
            "label_smoothing": 0.1,
            "best_val_acc": round(best["clean"], 6),
            "best_epoch": best["epoch"],
            "sklearn_val_acc": round(best["clean"], 6),
            "mean_epoch_sec": round(mean_sec, 2),
            "seed": SEED,
            "pipeline": "IV3",
            "aug": "F",
            "val_corrupt_acc": round(best["corrupt"], 6),
            "sklearn_val_corrupt": round(best["corrupt"], 6),
            "lr_layer3": LR_6E,
            "accum_steps": 1,
            "peak_vram_mb": round(peak_mb, 1),
            "base_aug": "F",
            "variant": "plain",
            "final_epoch": final["epoch"],
            "final_val_acc": round(final["clean"], 6),
            "final_corrupt": round(final["corrupt"], 6),
            "best_score": round(best["score"], 6),
            "note": (
                f"wd {WD}; lr Mixed_6e=lr_layer3 Mixed_7=lr_layer4; "
                f"TTA8 {tta_clean:.6f}/{tta_corrupt:.6f}; test not used"
            ),
        }
    )
    say(f"IV3 train done | epoch {mean_sec:.1f}s | vram {peak_mb:.0f}MB | ckpt {CKPT_PATH.as_posix()}")
    return payload


@torch.no_grad()
def tta_probs(model, images, img_size, batch):
    model.eval()
    chunks = []
    for start in range(0, images.shape[0], batch):
        base = tf.gpu_eval(images[start : start + batch].to(tf.DEVICE, non_blocking=True), img_size=img_size)
        probs = None
        for turns in range(4):
            view = base if turns == 0 else torch.rot90(base, turns, dims=(-2, -1))
            for flipped in (view, view.flip(-1)):
                batch_x = flipped.contiguous(memory_format=torch.channels_last)
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(batch_x)
                piece = torch.softmax(logits.float(), dim=1)
                probs = piece if probs is None else probs + piece
        chunks.append((probs / 8.0).cpu())
    return torch.cat(chunks, dim=0)


def pair_acc(probs, labels):
    pred = probs.argmax(dim=1).numpy()
    true = labels.numpy()
    score = float(accuracy_score(true, pred))
    if abs(score - float((pred == true).mean())) >= 1e-12:
        raise RuntimeError("sklearn mismatch")
    return score


def mean_probs(parts):
    acc = parts[0]
    for part in parts[1:]:
        acc = acc + part
    return acc / len(parts)


def ens_row(variant, clean, corrupt, extra):
    row = {
        "model": "inception_v3+resnet50",
        "weights": "IMAGENET1K_V1",
        "unfrozen": "IV3 Mixed_6e+Mixed_7abc+fc; F layer3+layer4+classifier",
        "img_size": "299+256",
        "batch_size": BATCH,
        "optimizer": "AdamW",
        "lr_head": LR_HEAD,
        "label_smoothing": 0.1,
        "best_val_acc": round(clean, 6),
        "sklearn_val_acc": round(clean, 6),
        "seed": "7",
        "pipeline": "IV3_ens",
        "aug": "F",
        "val_corrupt_acc": round(corrupt, 6),
        "sklearn_val_corrupt": round(corrupt, 6),
        "base_aug": "F",
        "variant": variant,
        "best_score": round((clean + corrupt) / 2.0, 6),
        "note": "mean softmax TTA8; test infer only",
    }
    row.update(extra)
    return row


def ensemble(val_u8, val_y, corrupt_u8):
    blob = json.loads(RESULT_PATH.read_text(encoding="utf-8"))
    iv3_ckpt = torch.load(CKPT_PATH, map_location="cpu", weights_only=False)
    if iv3_ckpt["epoch"] != blob["best_epoch"] or iv3_ckpt["seed"] != SEED:
        raise RuntimeError("iv3 checkpoint")
    f7_ckpt = torch.load(tf.CKPT_DIR / "resnet50_F_seed7.pt", map_location="cpu", weights_only=False)
    f42_ckpt = torch.load(tf.CKPT_DIR / "resnet50_F_seed42.pt", map_location="cpu", weights_only=False)
    iv3 = build_inception()
    tf.load_state(iv3, iv3_ckpt["state_dict"])
    p_clean = tta_probs(iv3, val_u8, IMG, BATCH)
    p_corrupt = tta_probs(iv3, corrupt_u8, IMG, BATCH)
    del iv3
    torch.cuda.empty_cache()
    f7 = tf.build_backbone("IMAGENET1K_V1")
    tf.load_state(f7, f7_ckpt["state_dict"])
    f7_clean = tta_probs(f7, val_u8, tf.IMG_SIZE, 64)
    f7_corrupt = tta_probs(f7, corrupt_u8, tf.IMG_SIZE, 64)
    f7_c = pair_acc(f7_clean, val_y)
    f7_k = pair_acc(f7_corrupt, val_y)
    if abs(f7_c - TTA_REF_CLEAN) > 1.5e-4 or abs(f7_k - TTA_REF_CORRUPT) > 1.5e-4:
        raise SystemExit(f"F7 TTA drifted {f7_c:.4f}/{f7_k:.4f}")
    del f7
    torch.cuda.empty_cache()
    f42 = tf.build_backbone("IMAGENET1K_V1")
    tf.load_state(f42, f42_ckpt["state_dict"])
    f42_clean = tta_probs(f42, val_u8, tf.IMG_SIZE, 64)
    f42_corrupt = tta_probs(f42, corrupt_u8, tf.IMG_SIZE, 64)
    del f42
    torch.cuda.empty_cache()
    rows = {
        "IV3": (pair_acc(p_clean, val_y), pair_acc(p_corrupt, val_y)),
        "F7+IV3": (
            pair_acc(mean_probs([f7_clean, p_clean]), val_y),
            pair_acc(mean_probs([f7_corrupt, p_corrupt]), val_y),
        ),
        "F7+F42+IV3": (
            pair_acc(mean_probs([f7_clean, f42_clean, p_clean]), val_y),
            pair_acc(mean_probs([f7_corrupt, f42_corrupt, p_corrupt]), val_y),
        ),
    }
    for name, (clean, corrupt) in rows.items():
        say(f"ENS {name} | {clean:.4f} | {corrupt:.4f} | {(clean + corrupt) / 2:.4f}")
    test_table = pd.read_csv(tf.DATA_DIR / "test.csv", dtype={"id": str})
    if list(test_table.columns) != ["id"] or len(test_table) != N_TEST or not test_table.id.is_unique:
        raise RuntimeError("test.csv")
    say("IV3 ens | test loaded for TTA only")
    test_u8 = tf.load_uint8(test_table, "test")
    if tuple(test_u8.shape) != (N_TEST, 3, 256, 256):
        raise RuntimeError(test_u8.shape)
    iv3 = build_inception()
    tf.load_state(iv3, iv3_ckpt["state_dict"])
    t0 = time.perf_counter()
    test_iv3 = tta_probs(iv3, test_u8, IMG, BATCH)
    torch.cuda.synchronize()
    iv3_test_sec = time.perf_counter() - t0
    del iv3
    torch.cuda.empty_cache()
    f7 = tf.build_backbone("IMAGENET1K_V1")
    tf.load_state(f7, f7_ckpt["state_dict"])
    test_f7 = tta_probs(f7, test_u8, tf.IMG_SIZE, 64)
    del f7, test_u8
    torch.cuda.empty_cache()
    labels = mean_probs([test_f7, test_iv3]).argmax(dim=1).numpy().astype(int)
    submission = pd.DataFrame({"id": test_table.id.to_numpy(), "label": labels})
    if submission.id.tolist() != test_table.id.tolist():
        raise RuntimeError("id order")
    if len(submission) != N_TEST or not submission.id.is_unique:
        raise RuntimeError("submission size")
    if not submission.label.between(0, 19).all() or not set(submission.label.tolist()) <= set(range(20)):
        raise RuntimeError("labels")
    SUB_PATH.parent.mkdir(parents=True, exist_ok=True)
    submission.to_csv(SUB_PATH, index=False, lineterminator="\n")
    back = pd.read_csv(SUB_PATH, dtype={"id": str})
    if list(back.columns) != ["id", "label"] or len(back) != N_TEST:
        raise RuntimeError("csv header")
    if back.id.tolist() != test_table.id.tolist() or not back.label.between(0, 19).all():
        raise RuntimeError("csv reread")
    ref = pd.read_csv(REF_SUB, dtype={"id": str})
    if ref.id.tolist() != back.id.tolist():
        raise RuntimeError("ref id order")
    agree = float((ref.label.to_numpy() == back.label.to_numpy()).mean())
    iv3_tta_test_est = blob["tta_val_sec"] * N_TEST / (2 * N_VAL)
    cache_sec = blob["cache_sec"]
    pipe = cache_sec + EPOCHS * F7_EPOCH_SEC + EPOCHS * blob["mean_sec"] + F_TTA_TEST_SEC + iv3_tta_test_est
    say(
        f"SUB {SUB_PATH.name} | agree F7F42 {agree:.4f} | "
        f"pipeline F7+IV3 {pipe / 60:.1f} min | iv3 test TTA {iv3_test_sec:.1f}s | "
        f"est from val {iv3_tta_test_est:.1f}s"
    )
    tf.append_exp(ens_row("IV3", *rows["IV3"], {"seed": "7", "model": "inception_v3", "img_size": IMG}))
    tf.append_exp(
        ens_row(
            "F7+IV3",
            *rows["F7+IV3"],
            {"seed": "7+7", "agree_ref": round(agree, 6), "pipeline_min": round(pipe / 60, 2)},
        )
    )
    tf.append_exp(ens_row("F7+F42+IV3", *rows["F7+F42+IV3"], {"seed": "7+42+7"}))
    blob["ens"] = {name: {"clean": clean, "corrupt": corrupt} for name, (clean, corrupt) in rows.items()}
    blob["agree_f7f42"] = agree
    blob["pipeline_min"] = pipe / 60.0
    blob["iv3_test_sec"] = iv3_test_sec
    blob["iv3_tta_test_est"] = iv3_tta_test_est
    RESULT_PATH.write_text(json.dumps(blob), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--ens", action="store_true")
    args = parser.parse_args()
    if sum((args.smoke, args.train, args.ens)) != 1:
        raise SystemExit("choose one of --smoke --train --ens")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if args.smoke or args.train:
        LOG_PATH.write_text("", encoding="utf-8")
    mark_cx_excluded()
    setup()
    if args.ens:
        _train_u8, _train_y, val_u8, val_y, corrupt_u8, _cache = load_fit()
        ensemble(val_u8, val_y, corrupt_u8)
        return
    train_u8, train_y, val_u8, val_y, corrupt_u8, cache_sec = load_fit()
    say(f"IV3 cache {cache_sec:.1f}s | train {train_u8.shape[0]} val {val_u8.shape[0]} | test not loaded")
    if args.smoke:
        smoke(train_u8, train_y)
        return
    train(train_u8, train_y, val_u8, val_y, corrupt_u8, cache_sec)


if __name__ == "__main__":
    main()
