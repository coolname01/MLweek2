"""Round 6. Profile F, then S1-S4. Test is loaded only for ensemble inference."""

import argparse
import json
import subprocess
import threading
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score
from torch.utils.data import TensorDataset
from torchvision import models

from src import train_f as tf

LOG_PATH = Path("outputs/logs/round6.log")
STATUS_PATH = Path("outputs/logs/round6_status.json")
RESULT_PATH = Path("outputs/logs/round6_result.json")
PROFILE_PATH = Path("outputs/logs/round6_profile.json")
REF_SUB = Path("outputs/submission_ens_F7F42_tta.csv")
F7_EPOCH_SEC = 62.49
F42_EPOCH_SEC = 62.82
F_TTA_TEST_SEC = 75.1
F7F42_MIN = 34.21
S1_FLOOR = 0.9485
E1_CLEAN = 0.8655
E1_CORRUPT = 0.8498
WRAPS = (
    ("_color_jitter", "color_jitter"),
    ("_gaussian_blur", "blur"),
    ("_gaussian_blur_wide", "blur"),
    ("_downscale", "downscale"),
    ("_downscale_uniform", "downscale"),
    ("_jpeg_or_noise", "jpeg_noise"),
)


def say(msg):
    tf.LOG_PATH = LOG_PATH
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with LOG_PATH.open("a", encoding="utf-8") as handle:
        handle.write(msg + "\n")
    print(msg.encode("cp1252", "replace").decode("cp1252"), flush=True)


def append_state(text):
    path = Path("docs/STATE.md")
    body = path.read_text(encoding="utf-8").rstrip()
    if "## Раунд 6" not in body:
        body += "\n\n## Раунд 6"
    path.write_text(body + "\n" + text.strip() + "\n", encoding="utf-8")


def load_result():
    if not RESULT_PATH.exists():
        return {}
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def save_result(blob):
    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(blob, indent=2), encoding="utf-8")


def exp_row(model, seed, variant, best, final, sec, peak_mb, batch, lr3, note, extra=None):
    row = {
        "model": model,
        "weights": "IMAGENET1K_V1",
        "unfrozen": "layer3+layer4+classifier",
        "img_size": tf.IMG_SIZE,
        "epochs": "" if best is None else best.get("epochs_ran", best.get("epoch", "")),
        "batch_size": batch,
        "optimizer": "AdamW",
        "lr_head": 1e-3,
        "lr_layer4": 1e-4,
        "label_smoothing": 0.1,
        "best_val_acc": "" if best is None else round(best["clean"], 6),
        "best_epoch": "" if best is None else best["epoch"],
        "sklearn_val_acc": "" if best is None else round(best["clean"], 6),
        "mean_epoch_sec": "" if sec is None else round(sec, 2),
        "seed": seed,
        "pipeline": "round6",
        "aug": "F",
        "val_corrupt_acc": "" if best is None else round(best["corrupt"], 6),
        "sklearn_val_corrupt": "" if best is None else round(best["corrupt"], 6),
        "lr_layer3": lr3,
        "accum_steps": 1,
        "peak_vram_mb": "" if peak_mb is None else round(peak_mb, 1),
        "base_aug": "F",
        "variant": variant,
        "final_epoch": "" if final is None else final["epoch"],
        "final_val_acc": "" if final is None else round(final["clean"], 6),
        "final_corrupt": "" if final is None else round(final["corrupt"], 6),
        "best_score": "" if best is None else round(best["score"], 6),
        "note": note,
    }
    if extra:
        row.update(extra)
    return row


def f_cfg(name, epochs, lr3, ckpt):
    return {
        "name": name,
        "epochs": epochs,
        "batch": tf.BATCH,
        "lr_layer3": lr3,
        "lr_layer4": 1e-4,
        "lr_head": 1e-3,
        "weights": "IMAGENET1K_V1",
        "ema_decay": None,
        "swa": None,
        "cutmix_p": 0.0,
        "ckpt": ckpt,
        "arch": "resnet50",
    }


def build_resnet18():
    backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    for param in backbone.parameters():
        param.requires_grad = False
    for block in (backbone.layer3, backbone.layer4):
        for param in block.parameters():
            param.requires_grad = True
    features = backbone.fc.in_features
    if features != 512:
        raise RuntimeError(features)
    backbone.fc = nn.Identity()
    head = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, tf.NUM_CLASSES))
    model = nn.Sequential(backbone, head).to(tf.DEVICE, memory_format=torch.channels_last)
    trainable = [name for name, param in model.named_parameters() if param.requires_grad]
    if not trainable or not all(
        name.startswith("0.layer3") or name.startswith("0.layer4") or name.startswith("1.")
        for name in trainable
    ):
        raise RuntimeError(trainable[:8])
    return model


def build_of(arch):
    if arch == "resnet50":
        return lambda: tf.build_backbone("IMAGENET1K_V1")
    if arch == "resnet18":
        return build_resnet18
    raise ValueError(arch)


def gpu_samples(stop, bucket):
    while not stop.is_set():
        try:
            proc = subprocess.run(
                ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                check=False,
                capture_output=True,
                text=True,
                timeout=5,
            )
            line = proc.stdout.strip().splitlines()
            if line:
                bucket.append(int(line[0].strip()))
        except (OSError, ValueError, subprocess.TimeoutExpired):
            pass
        stop.wait(1.0)


def start_dmon():
    try:
        return subprocess.Popen(
            ["nvidia-smi", "dmon", "-s", "u", "-d", "1", "-c", "240"],
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
        )
    except OSError:
        return None


def dmon_sm(proc):
    if proc is None:
        return []
    proc.terminate()
    try:
        out, _ = proc.communicate(timeout=10)
    except subprocess.TimeoutExpired:
        proc.kill()
        out, _ = proc.communicate(timeout=5)
    vals = []
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].isdigit():
            vals.append(int(parts[1]))
    return vals


def profile_epoch(data):
    """One F epoch, seed 7. Timers sync the device; RNG order matches train_one."""
    cfg = f_cfg("F", 15, 5e-5, "")
    tf.seed_everything(7)
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    model = tf.build_backbone("IMAGENET1K_V1")
    train_u8, train_y, val_u8, val_y, corrupt_u8 = data
    train_loader = tf.make_loader(TensorDataset(train_u8, train_y), True, 7, cfg["batch"])
    val_loader = tf.make_loader(TensorDataset(val_u8, val_y), False, 7, tf.BATCH)
    corrupt_loader = tf.make_loader(TensorDataset(corrupt_u8, val_y), False, 7, tf.BATCH)
    buckets = {
        "color_jitter": 0.0,
        "blur": 0.0,
        "downscale": 0.0,
        "jpeg_noise": 0.0,
        "aug_total": 0.0,
        "h2d": 0.0,
        "forward": 0.0,
        "backward": 0.0,
        "opt_step": 0.0,
        "eval_clean": 0.0,
        "eval_corrupt": 0.0,
    }
    saved = {name: getattr(tf, name) for name, _bucket in WRAPS}

    def wrapped(fn, bucket):
        def inner(*args, **kwargs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            out = fn(*args, **kwargs)
            torch.cuda.synchronize()
            buckets[bucket] += time.perf_counter() - t0
            return out

        return inner

    for name, bucket in WRAPS:
        setattr(tf, name, wrapped(saved[name], bucket))
    stop = threading.Event()
    util = []
    thread = threading.Thread(target=gpu_samples, args=(stop, util), daemon=True)
    dmon = start_dmon()
    thread.start()
    try:
        criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        optimizer = tf.optimizer_of(model, cfg["lr_layer3"], cfg["lr_layer4"], cfg["lr_head"])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])
        scaler = torch.amp.GradScaler("cuda")
        torch.cuda.synchronize()
        t_epoch = time.perf_counter()
        model.train()
        tf.freeze_running_bn(model)
        optimizer.zero_grad(set_to_none=True)
        pending = 0
        n_seen = 0
        total_batches = len(train_loader)
        accum_steps = 1
        for images, labels in train_loader:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            images = images.to(tf.DEVICE, non_blocking=True)
            labels = labels.to(tf.DEVICE, non_blocking=True)
            torch.cuda.synchronize()
            buckets["h2d"] += time.perf_counter() - t0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            images = tf.gpu_augment(images, cfg="F", img_size=tf.IMG_SIZE)
            torch.cuda.synchronize()
            buckets["aug_total"] += time.perf_counter() - t0
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                logits = model(images)
                loss = criterion(logits, labels)
                torch.cuda.synchronize()
                buckets["forward"] += time.perf_counter() - t0
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            scaler.scale(loss).backward()
            torch.cuda.synchronize()
            buckets["backward"] += time.perf_counter() - t0
            pending += 1
            n_seen += 1
            if pending == accum_steps or n_seen == total_batches:
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                torch.cuda.synchronize()
                buckets["opt_step"] += time.perf_counter() - t0
                pending = 0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        clean, sk_clean = tf.accuracy_of(model, val_loader)
        torch.cuda.synchronize()
        buckets["eval_clean"] += time.perf_counter() - t0
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        corrupt, sk_corrupt = tf.accuracy_of(model, corrupt_loader)
        torch.cuda.synchronize()
        buckets["eval_corrupt"] += time.perf_counter() - t0
        scheduler.step()
        torch.cuda.synchronize()
        epoch_sec = time.perf_counter() - t_epoch
    finally:
        stop.set()
        thread.join(timeout=3)
        sm = dmon_sm(dmon)
        for name, fn in saved.items():
            setattr(tf, name, fn)
    if abs(clean - sk_clean) >= 1e-12 or abs(corrupt - sk_corrupt) >= 1e-12:
        raise RuntimeError("sklearn mismatch")
    say(f"P0 epoch1 | {clean:.4f}/{corrupt:.4f} | ref {E1_CLEAN}/{E1_CORRUPT}")
    if abs(clean - E1_CLEAN) > 5e-4 or abs(corrupt - E1_CORRUPT) > 5e-4:
        raise SystemExit(f"P0 drifted {clean:.4f}/{corrupt:.4f} vs {E1_CLEAN}/{E1_CORRUPT}")
    aug_other = buckets["aug_total"] - (
        buckets["color_jitter"] + buckets["blur"] + buckets["downscale"] + buckets["jpeg_noise"]
    )
    parts = {
        "color_jitter": buckets["color_jitter"],
        "blur": buckets["blur"],
        "downscale": buckets["downscale"],
        "jpeg_noise": buckets["jpeg_noise"],
        "aug_other": max(0.0, aug_other),
        "h2d": buckets["h2d"],
        "forward": buckets["forward"],
        "backward": buckets["backward"],
        "opt_step": buckets["opt_step"],
        "eval_clean": buckets["eval_clean"],
        "eval_corrupt": buckets["eval_corrupt"],
    }
    used = sum(parts.values())
    parts["other"] = max(0.0, epoch_sec - used)
    sm_vals = sm or util
    sm_mean = float(sum(sm_vals) / len(sm_vals)) if sm_vals else float("nan")
    say(
        f"P0 seed 7 | epoch 1 | plain {clean:.4f}/{corrupt:.4f} | {epoch_sec:.1f}s | "
        f"gpu sm {sm_mean:.1f}% n {len(sm_vals)} source {'dmon' if sm else 'query'}"
    )
    say("P0 stage | sec | share")
    rows = []
    for name, sec in parts.items():
        share = sec / epoch_sec if epoch_sec else 0.0
        say(f"P0 stage {name} | {sec:.2f}s | {share:.1%}")
        rows.append({"stage": name, "sec": round(sec, 3), "share": round(share, 4)})
    say(f"P0 stage epoch | {epoch_sec:.2f}s | 100%")
    tips = speed_tips(parts, epoch_sec)
    for tip in tips:
        say(f"P0 tip | {tip}")
    payload = {
        "clean": clean,
        "corrupt": corrupt,
        "epoch_sec": epoch_sec,
        "gpu_sm_mean": None if sm_vals == [] else round(sm_mean, 2),
        "gpu_n": len(sm_vals),
        "gpu_source": "dmon" if sm else "query",
        "stages": rows,
        "tips": tips,
    }
    PROFILE_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    note = "profile 1 epoch; " + ", ".join(f"{row['stage']} {row['share']:.1%}" for row in rows)
    tf.append_exp(
        exp_row(
            "resnet50",
            7,
            "P0",
            tf.pack_row(1, clean, corrupt),
            tf.pack_row(1, clean, corrupt),
            epoch_sec,
            torch.cuda.max_memory_reserved() / (1024**2),
            cfg["batch"],
            cfg["lr_layer3"],
            note,
            {"epochs": 1},
        )
    )
    share_txt = ", ".join(f"{row['stage']} {row['share']:.1%}" for row in rows if row["share"] >= 0.05)
    append_state(
        f"- P0 F seed 7, 1 эпоха: {clean:.4f}/{corrupt:.4f} за {epoch_sec:.1f} с, "
        f"GPU sm {sm_mean:.0f}% ({payload['gpu_source']}, n={len(sm_vals)}). Доли: {share_txt}. "
        f"Ускорения без смены семантики: {'; '.join(tips)} "
        f"Чекпоинт не сохранён. test не использовался. cudnn deterministic=True, benchmark=False."
    )
    del model
    torch.cuda.empty_cache()
    return payload


def speed_tips(parts, epoch_sec):
    share = {key: val / epoch_sec for key, val in parts.items()}
    tips = []
    if share["color_jitter"] >= 0.05:
        tips.append("ColorJitter: один batched проход вместо python-цикла, факторы и порядок те же")
    if share["blur"] >= 0.08:
        tips.append("blur: вынести буферы coords из батча, sigma и радиус ядра не менять")
    if share["downscale"] >= 0.08:
        tips.append("downscale: оставить antialias и факторы, убрать синхронизации между стадиями")
    if share["jpeg_noise"] >= 0.08:
        tips.append("JPEG/шум: те же уровни и std, без поэлементного Python вокруг pool")
    aug = share["color_jitter"] + share["blur"] + share["downscale"] + share["jpeg_noise"] + share["aug_other"]
    if aug < 0.25:
        tips.append("аугментации не доминируют, их слияние почти не сократит эпоху")
    if share["eval_clean"] + share["eval_corrupt"] >= 0.2:
        tips.append("два val — доля эпохи, на пиксели аугментаций не влияют")
    if not tips:
        tips.append("узкое место forward/backward, семантику аугментаций трогать не нужно")
    return tips


def train_run(cfg, seed, data):
    say(
        f"run {cfg['name']} seed {seed} | epochs {cfg['epochs']} | batch {cfg['batch']} | "
        f"lr3 {cfg['lr_layer3']} | arch {cfg['arch']}"
    )
    tf.seed_everything(seed)
    if torch.backends.cudnn.deterministic is not True or torch.backends.cudnn.benchmark is not False:
        raise RuntimeError("cudnn flags")
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    build = build_of(cfg["arch"])
    model = build()
    train_u8, train_y, val_u8, val_y, corrupt_u8 = data
    train_loader = tf.make_loader(TensorDataset(train_u8, train_y), True, seed, cfg["batch"])
    val_loader = tf.make_loader(TensorDataset(val_u8, val_y), False, seed, tf.BATCH)
    corrupt_loader = tf.make_loader(TensorDataset(corrupt_u8, val_y), False, seed, tf.BATCH)
    result = tf.train_one(model, None, train_loader, val_loader, corrupt_loader, cfg, seed, False)
    best = result["best_plain"]
    final = result["final_plain"]
    holder = build()
    tf.load_state(holder, result["best_plain_state"])
    tta_clean, tta_corrupt, tta_sec = tf.tta_pair(holder, val_loader, corrupt_loader)
    say(
        f"tta8 {cfg['name']} seed {seed} | {tta_clean:.4f}/{tta_corrupt:.4f} | "
        f"best e{best['epoch']} {best['clean']:.4f}/{best['corrupt']:.4f} | "
        f"final e{final['epoch']} {final['clean']:.4f}/{final['corrupt']:.4f} | "
        f"epoch {result['mean_sec']:.1f}s | val tta {tta_sec:.1f}s"
    )
    path = tf.CKPT_DIR / cfg["ckpt"].format(seed=seed)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": result["best_plain_state"],
            "epoch": best["epoch"],
            "val_acc": best["clean"],
            "val_corrupt": best["corrupt"],
            "seed": seed,
            "img_size": tf.IMG_SIZE,
            "arch": cfg["arch"],
            "lr_layer3": cfg["lr_layer3"],
            "epochs": cfg["epochs"],
        },
        path,
    )
    say(f"saved {path.as_posix()}")
    model_name = "resnet18" if cfg["arch"] == "resnet18" else "resnet50"
    note = f"cosine T_max={cfg['epochs']}; best and final plain; test not used"
    tf.append_exp(
        exp_row(
            model_name,
            seed,
            cfg["name"],
            best,
            final,
            result["mean_sec"],
            result["peak_mb"],
            cfg["batch"],
            cfg["lr_layer3"],
            note,
            {"epochs": result["epochs_ran"]},
        )
    )
    tta = tf.pack_row(best["epoch"], tta_clean, tta_corrupt)
    tf.append_exp(
        exp_row(
            model_name,
            seed,
            f"{cfg['name']}_tta8",
            tta,
            None,
            None,
            None,
            cfg["batch"],
            cfg["lr_layer3"],
            "TTA8 of best epoch",
            {"epochs": "", "best_epoch": best["epoch"]},
        )
    )
    summary = {
        "name": cfg["name"],
        "seed": seed,
        "arch": cfg["arch"],
        "best_epoch": best["epoch"],
        "best_clean": best["clean"],
        "best_corrupt": best["corrupt"],
        "best_score": best["score"],
        "final_epoch": final["epoch"],
        "final_clean": final["clean"],
        "final_corrupt": final["corrupt"],
        "final_score": final["score"],
        "tta_clean": tta_clean,
        "tta_corrupt": tta_corrupt,
        "tta_score": (tta_clean + tta_corrupt) / 2.0,
        "tta_sec": tta_sec,
        "mean_sec": result["mean_sec"],
        "peak_mb": result["peak_mb"],
        "epochs": cfg["epochs"],
        "ckpt": path.name,
    }
    result["best_plain_state"] = None
    del model, holder
    torch.cuda.empty_cache()
    return summary


def state_train(tag, row):
    append_state(
        f"- {tag}: лучшая e{row['best_epoch']} {row['best_clean']:.4f}/{row['best_corrupt']:.4f} "
        f"(score {row['best_score']:.4f}), финал e{row['final_epoch']} "
        f"{row['final_clean']:.4f}/{row['final_corrupt']:.4f}, "
        f"TTA8 {row['tta_clean']:.4f}/{row['tta_corrupt']:.4f}, эпоха {row['mean_sec']:.1f} с. "
        f"Чекпоинт {row['ckpt']}. test не использовался. cudnn deterministic=True, benchmark=False."
    )


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


def combine(parts, weights):
    acc = parts[0] * weights[0]
    total = weights[0]
    for part, weight in zip(parts[1:], weights[1:]):
        acc = acc + part * weight
        total += weight
    return acc / total


def val_tta_model(arch, path, images, batch):
    build = build_of(arch)
    model = build()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tf.load_state(model, blob["state_dict"])
    t0 = time.perf_counter()
    probs = tta_probs(model, images, tf.IMG_SIZE, batch)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return probs, sec, blob


def test_tta_sec(arch, path, images):
    build = build_of(arch)
    model = build()
    blob = torch.load(path, map_location="cpu", weights_only=False)
    tf.load_state(model, blob["state_dict"])
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    probs = tta_probs(model, images, tf.IMG_SIZE, tf.BATCH)
    torch.cuda.synchronize()
    sec = time.perf_counter() - t0
    del model
    torch.cuda.empty_cache()
    return probs, sec


def est_test_tta(val_tta_sec):
    return val_tta_sec * tf.N_TEST / (2 * tf.N_VAL)


def write_submission(name, test_table, labels):
    path = Path("outputs") / name
    submission = pd.DataFrame({"id": test_table.id.to_numpy(), "label": labels})
    if submission.id.tolist() != test_table.id.tolist():
        raise RuntimeError("id order")
    if len(submission) != tf.N_TEST or not submission.id.is_unique:
        raise RuntimeError("submission size")
    if not submission.label.between(0, 19).all():
        raise RuntimeError("labels")
    submission.to_csv(path, index=False, lineterminator="\n")
    back = pd.read_csv(path, dtype={"id": str})
    if list(back.columns) != ["id", "label"] or len(back) != tf.N_TEST:
        raise RuntimeError("csv header")
    if back.id.tolist() != test_table.id.tolist() or not back.label.between(0, 19).all():
        raise RuntimeError("csv reread")
    ref = pd.read_csv(REF_SUB, dtype={"id": str})
    if ref.id.tolist() != back.id.tolist():
        raise RuntimeError("ref id order")
    agree = float((ref.label.to_numpy() == back.label.to_numpy()).mean())
    same = int((ref.label.to_numpy() == back.label.to_numpy()).sum())
    return path.name, agree, same


def ensemble(data, blob, cache_sec):
    _train_u8, _train_y, val_u8, val_y, corrupt_u8 = data
    specs = {
        "F7": ("resnet50", tf.CKPT_DIR / "resnet50_F_seed7.pt"),
        "F42": ("resnet50", tf.CKPT_DIR / "resnet50_F_seed42.pt"),
    }
    if "s1" in blob:
        specs["S1"] = ("resnet50", tf.CKPT_DIR / blob["s1"]["ckpt"])
    if "s2" in blob:
        specs["S2"] = ("resnet50", tf.CKPT_DIR / blob["s2"]["ckpt"])
    if "s3" in blob:
        specs["S3"] = ("resnet18", tf.CKPT_DIR / blob["s3"]["ckpt"])
    for seed_key, row in blob.get("s4", {}).items():
        specs[f"S1_{seed_key}"] = ("resnet50", tf.CKPT_DIR / row["ckpt"])

    clean_p, corrupt_p, meta = {}, {}, {}
    for key, (arch, path) in specs.items():
        clean_p[key], sec_c, info = val_tta_model(arch, path, val_u8, tf.BATCH)
        corrupt_p[key], sec_k, _info = val_tta_model(arch, path, corrupt_u8, tf.BATCH)
        meta[key] = {
            "epoch": info.get("epoch"),
            "val_sec": sec_c + sec_k,
            "arch": arch,
            "path": path.name,
        }
        say(f"ENS load {key} | epoch {info.get('epoch')} | val tta {sec_c + sec_k:.1f}s")

    f7_c = pair_acc(clean_p["F7"], val_y)
    f7_k = pair_acc(corrupt_p["F7"], val_y)
    if abs(f7_c - tf.TTA_REF_CLEAN) > 1.5e-4 or abs(f7_k - tf.TTA_REF_CORRUPT) > 1.5e-4:
        raise SystemExit(f"F7 TTA drifted {f7_c:.4f}/{f7_k:.4f}")

    def add(name, keys, weights):
        clean = pair_acc(combine([clean_p[key] for key in keys], weights), val_y)
        corrupt = pair_acc(combine([corrupt_p[key] for key in keys], weights), val_y)
        return {"name": name, "keys": keys, "weights": weights, "clean": clean, "corrupt": corrupt, "score": (clean + corrupt) / 2.0}

    plans = [add("F7+F42", ["F7", "F42"], [1.0, 1.0])]
    if abs(plans[0]["score"] - 0.9607) > 0.002:
        say(f"WARN F7+F42 score {plans[0]['score']:.4f} vs 0.9607")
    if "S1" in clean_p:
        plans.append(add("F7+F42+S1", ["F7", "F42", "S1"], [1.0, 1.0, 1.0]))
    if "S3" in clean_p:
        plans.append(add("F7+F42+S3", ["F7", "F42", "S3"], [1.0, 1.0, 1.0]))
        plans.append(add("F7+F42+0.5*S3", ["F7", "F42", "S3"], [1.0, 1.0, 0.5]))
        plans.append(add("F7+S3", ["F7", "S3"], [1.0, 1.0]))
    s1_keys = [key for key in ("S1", "S1_42", "S1_123") if key in clean_p]
    if len(s1_keys) == 3:
        plans.append(add("S1x3", s1_keys, [1.0, 1.0, 1.0]))
        if "S2" in clean_p:
            plans.append(add("S1x3+S2", s1_keys + ["S2"], [1.0, 1.0, 1.0, 1.0]))

    s3_test_sec = None
    test_table = pd.read_csv(tf.DATA_DIR / "test.csv", dtype={"id": str})
    if list(test_table.columns) != ["id"] or len(test_table) != tf.N_TEST or not test_table.id.is_unique:
        raise RuntimeError("test.csv")
    say("ENS | test loaded for TTA only")
    test_u8 = tf.load_uint8(test_table, "test")
    if tuple(test_u8.shape) != (tf.N_TEST, 3, 256, 256):
        raise RuntimeError(test_u8.shape)
    test_probs = {}
    if "S3" in specs:
        test_probs["S3"], s3_test_sec = test_tta_sec("resnet18", specs["S3"][1], test_u8)
        say(f"ENS S3 test TTA | {s3_test_sec:.1f}s | n {tf.N_TEST}")

    s1 = blob.get("s1")
    s2 = blob.get("s2")
    s3 = blob.get("s3")
    s4 = blob.get("s4", {})

    def minutes_of(plan):
        keys = plan["keys"]
        if "S3" in keys and "F7" in keys and "F42" in keys:
            return F7F42_MIN + 12 * s3["mean_sec"] / 60.0 + s3_test_sec / 60.0 + 0.2
        total = cache_sec
        if "F7" in keys:
            total += 15 * F7_EPOCH_SEC + F_TTA_TEST_SEC
        if "F42" in keys:
            total += 15 * F42_EPOCH_SEC + F_TTA_TEST_SEC
        if "S1" in keys and s1:
            total += s1["epochs"] * s1["mean_sec"] + est_test_tta(s1["tta_sec"])
        if "S2" in keys and s2:
            total += s2["epochs"] * s2["mean_sec"] + est_test_tta(s2["tta_sec"])
        if "S3" in keys and s3:
            total += 12 * s3["mean_sec"] + s3_test_sec + 12.0
        for seed_key, label in (("42", "S1_42"), ("123", "S1_123")):
            if label in keys and seed_key in s4:
                row = s4[seed_key]
                total += row["epochs"] * row["mean_sec"] + est_test_tta(row["tta_sec"])
        return total / 60.0

    say("ENS name | clean | corrupt | score | min")
    for plan in plans:
        plan["minutes"] = minutes_of(plan)
        say(
            f"ENS {plan['name']} | {plan['clean']:.4f} | {plan['corrupt']:.4f} | "
            f"{plan['score']:.4f} | {plan['minutes']:.2f} min"
        )
        tf.append_exp(
            exp_row(
                "+".join(sorted({meta[key]["arch"] for key in plan["keys"]})),
                "+".join(plan["keys"]),
                plan["name"],
                tf.pack_row(0, plan["clean"], plan["corrupt"]),
                None,
                None,
                None,
                tf.BATCH,
                "",
                "mean softmax TTA8; test infer only",
                {"pipeline_min": round(plan["minutes"], 2), "epochs": ""},
            )
        )

    fit = [plan for plan in plans if plan["minutes"] <= 42.0 + 1e-9]
    fit.sort(key=lambda plan: (-plan["score"], plan["minutes"], plan["name"]))
    top = fit[:3]
    needed = {key for plan in top for key in plan["keys"] if key not in test_probs}
    for key in needed:
        arch, path = specs[key]
        test_probs[key], sec = test_tta_sec(arch, path, test_u8)
        say(f"ENS {key} test TTA | {sec:.1f}s")
    del test_u8
    torch.cuda.empty_cache()
    agrees = []
    for rank, plan in enumerate(top, start=1):
        labels = combine([test_probs[key] for key in plan["keys"]], plan["weights"]).argmax(dim=1).numpy().astype(int)
        fname, agree, same = write_submission(f"submission_cand{rank}_tta.csv", test_table, labels)
        agrees.append(f"{fname} {plan['name']} {plan['clean']:.4f}/{plan['corrupt']:.4f}/{plan['score']:.4f} {plan['minutes']:.2f} мин, совпадение {same}/{tf.N_TEST}={agree:.4f}")
        say(f"SUB cand{rank} | {plan['name']} | agree {same}/{tf.N_TEST}={agree:.4f} | {plan['minutes']:.2f} min")
        tf.append_exp(
            exp_row(
                "+".join(sorted({meta[key]["arch"] for key in plan["keys"]})),
                "+".join(plan["keys"]),
                f"cand{rank}",
                tf.pack_row(0, plan["clean"], plan["corrupt"]),
                None,
                None,
                None,
                tf.BATCH,
                "",
                fname,
                {"pipeline_min": round(plan["minutes"], 2), "agree_ref": round(agree, 6), "epochs": ""},
            )
        )
    f42_epoch = meta["F42"]["epoch"]
    line = (
        f"- Ансамбли TTA8 (чистый | искажённый | среднее | мин): "
        + "; ".join(
            f"{plan['name']} {plan['clean']:.4f}|{plan['corrupt']:.4f}|{plan['score']:.4f}|{plan['minutes']:.2f}"
            for plan in plans
        )
        + f". F42 чекпоинт epoch {f42_epoch}. S3 test TTA {None if s3_test_sec is None else round(s3_test_sec, 1)} с. "
        + " ".join(agrees)
        + " Лимит 42 мин. test только в TTA. Чекпоинты не коммитить."
    )
    append_state(line)
    return {"plans": [{k: plan[k] for k in ("name", "clean", "corrupt", "score", "minutes")} for plan in plans], "cands": agrees}


def boot():
    tf.LOG_PATH = LOG_PATH
    tf.STATUS_PATH = STATUS_PATH
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    tf.seed_everything(7)
    if torch.backends.cudnn.deterministic is not True or torch.backends.cudnn.benchmark is not False:
        raise SystemExit("cudnn flags")
    tf._RN_MEAN = torch.tensor(tf.RN_MEAN, device=tf.DEVICE).view(1, 3, 1, 1)
    tf._RN_STD = torch.tensor(tf.RN_STD, device=tf.DEVICE).view(1, 3, 1, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage", default="all", choices=["all", "p0", "s1", "s2", "s3", "s4", "ens"])
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    boot()
    blob = load_result()
    say(f"round6 start | {torch.cuda.get_device_name(0)} | stage {args.stage}")
    if "cache_sec" not in blob or args.stage in ("all", "p0", "s1"):
        data, cache_sec = tf.load_data()
        blob["cache_sec"] = cache_sec
        save_result(blob)
    else:
        data, cache_sec = tf.load_data()
        cache_sec = blob["cache_sec"]
    stages = ["p0", "s1", "s2", "s3", "s4", "ens"] if args.stage == "all" else [args.stage]
    for stage in stages:
        if args.resume and stage in blob:
            say(f"round6 skip | {stage}")
            continue
        if stage == "p0":
            blob["p0"] = profile_epoch(data)
        elif stage == "s1":
            row = train_run(f_cfg("S1", 10, 5e-5, "resnet50_F10_seed{seed}.pt"), 7, data)
            blob["s1"] = row
            blob["s1_pass"] = row["best_score"] + 1e-12 >= S1_FLOOR
            state_train("S1 F10 seed 7", row)
            append_state(
                f"- S1 порог S4: score {row['best_score']:.4f} {'проходит' if blob['s1_pass'] else 'ниже'} {S1_FLOOR:.4f}."
            )
        elif stage == "s2":
            row = train_run(f_cfg("S2", 10, 1e-4, "resnet50_E10_seed{seed}.pt"), 7, data)
            blob["s2"] = row
            state_train("S2 E10 seed 7 layer3 1e-4", row)
        elif stage == "s3":
            cfg = f_cfg("S3", 12, 5e-5, "resnet18_F_seed{seed}.pt")
            cfg["arch"] = "resnet18"
            row = train_run(cfg, 7, data)
            blob["s3"] = row
            state_train("S3 ResNet18 F seed 7", row)
        elif stage == "s4":
            if not blob.get("s1_pass"):
                say(f"S4 skip | S1 score {blob.get('s1', {}).get('best_score')} < {S1_FLOOR}")
                append_state(f"- S4 не запускался: S1 score ниже {S1_FLOOR:.4f}.")
                blob["s4"] = {}
            else:
                blob["s4"] = {}
                for seed in (42, 123):
                    row = train_run(f_cfg("S4", 10, 5e-5, "resnet50_F10_seed{seed}.pt"), seed, data)
                    blob["s4"][str(seed)] = row
                    state_train(f"S4 F10 seed {seed}", row)
        elif stage == "ens":
            blob["ens"] = ensemble(data, blob, cache_sec)
        save_result(blob)
        say(f"round6 stage done | {stage}")
    say("round6 done")


if __name__ == "__main__":
    main()
