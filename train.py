import copy
import gc
import json
import os
import random
import shutil
import time
from dataclasses import dataclass

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from PIL import Image
from scipy.stats import rankdata
from sklearn.metrics import balanced_accuracy_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedKFold
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms


class CFG:
    data_dir = "data"
    train_img_dir = os.path.join(data_dir, "train_images")
    eval_img_dir = os.path.join(data_dir, "eval_images")
    train_csv = os.path.join(data_dir, "train_metadata.csv")
    eval_csv = os.path.join(data_dir, "test_metadata.csv")
    weights_dir = "weights"
    oof_dir = "oof"
    ensemble_path = "ensemble.json"
    submission_path = "submission.csv"
    expected_rows = 2000
    seed = 42
    img_size = 256
    n_folds = 5
    batch_size = 32
    num_workers = 0
    weight_decay = 3e-4
    max_epochs = 100
    patience = 10
    lr_patience = 3
    lr_factor = 0.5
    unet_channels = (16, 32, 64, 128)
    unet_neck = 256
    hc_iters = 100
    hc_patience = 15
    tol = 1e-6
    nested_repeats = 3


@dataclass(frozen=True)
class ModelSpec:
    name: str
    arch: str
    lr: float = 1e-4
    dropout: float = 0.0
    augment: bool = False
    mean: float = 0.5
    std: float = 0.5


MODELS = (
    ModelSpec("mobilevitv2_100", "mobilevitv2_100", augment=True),
    ModelSpec(
        "convnext_nano",
        "convnext_nano.in12k_ft_in1k",
        dropout=0.4,
        mean=0.449,
        std=0.226,
    ),
    ModelSpec("edgenext_small", "edgenext_small", augment=True),
    ModelSpec("unet", "unet", augment=True),
    ModelSpec(
        "vit_medium", "vit_medium_patch16_gap_256.sw_in12k_ft_in1k", lr=5e-5
    ),
)


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if shutil.which("nvidia-smi"):
        print("WARNING: NVIDIA GPU found but this torch build is CPU-only")
    return torch.device("cpu")


def free_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def make_dirs():
    for path in (CFG.weights_dir, CFG.oof_dir):
        os.makedirs(path, exist_ok=True)


def weight_path(spec, fold):
    return os.path.join(CFG.weights_dir, f"{spec.name}_fold{fold}.pt")


def load_metadata():
    return pd.read_csv(CFG.train_csv), pd.read_csv(CFG.eval_csv)


DEVICE = get_device()


def normalize_shadow(img, azimuth):
    img = img.convert("L")
    fill = int(np.asarray(img).mean())
    return img.rotate(
        -azimuth,
        resample=Image.Resampling.BICUBIC,
        expand=False,
        fillcolor=fill,
    )


def load_images(df, img_dir, size):
    images = np.empty((len(df), size, size), dtype=np.uint8)
    pairs = zip(df["image_id"], df["sun_azimuth_angle"])
    for i, (name, azimuth) in enumerate(pairs):
        with Image.open(os.path.join(img_dir, name)) as img:
            img = normalize_shadow(img, azimuth)
            img = img.resize((size, size), Image.Resampling.BICUBIC)
            images[i] = np.asarray(img)
    return images


def build_transform(spec, train):
    steps = []
    if train and spec.augment:
        steps += [
            transforms.ColorJitter(brightness=0.2, contrast=0.2),
            transforms.RandomAffine(degrees=0, translate=(0.03, 0.03)),
        ]
    steps += [
        transforms.ToTensor(),
        transforms.Normalize((spec.mean,), (spec.std,)),
    ]
    return transforms.Compose(steps)


class LunarDataset(Dataset):
    def __init__(self, images, labels=None, transform=None):
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = Image.fromarray(self.images[idx])
        if self.transform is not None:
            image = self.transform(image)
        if self.labels is None:
            return image
        return image, int(self.labels[idx])


def make_loader(images, labels, spec, train):
    dataset = LunarDataset(images, labels, build_transform(spec, train))
    return DataLoader(
        dataset,
        batch_size=CFG.batch_size,
        shuffle=train,
        num_workers=CFG.num_workers,
        pin_memory=DEVICE.type == "cuda",
    )


def get_class_weights(labels):
    counts = np.bincount(labels, minlength=2)
    weights = 1.0 / counts
    weights = weights / weights.sum() * len(counts)
    return torch.tensor(weights, dtype=torch.float32, device=DEVICE)


def make_folds(labels):
    splitter = StratifiedKFold(
        n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed
    )
    return list(splitter.split(np.zeros(len(labels)), labels))


def double_conv(in_ch, out_ch):
    layers = []
    for i in range(2):
        layers += [
            nn.Conv2d(
                in_ch if i == 0 else out_ch,
                out_ch,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_ch),
            nn.PReLU(),
        ]
    return nn.Sequential(*layers)


class UNetClassifier(nn.Module):
    def __init__(self, channels, neck_channels, dropout, num_classes=2):
        super().__init__()
        in_channels = (1,) + tuple(channels[:-1])
        up_in = (neck_channels,) + tuple(reversed(channels[1:]))
        up_out = tuple(reversed(channels))
        self.encoders = nn.ModuleList(
            double_conv(c_in, c_out)
            for c_in, c_out in zip(in_channels, channels)
        )
        self.down = nn.MaxPool2d(2)
        self.neck = double_conv(channels[-1], neck_channels)
        self.ups = nn.ModuleList(
            nn.ConvTranspose2d(c_in, c_out, kernel_size=2, stride=2)
            for c_in, c_out in zip(up_in, up_out)
        )
        self.decoders = nn.ModuleList(
            double_conv(2 * c_out, c_out) for c_out in up_out
        )
        self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten())
        self.drop = nn.Dropout(dropout)
        self.fc = nn.Linear(neck_channels + channels[0], num_classes)

    def forward(self, x):
        skips = []
        for encoder in self.encoders:
            x = encoder(x)
            skips.append(x)
            x = self.down(x)
        x = self.neck(x)
        pooled = [self.gap(x)]
        for up, decoder, skip in zip(self.ups, self.decoders, reversed(skips)):
            x = decoder(torch.cat([skip, up(x)], dim=1))
        pooled.append(self.gap(x))
        return self.fc(self.drop(torch.cat(pooled, dim=1)))


def build_model(spec, pretrained):
    if spec.arch == "unet":
        model = UNetClassifier(CFG.unet_channels, CFG.unet_neck, spec.dropout)
    else:
        model = timm.create_model(
            spec.arch,
            pretrained=pretrained,
            in_chans=1,
            num_classes=2,
            drop_rate=spec.dropout,
        )
    return model.to(DEVICE)


def run_epoch(model, loader, criterion, optimizer=None):
    training = optimizer is not None
    model.train(training)
    total_loss, preds, targets = 0.0, [], []
    with torch.set_grad_enabled(training):
        for images, labels in loader:
            images = images.to(DEVICE, non_blocking=True)
            labels = labels.to(DEVICE, non_blocking=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * len(labels)
            preds.append(logits.argmax(dim=1).cpu())
            targets.append(labels.cpu())
    preds = torch.cat(preds).numpy()
    targets = torch.cat(targets).numpy()
    return total_loss / len(targets), balanced_accuracy_score(targets, preds)


def predict_proba(model, loader):
    model.eval()
    probs = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            logits = model(images.to(DEVICE, non_blocking=True))
            probs.append(torch.softmax(logits, dim=1)[:, 1].cpu())
    return torch.cat(probs).numpy()


def fit(spec, train_loader, val_loader, class_weights, seed):
    seed_everything(seed)
    model = build_model(spec, pretrained=True)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.RAdam(
        model.parameters(), lr=spec.lr, weight_decay=CFG.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=CFG.lr_factor,
        patience=CFG.lr_patience,
    )
    history, best_score, best_epoch, best_state = [], -1.0, 0, None
    for epoch in range(1, CFG.max_epochs + 1):
        train_loss, train_score = run_epoch(
            model, train_loader, criterion, optimizer
        )
        val_loss, val_score = run_epoch(model, val_loader, criterion)
        scheduler.step(val_score)
        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "train_bal_acc": train_score,
                "val_loss": val_loss,
                "val_bal_acc": val_score,
            }
        )
        if val_score > best_score + CFG.tol:
            best_score, best_epoch = val_score, epoch
            best_state = copy.deepcopy(model.state_dict())
        elif epoch - best_epoch >= CFG.patience:
            break
    model.load_state_dict(best_state)
    return model, pd.DataFrame(history), best_epoch


def train_fold(spec, fold, split, images, labels, eval_images):
    train_idx, val_idx = split
    train_loader = make_loader(
        images[train_idx], labels[train_idx], spec, train=True
    )
    val_loader = make_loader(
        images[val_idx], labels[val_idx], spec, train=False
    )
    eval_loader = make_loader(eval_images, None, spec, train=False)
    class_weights = get_class_weights(labels[train_idx])
    start = time.perf_counter()
    model, history, best_epoch = fit(
        spec, train_loader, val_loader, class_weights, CFG.seed + fold
    )
    torch.save(model.state_dict(), weight_path(spec, fold))
    val_probs = predict_proba(model, val_loader)
    eval_probs = predict_proba(model, eval_loader)
    best = history.loc[history["epoch"] == best_epoch].iloc[0]
    print(
        f"{spec.name} fold {fold} | epochs={len(history)} | "
        f"best_epoch={best_epoch} | val_bal_acc={best['val_bal_acc']:.4f} | "
        f"{time.perf_counter() - start:.0f}s"
    )
    del model
    free_memory()
    return val_probs, eval_probs


def run_model(spec, folds, images, labels, eval_images):
    cache = os.path.join(CFG.oof_dir, f"{spec.name}.npz")
    if os.path.exists(cache):
        saved = np.load(cache)
        print(f"{spec.name}: loaded cached predictions")
        return saved["oof"], saved["eval"]
    oof = np.zeros(len(labels))
    eval_probs = np.zeros(len(eval_images))
    for fold, split in enumerate(folds):
        val_probs, fold_eval = train_fold(
            spec, fold, split, images, labels, eval_images
        )
        oof[split[1]] = val_probs
        eval_probs += fold_eval / len(folds)
    np.savez(cache, oof=oof, eval=eval_probs)
    return oof, eval_probs


def train_all(folds, images, labels, eval_images):
    oofs, evals = [], []
    for spec in MODELS:
        oof, eval_probs = run_model(spec, folds, images, labels, eval_images)
        oofs.append(oof)
        evals.append(eval_probs)
    names = [spec.name for spec in MODELS]
    return names, np.stack(oofs), np.stack(evals)


def predict_labels(scores, threshold):
    return (scores >= threshold).astype(int)


def best_threshold(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores)
    return float(thresholds[int(np.argmax(tpr + 1 - fpr))])


def evaluate(labels, scores):
    threshold = best_threshold(labels, scores)
    return {
        "auc": roc_auc_score(labels, scores),
        "threshold": threshold,
        "bal_acc_thr": balanced_accuracy_score(
            labels, predict_labels(scores, threshold)
        ),
    }


def report_models(names, oof, labels):
    table = pd.DataFrame(
        [{"model": n, **evaluate(labels, p)} for n, p in zip(names, oof)]
    )
    table = table.sort_values("bal_acc_thr", ascending=False)
    print(table.round(4).to_string(index=False))


def score_auc(labels, cands):
    ranks = rankdata(cands, axis=1)
    pos = labels == 1
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[:, pos].sum(axis=1) - n_pos * (n_pos + 1) / 2) / (
        n_pos * n_neg
    )


def score_bal_acc(labels, cands):
    order = np.argsort(-cands, axis=1, kind="stable")
    ranked = np.take_along_axis(cands, order, axis=1)
    is_pos = (labels == 1)[order]
    tpr = np.cumsum(is_pos, axis=1) / is_pos[0].sum()
    fpr = np.cumsum(~is_pos, axis=1) / (~is_pos[0]).sum()
    edge = np.ones(ranked.shape, dtype=bool)
    edge[:, :-1] = ranked[:, :-1] != ranked[:, 1:]
    gap = np.where(edge, tpr - fpr, -np.inf).max(axis=1)
    return (1 + gap) / 2


def rank_candidates(labels, cands):
    scores = score_bal_acc(labels, cands)
    tie_break = score_auc(labels, cands)
    order = np.lexsort((tie_break, np.round(scores, 6)))[::-1]
    return order, scores


def blend(feats, weights):
    return np.tensordot(weights, feats, axes=1)


def hill_climb(feats, labels):
    running = np.zeros(feats.shape[1])
    picks, trace = [], []
    best_score, best_step, stale = -np.inf, 0, 0
    for step in range(1, CFG.hc_iters + 1):
        order, scores = rank_candidates(labels, (running + feats) / step)
        pick = int(order[0])
        picks.append(pick)
        trace.append(scores[pick])
        running += feats[pick]
        if scores[pick] > best_score + CFG.tol:
            best_score, best_step, stale = scores[pick], step, 0
        else:
            stale += 1
            if stale >= CFG.hc_patience:
                break
    counts = np.bincount(picks[:best_step], minlength=len(feats))
    return counts / counts.sum(), picks[:best_step], trace[:best_step]


def fit_ensemble(oof, labels):
    weights = hill_climb(oof, labels)[0]
    return weights, evaluate(labels, blend(oof, weights))


def nested_cv_score(oof, labels):
    scores = []
    for repeat in range(CFG.nested_repeats):
        splitter = StratifiedKFold(
            n_splits=CFG.n_folds, shuffle=True, random_state=CFG.seed + repeat
        )
        preds = np.zeros(len(labels), dtype=int)
        for fit_idx, hold_idx in splitter.split(oof.T, labels):
            weights, stats = fit_ensemble(oof[:, fit_idx], labels[fit_idx])
            hold_scores = blend(oof[:, hold_idx], weights)
            preds[hold_idx] = predict_labels(hold_scores, stats["threshold"])
        scores.append(balanced_accuracy_score(labels, preds))
    return float(np.mean(scores)), float(np.std(scores))


def report_ensemble(names, oof, labels, weights, stats):
    uniform = np.full(len(names), 1.0 / len(names))
    soft = evaluate(labels, blend(oof, uniform))
    nested_mean, nested_std = nested_cv_score(oof, labels)
    table = pd.DataFrame({"model": names, "weight": weights})
    table = table[table["weight"] > 0].sort_values("weight", ascending=False)
    print(table.round(4).to_string(index=False))
    print(f"soft vote     OOF bal_acc = {soft['bal_acc_thr']:.4f}")
    print(f"hill climb    OOF bal_acc = {stats['bal_acc_thr']:.4f}")
    print(
        f"hill climb nested-CV bal_acc = {nested_mean:.4f} "
        f"+/- {nested_std:.4f}"
    )
    print(f"threshold = {stats['threshold']:.4f} | auc = {stats['auc']:.4f}")
    return {"soft_vote": soft, "nested_mean": nested_mean}


def save_ensemble(names, weights, threshold):
    payload = {
        "models": list(names),
        "weights": [float(w) for w in weights],
        "threshold": float(threshold),
    }
    with open(CFG.ensemble_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def load_ensemble():
    with open(CFG.ensemble_path, encoding="utf-8") as f:
        return json.load(f)


def validate_submission(submission, eval_df):
    if list(submission.columns) != ["image_id", "label"]:
        raise ValueError(f"bad columns: {list(submission.columns)}")
    if len(submission) != CFG.expected_rows:
        raise ValueError(f"expected {CFG.expected_rows} rows")
    if submission.isnull().any().any():
        raise ValueError("submission contains null values")
    if not submission["label"].isin([0, 1]).all():
        raise ValueError("labels must be 0 or 1")
    if not submission["image_id"].equals(eval_df["image_id"]):
        raise ValueError("image_id order differs from test metadata")


def make_submission(eval_df, scores, threshold, path):
    submission = pd.DataFrame(
        {
            "image_id": eval_df["image_id"],
            "label": predict_labels(scores, threshold),
        }
    )
    validate_submission(submission, eval_df)
    submission.to_csv(path, index=False)
    return submission


def report_submission(submission, labels):
    print(f"saved {CFG.submission_path}: {submission.shape}")
    print(submission["label"].value_counts().to_string())
    print(
        f"positive rate: train labels={labels.mean():.3f} | "
        f"eval predicted={submission['label'].mean():.3f}"
    )


def main():
    seed_everything(CFG.seed)
    make_dirs()
    train_df, eval_df = load_metadata()
    labels = train_df["label"].to_numpy().astype(int)
    images = load_images(train_df, CFG.train_img_dir, CFG.img_size)
    eval_images = load_images(eval_df, CFG.eval_img_dir, CFG.img_size)
    folds = make_folds(labels)
    names, oof, eval_probs = train_all(folds, images, labels, eval_images)
    report_models(names, oof, labels)
    weights, stats = fit_ensemble(oof, labels)
    summary = report_ensemble(names, oof, labels, weights, stats)
    save_ensemble(names, weights, stats["threshold"])
    submission = make_submission(
        eval_df,
        blend(eval_probs, weights),
        stats["threshold"],
        CFG.submission_path,
    )
    report_submission(submission, labels)
    print(
        f"FINAL ensemble balanced accuracy | OOF={stats['bal_acc_thr']:.4f} | "
        f"nested-CV={summary['nested_mean']:.4f}"
    )


if __name__ == "__main__":
    main()
