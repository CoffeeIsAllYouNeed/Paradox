import argparse
import os
import shutil
import time
import traceback

import pandas as pd

import inference
import train
from train import CFG

DRY_DIR = "dry_run"
DELIVERABLES = ("README.md", "requirements.txt", "train.py", "inference.py")
LINK_PLACEHOLDER = "PASTE_DOWNLOAD_LINK_HERE"
TRAIN_COLUMNS = ("image_id", "label", "sun_azimuth_angle")
EVAL_COLUMNS = ("image_id", "sun_azimuth_angle")


def missing_images(df, img_dir):
    return [
        name
        for name in df["image_id"]
        if not os.path.exists(os.path.join(img_dir, name))
    ]


def check_columns(df, columns, name):
    absent = [col for col in columns if col not in df.columns]
    if absent:
        raise ValueError(f"{name} metadata is missing columns: {absent}")


def check_data():
    required = (
        CFG.train_img_dir,
        CFG.eval_img_dir,
        CFG.train_csv,
        CFG.eval_csv,
    )
    missing = [path for path in required if not os.path.exists(path)]
    if missing:
        raise FileNotFoundError(f"missing data paths: {missing}")
    train_df, eval_df = train.load_metadata()
    check_columns(train_df, TRAIN_COLUMNS, "train")
    check_columns(eval_df, EVAL_COLUMNS, "eval")
    if len(eval_df) != CFG.expected_rows:
        raise ValueError(
            f"eval has {len(eval_df)} rows, expected {CFG.expected_rows}"
        )
    absent = missing_images(train_df, CFG.train_img_dir)
    absent += missing_images(eval_df, CFG.eval_img_dir)
    if absent:
        raise FileNotFoundError(
            f"{len(absent)} images not found, e.g. {absent[:3]}"
        )
    print(f"data ok: train={len(train_df)} eval={len(eval_df)}")


def reproduce():
    reference = pd.read_csv(CFG.submission_path)
    inference.main()
    rebuilt = pd.read_csv(CFG.submission_path)
    mismatches = int((reference["label"] != rebuilt["label"]).sum())
    print(f"inference vs train.py label mismatches: {mismatches}")
    if mismatches:
        print("WARNING: inference.py did not reproduce train.py exactly")


def verify_deliverables():
    problems, todo = [], []
    required = DELIVERABLES + (CFG.ensemble_path, CFG.submission_path)
    for name in required:
        if not os.path.exists(name):
            problems.append(f"missing file: {name}")
    expected = len(train.MODELS) * CFG.n_folds
    found = 0
    if os.path.isdir(CFG.weights_dir):
        found = len(os.listdir(CFG.weights_dir))
    if found != expected:
        problems.append(f"expected {expected} weight files, found {found}")
    if os.path.exists(CFG.submission_path):
        submission = pd.read_csv(CFG.submission_path)
        eval_df = pd.read_csv(CFG.eval_csv)
        try:
            train.validate_submission(submission, eval_df)
        except ValueError as error:
            problems.append(f"submission invalid: {error}")
    if os.path.exists("README.md"):
        with open("README.md", encoding="utf-8") as f:
            if LINK_PLACEHOLDER in f.read():
                todo.append("paste the weights download link into README.md")
    for line in problems:
        print("PROBLEM:", line)
    for line in todo:
        print("TODO:", line)
    if not problems:
        print("deliverable checks passed")


def run_train():
    check_data()
    train.main()


def run_inference():
    check_data()
    inference.main()


def run_pipeline():
    check_data()
    train.main()
    reproduce()
    verify_deliverables()


def dry_run():
    check_data()
    shutil.rmtree(DRY_DIR, ignore_errors=True)
    os.makedirs(DRY_DIR)
    patch = {
        "max_epochs": 1,
        "n_folds": 2,
        "weights_dir": os.path.join(DRY_DIR, "weights"),
        "oof_dir": os.path.join(DRY_DIR, "oof"),
        "ensemble_path": os.path.join(DRY_DIR, "ensemble.json"),
        "submission_path": os.path.join(DRY_DIR, "submission.csv"),
    }
    saved = {key: getattr(CFG, key) for key in patch}
    for key, value in patch.items():
        setattr(CFG, key, value)
    try:
        train.main()
        reproduce()
    finally:
        for key, value in saved.items():
            setattr(CFG, key, value)
    shutil.rmtree(DRY_DIR)
    print("dry run passed, temporary files removed")


STAGES = {
    "dry-run": ("Dry run (2 folds, 1 epoch, isolated folder)", dry_run),
    "full": ("Full pipeline (check, train, inference, verify)", run_pipeline),
    "train": ("Train only (train.py)", run_train),
    "inference": ("Inference only (inference.py)", run_inference),
    "verify": ("Verify deliverables", verify_deliverables),
}


def prompt_stage():
    keys = list(STAGES)
    print()
    for i, key in enumerate(keys, 1):
        print(f"[{i}] {STAGES[key][0]}")
    print("[q] Quit")
    choice = input("Select: ").strip().lower()
    if choice in ("", "q"):
        return None
    if choice.isdigit() and 1 <= int(choice) <= len(keys):
        return keys[int(choice) - 1]
    if choice in STAGES:
        return choice
    print(f"Unknown choice: {choice}")
    return ""


def run_stage(key):
    start = time.perf_counter()
    try:
        STAGES[key][1]()
    except Exception:
        traceback.print_exc()
        return False
    print(f"{key} finished in {time.perf_counter() - start:.0f}s")
    return True


def main():
    parser = argparse.ArgumentParser(description="Run the pipeline stages")
    parser.add_argument("--stage", choices=list(STAGES))
    args = parser.parse_args()
    if args.stage:
        raise SystemExit(0 if run_stage(args.stage) else 1)
    while True:
        key = prompt_stage()
        if key is None:
            break
        if key:
            run_stage(key)


if __name__ == "__main__":
    main()
