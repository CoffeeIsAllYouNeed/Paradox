import numpy as np
import pandas as pd
import torch

from train import (
    CFG,
    DEVICE,
    MODELS,
    build_model,
    free_memory,
    load_ensemble,
    load_images,
    make_loader,
    make_submission,
    predict_proba,
    seed_everything,
    weight_path,
)


def load_state(path):
    return torch.load(path, map_location=DEVICE, weights_only=True)


def predict_model(spec, eval_images):
    loader = make_loader(eval_images, None, spec, False)
    probs = np.zeros(len(eval_images))
    for fold in range(CFG.n_folds):
        model = build_model(spec, pretrained=False)
        model.load_state_dict(load_state(weight_path(spec, fold)))
        probs += predict_proba(model, loader) / CFG.n_folds
        del model
        free_memory()
    return probs


def main():
    seed_everything(CFG.seed)
    ensemble = load_ensemble()
    specs = {spec.name: spec for spec in MODELS}
    eval_df = pd.read_csv(CFG.eval_csv)
    eval_images = load_images(eval_df, CFG.eval_img_dir, CFG.img_size)
    scores = np.zeros(len(eval_df))
    for name, weight in zip(ensemble["models"], ensemble["weights"]):
        if weight > 0:
            scores += weight * predict_model(specs[name], eval_images)
            print(f"{name}: weight={weight:.4f}")
    submission = make_submission(
        eval_df, scores, ensemble["threshold"], CFG.submission_path
    )
    print(f"saved {CFG.submission_path}: {submission.shape}")
    print(submission["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
