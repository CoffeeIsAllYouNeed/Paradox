# Lunar crop classification: 5-fold stratified ensemble

Binary classification of lunar image crops. Five models are trained with
5-fold stratified cross-validation on the full training set, blended with a
hill-climbing ensemble, and thresholded on out-of-fold (OOF) predictions to
produce `submission.csv` (`image_id,label`, 2,000 rows, no nulls).

## Folder contents

| File | Purpose |
| --- | --- |
| `train.py` | Loads data, trains 5 models x 5 folds, fits the ensemble, writes weights, `ensemble.json` and `submission.csv` |
| `inference.py` | Loads the saved fold weights and `ensemble.json`, predicts the eval images and writes `submission.csv` |
| `run_all.py` | Interactive prompt / CLI that runs the stages below in the right order |
| `requirements.txt` | Python dependencies |
| `Final_Ensemble.ipynb` | Notebook version of the same pipeline (Setup, EDA, Preprocess, Model Training, Ensemble, Submission) |

## How to run

Place the data in a `data/` folder next to the scripts:

```
data/train_images/  data/eval_images/
data/train_metadata.csv  data/test_metadata.csv
```

```
pip install -r requirements.txt
python run_all.py
```

`run_all.py` opens a menu: dry run (2 folds, 1 epoch, isolated folder), full
pipeline (data check, training, inference, verification), train only,
inference only, and deliverable verification. Stages can also be run directly,
for example `python run_all.py --stage dry-run` then
`python run_all.py --stage full`. The same steps can be run by hand:

```
python train.py
python inference.py
```

`train.py` caches each model's OOF and eval predictions in `oof/`, so an
interrupted run resumes at the next unfinished model. Delete `oof/` to retrain
from scratch. Pretrained timm weights are downloaded on the first run.

## Model weights

Download link: **PASTE_DOWNLOAD_LINK_HERE**

The link must be shared as "Anyone with the link can view". Extract the
archive so the fold weights sit in `weights/` (`weights/<model>_fold<k>.pt`)
and `ensemble.json` sits next to `inference.py`, then run
`python inference.py`.

## Methodology summary

### Handling `sun_azimuth_angle`

The sun azimuth changes the direction in which shadows fall, so the same
terrain looks different under different illumination. Every image, in both
train and eval, is passed through the same shadow normalization before any
model sees it:

1. Convert to grayscale.
2. Rotate by `-sun_azimuth_angle` degrees (PIL bicubic resampling, canvas size
   unchanged), so the illumination direction is expressed relative to a fixed
   reference orientation instead of the image frame. The network therefore
   does not have to learn to be invariant to the sun azimuth on its own.
3. Fill the empty corners created by the rotation with the image's mean gray
   level, so they do not appear as dark artifacts.
4. Resize to 256 x 256 (bicubic).

Training augmentation contains no rotations or flips, so the normalized
shadow orientation is preserved. MobileViTv2, EdgeNeXt and the U-Net use
brightness/contrast jitter (0.2) and translations of up to 3%; ConvNeXt and
ViT train without augmentation.

### Models

| Model | Source | Notes |
| --- | --- | --- |
| `mobilevitv2_100` | timm, pretrained | lr 1e-4, augmentation |
| `convnext_nano.in12k_ft_in1k` | timm, pretrained | lr 1e-4, head dropout 0.4, no augmentation |
| `edgenext_small` | timm, pretrained | lr 1e-4, augmentation |
| `unet` | custom U-Net encoder/decoder classifier, from scratch | channels 16-32-64-128, neck 256, augmentation |
| `vit_medium_patch16_gap_256.sw_in12k_ft_in1k` | timm, pretrained | lr 5e-5, no augmentation |

The five models are the ones selected by the hill-climb ensemble experiment.
All use a single input channel, RAdam with weight decay 3e-4, batch size 32,
and class-weighted cross-entropy.

### Training and validation

- 5-fold stratified split (seed 42) over the full training set.
- Up to 100 epochs per fold with early stopping (patience 10) on validation
  balanced accuracy, and `ReduceLROnPlateau` (factor 0.5, patience 3). The
  best-epoch checkpoint of each fold is saved.
- Each sample gets an OOF prediction from the one model that did not train on
  it. Eval predictions are the mean over the 5 fold models.

### Ensemble

Hill-climbing (with replacement) over the five models' OOF probabilities
optimizes balanced accuracy, with AUC as tie-break. The resulting weights are
applied to the fold-averaged eval probabilities, and the decision threshold
is the Youden-optimal cut on the blended OOF scores. `train.py` also reports
a nested-CV estimate of the ensemble (weights and threshold refit inside each
split) as a less optimistic score.


cd /d C:\Users\vm\Desktop\Paradox_Folder && python -m venv .venv && call .venv\Scripts\activate && python -m pip install --upgrade pip && pip install torch torchvision --index-url https://download.pytorch.org/whl/cu126 && pip install -r requirements.txt && python -c "import torch, timm; assert torch.cuda.is_available(), 'CUDA not available'; print('torch', torch.__version__, '| CUDA OK')" && python run_all.py