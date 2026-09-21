# VGGT Evaluation

This repository contains code to reproduce the evaluation results presented in the VGGT paper.

## TD-HOMES (FP_Stru3D, TD_HM3D, and TD_Mansion)

`eval_td_homes.py` evaluates VGGT camera pose using the exact LO, normal, and
NO sequence maps from `Pi3_early_fusion`. For every sample the top-down image
is prepended to the mapped perspective images, so it is VGGT input/reference
index 0. Only calibrated perspective-pair predictions are scored. Results,
raw pair errors, per-sequence rows, a reproducibility manifest, and resumable
per-sample predictions are written below `evals/`.

The TD files follow the active Pi3 dataset configs exactly: FP_Stru3D uses
`rgb_td/rgb_td_mirror.png`, while TD_HM3D and TD_mansion use
`rgb_td/rgb_td.png`. Missing configured TD files are treated as errors; the
evaluator does not silently substitute another render.
The few one-perspective samples retained by Pi3's
`allow_partial_no_overlap_eval` policy are inferred and saved, but correctly
contribute zero perspective-pair errors.

Create the isolated lowercase environment requested for this evaluation:

```bash
bash scripts/create_vggt_env.sh
```

Validate all 1,629 benchmark samples and their TD-reference ordering without
loading the model:

```bash
conda run -n vggt python evaluation/eval_td_homes.py --dry-run
```

Run the complete evaluation (the model id is downloaded by Hugging Face on the
first run):

```bash
conda run -n vggt python evaluation/eval_td_homes.py \
  --model facebook/VGGT-1B \
  --output-dir evals/relpose-angular_VGGT
```

To split the complete run deterministically over several GPUs while sharing the
same resumable sample store, use the coordinator script:

```bash
TDHOMES_GPUS="0 1 2 3" bash scripts/eval_td_homes.sh
```

By default, missing local `data/` is resolved to the data symlink beside the
reference `Pi3_TD_Fusion` checkout. Use `--data-root`, `--pi3-root`, or
`--map-dir` to override those locations. For a quick end-to-end check, add
`--max-sequences 1`; this evaluates one sequence per dataset and overlap and
marks the generated metrics as partial. Existing per-sample `.npz` files are
resumed unless `--overwrite` is passed.

### Multi-view reconstruction

`eval_mv_recon_td_homes.py` evaluates the same 1,629 mapped samples with the
Pi3 multi-view reconstruction setting: width 512, dataset-specific depth
filtering and camera calibration, pixel-corresponding Umeyama Sim(3),
point-to-point ICP with a 0.1 m threshold, and bidirectional nearest-neighbor
Accuracy, Completion, and normal-consistency metrics. The TD image remains
VGGT input/reference index 0 and is excluded from GT point scoring.

Validate the full sample set without inference:

```bash
conda run -n vggt python evaluation/eval_mv_recon_td_homes.py --dry-run
```

Run on multiple GPUs with resumable per-sample JSON outputs:

```bash
TDHOMES_GPUS="0 1 2 3" bash scripts/eval_mv_recon_td_homes.sh
```

The default output is `evals/mv-recon_VGGT`. As in Pi3, aligned prediction and
GT PLY files plus RGB grids are saved by default. Add
`TDHOMES_EXTRA_ARGS="--no-save-pointclouds"` when only metrics are needed.

For the perspective-only ablation, add `--no-td` and use a distinct output
directory. This keeps the same Pi3 maps, GT preprocessing, alignment, and
metrics while omitting the TD image from the VGGT input:

```bash
TDHOMES_GPUS="0 1" \
TDHOMES_OUTPUT="evals/relpose-angular_VGGT_no-TD" \
TDHOMES_EXTRA_ARGS="--no-td" \
bash scripts/eval_td_homes.sh

TDHOMES_GPUS="2 3" \
TDHOMES_OUTPUT="evals/mv-recon_VGGT_no-TD" \
TDHOMES_EXTRA_ARGS="--no-td" \
bash scripts/eval_mv_recon_td_homes.sh
```

## Table of Contents

- [Camera Pose Estimation on Co3D](#camera-pose-estimation-on-co3d)
  - [Model Weights](#model-weights)
  - [Setup](#setup)
  - [Dataset Preparation](#dataset-preparation)
  - [Running the Evaluation](#running-the-evaluation)
  - [Expected Results](#expected-results)
- [Checklist](#checklist)

## Camera Pose Estimation on Co3D

### Model Weights

We have addressed a minor bug in the publicly released checkpoint related to the TrackHead configuration. Specifically, the `pos_embed` flag was incorrectly set to `False`. The following checkpoint incorporates this fix by fine-tuning the tracker head with `pos_embed` as `True` while preserving all other parameters. This fix will be merged into the main branch in a future update.

```bash
wget https://huggingface.co/facebook/VGGT_tracker_fixed/resolve/main/model_tracker_fixed_e20.pt
```

Note: The default checkpoint remains functional, though you may observe a slight performance decrease (approximately 0.3% in AUC@30) when using Bundle Adjustment (BA). If using the default checkpoint, ensure you set `pos_embed` to `False` for the TrackHead. This modification only affects tracking-based evaluations and has no impact on feed-forward estimation performance, as tracking is not utilized in the feed-forward approach.

### Setup

Install the required dependencies:

```bash
# Install VGGT as a package
pip install -e .

# Install evaluation dependencies
pip install pycolmap==3.10.0 pyceres==2.3

# Install LightGlue for keypoint detection
git clone https://github.com/cvg/LightGlue.git
cd LightGlue
python -m pip install -e .
cd ..
```

### Dataset Preparation

1. Download the Co3D dataset from the [official repository](https://github.com/facebookresearch/co3d)

2. Preprocess the dataset (approximately 5 minutes):
```bash
python preprocess_co3d.py --category all \
    --co3d_v2_dir /YOUR/CO3D/PATH \
    --output_dir /YOUR/CO3D/ANNO/PATH
```

   Replace `/YOUR/CO3D/PATH` with the path to your downloaded Co3D dataset, and `/YOUR/CO3D/ANNO/PATH` with the desired output directory for the processed annotations. Note that the processed data here uses the PyTorch3D camera convention, while the annotation files we provided for training on Hugging Face have already been converted to the OpenCV convention.



### Running the Evaluation

Choose one of these evaluation modes:

```bash
# Standard VGGT evaluation
python test_co3d.py \
    --model_path /YOUR/MODEL/PATH \
    --co3d_dir /YOUR/CO3D/PATH \
    --co3d_anno_dir /YOUR/CO3D/ANNO/PATH \
    --seed 0

# VGGT with Bundle Adjustment
python test_co3d.py \
    --model_path /YOUR/MODEL/PATH \
    --co3d_dir /YOUR/CO3D/PATH \
    --co3d_anno_dir /YOUR/CO3D/ANNO/PATH \
    --seed 0 \
    --use_ba
```




### Expected Results

#### Quick Evaluation
Full evaluation on Co3D can take a long time. For faster trials, you can run with ```--fast_eval```. This does exactly the same but limiting to evaluate over at most 10 sequence per category.

Use `--fast_eval` to test on a subset of data (max 10 sequences per category):

- Feed-forward estimation:
  - AUC@30: 89.98
  - AUC@15: 83.89
  - AUC@5: 67.45
  - AUC@3: 56.65

- With Bundle Adjustment (`--use_ba`):
  - AUC@30: 90.52
  - AUC@15: 85.08
  - AUC@5: 70.69
  - AUC@3: 61.32

#### Full Evaluation

- Feedforward estimation achieves a Mean AUC@30 of 89.5% (slightly higher than the 88.2% reported in the paper due to implementation differences)
- With Bundle Adjustment, you can expect a Mean AUC@30 between 90.5% and 92.5%

> **Note:** For simplicity, this script did not optimize the inference speed, so timing results may differ from those reported in the paper. For example, when using ba, keypoint extractor models are re-initialized for each sequence rather than being loaded once.
