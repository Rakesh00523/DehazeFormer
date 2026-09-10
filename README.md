# Fog Removal for Autonomous Vehicles

A single, self-contained Jupyter notebook (`fogfinal.ipynb`) that documents and runs an end-to-end fog/haze removal pipeline built on **DehazeFormer-S**, aimed at restoring visibility in foggy driving footage for autonomous-vehicle perception.

The notebook doubles as both a **project report** (methodology, training config, checkpoint comparisons, cost estimate, qualitative results) and a **runnable demo** (loads the final trained checkpoint and dehazes a video/image end-to-end).

## What it does

1. **Dataset preparation** — expects RESIDE-IN in a train/test layout for fine-tuning.
2. **Model fine-tuning** — fine-tunes DehazeFormer-S for three durations (15 / 50 / 300 epochs) on 4×H200 GPUs via Modal.
3. **Evaluation** — computes PSNR/SSIM per checkpoint on the RESIDE-IN test split.
4. **Checkpoint selection** — picks the best-performing (300-epoch) checkpoint.
5. **Inference** — runs the selected model on:
   - a bundled foggy demo video (frame-by-frame dehazing with temporal smoothing + optional audio passthrough),
   - a batch of real-world hazy images from the **RTTS** dataset,
   - ad-hoc single images/videos (Colab-style cells at the end of the notebook).
6. **Reporting** — exports restored video/images, a metrics JSON, and a Markdown summary for each run.

## Results

Evaluated on the RESIDE-IN test split (500 images):

| Run        | Epochs | GPU     | Mean PSNR | Mean SSIM |
|------------|--------|---------|-----------|-----------|
| 15 epochs  | 15     | 4×H200  | 22.93     | 0.8926    |
| 50 epochs  | 50     | 4×H200  | 25.28     | 0.9111    |
| 300 epochs | 300    | 4×H200  | **27.56** | **0.9451** |

The 300-epoch checkpoint is used as the final deployed model.

## Training configuration

| Parameter          | Value           |
|---------------------|-----------------|
| Model               | DehazeFormer-S  |
| Dataset             | RESIDE-IN       |
| Optimizer           | AdamW           |
| Learning rate       | 2e-4            |
| Patch size          | 256             |
| Global batch size   | 16              |
| Eval frequency      | Every epoch     |
| Training platform   | Modal           |
| GPU allocation      | 4× H200         |

## Repository layout

Currently the repository only contains the notebook itself:

```
IOMP/
└── fogfinal.ipynb
```

Everything else — model weights, evaluation JSONs, comparison images, and the upstream DehazeFormer source — is fetched automatically the first time the notebook runs:

- **Checkpoints, eval JSONs, and comparison assets** are pulled from the Hugging Face model repo `Pranavz/minor32-fog-removal` into `runs/modal/...` and `notebooks/assets/...`.
- **Upstream DehazeFormer code** is git-cloned from [IDKiro/DehazeFormer](https://github.com/IDKiro/DehazeFormer) (pinned to a fixed commit) into `external/dehazeformer/`.
- **RTTS sample images** are pulled from the Hugging Face dataset `Leoxuhan/RTTS_YOLO` into `data/RTTS/`.
- **Notebook demo output** (restored video, metrics, report) is written to `runs/notebook_demo/`.
- **RTTS inference output** is written to `runs/rtts_inference/`.

## Requirements

- Python 3.13
- `ffmpeg` available on `PATH` (used for audio remuxing and video re-encoding)
- A CUDA GPU is recommended but not required — inference falls back to CPU automatically

Python packages (auto-installed by the notebook's first cell if missing):

```
numpy pandas matplotlib seaborn pillow opencv-python-headless
torch timm pytorch-msssim ipython huggingface_hub tqdm
```

## Running the notebook

1. Open `fogfinal.ipynb` in Jupyter/JupyterLab (or VS Code's notebook viewer).
2. Run the cells from top to bottom:
   - The first two cells install dependencies and download all required assets (checkpoints, upstream code, sample data) — this requires internet access.
   - The report/analysis section (tables, cost estimate, PSNR/SSIM plots, before/after comparisons) can be run as-is to reproduce the write-up.
   - The **Final Inference Workflow** section runs the 300-epoch checkpoint on the bundled `runs/test_assets/hazy.mp4` by default. Edit `INPUT_VIDEO` / `REFERENCE_VIDEO` in that cell to point at your own footage.
   - The **RTTS** section downloads and dehazes a small batch of real-world foggy images for a sanity check outside the training distribution.

> **Note:** The last few cells (image/video inference against `/content/...` paths) were authored for a Google Colab session and hard-code Colab file paths. Update those paths (or delete the cells) before running locally.

## Notes on cost

The notebook includes a rough GPU-cost estimate based on Modal's H200 pricing, extrapolated from the 300-epoch run's logged throughput (~0.024 GPU-hours/epoch on a 4×H200 cluster). This is for budgeting reference only, not a billed total.

## Acknowledgements

- [DehazeFormer](https://github.com/IDKiro/DehazeFormer) — backbone dehazing architecture.
- RESIDE-IN and RTTS — training and real-world evaluation datasets.
- Trained checkpoints and evaluation artifacts hosted at `Pranavz/minor32-fog-removal` on Hugging Face.
