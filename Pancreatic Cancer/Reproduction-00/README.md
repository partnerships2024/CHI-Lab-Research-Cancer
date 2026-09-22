# Pancreatic Tumor Segmentation (PanTS)

A deep learning project for automatic segmentation of pancreatic tumors from medical imaging scans using a U-Net architecture with batch normalization and Dice loss.

## Description

This project implements a U-Net convolutional neural network for pixel-wise segmentation of pancreatic tumors. The model processes 2D medical images and produces binary masks that highlight tumor regions, assisting in pancreatic cancer diagnosis and treatment planning.

### Key Features
- U-Net architecture with BatchNormalization and 1024-filter bottleneck
- Dice loss function optimized for segmentation
- Data augmentation for improved generalization
- Comprehensive evaluation with F1, IoU, Dice, Recall, and Precision metrics
- Reproducible training with fixed random seed

## Dataset

Data source: [PanTS Dataset (Google Drive)](https://drive.google.com/file/d/1LbpsDu2MkSJpzq5pIAEm6Z0TAVZO8162/view?usp=drive_link)

The dataset contains pancreatic medical images with corresponding expert-annotated binary masks. Ground truth masks distinguish tumor tissue (label 1) from background (label 0).

## Model Architecture

The U-Net consists of:

**Encoder (Contraction Path)**
- 4 encoder blocks, each with two Conv2D + BatchNormalization + ReLU followed by MaxPooling
- Filter sizes: 64, 128, 256, 512

**Bottleneck**
- Two Conv2D + BatchNormalization + ReLU layers with 1024 filters

**Decoder (Expansion Path)**
- 4 decoder blocks, each with Conv2DTranspose + skip connection + two Conv2D + BatchNormalization + ReLU
- Filter sizes: 512, 256, 128, 64

**Output**
- 1×1 Conv2D with sigmoid activation for binary classification

### Training Configuration
- Image size: 256×256×3
- Batch size: 16
- Optimizer: Adam (learning rate 1e-4)
- Loss: Dice loss
- Epochs: 20
- Callbacks: ModelCheckpoint, ReduceLROnPlateau, CSVLogger, EarlyStopping

## Installation

```bash
pip install -r requirements.txt
```

Or install manually:

```bash
pip install tensorflow numpy pandas opencv-python scikit-learn tqdm matplotlib
```

## Dataset Preparation

Place your dataset in the following structure:

```
data/
├── images/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── masks/
    ├── image_001_mask.png
    ├── image_002_mask.png
    └── ...
```

Each image must have a corresponding mask file with the `_mask` suffix. Images are RGB (3 channels), masks are grayscale.

## Training

```bash
python train.py
```

This trains the U-Net model on your dataset and saves:
- Best model checkpoint: `checkpoints/model.h5`
- Training log: `results/log.csv`
- Training curves: `results/training_history.png`

To customize training, edit the hyperparameters at the top of `train.py`.

## Testing and Evaluation

```bash
python test.py
```

This loads the trained model, runs predictions on the test set, and produces:

- `results/summary_metrics.csv` — overall metrics (F1, IoU, Dice, Recall, Precision)
- `results/scores.csv` — per-image metrics
- `results/metrics_barplot.png` — bar chart of overall metrics
- `results/sample_predictions.png` — side-by-side comparison of input, ground truth, and prediction
- `results/predictions/` — individual prediction images

## Evaluation Metrics

The model is evaluated using:

| Metric    | Description                                    |
|-----------|------------------------------------------------|
| Dice      | Overlap between prediction and ground truth    |
| F1 Score  | Harmonic mean of precision and recall          |
| IoU       | Intersection over Union (Jaccard index)        |
| Recall    | Sensitivity — fraction of tumor pixels found   |
| Precision | Fraction of predicted tumor pixels that are correct |

Pixel accuracy is not the primary metric due to class imbalance in tumor segmentation.

## In-Distribution Results

After running `test.py`, the results are saved to `results/summary_metrics.csv`. Typical format:

| Metric    | Value  |
|-----------|--------|
| Dice      | 0.XXXX |
| F1        | 0.XXXX |
| IoU       | 0.XXXX |
| Recall    | 0.XXXX |
| Precision | 0.XXXX |

## Project Files

| File | Purpose |
|------|---------|
| `train.py` | Trains the U-Net model and saves the best checkpoint |
| `test.py` | Evaluates the trained model and generates in-distribution results |
| `README.md` | Project documentation |
| `checkpoints/model.h5` | Saved model checkpoint (after training) |
| `results/` | Output metrics, plots, and prediction images (after testing) |

## Requirements

- Python 3.8 or higher
- TensorFlow 2.10 or higher
- NumPy, Pandas, OpenCV
- Scikit-learn, tqdm, Matplotlib

GPU is recommended for training but not required.

## Notes

- Input images are resized to 256×256 before feeding into the model.
- Masks are binarized during training and evaluation.
- If a CUDA out-of-memory error occurs, reduce `BATCH_SIZE` in `train.py`.
- For models larger than 100 MB, use Git LFS or GitHub Releases instead of committing directly.

## Acknowledgements

- U-Net: Ronneberger et al., 2015 — [https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)
- Dataset: PanTS dataset providers

## License

MIT License.


