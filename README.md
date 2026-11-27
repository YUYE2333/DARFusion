# DARFusion

DARFusion classifies land cover from hyperspectral and LiDAR observations under sensor degradation. Each modality routes its features to a degradation-family repair expert before cross-modal fusion.

## Setup

Install the dependencies with `pip install -r requirements.txt`. Supply the HSI and LiDAR data separately. Each dataset has a configuration in `configs/datasets/` defining its data directory, MAT filenames and array keys. For Houston2013, the default inputs are `data/Houston/HSI.mat` and `data/Houston/LiDAR.mat`. The fixed label splits are included in `splits/`.

## Train and evaluate

From the repository root, train both stages on Houston2013 with one model seed:

```bash
python train.py --config configs/experiments/houston2013.json --model-seed 1234
```

Evaluate the resulting checkpoint, replacing the example checkpoint path with the one from your training run:

```bash
python test.py --config configs/experiments/houston2013.json --checkpoint path/to/darfusion.pt --eval-profile full
```

For Trento or MUUFL, use the corresponding experiment configuration and checkpoint. The reported protocol uses the official test labels for checkpoint selection. The full evaluation covers one clean condition and 90 degraded conditions.
