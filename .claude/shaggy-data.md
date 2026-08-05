# DATASET
The `data/` folder contains two PyTorch tensors, `x.pt` and `mask.pt`, used to train and evaluate 
architectures and other components of the `shaggy` package.

## `x.pt`
Tensor of shape `(N, C, Y, X)`.
Current shape: `(16, 8, 128, 256)`.
- `N` — number of samples (currently `16`)
- `C` — number of channels (currently `8`)
- `Y` — number of longitude grid points (currently `128`)
- `X` — number of latitude grid points (currently `256`)

### VARIABLES
- Channels `0-3`: temperature
- Channels `4-7`: oxygen
- Within each variable, channels are ordered by depth: channel 0 = surface, increasing depth with channel index, last channel = deepest.

### PREPROCESSING
- Data is normalized to `[0, 1]`.
- Land values are stored as `NaN`. Replace them with `0` before training.

## `mask.pt`
Use this mask to exclude land pixels from training and evaluation losses/metrics.
Tensor of shape `(8, Y, X)`, one mask per channel/level.
- `1` = valid data (sea)
- `0` = invalid (land, undefined)

## PLOTTING
The data underwent a vertical flip during preprocessing. Before plotting any level, apply `torch.flipud` to restore the correct Black Sea orientation.