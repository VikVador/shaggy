# DATASET
The `data/` folder holds two dataset pairs (tensor `x`, mask `m`), used to train and evaluate
architectures and other components of the `shaggy` package.

## `x.pt` / `m.pt` — main dataset
Tensor `x.pt` of shape `(N, C, Y, X)`.
Current shape: `(256, 20, 128, 256)`.
- `N` — number of samples (currently `256`)
- `C` — number of channels (currently `20`)
- `Y` — number of longitude grid points (currently `128`)
- `X` — number of latitude grid points (currently `256`)

### VARIABLES
4 variables, each on 5 depth levels (`4 * 5 = 20` channels):
- Channels `0-4`: temperature
- Channels `5-9`: salinity
- Channels `10-14`: oxygen
- Channels `15-19`: chlorophyll
- Within each variable, channels are ordered by depth: first channel = surface, increasing depth with channel index, last channel = deepest.

### PREPROCESSING
- Data is standardized per variable (not bounded to `[0, 1]`); ranges differ across variables.
- Land values are stored as `NaN`. Replace them with `0` before training.

Tensor `m.pt` of shape `(20, Y, X)`, one mask per channel/level.
Use this mask to exclude land pixels from training and evaluation losses/metrics.
- `1` = valid data (sea)
- `0` = invalid (land, undefined)

## `x-dummy.pt` / `m-dummy.pt` — small dummy dataset
Tensor `x-dummy.pt` of shape `(N, C, Y, X)`.
Current shape: `(16, 8, 128, 256)`.
- `N` — number of samples (currently `16`)
- `C` — number of channels (currently `8`)
- `Y` — number of longitude grid points (currently `128`)
- `X` — number of latitude grid points (currently `256`)

### VARIABLES
2 variables, each on 4 depth levels (`2 * 4 = 8` channels):
- Channels `0-3`: temperature
- Channels `4-7`: oxygen
- Within each variable, channels are ordered by depth: first channel = surface, increasing depth with channel index, last channel = deepest.

### PREPROCESSING
- Data is normalized to `[0, 1]`.
- Land values are stored as `NaN`. Replace them with `0` before training.

Tensor `m-dummy.pt` of shape `(8, Y, X)`, one mask per channel/level.
Use this mask to exclude land pixels from training and evaluation losses/metrics.
- `1` = valid data (sea)
- `0` = invalid (land, undefined)

## PLOTTING
The data underwent a vertical flip during preprocessing. Before plotting any level, apply `torch.flipud` to restore the correct Black Sea orientation.
