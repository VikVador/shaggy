```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       Python: how to write, test and deliver code
     ┌─┴───┴─┐       in my projects (Heimdall at NYU, Neptune at ULiège).
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# PYTHON

Read this file **before writing or changing any code**. Libraries and projects to know: the skill `research`.

**Contents**

1. Before coding
2. Philosophy
3. Functions
4. Docstrings
5. Comments
6. Style
7. Data and deep learning
8. Tests
9. Configurations, notebooks and jobs
10. Working with me

---

## 1. BEFORE CODING

- **Activate the project environment** (e.g. `heimdall`, `neptune`) before running anything: every dependency is in it.
  A missing package goes into `pyproject.toml` first, then into the environment.
- **Check what already exists** in the codebase, then in my libraries (`shaggy`, `albus`) and in `azula` / `zuko`,
  before writing anything new. Reuse and extend rather than rewrite.
- **Read the surrounding code** and write like it: same names, same structure, same comment density.
- For a non-trivial change, **propose a short plan first** and wait for my answer when a choice is mine to make.

## 2. PHILOSOPHY

- **KISS**: clear, concise, as simple as possible. Readable beats clever.
- **No over-engineering**: no abstraction, class hierarchy, option or feature that was not asked for.
- **Minimal changes**: touch only what the task needs, keep the rest of the code as it is.
- **Uniform**: same notations, names and patterns across functions, files and projects.
- **Built-in idioms first**: Python, NumPy, PyTorch, xarray, before any custom implementation.

## 3. FUNCTIONS

- Short and single-purpose.
- **Explicit names** reflecting the physical or mathematical meaning (`thickness`, `mask_ocean`, `lat_channels`).
- **Type hints** on every argument and return value.
- Security checks at the top of the function, with explicit messages:

```python
assert window > 0, f"ERROR - window must be positive, got {window}."
raise ValueError("ERROR - The format is incorrect, it should be YYYY-MM-DD.")
```

## 4. DOCSTRINGS

Every function, class and method has a docstring in **exactly** this format:

- `r"""` then **one line**: a clear, concise summary (classes start with "Creates ...").
- A blank line, then **directly** `Arguments:` and `Returns:` (no other paragraph in between).
- Names aligned on the colon, units in brackets, shapes in parentheses.
- A "why" that seems worth keeping goes into a short code comment, not into the docstring.

```python
def compute_hypoxic_layer(depths: np.ndarray, dox: np.ndarray, bathymetry: float) -> float:
    r"""Compute the thickness of the bottom hypoxic layer for a single profile.

    Arguments:
        depths     : Observation depths for the profile, sorted in ascending order [m].
        dox        : Dissolved oxygen values for the profile [µmol/kg].
        bathymetry : Seafloor depth at the profile location [m].

    Returns:
        thickness : Thickness of the bottom hypoxic layer [m].
    """
```

## 5. COMMENTS

- Only for the parts that are a bit complex: **one simple, clear sentence**.
- **Never** the `# Topic | explanation` style with a pipe separator.

```python
# Each worker opens the store at its first sample        ✓
# Store | Opened by each worker at its first sample      ✗
```

## 6. STYLE

- **Ruff** (version pinned in `pyproject.toml`, line length 99): `ruff check` and `ruff format` must pass.
- Aligned tables of constants or configurations between `# fmt: off` markers:

```python
# fmt: off
#
# ----- Domain
#
DATASET_REGION = {
    "glorys" : {"x": slice(0, 4320), "y": slice(25, 2041)},
    "sst"    : {"x": slice(0, 3600), "y": slice(0,  1600)},
}
```

- Constants in `UPPER_CASE` in the `__init__.py` of their module, paths in the `config` module.
- `pathlib.Path` for paths, f-strings for text.
- Code, comments and docstrings in **English**.

## 7. DATA AND DEEP LEARNING

- **Never load a whole dataset**: open lazily (xarray, zarr), select, then load only what is needed.
- **float32**, never float64 (watch xarray decoding packed int16 into float64); bfloat16 for the inputs of the
  models when the precision allows it.
- **Measure before launching**: memory and time of one step on a small test job before any big training.
- **Check numerically sensitive calls** (e.g. `torch.cdist` switches to an imprecise formula above 25 samples).
- PyTorch `DataLoader` reading remote stores (s3fs): spawned workers (`multiprocessing_context="spawn"`).
- Masks, meshes and statistics are computed once, never in the training loop.

## 8. TESTS

- `pytest` for every new function with non-trivial logic: a typical case and an edge case.
- Tests needing the datasets are marked `@pytest.mark.integration` (skipped in CI).
- Template (change only the name, the inputs and the assertions):

```python
def test_compute_hypoxic_layer_typical():
    r"""Determines if it computes the correct thickness for a typical hypoxic layer."""

    # CODE HERE

```

- Before saying that something works: **run** ruff and the tests (on a compute node on a cluster), and show
  the result.

## 9. CONFIGURATIONS, NOTEBOOKS AND JOBS

- Experiments are driven by `.yml` files: every value is a list (grid search), a list value is therefore wrapped
  twice, e.g. `"hid_channels" : [[8, 16, 32]]`. Same order of keys in every file of an experiment.
- Notebooks are for exploration and checks: short cells, plotting helpers defined once.
- Cluster jobs: only the `#SBATCH` configuration and the command, see the HPC files.

## 10. WORKING WITH ME

- Answer **in French**, clearly, simply and concisely; tables when comparing options; numbers rather than adjectives.
- Give a recommendation, not a list of every possibility.
- **Ask before** anything destructive or visible to others: deleting data, cancelling jobs, pushing, opening PRs.
- **Never read** credential files (`~/.config/rclone/rclone.conf`, `~/.aws/credentials`, `~/.config/gh/hosts.yml`,
  tokens, `.netrc`), and never write keys or tokens in a repository.
- Say what was verified and what was not; never present an estimate as a measurement.
