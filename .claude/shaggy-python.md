# GUIDELINES
PLEASE REFER TO THIS DOCUMENT BEFORE WRITING ANY NEW CODE OR MAKING CHANGES TO EXISTING CODE. YOU SHOULD
DO NOT FORGET TO ACTIVATE THE `shaggy` ENVIRONMENT BEFORE RUNNING ANY CODE, AS ALL NECESSARY DEPENDENCIES
ARE INSTALLED THERE AND MAY NOT BE AVAILABLE IN OTHER ENVIRONMENTS. IF YOU ENCOUNTER MISSING DEPENDENCIES,
CHECK THE `pyproject.toml` FILE AND INSTALL THE REQUIRED PACKAGES IN THE `shaggy` ENVIRONMENT.

# PHILOSOPHY
Write Python code that is **clear, concise, and as simple as possible**.
Avoid over-engineering, prefer readable solutions over clever ones.
Always remember KISS: "Keep It Simple Stupid".

# BEFORE YOU START CODING
BEFORE YOU START CODING, CHECK IF WHAT YOU WANT TO IMPLEMENT ALREADY EXISTS IN THE CODEBASE.

# FUNCTIONS | RULES
- Keep functions **short and single-purpose**.
- Use **explicit variable names** that reflect the physical or mathematical meaning.
- Avoid unnecessary abstractions or complex class hierarchies.
- Prefer **built-in Python and Torch/NumPy/Xarray idioms** over custom implementations.
- Include **type hints** for function arguments and return values to improve readability and facilitate debugging.
- Include a **docstring** for every function following our personnal docstring format:

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

# FUNCTIONS | TESTING
- Use `pytest` for unit tests.
- For each new function with non-trivial logic or in a core workflow, add a `pytest` test that covers both typical and edge-case inputs.
- Base each new test on the template below, changing only the test name, inputs, and assertions for the function under test.

```python
def test_compute_hypoxic_layer_typical():
    r"""Determines if it computes the correct thickness for a typical hypoxic layer."""

    # CODE HERE

```
