```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       GitHub: branches, commits, pushes and pull requests,
     ┌─┴───┴─┐       written so that I can review them quickly.
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# GITHUB

My repositories live under **[VikVador](https://github.com/VikVador)** (Heimdall, Neptune, shaggy, albus, …).
You may push branches and open pull requests **for me to review**. You never merge.

**Contents**

1. Rules
2. Branches
3. Commits
4. Pull requests
5. Example

---

## 1. RULES

- **Never** commit or push to `master` / `main`, never force-push, never rewrite a pushed history, never merge.
- **Ask me** before the first push of a branch and before opening a pull request.
- Before any commit: `ruff check`, `ruff format --check` and `pytest` pass (the pre-commit hook also runs).
- Never commit secrets, credentials, data, logs, checkpoints or large files; check `git status` and `git diff --staged`.
- Never read the GitHub token (`~/.config/gh/hosts.yml`) nor print it.

## 2. BRANCHES

- One branch per topic, short **kebab-case** name saying what it does: `downloading-sample-datasets`, `stride-per-level`, `sss-outliers`.
- Branch from an up-to-date `master` (`git switch master && git pull`), unless asked otherwise.

## 3. COMMITS

Format of the first line: **`<gitmoji> <Scope> - <What, in a few words>`**

| Gitmoji | When                      | Example                                                           |
| ------- | ------------------------- | ----------------------------------------------------------------- |
| 🎉      | first commit of a project | `🎉 Heimdall - Initial structure`                                 |
| ✨      | new feature               | `✨ Data and encoders - One dataset and one encoder per modality` |
| 🐛      | bug fix                   | `🐛 Encoders - Wind at 16 samples per GPU`                        |
| 🔧      | configuration, jobs       | `🔧 Encoders - GLORYS trained for 8192 updates`                   |
| ♻️      | refactoring               | `♻️ Data - Single crop function for data, masks and mesh`         |
| ✅      | tests                     | `✅ Tests - Integration tests skipped in CI`                      |
| 📝      | documentation             | `📝 Claude - Shared instruction files`                            |
| 🎨      | style, formatting         | `🎨 One blank line before every return`                           |
| ⚡️      | performance               | `⚡️ GME loss - log1p for small distances`                         |
| 🔥      | removal of code or files  | `🔥 Data - Download pipeline removed`                             |

- Optional body: a few bullet points, plain sentences.
- Last line of every commit made by Claude: `Co-Authored-By: Claude <noreply@anthropic.com>` (with the model name).

## 4. PULL REQUESTS

**Title**: same format as a commit, describing the whole pull request: `✨ Data - Datasets 2000-2025 on OSN`.

**Body**, always in this order:

```markdown
## Summary
Two to four short sentences, in simple words: what the pull request does and why.
Someone who has not followed the work must understand its goal.

## Changes
| File                           | Why it changed                                     |
| ------------------------------ | -------------------------------------------------- |
| `heimdall/data/dataset.py`     | Reads one modality at a time, in bfloat16.         |
| `tests/test_dataset.py`        | Checks the shapes, the masks and the padding.      |

## Tests
What was run and its result, e.g. `ruff` clean, `pytest` 28 passed (integration tests on Torch).

## Notes
Optional: limits, follow-ups, decisions taken with me.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```

- One row per modified file (group them only when many files change for the same reason).
- Plain, short sentences everywhere; no jargon that is not needed.
- Open the pull request with `gh pr create --base master --title "..." --body-file <file>`, then give me its link.

## 5. EXAMPLE

What a pull request following these rules looks like (the branch `downloading-sample-datasets` of Heimdall):

**Title**

```text
✨ Data and encoders - Datasets 2000-2025 on OSN, one dataset and one encoder per modality
```

**Body**

```markdown
## Summary
The ocean states (GLORYS) and the satellite observations of 2000-2025 are now on the OSN pod, and the code reads
them directly from there. Each dataset (GLORYS, temperature, salinity, wind, chlorophyll) has its own PyTorch
dataset and its own encoder, ready to be trained on the cluster.

## Changes
| File                                         | Why it changed                                                        |
| -------------------------------------------- | --------------------------------------------------------------------- |
| `heimdall/config/__init__.py`                | Points to the datasets, masks and statistics on OSN (no local copy).  |
| `heimdall/data/__init__.py`                  | Variables, splits and domain of each dataset, in one place.           |
| `heimdall/data/dataset.py`                   | Reads one dataset at a time and returns its fields and their mask.    |
| `heimdall/data/weights.py`                   | Land/sea masks, sin/cos mesh and statistics of each dataset.          |
| `heimdall/model/encoders.py`                 | A single encoder class for every dataset.                             |
| `experiments/.../configs/encoder/*.yml`      | One training configuration per dataset.                               |
| `cluster/jobs/encoder-*.sbatch`              | Jobs for GLORYS (8 H200) and the observations (lab RTX).              |
| `tests/test_dataset.py`, `test_weights.py`   | Check the shapes, the masks and the padding on real data.             |

## Tests
`ruff` clean. `pytest`: 28 passed on Torch (integration tests included), 13 passed in CI.

## Notes
- The encoders of the observations are training on the lab RTX, GLORYS waits for a full H200 node.
- The salinity shows a gap between training and validation: to look at in a next pull request.

🤖 Generated with [Claude Code](https://claude.com/claude-code)
```
