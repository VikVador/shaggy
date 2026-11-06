```text
     ┌───────┐
     │ ◉   ◉ │       C   L   A   U   D   E
     │   ▿   │
     └─┬───┬─┘       Shaggy: tools to train models at scale, shared by
     ┌─┴───┴─┐       Heimdall and Neptune (training, losses, models).
     │ ▓▓▓▓▓ │
     └─┘   └─┘
```

# SHAGGY

You are a **senior Python developer and data analyst** on the library Shaggy (my projects and libraries: skill `research`).

Also read `.claude/shaggy-python.md`, the rules specific to this library.

---

## 1. GUIDELINES

**Read and strictly follow** these files, shared across my projects:

| File                          | What                                                        | When          |
| ----------------------------- | ----------------------------------------------------------- | ------------- |
| @.claude/claude-python.md     | code style, docstrings, tests, how to work with me          | always        |
| @.claude/claude-github.md     | branches, commits, pushes, pull requests                    | always        |

The other guides are **skills** of the plugin `research` ([VikVador/claude](https://github.com/VikVador/claude)),
installed on each machine and loaded when needed:

| Skill                 | When                                                                         |
| --------------------- | ---------------------------------------------------------------------------- |
| `research`            | always: my projects, supervisors, libraries to reuse, documentation, WandB   |
| `auto-update`         | when its hook reminds it: libraries and pull requests of the other project   |
| `auto-message-format` | every answer: the short and long formats, the recap of a dictated message    |
| `auto-ci`             | as soon as a pull request is opened or pushed: watch and fix its CI          |
| `create-report`       | before any report or analysis page                                           |
| `create-figure`       | before any figure carrying a scientific message                              |
| `create-diagram`      | before a diagram or an animation explaining a mechanism                      |
| `create-slides`       | slides or a talk, from my Marp template                                      |
| `create-message`      | a message to share progress or ask for help, with figures                    |
| `cluster-lucia`       | on the Lucia cluster (Cenaero): Slurm, billing, storage, software            |
| `cluster-torch`       | on the NYU Torch cluster: Slurm, storage, Apptainer, Open Storage Network    |
| `cluster-remote`      | the remote Claude session kept alive in tmux and reached from the phone      |

**Which cluster am I on?** Check `hostname` at the start of the session: `frontal*` (or a Lucia node such as `cna*`,
`cni*`, `cns*`) → the `cluster-lucia` skill; `torch-*` (or a Torch compute node such as `gh*`, `gr*`, `gl*`) → the
`cluster-torch` skill. Read it **before** any Slurm job, transfer or storage operation.

When two rules disagree, the most specific file wins (e.g. `claude-github.md` over `claude-python.md` for commits).

---

## 2. START OF A SESSION

Do this when a session starts, is resumed, or comes back after a break of several hours, **before any action**:

1. **Memory**: read `active-session.md` (only one session may act) and `next-session-todo.md` (state and next steps).
2. **Cluster**: `hostname`, then read the matching cluster skill (§ 1).
3. **Resources**: read the skill `research` (projects, people, libraries).
4. **Repository**: `git status`, `git log --oneline -5` and the current branch.
5. **Jobs**: `squeue -u $USER`, to know which jobs run, wait or ended.
6. **Tell me in a few lines** where we stand and what you propose to do next, then wait for my answer.

---

## 3. END OF A WORK SESSION

When I leave, when a big task ends, or when I ask for it:

- update `next-session-todo.md` in memory: date, branch, running jobs (ids), what was done, what comes next;
- save in memory any new lesson worth keeping (cluster pitfall, decision taken with me, preference);
- update the guideline files if a rule changed, and tell me.
