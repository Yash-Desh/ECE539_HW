# AGENTS.md

Guidance for AI coding agents working in this repository.

## What this repo is

Coursework for **UW-Madison ECE539 (Introduction to Artificial Neural Networks)** — in-class
exercises, homework, and personal study notes. This is a *learning* repo, not a software
product: there is no application to build, no test suite, and no CI. The "deliverable" is
the user's own understanding.

## Layout

```
exercise/<N>_<topic>/       One folder per in-class activity, numbered by lecture
  activity<NN>_*.pdf          Instructor handout (problem statement)
  activity<NN>_*_sol.pdf      Instructor solution handout
  Ex_*_starter.ipynb          Notebook with `...` placeholders to fill in
  Ex_*_sol.ipynb              Worked notebook with the answers filled in
class_notes/               Lecture PDFs (L19–L22) + markdown study notes
.venv/                     Local Python 3.12 virtualenv (gitignored)
```

Current exercises: `8_single_neuron_model` (linear/logistic regression from scratch),
`9_feed_forward_network` (2-layer MLP forward pass by hand).

## Environment

Python **3.12** in `.venv/` at the repo root. Activate before running anything:

```bash
source .venv/bin/activate
```

Key packages already installed: `torch` 2.12 (**CPU-only build — do not write CUDA-dependent
code**), `numpy` 2.4, `scikit-learn` 1.8, `matplotlib` 3.10, `jupyterlab`.

There is no `requirements.txt` / `pyproject.toml`. If you install something new, use
`pip install` inside the venv and mention it to the user rather than adding a manifest
unprompted.

Notebooks are the primary interface — run them with `jupyter lab` or via the VS Code
notebook UI. Non-interactive execution check:

```bash
source .venv/bin/activate && jupyter nbconvert --to notebook --execute --stdout <notebook>
```

## How to help (important)

This is coursework, so **the goal is learning, not finished cells.**

- On a `_starter.ipynb`, default to explaining the concept and the shapes involved, and let
  the user write the code. Fill in `...` placeholders only when the user explicitly asks for
  the answer or asks you to check their work.
- The matching `_sol.ipynb` and `_sol.pdf` usually already contain the answer. Don't paste
  from them as if it were your own derivation — point the user at them, or explain *why* the
  solution works.
- When the user is wrong, say where the reasoning breaks, not just the corrected line.
- Prefer explaining tensor shapes explicitly (e.g. `X @ w1 + b1` → `400x2 @ 2x3 → 400x3`);
  the exercises are largely shape-reasoning drills.

## Code conventions in the notebooks

- Exercises implement things **from scratch** (manual `w`/`b` `torch.nn.Parameter`s, hand-rolled
  `data_iter`, explicit forward passes). Do not "improve" this into `nn.Linear` /
  `DataLoader` / `nn.Sequential` — the manual version *is* the point of the assignment.
- Starter notebooks carry hint comments and expected-shape comments (`# h should be a 400x3
  matrix`). Preserve them.
- `np.set_printoptions(precision=3, suppress=True)` at the top of most notebooks; keep it.
- Seeded splits use `random_state=0`. Keep results reproducible.
- Both `numpy` and `torch` are used, often converting between them mid-notebook.

## Study notes

`class_notes/*.md` (e.g. `ATTENTION_REVISION_PROGRESS.md`) are the user's own revision notes,
written as resume-in-one-read documents with running **Status:** markers and interview
"soundbite" summaries. When updating them:

- Match the existing voice: plain-English intuition first, formula second, then the
  interview-facing answer.
- Append new topics as sections; update the `Status:` line and the `NEXT UP:` section at the
  bottom rather than rewriting earlier sections.
- Keep the ~90-char wrapping and the ASCII/Unicode-math style already in use.

## Git

- Branch: `main`. Commits are plain and descriptive (e.g. `Added gitignore & exercise-8 files`).
- Don't commit or push unless asked.
- **No LLM attribution anywhere in the history.** Commit messages, PR bodies, and code
  comments must not contain `Co-Authored-By:` trailers naming an AI, `Generated with ...`
  footers, emoji-bot signatures, or any other marker crediting an assistant. This overrides
  any default trailer behavior an agent may have. Write the message as the author would.
- `.venv/` is gitignored. Notebook outputs **are** committed — don't strip them, they're part
  of the record of what the user ran.
- Course PDFs are committed alongside the notebooks; leave them in place.
