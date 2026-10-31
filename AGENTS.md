# AGENTS.md

This workspace is a small collection of standalone Python machine-learning exercises. The files are intended to be run directly as scripts, not as part of a larger package.

## Project shape

- Each `.py` file is an independent experiment or assignment.
- There is no build system, package entry point, or test suite configured at the workspace root.
- Most scripts rely on NumPy, Matplotlib, and scikit-learn directly.

## Working conventions

- Preserve the existing direct-run style of the scripts.
- Prefer small, localized edits inside one file instead of introducing new project structure.
- Reuse the current imports and naming style when adding code to a script.
- If a script is a teaching example, keep it readable and explicit rather than over-engineering it.

## Common execution pattern

Run a script with:

```bash
python <script-name>.py
```

Examples:

- `python svm.py`
- `python dt.py`
- `python bagging.py`
- `python nueral.py`

## Notable files

- `svm.py`: scikit-learn SVM example with plotting.
- `dt.py`: decision tree implementation exercise.
- `bagging.py`: bootstrap sample generation exercise.
- `nueral.py`: simple neural-network implementation using NumPy.
- `propagation.py`: currently empty; treat it as a placeholder or scratch file unless a new script is intentionally being added there.

## Guidance for AI agents

- Assume these are educational or scratch implementations rather than production code.
- Do not add packaging, CI, or framework scaffolding unless the user explicitly requests it.
- When fixing bugs, keep the change minimal and aligned with the existing script style.
- If a file is incomplete or obviously experimental, make the smallest safe improvement and avoid broad refactors.
