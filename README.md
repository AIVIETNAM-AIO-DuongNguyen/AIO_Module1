# AIO_Module1 — Generalization Levers on Low-Data Medical Image Classification

A controlled study of two generalization levers for medical image classification when labeled data is scarce:

- **Input-space lever (preprocessing):** none / global histogram equalization / local contrast (CLAHE)
- **Parameter-space lever (transfer strategy):** linear probe / full fine-tune / LoRA

We measure how these levers interact across shrinking data regimes on the public **MedMNIST** benchmark, and produce a practical decision guide plus a reproducible, config-driven experiment harness.

This is a **methodological** study of generalization techniques. It is **not** a clinical diagnostic tool, and MedMNIST is not intended for clinical use.

## Project status

Foundation harness complete: data loading, the three preprocessing arms, the
three transfer strategies, training/evaluation, config generation for the core
matrix, and CPU smoke tests all run end to end. Next is baseline reproduction on
a GPU server and running the 162-run core matrix. See
[docs/ROADMAP.md](docs/ROADMAP.md) for the detailed status.

## Repository layout

```
configs/      One YAML per experiment run
src/          Source packages (data, models, training, evaluation, utils)
tests/        CPU-friendly pytest smoke tests
scripts/      Standalone scripts (e.g. GPU environment check)
docs/note/    Research notes, indexed in docs/note/README.md
```

## Getting started

A virtual environment is required. Do not install dependencies into system Python.

```bash
python -m venv .venv
# Windows PowerShell
.venv\Scripts\Activate.ps1
# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

Verify a GPU environment (run on the server, not the CPU dev machine):

```bash
python scripts/check_gpu_env.py
```

Run one experiment from a config (one YAML describes one run):

```bash
python scripts/run_experiment.py --config configs/example_run.yaml
```

Results are written to `<output_dir>/<run_name>/` (checkpoint and `metrics.json`)
and appended to `<output_dir>/results.csv`.

## Tests

CPU-friendly smoke tests cover every module and run on the local machine:

```bash
pytest -q
```

The training smoke test skips cleanly if the MedMNIST download is unavailable.

## Documentation

- [docs/ROADMAP.md](docs/ROADMAP.md) — progress tracker and next steps.
- [docs/note/](docs/note/README.md) — research notes, indexed by topic.

## Experiment scope (locked core matrix)

3 preprocessing × 3 transfer strategy × 3 data regime {100%, 10%, 5%} × 2 datasets × 3 seeds = 162 runs, sized for free-tier compute. Backbone: ResNet-50 (ImageNet-pretrained), resolution 64×64. Primary metric: AUROC on the official MedMNIST test split.
