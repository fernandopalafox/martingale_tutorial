# martingale_tutorial

Demo of prior-free uncertainty quantification via martingales, for a simple
linear model with an unknown slope. Companion code for Part 2 of the blog
post, at `content/research/martingale2.md` in `fernandopalafox.github.io`.

## Setup

```
uv sync
```

## Run

```
uv run scripts/generate_linear.py --config config/default.toml
uv run scripts/run_martingale.py --config config/default.toml
uv run scripts/plot_results.py --config config/default.toml

uv run scripts/generate_linear.py --config config/uninformative.toml
uv run scripts/run_martingale.py --config config/uninformative.toml
uv run scripts/plot_results.py --config config/uninformative.toml

uv run scripts/plot_comparison.py
```

Each run generates a small linear dataset, runs `N` independent martingale
trajectories that each resample from the data and update the slope estimate
in closed form (recursive least squares, no learning rate to tune), and
plots the data, the trajectories, and a histogram of the resulting
`theta_infty` values — a posterior over the slope obtained without ever
specifying a prior.

`config/default.toml` (label `informative`) samples `x` from `[-1, 1]`;
`config/uninformative.toml` samples `x` from a much narrower `[-0.1, 0.1]`,
giving the martingale procedure far less information per observation.
Both configs use the same reduced noise level so the two runs differ only
in what's sampled. Each script reads `config/default.toml` by default; pass
`--config` to point at a different config. Data and figures are saved with
the config's `label` in the filename (e.g. `data/dataset_informative.npz`,
`figures/trajectories_uninformative.png`).

`scripts/plot_comparison.py` loads both runs' saved `data/` and produces
three side-by-side comparison figures, informative in blue and
uninformative in orange throughout: `figures/data_comparison.png` (both
samples' `(x, y)` scatter), `figures/trajectories_comparison.png` (both
sets of trajectories overlaid), and
`figures/posterior_histogram_comparison.png` (both `theta_infty`
histograms overlaid).

<p align="center">
  <img src="figures/data_comparison.png" alt="Observed data, informative vs. uninformative sampling" width="32%">
  <img src="figures/trajectories_comparison.png" alt="Martingale trajectories, informative vs. uninformative sampling" width="32%">
  <img src="figures/posterior_histogram_comparison.png" alt="Prior-free posterior, informative vs. uninformative sampling" width="32%">
</p>

## Layout

- `config/` — TOML files with all the run parameters. Each has a `label`
  used to namespace its `data/` and `figures/` outputs.
- `scripts/generate_linear.py` — draws synthetic `(x, y)` data from a known
  linear model. Named for the model (`_linear`) so other data-generating
  scripts can live alongside it later.
- `scripts/run_martingale.py` — runs the martingale procedure.
- `scripts/plot_results.py` — makes the data, trajectory, and histogram
  figures for a single config.
- `scripts/plot_comparison.py` — makes comparison figures overlaying the
  informative and uninformative runs.
- `data/` — generated datasets and results (gitignored, regenerate anytime).
- `figures/` — output PNGs.
