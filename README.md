# martingale_tutorial

Demo of prior-free uncertainty quantification via martingales, for a simple
linear model with an unknown slope. Companion code for the blog post at
`content/research/martingale_tutorial.md` in `fernandopalafox.github.io`.

## Setup

```
uv sync
```

## Run

```
uv run scripts/generate_linear.py
uv run scripts/run_martingale.py
uv run scripts/plot_results.py
```

This generates a small linear dataset, runs `N` independent martingale
trajectories that each resample from the data and update the slope estimate
in closed form (recursive least squares, no learning rate to tune), and
plots the data, the trajectories, and a histogram of the resulting
`theta_infty` values — a posterior over the slope obtained without ever
specifying a prior. Trajectories and histogram bars are colored by their
`theta_infty` value on the same diverging scale, so the two figures read
together.

Each script reads `config/default.toml` by default; pass `--config` to
point at a different config (e.g. a different `theta_star`, noise level, or
sample size).

## Layout

- `config/` — TOML files with all the run parameters.
- `scripts/generate_linear.py` — draws synthetic `(x, y)` data from a known
  linear model. Named for the model (`_linear`) so other data-generating
  scripts can live alongside it later.
- `scripts/run_martingale.py` — runs the martingale procedure.
- `scripts/plot_results.py` — makes the data, trajectory, and histogram figures.
- `data/` — generated datasets and results (gitignored, regenerate anytime).
- `figures/` — output PNGs.
