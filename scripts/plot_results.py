"""Plot the observed data, martingale trajectories, and theta_infty histogram."""

import argparse
import tomllib

import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
import numpy as np

BG_COLOR = "#F2E4D4"
ACCENT_COLOR = "#3B6E8F"
CONVERGENCE_CMAP = "plasma"


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def style_axes(ax, fig):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(labelsize=14, width=2.0, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)


def plot_data(x, y, theta_star, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

    ax.axhline(0, color="gray", linewidth=1, alpha=0.4, zorder=0)
    ax.axvline(0, color="gray", linewidth=1, alpha=0.4, zorder=0)

    x_line = np.array([x.min(), x.max()])
    ax.plot(x_line, theta_star * x_line, color="black", linestyle="--", linewidth=2,
            label="True line", zorder=1)
    ax.scatter(x, y, color=ACCENT_COLOR, s=60, edgecolor=BG_COLOR, linewidth=1.2,
               zorder=2, label="Observed data")

    ax.set_xlabel("x", fontsize=16, fontweight="bold")
    ax.set_ylabel("y", fontsize=16, fontweight="bold")
    ax.set_title("Observed data", fontsize=17, fontweight="bold")
    ax.legend(fontsize=14)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_trajectories(theta_traj, x_traj, theta_star, S_n, n, M, num_plot, path):
    m = np.arange(n, M + 1)
    cmap = plt.get_cmap(CONVERGENCE_CMAP)
    alpha_min, alpha_max = 0.15, 0.9

    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

    for traj, x_seq in zip(theta_traj[:num_plot], x_traj[:num_plot]):
        # S_m = sum of x_i^2 seen so far: the accumulated information that drives
        # the update's step size down. Use it (not raw m) as our convergence measure,
        # both for color (dark/purple = unconverged, bright/yellow = converged) and
        # for fading each line in as it goes.
        S = np.concatenate([[S_n], S_n + np.cumsum(x_seq**2)])
        progress = (S - S_n) / (S[-1] - S_n)
        alpha = alpha_min + (alpha_max - alpha_min) * progress

        points = np.array([m, traj]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        seg_colors = [(*cmap(p)[:3], a) for p, a in zip(progress[:-1], alpha[:-1])]
        lc = LineCollection(segments, colors=seg_colors, linewidths=2.2)
        ax.add_collection(lc)
        ax.plot(m[-1], traj[-1], "o", color=cmap(1.0), markersize=6, alpha=alpha_max)

    ymin, ymax = theta_traj[:num_plot].min(), theta_traj[:num_plot].max()
    pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.axhline(theta_star, color="black", linestyle="--", linewidth=2, label=r"$\theta^*$")

    theta_hat_n = float(theta_traj[0, 0])
    ax.plot(n, theta_hat_n, "o", color="dimgray", markersize=8, zorder=5,
            label=rf"$\hat\theta_n \approx {theta_hat_n:.2f}$")

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=Normalize(0, 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label("Convergence", fontsize=16, fontweight="bold")
    cbar.set_ticks([0, 1])
    cbar.set_ticklabels(["Unconverged", "Converged"])
    cbar.ax.tick_params(labelsize=12, width=2.0)
    cbar.outline.set_linewidth(2.0)

    ax.set_xlabel("Iteration", fontsize=16, fontweight="bold")
    ax.set_ylabel(r"$\hat\theta_m$", fontsize=16, fontweight="bold")
    ax.set_title("Martingale trajectories", fontsize=17, fontweight="bold")
    ax.set_xlim(n, M)
    ax.legend(fontsize=14)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_histogram(theta_infty, theta_star, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

    ax.hist(theta_infty, bins=40, color=ACCENT_COLOR, edgecolor=BG_COLOR)
    ax.axvline(theta_star, color="black", linestyle="--", linewidth=2, label=r"$\theta^*$")

    ax.set_xlabel(r"$\theta_\infty$", fontsize=16, fontweight="bold")
    ax.set_ylabel("count", fontsize=16, fontweight="bold")
    ax.set_title("Prior-free posterior", fontsize=17, fontweight="bold")
    ax.legend(fontsize=14)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = np.load("data/dataset.npz")
    results = np.load("data/martingale_results.npz")

    plot_data(dataset["x"], dataset["y"], config["theta_star"], "figures/data.png")
    plot_trajectories(
        results["theta_traj"],
        results["x_traj"],
        config["theta_star"],
        float(np.sum(dataset["x"] ** 2)),
        config["n"],
        config["M"],
        config["num_trajectories_plot"],
        "figures/trajectories.png",
    )
    plot_histogram(
        results["theta_infty"], config["theta_star"], "figures/posterior_histogram.png"
    )
    print("Saved figures/data.png, figures/trajectories.png, figures/posterior_histogram.png")


if __name__ == "__main__":
    main()
