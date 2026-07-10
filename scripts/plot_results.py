"""Plot the observed data, martingale trajectories, and theta_infty histogram."""

import argparse
import tomllib

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np

BG_COLOR = "#F2E4D4"
ACCENT_COLOR = "#8B5E3C"
CMAP = "coolwarm"


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def style_axes(ax, fig):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(labelsize=14, width=2.0, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)


def value_norm(values, center):
    """Diverging norm centered on `center`, e.g. to color by distance from theta_star."""
    spread = max(abs(np.max(values) - center), abs(np.min(values) - center))
    spread = spread if spread > 0 else 1.0
    return TwoSlopeNorm(vmin=center - spread, vcenter=center, vmax=center + spread)


def plot_data(x, y, theta_star, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

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


def plot_trajectories(theta_traj, theta_infty, theta_star, n, M, num_plot, path):
    m = np.arange(n, M + 1)
    cmap = plt.get_cmap(CMAP)
    norm = value_norm(theta_infty, theta_star)

    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

    for traj in theta_traj[:num_plot]:
        color = cmap(norm(traj[-1]))
        ax.plot(m, traj, color=color, alpha=0.75, linewidth=2.2)
        ax.plot(m[-1], traj[-1], "o", color=color, markersize=6)
    ax.axhline(theta_star, color="black", linestyle="--", linewidth=2, label="True value")

    theta_hat_n = float(theta_traj[0, 0])
    ax.plot(n, theta_hat_n, "x", color="dimgray", markersize=8, markeredgewidth=2)
    ax.annotate(
        rf"initial estimate $\hat\theta_n \approx {theta_hat_n:.2f}$",
        xy=(n, theta_hat_n), xytext=(10, 10), textcoords="offset points",
        fontsize=9, color="dimgray", alpha=0.8,
    )

    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, pad=0.01)
    cbar.set_label(r"$\theta_\infty$", fontsize=16, fontweight="bold")
    cbar.ax.tick_params(labelsize=14, width=2.0)
    cbar.outline.set_linewidth(2.0)

    ax.set_xlabel("m", fontsize=16, fontweight="bold")
    ax.set_ylabel(r"$\theta_m$", fontsize=16, fontweight="bold")
    ax.set_title("Martingale trajectories", fontsize=17, fontweight="bold")
    ax.set_xlim(n, M)
    ax.legend(fontsize=14)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_histogram(theta_infty, theta_star, path):
    cmap = plt.get_cmap(CMAP)
    norm = value_norm(theta_infty, theta_star)

    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

    _, bin_edges, patches = ax.hist(theta_infty, bins=40, edgecolor=BG_COLOR)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    for center, patch in zip(bin_centers, patches):
        patch.set_facecolor(cmap(norm(center)))
    ax.axvline(theta_star, color="black", linestyle="--", linewidth=2, label="True value")

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
        results["theta_infty"],
        config["theta_star"],
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
