"""Plot martingale trajectories and the resulting theta_infty histogram."""

import argparse
import tomllib

import matplotlib.pyplot as plt
import numpy as np

BG_COLOR = "#F2E4D4"
ACCENT_COLOR = "#8B5E3C"


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def style_axes(ax, fig):
    fig.patch.set_facecolor(BG_COLOR)
    ax.set_facecolor(BG_COLOR)
    ax.tick_params(labelsize=14, width=2.0, which="both")
    for spine in ax.spines.values():
        spine.set_linewidth(2.0)


def plot_trajectories(theta_traj, theta_star, n, M, num_plot, path):
    m = np.arange(n, M + 1)
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

    for traj in theta_traj[:num_plot]:
        ax.plot(m, traj, color=ACCENT_COLOR, alpha=0.5, linewidth=2.2)
        ax.plot(m[-1], traj[-1], "o", color=ACCENT_COLOR, markersize=6, alpha=0.9)
    ax.axhline(theta_star, color="black", linestyle="--", linewidth=2, label=r"$\theta^*$")

    ax.set_xlabel("m", fontsize=16, fontweight="bold")
    ax.set_ylabel(r"$\theta_m$", fontsize=16, fontweight="bold")
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
    ax.set_title("Prior-free posterior over the slope", fontsize=17, fontweight="bold")
    ax.legend(fontsize=14)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    results = np.load("data/martingale_results.npz")

    plot_trajectories(
        results["theta_traj"],
        config["theta_star"],
        config["n"],
        config["M"],
        config["num_trajectories_plot"],
        "figures/trajectories.png",
    )
    plot_histogram(
        results["theta_infty"], config["theta_star"], "figures/posterior_histogram.png"
    )
    print("Saved figures/trajectories.png and figures/posterior_histogram.png")


if __name__ == "__main__":
    main()
