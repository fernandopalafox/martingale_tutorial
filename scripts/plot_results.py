"""Plot the observed data, martingale trajectories, and theta_infty histogram."""

import argparse
import tomllib

import matplotlib.pyplot as plt
import numpy as np

BG_COLOR = "#F2E4D4"
ACCENT_COLOR = "#3B6E8F"


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
    ax.scatter(x, y, color=ACCENT_COLOR, s=110, edgecolor=BG_COLOR, linewidth=2.2,
               zorder=2, label="Observed data")

    ax.set_xlabel("x", fontsize=16, fontweight="bold")
    ax.set_ylabel("y", fontsize=16, fontweight="bold")
    ax.set_title("Observed data", fontsize=17, fontweight="bold")
    ax.legend(fontsize=14)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_trajectories(theta_traj, theta_star, n, M, num_plot, path):
    m = np.arange(M - n + 1)  # iterations of the resampling loop, starting at 0

    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

    for i, traj in enumerate(theta_traj[:num_plot]):
        ax.plot(m, traj, color=ACCENT_COLOR, linewidth=1.8, alpha=0.7,
                label="Trajectories" if i == 0 else None)

    ymin, ymax = theta_traj[:num_plot].min(), theta_traj[:num_plot].max()
    pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.axhline(theta_star, color="black", linestyle="--", linewidth=2, label=r"$\theta^*$")

    theta_hat_n = float(theta_traj[0, 0])
    ax.plot(0, theta_hat_n, "o", color="black", markersize=8, zorder=5, clip_on=False,
            label=rf"$\hat\theta_0 \approx {theta_hat_n:.2f}$")

    ax.set_xlabel("Iteration ($m$)", fontsize=16, fontweight="bold")
    ax.set_ylabel(r"Parameter estimate ($\hat\theta_m$)", fontsize=16, fontweight="bold")
    ax.set_title("Martingale trajectories", fontsize=17, fontweight="bold")
    ax.set_xlim(0, M - n)
    ax.legend(fontsize=14, loc="lower right")
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
    label = config["label"]
    dataset = np.load(f"data/dataset_{label}.npz")
    results = np.load(f"data/martingale_results_{label}.npz")

    data_path = f"figures/data_{label}.png"
    trajectories_path = f"figures/trajectories_{label}.png"
    histogram_path = f"figures/posterior_histogram_{label}.png"

    plot_data(dataset["x"], dataset["y"], config["theta_star"], data_path)
    plot_trajectories(
        results["theta_traj"],
        config["theta_star"],
        config["n"],
        config["M"],
        config["num_trajectories_plot"],
        trajectories_path,
    )
    plot_histogram(results["theta_infty"], config["theta_star"], histogram_path)
    print(f"Saved {data_path}, {trajectories_path}, {histogram_path}")


if __name__ == "__main__":
    main()
