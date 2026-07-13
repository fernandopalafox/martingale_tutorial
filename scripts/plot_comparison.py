"""Plot the informative and uninformative distributions' runs together for comparison."""

import argparse

import matplotlib.pyplot as plt
import numpy as np

from plot_results import ACCENT_COLOR, BG_COLOR, load_config, style_axes

ORANGE_COLOR = "#C1652F"


def plot_data_comparison(informative, uninformative, theta_star, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

    ax.axhline(0, color="gray", linewidth=1, alpha=0.4, zorder=0)
    ax.axvline(0, color="gray", linewidth=1, alpha=0.4, zorder=0)

    x_all = np.concatenate([informative["x"], uninformative["x"]])
    x_line = np.array([x_all.min(), x_all.max()])
    ax.plot(x_line, theta_star * x_line, color="black", linestyle="--", linewidth=2,
            label="True line", zorder=1)

    ax.scatter(informative["x"], informative["y"], color=ACCENT_COLOR, s=110,
               edgecolor=BG_COLOR, linewidth=2.2, zorder=2, label="Informative")
    ax.scatter(uninformative["x"], uninformative["y"], color=ORANGE_COLOR, s=110,
               edgecolor=BG_COLOR, linewidth=2.2, zorder=3, label="Uninformative")

    ax.set_xlabel("x", fontsize=16, fontweight="bold")
    ax.set_ylabel("y", fontsize=16, fontweight="bold")
    ax.set_title("Observed data", fontsize=17, fontweight="bold")
    ax.legend(fontsize=14)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def _draw_trajectories(ax, theta_traj, num_plot, color, label):
    m = np.arange(theta_traj.shape[1])
    for i, traj in enumerate(theta_traj[:num_plot]):
        ax.plot(m, traj, color=color, linewidth=1.8, alpha=0.7,
                label=label if i == 0 else None)
    return m


def plot_trajectories_comparison(informative, uninformative, num_plot, theta_star, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

    m = _draw_trajectories(ax, informative["theta_traj"], num_plot, ACCENT_COLOR, "Informative")
    _draw_trajectories(ax, uninformative["theta_traj"], num_plot, ORANGE_COLOR, "Uninformative")

    all_traj = np.concatenate([informative["theta_traj"][:num_plot],
                                uninformative["theta_traj"][:num_plot]])
    ymin, ymax = all_traj.min(), all_traj.max()
    pad = 0.05 * (ymax - ymin)
    ax.set_ylim(ymin - pad, ymax + pad)
    ax.axhline(theta_star, color="black", linestyle="--", linewidth=2,
               label=r"True estimate ($\theta^*$)")

    theta_hat_inf = float(informative["theta_traj"][0, 0])
    theta_hat_uninf = float(uninformative["theta_traj"][0, 0])
    ax.plot(0, theta_hat_inf, "o", color="black", markersize=8, zorder=5, clip_on=False,
            label=r"Initial estimate ($\hat\theta_0$)")
    ax.plot(0, theta_hat_uninf, "o", color="black", markersize=8, zorder=5, clip_on=False)

    ax.set_xlabel("Iteration ($m$)", fontsize=16, fontweight="bold")
    ax.set_ylabel(r"Parameter estimate ($\hat\theta_m$)", fontsize=16, fontweight="bold")
    ax.set_title("Martingale trajectories", fontsize=17, fontweight="bold")
    ax.set_xlim(0, m[-1])
    ax.legend(fontsize=13, loc="lower right")
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def plot_histogram_comparison(inf_theta_infty, uninf_theta_infty, theta_star, path):
    fig, ax = plt.subplots(figsize=(8, 5))
    style_axes(ax, fig)

    combined = np.concatenate([inf_theta_infty, uninf_theta_infty])
    bins = np.linspace(combined.min(), combined.max(), 60)

    ax.hist(uninf_theta_infty, bins=bins, color=ORANGE_COLOR, edgecolor=BG_COLOR,
            linewidth=0.4, alpha=0.8, label="Uninformative")
    ax.hist(inf_theta_infty, bins=bins, color=ACCENT_COLOR, edgecolor=BG_COLOR,
            linewidth=0.4, alpha=0.8, label="Informative")
    ax.axvline(theta_star, color="black", linestyle="--", linewidth=2, label=r"$\theta^*$")

    ax.set_xlabel(r"$\theta_\infty$", fontsize=16, fontweight="bold")
    ax.set_ylabel("count", fontsize=16, fontweight="bold")
    ax.set_title("Prior-free posterior", fontsize=17, fontweight="bold")
    ax.legend(fontsize=14)
    fig.tight_layout(pad=0.5)
    fig.savefig(path, dpi=150, bbox_inches="tight")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--informative-config", default="config/default.toml")
    parser.add_argument("--uninformative-config", default="config/uninformative.toml")
    args = parser.parse_args()

    inf_config = load_config(args.informative_config)
    uninf_config = load_config(args.uninformative_config)
    theta_star = inf_config["theta_star"]

    inf_dataset = np.load(f"data/dataset_{inf_config['label']}.npz")
    uninf_dataset = np.load(f"data/dataset_{uninf_config['label']}.npz")
    inf_results = np.load(f"data/martingale_results_{inf_config['label']}.npz")
    uninf_results = np.load(f"data/martingale_results_{uninf_config['label']}.npz")

    plot_data_comparison(inf_dataset, uninf_dataset, theta_star, "figures/data_comparison.png")

    plot_trajectories_comparison(
        inf_results, uninf_results,
        inf_config["num_trajectories_plot"], theta_star,
        "figures/trajectories_comparison.png",
    )

    plot_histogram_comparison(
        inf_results["theta_infty"], uninf_results["theta_infty"], theta_star,
        "figures/posterior_histogram_comparison.png",
    )

    print("Saved figures/data_comparison.png, figures/trajectories_comparison.png, "
          "figures/posterior_histogram_comparison.png")


if __name__ == "__main__":
    main()
