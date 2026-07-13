"""Run the prior-free martingale procedure for the linear model's slope."""

import argparse
import tomllib

import jax
import jax.numpy as jnp
import numpy as np


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def run_martingale(key, x, y, config):
    n, M, sigma = config["n"], config["M"], config["sigma"]
    theta_hat_n = jnp.sum(x * y) / jnp.sum(x**2)
    S_n = jnp.sum(x**2)

    def step(carry, step_key):
        theta, S = carry
        key_idx, key_noise = jax.random.split(step_key)
        x_next = x[jax.random.randint(key_idx, (), 0, n)]
        y_next = theta * x_next + sigma * jax.random.normal(key_noise, ())
        S_next = S + x_next**2
        theta_next = theta + x_next * (y_next - x_next * theta) / S_next
        return (theta_next, S_next), (theta_next, x_next, y_next)

    def single_run(run_key):
        step_keys = jax.random.split(run_key, M - n)
        (theta_final, _), (theta_seq, x_seq, y_seq) = jax.lax.scan(
            step, (theta_hat_n, S_n), step_keys
        )
        theta_traj = jnp.concatenate([theta_hat_n[None], theta_seq])
        return theta_final, theta_traj, x_seq, y_seq

    run_keys = jax.random.split(key, config["N"])
    return jax.vmap(single_run)(run_keys)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    dataset = np.load(f"data/dataset_{config['label']}.npz")
    x, y = jnp.array(dataset["x"]), jnp.array(dataset["y"])

    key = jax.random.PRNGKey(config["seed"])
    theta_infty, theta_traj, x_traj, y_traj = run_martingale(key, x, y, config)

    out_path = f"data/martingale_results_{config['label']}.npz"
    np.savez(
        out_path,
        theta_infty=np.asarray(theta_infty),
        theta_traj=np.asarray(theta_traj),
        x_traj=np.asarray(x_traj),
        y_traj=np.asarray(y_traj),
    )
    print(f"Ran {config['N']} martingale trajectories to {out_path}")


if __name__ == "__main__":
    main()
