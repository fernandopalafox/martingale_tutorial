"""Generate synthetic data for the linear model y = theta_star * x + noise."""

import argparse
import tomllib

import jax
import jax.numpy as jnp
import numpy as np


def load_config(path):
    with open(path, "rb") as f:
        return tomllib.load(f)


def generate_data(key, config):
    x = jnp.linspace(config["x_min"], config["x_max"], config["n"])
    noise = config["sigma"] * jax.random.normal(key, (config["n"],))
    y = config["theta_star"] * x + noise
    return x, y


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config/default.toml")
    args = parser.parse_args()

    config = load_config(args.config)
    key = jax.random.PRNGKey(config["seed"])
    x, y = generate_data(key, config)

    np.savez("data/dataset.npz", x=np.asarray(x), y=np.asarray(y))
    print(f"Saved {config['n']} points to data/dataset.npz")


if __name__ == "__main__":
    main()
