"""
Generate the LHS-sampled parameter pool + fixed held-out test set for the
TabPFN few-shot SI/PI regression study.

Usage:
    python 01_generate_data.py
Outputs:
    data/pool.csv   (1700 rows: draw few-shot training subsets from here)
    data/test.csv   (300 rows: fixed held-out evaluation set, used for every budget/seed)
    figures/design_space_coverage.png
"""
import numpy as np
import pandas as pd
from scipy.stats.qmc import LatinHypercube, scale
import matplotlib.pyplot as plt

from physics import PARAM_NAMES, PARAM_BOUNDS, simulate

RNG_SEED = 42
N_TOTAL = 2000
N_TEST = 300


def generate_pool(n_total=N_TOTAL, seed=RNG_SEED):
    d = len(PARAM_NAMES)
    sampler = LatinHypercube(d=d, seed=seed)
    unit = sampler.random(n=n_total)  # in [0,1)^d

    lowers = np.array([PARAM_BOUNDS[p][0] for p in PARAM_NAMES])
    uppers = np.array([PARAM_BOUNDS[p][1] for p in PARAM_NAMES])
    scaled = scale(unit, lowers, uppers)

    df = pd.DataFrame(scaled, columns=PARAM_NAMES)
    df["insertion_loss_db"] = simulate(df)
    return df


def main():
    rng = np.random.default_rng(RNG_SEED)
    df = generate_pool()

    # fixed held-out test set, drawn once and never touched again
    test_idx = rng.choice(df.index, size=N_TEST, replace=False)
    test_df = df.loc[test_idx].reset_index(drop=True)
    pool_df = df.drop(index=test_idx).reset_index(drop=True)

    pool_df.to_csv("data/pool.csv", index=False)
    test_df.to_csv("data/test.csv", index=False)

    print(f"Pool: {len(pool_df)} rows -> data/pool.csv")
    print(f"Test: {len(test_df)} rows -> data/test.csv")
    print(pool_df.describe())

    # design-space coverage figure (2D projections of a couple of key params)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(pool_df["w_mm"], pool_df["s_mm"], s=6, alpha=0.5, label="pool")
    axes[0].scatter(test_df["w_mm"], test_df["s_mm"], s=6, alpha=0.5, color="red", label="test")
    axes[0].set_xlabel("w_mm"); axes[0].set_ylabel("s_mm"); axes[0].legend()
    axes[0].set_title("LHS coverage: width vs spacing")

    axes[1].hist(pool_df["insertion_loss_db"], bins=40, alpha=0.7, label="pool")
    axes[1].hist(test_df["insertion_loss_db"], bins=40, alpha=0.7, label="test")
    axes[1].set_xlabel("insertion_loss_db"); axes[1].legend()
    axes[1].set_title("Target distribution")

    plt.tight_layout()
    plt.savefig("figures/design_space_coverage.png", dpi=150)
    print("Saved figures/design_space_coverage.png")


if __name__ == "__main__":
    main()
