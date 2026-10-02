"""
Core few-shot experiment: TabPFN vs XGBoost vs MLP vs GP at training
budgets N in {10, 20, 50, 100, 200}, each over 10 random seeds, evaluated
on the SAME fixed 300-row held-out test set every time.

Requires (install once, e.g. in a Colab cell):
    pip install tabpfn xgboost scikit-learn scipy pandas

Usage:
    python 02_run_experiment.py
Output:
    results/raw_results.csv   (one row per model x budget x seed)
"""
import time
import warnings
import os
from pathlib import Path
from statistics import NormalDist
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel, ConstantKernel
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
try:
    from sklearn.metrics import root_mean_squared_error
except ImportError:  # older sklearn
    from sklearn.metrics import mean_squared_error
    def root_mean_squared_error(y_true, y_pred):
        return mean_squared_error(y_true, y_pred) ** 0.5

from physics import PARAM_NAMES

warnings.filterwarnings("ignore")

BUDGETS = [10, 20, 50, 100, 200]
# Increase the statistically ambiguous middle budget while retaining the
# original 10 matched draws at the other budgets.
SEEDS_BY_BUDGET = {10: 10, 20: 10, 50: 30, 100: 10, 200: 10}
TARGET = "insertion_loss_db"
INTERVAL_COVERAGE = 0.90

# ---------------------------------------------------------------------
# Model factory. TabPFN / XGBoost imported lazily so the script still runs
# (skipping those methods with a clear warning) if they aren't installed
# in the current environment -- handy for quick local sanity checks before
# moving to Colab, where you'll `pip install tabpfn xgboost` first.
# ---------------------------------------------------------------------

def _try_import_tabpfn():
    try:
        from tabpfn import TabPFNRegressor
        return TabPFNRegressor
    except ImportError:
        return None


def _try_import_xgboost():
    try:
        from xgboost import XGBRegressor
        return XGBRegressor
    except ImportError:
        return None


TabPFNRegressor = _try_import_tabpfn()
XGBRegressor = _try_import_xgboost()


def _has_tabpfn_token():
    """Check for an existing non-interactive TabPFN credential.

    Do not instantiate TabPFN when this is absent: doing so triggers its
    browser-login handler.  The handler cannot complete from many terminals
    and used to be retried once per seed by this script.
    """
    if os.environ.get("TABPFN_TOKEN", "").strip():
        return True

    home = Path.home()
    return any(
        path.is_file() and path.read_text(encoding="utf-8").strip()
        for path in (home / ".cache" / "tabpfn" / "auth_token", home / ".tabpfn" / "token")
    )


def get_models():
    models = {}

    if TabPFNRegressor is not None and _has_tabpfn_token():
        models["TabPFN"] = lambda: TabPFNRegressor()  # zero-shot, no tuning
    elif TabPFNRegressor is not None:
        print(
            "[warn] TabPFN is installed but no local credential was found; "
            "skipping it without opening a browser. Set TABPFN_TOKEN, then rerun."
        )
    else:
        print("[warn] tabpfn not installed -- skipping TabPFN. "
              "Run: pip install tabpfn")

    if XGBRegressor is not None:
        models["XGBoost"] = lambda: XGBRegressor(
            n_estimators=200, max_depth=4, learning_rate=0.1,
            subsample=0.9, colsample_bytree=0.9, random_state=0, verbosity=0,
        )
    else:
        print("[warn] xgboost not installed -- skipping XGBoost. "
              "Run: pip install xgboost")

    models["MLP"] = lambda: MLPRegressor(
        hidden_layer_sizes=(64, 32), activation="relu", solver="adam",
        max_iter=3000, random_state=0,
    )

    kernel = ConstantKernel(1.0) * RBF(length_scale=np.ones(len(PARAM_NAMES))) \
        + WhiteKernel(noise_level=1e-3)
    models["GP"] = lambda: GaussianProcessRegressor(
        kernel=kernel, normalize_y=True, n_restarts_optimizer=2, random_state=0,
    )

    return models


def evaluate_once(model_name, model_factory, X_train, y_train, X_test, y_test, needs_scaling):
    if needs_scaling:
        x_scaler = StandardScaler().fit(X_train)
        X_train_ = x_scaler.transform(X_train)
        X_test_ = x_scaler.transform(X_test)
    else:
        X_train_, X_test_ = X_train, X_test

    model = model_factory()
    t0 = time.perf_counter()
    model.fit(X_train_, y_train)
    fit_time = time.perf_counter() - t0

    t0 = time.perf_counter()
    interval_lower = interval_upper = None
    if model_name == "GP":
        y_pred, y_std = model.predict(X_test_, return_std=True)
        z_value = NormalDist().inv_cdf((1 + INTERVAL_COVERAGE) / 2)
        interval_lower, interval_upper = y_pred - z_value * y_std, y_pred + z_value * y_std
    elif model_name == "TabPFN":
        # `main` computes the point prediction and requested quantiles in a
        # single forward pass, so the recorded prediction time is honest.
        output = model.predict(
            X_test_, output_type="main", quantiles=[0.05, 0.95],
        )
        y_pred = output["mean"]
        interval_lower, interval_upper = output["quantiles"]
    else:
        y_pred = model.predict(X_test_)
    predict_time = time.perf_counter() - t0

    rmse = root_mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    result = {
        "model": model_name, "rmse": rmse, "mae": mae, "r2": r2,
        "fit_time_s": fit_time, "predict_time_s": predict_time,
    }
    if interval_lower is not None:
        result["picp_90"] = np.mean((y_test >= interval_lower) & (y_test <= interval_upper))
        result["mpiw_90_db"] = np.mean(interval_upper - interval_lower)
    else:
        result["picp_90"] = np.nan
        result["mpiw_90_db"] = np.nan
    return result


def main():
    pool = pd.read_csv("data/pool.csv")
    test = pd.read_csv("data/test.csv")

    X_pool = pool[PARAM_NAMES].values
    y_pool = pool[TARGET].values
    X_test = test[PARAM_NAMES].values
    y_test = test[TARGET].values

    models = get_models()

    # Validate an available credential exactly once, before the experiment.
    # If it is expired/invalid, exclude TabPFN instead of launching the same
    # authentication flow for every budget and seed.
    if "TabPFN" in models:
        # A valid token proceeds normally.  An invalid token or an unaccepted
        # license raises immediately rather than invoking TabPFN's GUI login.
        os.environ.setdefault("TABPFN_NO_BROWSER", "1")
        try:
            probe = models["TabPFN"]()
            probe.fit(X_pool[:10], y_pool[:10])
            print("TabPFN credential/model preflight succeeded.")
        except Exception as exc:
            del models["TabPFN"]
            print(f"[warn] TabPFN preflight failed; skipping it: {exc}")

    # MLP and GP benefit from feature scaling; tree models / TabPFN handle raw scale fine
    needs_scaling = {"MLP": True, "GP": True, "XGBoost": False, "TabPFN": False}

    rows = []
    total_runs = sum(SEEDS_BY_BUDGET[N] for N in BUDGETS) * len(models)
    run_i = 0

    for N in BUDGETS:
        for seed in range(SEEDS_BY_BUDGET[N]):
            rng = np.random.default_rng(seed * 1000 + N)
            idx = rng.choice(len(X_pool), size=N, replace=False)
            X_train, y_train = X_pool[idx], y_pool[idx]

            for model_name, factory in models.items():
                run_i += 1
                try:
                    result = evaluate_once(
                        model_name, factory, X_train, y_train, X_test, y_test,
                        needs_scaling.get(model_name, False),
                    )
                except Exception as e:
                    print(f"[error] {model_name} N={N} seed={seed}: {e}")
                    continue
                result.update({"N": N, "seed": seed})
                rows.append(result)
                print(f"[{run_i}/{total_runs}] N={N:4d} seed={seed:2d} "
                      f"{model_name:8s} RMSE={result['rmse']:.4f} "
                      f"fit={result['fit_time_s']:.3f}s")

    out = pd.DataFrame(rows)
    out.to_csv("results/raw_results.csv", index=False)
    print(f"\nSaved {len(out)} rows -> results/raw_results.csv")


if __name__ == "__main__":
    main()
