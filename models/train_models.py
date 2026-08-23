from __future__ import annotations

import argparse
import json
import pickle
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR

try:
    from xgboost import XGBRegressor
except Exception:  # pragma: no cover - optional dependency
    XGBRegressor = None

try:
    from lightgbm import LGBMRegressor
except Exception:  # pragma: no cover - optional dependency
    LGBMRegressor = None

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FEATURES_DIR = ROOT / "features"
MODELS_DIR = ROOT / "models"
ARTIFACTS_DIR = MODELS_DIR / "artifacts"
RESULTS_PATH = MODELS_DIR / "results.csv"
SUMMARY_PATH = MODELS_DIR / "training_summary.json"

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    estimator_factory: Any
    param_grid: list[dict[str, Any]]
    train_cap: int
    sample_for_ci: int = 1000


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def load_features() -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    with (FEATURES_DIR / "drug_features.pkl").open("rb") as f:
        drug_features = pickle.load(f)
    with (FEATURES_DIR / "protein_features.pkl").open("rb") as f:
        protein_features = pickle.load(f)
    return drug_features, protein_features


def load_split(split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split_name}_train.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def load_test_split(split_name: str) -> pd.DataFrame:
    path = DATA_DIR / f"{split_name}_test.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing split file: {path}")
    return pd.read_csv(path)


def sample_frame(df: pd.DataFrame, limit: int | None, seed: int) -> pd.DataFrame:
    if limit is None or limit <= 0 or len(df) <= limit:
        return df.reset_index(drop=True)
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def build_pair_matrix(
    df: pd.DataFrame,
    drug_features: dict[str, np.ndarray],
    protein_features: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    y = np.empty(len(df), dtype=np.float32)

    for idx, row in enumerate(df.itertuples(index=False)):
        drug_vec = drug_features.get(row.drug_id)
        protein_vec = protein_features.get(row.protein_id)
        if drug_vec is None or protein_vec is None:
            raise KeyError(f"Missing features for pair: {row.drug_id} / {row.protein_id}")
        rows.append(np.concatenate([drug_vec, protein_vec]).astype(np.float32))
        y[idx] = float(row.affinity)

    return np.vstack(rows), y


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    concordant = 0.0
    permissible = 0.0
    n = len(y_true)
    for i in range(n):
        yi = y_true[i]
        pi = y_pred[i]
        for j in range(i + 1, n):
            yj = y_true[j]
            if yi == yj:
                continue
            permissible += 1.0
            pj = y_pred[j]
            diff = (pi - pj) * (yi - yj)
            if diff > 0:
                concordant += 1.0
            elif diff == 0:
                concordant += 0.5
    if permissible == 0.0:
        return float("nan")
    return concordant / permissible


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray, ci_sample: int, seed: int) -> dict[str, float]:
    metrics = {
        "mse": float(mean_squared_error(y_true, y_pred)),
        "rmse": float(mean_squared_error(y_true, y_pred, squared=False)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }

    if len(y_true) > ci_sample:
        rng = np.random.default_rng(seed)
        sample_idx = rng.choice(len(y_true), size=ci_sample, replace=False)
        ci_true = y_true[sample_idx]
        ci_pred = y_pred[sample_idx]
    else:
        ci_true = y_true
        ci_pred = y_pred
    metrics["ci"] = float(concordance_index(ci_true, ci_pred))

    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        metrics["pearson_r"] = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        metrics["pearson_r"] = float("nan")
    return metrics


def make_model_specs(seed: int) -> list[ModelSpec]:
    specs: list[ModelSpec] = [
        ModelSpec(
            name="rf",
            estimator_factory=lambda params: RandomForestRegressor(
                random_state=seed, n_jobs=-1, **params
            ),
            param_grid=[
                {"n_estimators": 200, "max_depth": None, "min_samples_leaf": 1, "max_features": "sqrt"},
                {"n_estimators": 400, "max_depth": 24, "min_samples_leaf": 2, "max_features": 0.5},
            ],
            train_cap=20000,
        ),
        ModelSpec(
            name="xgboost",
            estimator_factory=lambda params: XGBRegressor(
                random_state=seed,
                objective="reg:squarederror",
                tree_method="hist",
                n_jobs=-1,
                **params,
            ),
            param_grid=[
                {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85},
                {"n_estimators": 700, "max_depth": 8, "learning_rate": 0.05, "subsample": 0.8, "colsample_bytree": 0.8},
            ],
            train_cap=20000,
        ),
        ModelSpec(
            name="lightgbm",
            estimator_factory=lambda params: LGBMRegressor(
                random_state=seed,
                n_jobs=-1,
                objective="regression",
                **params,
            ),
            param_grid=[
                {"n_estimators": 500, "num_leaves": 31, "learning_rate": 0.05, "subsample": 0.85, "colsample_bytree": 0.85},
                {"n_estimators": 700, "num_leaves": 63, "learning_rate": 0.03, "subsample": 0.8, "colsample_bytree": 0.8},
            ],
            train_cap=20000,
        ),
        ModelSpec(
            name="svr",
            estimator_factory=lambda params: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("svr", SVR(**params)),
                ]
            ),
            param_grid=[
                {"svr__C": 10.0, "svr__epsilon": 0.1, "svr__gamma": "scale", "svr__kernel": "rbf"},
                {"svr__C": 30.0, "svr__epsilon": 0.1, "svr__gamma": "scale", "svr__kernel": "rbf"},
            ],
            train_cap=6000,
        ),
        ModelSpec(
            name="gpr",
            estimator_factory=lambda params: Pipeline(
                [
                    ("scaler", StandardScaler()),
                    (
                        "gpr",
                        GaussianProcessRegressor(
                            kernel=ConstantKernel(1.0, (0.1, 10.0))
                            * Matern(length_scale=1.0, length_scale_bounds=(0.1, 10.0), nu=1.5)
                            + WhiteKernel(noise_level=1.0, noise_level_bounds=(1e-5, 10.0)),
                            alpha=params.get("alpha", 1e-2),
                            normalize_y=True,
                            optimizer=None,
                            random_state=seed,
                        ),
                    ),
                ]
            ),
            param_grid=[
                {"alpha": 1e-2},
                {"alpha": 5e-2},
            ],
            train_cap=2500,
            sample_for_ci=500,
        ),
    ]

    available = []
    for spec in specs:
        if spec.name == "xgboost" and XGBRegressor is None:
            continue
        if spec.name == "lightgbm" and LGBMRegressor is None:
            continue
        available.append(spec)
    return available


def fit_and_select(
    spec: ModelSpec,
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    seed: int,
) -> tuple[Any, dict[str, Any], dict[str, float]]:
    best_model = None
    best_params: dict[str, Any] = {}
    best_metrics: dict[str, float] | None = None
    best_rmse = float("inf")

    for params in spec.param_grid:
        model = spec.estimator_factory(params)
        model.fit(x_train, y_train)
        preds = model.predict(x_val)
        metrics = evaluate_predictions(y_val, preds, spec.sample_for_ci, seed)
        if metrics["rmse"] < best_rmse:
            best_rmse = metrics["rmse"]
            best_model = model
            best_params = params
            best_metrics = metrics

    assert best_model is not None and best_metrics is not None
    return best_model, best_params, best_metrics


def train_for_split(
    split_name: str,
    spec: ModelSpec,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    drug_features: dict[str, np.ndarray],
    protein_features: dict[str, np.ndarray],
    seed: int,
) -> dict[str, Any]:
    ensure_dir(ARTIFACTS_DIR / split_name)

    train_df = sample_frame(train_df, spec.train_cap, seed)
    x_all_train, y_all_train = build_pair_matrix(train_df, drug_features, protein_features)
    x_test, y_test = build_pair_matrix(test_df, drug_features, protein_features)

    x_train, x_val, y_train, y_val = train_test_split(
        x_all_train,
        y_all_train,
        test_size=0.15,
        random_state=seed,
    )

    model, best_params, val_metrics = fit_and_select(spec, x_train, y_train, x_val, y_val, seed)

    final_model = spec.estimator_factory(best_params)
    final_model.fit(x_all_train, y_all_train)
    test_preds = final_model.predict(x_test)
    test_metrics = evaluate_predictions(y_test, test_preds, spec.sample_for_ci, seed)

    artifact_path = ARTIFACTS_DIR / split_name / f"{spec.name}.joblib"
    joblib.dump(final_model, artifact_path)

    return {
        "split": split_name,
        "model": spec.name,
        "train_rows_used": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "feature_dim": int(x_all_train.shape[1]),
        "best_params": json.dumps(best_params),
        "val_mse": float(val_metrics["mse"]),
        "val_rmse": float(val_metrics["rmse"]),
        "val_mae": float(val_metrics["mae"]),
        "val_r2": float(val_metrics["r2"]),
        "val_ci": float(val_metrics["ci"]),
        "val_pearson_r": float(val_metrics["pearson_r"]),
        "test_mse": float(test_metrics["mse"]),
        "test_rmse": float(test_metrics["rmse"]),
        "test_mae": float(test_metrics["mae"]),
        "test_r2": float(test_metrics["r2"]),
        "test_ci": float(test_metrics["ci"]),
        "test_pearson_r": float(test_metrics["pearson_r"]),
        "artifact_path": str(artifact_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train classical regressors on KIBA split features.")
    parser.add_argument("--splits", nargs="+", default=["random", "cold_drug", "cold_protein"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--results-path", type=Path, default=RESULTS_PATH)
    args = parser.parse_args()

    ensure_dir(MODELS_DIR)
    ensure_dir(ARTIFACTS_DIR)

    drug_features, protein_features = load_features()
    specs = make_model_specs(args.seed)

    all_records: list[dict[str, Any]] = []
    for split_name in args.splits:
        train_df = load_split(split_name)
        test_df = load_test_split(split_name)
        for spec in specs:
            print(f"Training {spec.name} on {split_name} split...")
            record = train_for_split(
                split_name=split_name,
                spec=spec,
                train_df=train_df,
                test_df=test_df,
                drug_features=drug_features,
                protein_features=protein_features,
                seed=args.seed,
            )
            all_records.append(record)
            print(
                f"  test RMSE={record['test_rmse']:.4f}, "
                f"test Pearson r={record['test_pearson_r']:.4f}"
            )

    results_df = pd.DataFrame(all_records)
    results_df.to_csv(args.results_path, index=False)

    summary = {
        "splits": args.splits,
        "models": [spec.name for spec in specs],
        "rows": int(len(results_df)),
        "best_by_split": {},
    }
    for split_name in args.splits:
        subset = results_df[results_df["split"] == split_name]
        if subset.empty:
            continue
        best_row = subset.sort_values("test_rmse").iloc[0]
        summary["best_by_split"][split_name] = {
            "model": best_row["model"],
            "test_rmse": float(best_row["test_rmse"]),
            "artifact_path": best_row["artifact_path"],
        }

    SUMMARY_PATH.write_text(json.dumps(summary, indent=2))
    print(f"Saved results to {args.results_path}")
    print(f"Saved summary to {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
