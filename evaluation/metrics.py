"""
Evaluation Metrics Module for Drug-Target Interaction (DTI) Affinity Prediction.

Implements standard regression and ranking metrics used in DTI benchmarks:
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Pearson Correlation Coefficient (Pearson r)
- Concordance Index (CI)
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Union, List, Tuple


def mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Mean Squared Error (MSE)."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return float(np.mean((y_true - y_pred) ** 2))


def root_mean_squared_error(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Calculate Root Mean Squared Error (RMSE)."""
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))


def pearson_correlation(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Pearson Correlation Coefficient (r).
    Returns 0.0 if variance is zero.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    if len(y_true) < 2:
        return 0.0
        
    std_true = np.std(y_true)
    std_pred = np.std(y_pred)
    
    if std_true == 0 or std_pred == 0:
        return 0.0
        
    corr = np.corrcoef(y_true, y_pred)[0, 1]
    return float(corr) if not np.isnan(corr) else 0.0


def concordance_index(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Calculate Concordance Index (CI) for affinity regression.
    
    CI measures the proportion of pairs where predicted order matches true order:
    CI = sum_{i < j and y_true[i] != y_true[j]} h(y_pred[i], y_pred[j]) / total_valid_pairs
    where h = 1 for concordant ordering, 0.5 for tied predictions, 0 for discordant ordering.
    
    Standard metric in DTI literature (KronRLS, SimBoost, DeepDTA).
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    
    n = len(y_true)
    if n < 2:
        return 0.0
        
    diff_true = y_true[:, None] - y_true[None, :]
    diff_pred = y_pred[:, None] - y_pred[None, :]
    
    # Upper triangle mask for i < j where y_true[i] != y_true[j]
    upper_mask = np.triu(np.ones((n, n), dtype=bool), k=1)
    valid_pairs_mask = upper_mask & (diff_true != 0)
    
    num_valid_pairs = np.sum(valid_pairs_mask)
    if num_valid_pairs == 0:
        return 0.0
        
    # Product of differences: > 0 means both diffs have the same sign (concordant)
    prod = diff_true[valid_pairs_mask] * diff_pred[valid_pairs_mask]
    
    concordant = np.sum(prod > 0)
    tied = np.sum(prod == 0)
    
    ci = (concordant + 0.5 * tied) / num_valid_pairs
    return float(ci)


def evaluate_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute full evaluation metrics dictionary for given true and predicted affinity scores.
    """
    return {
        "mse": mean_squared_error(y_true, y_pred),
        "rmse": root_mean_squared_error(y_true, y_pred),
        "pearson_r": pearson_correlation(y_true, y_pred),
        "ci": concordance_index(y_true, y_pred),
    }


def format_results_matrix(
    results_data: Dict[str, Dict[str, Dict[str, float]]]
) -> pd.DataFrame:
    """
    Format evaluation results into a 12-cell matrix (Algorithms x Splits).
    
    Expected results_data format:
    {
        "Random Split": {
            "Random Forest": {"rmse": 0.45, "ci": 0.78, ...},
            "XGBoost": {"rmse": 0.42, "ci": 0.81, ...},
            ...
        },
        "Cold-Drug Split": { ... },
        "Cold-Protein Split": { ... }
    }
    """
    rows = []
    for split_name, algos in results_data.items():
        for algo_name, metrics in algos.items():
            row = {
                "Split": split_name,
                "Algorithm": algo_name,
                "MSE": metrics.get("mse", np.nan),
                "RMSE": metrics.get("rmse", np.nan),
                "Pearson r": metrics.get("pearson_r", np.nan),
                "CI": metrics.get("ci", np.nan),
            }
            rows.append(row)
            
    df = pd.DataFrame(rows)
    return df
