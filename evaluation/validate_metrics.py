"""
Validation script for evaluation metrics pipeline.
Run this script to verify metric computations on synthetic benchmark test cases.
"""

import sys
import os
import numpy as np

# Ensure parent directory is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from evaluation.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    pearson_correlation,
    concordance_index,
    evaluate_predictions,
    format_results_matrix,
)


def test_perfect_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])

    metrics = evaluate_predictions(y_true, y_pred)
    assert np.isclose(metrics["mse"], 0.0), f"Expected MSE 0.0, got {metrics['mse']}"
    assert np.isclose(metrics["rmse"], 0.0), f"Expected RMSE 0.0, got {metrics['rmse']}"
    assert np.isclose(metrics["pearson_r"], 1.0), f"Expected Pearson r 1.0, got {metrics['pearson_r']}"
    assert np.isclose(metrics["ci"], 1.0), f"Expected CI 1.0, got {metrics['ci']}"
    print("[PASS] test_perfect_prediction")


def test_inverted_prediction():
    y_true = np.array([1.0, 2.0, 3.0, 4.0])
    y_pred = np.array([4.0, 3.0, 2.0, 1.0])

    metrics = evaluate_predictions(y_true, y_pred)
    assert np.isclose(metrics["pearson_r"], -1.0), f"Expected Pearson r -1.0, got {metrics['pearson_r']}"
    assert np.isclose(metrics["ci"], 0.0), f"Expected CI 0.0, got {metrics['ci']}"
    print("[PASS] test_inverted_prediction")


def test_tied_prediction():
    y_true = np.array([1.0, 2.0, 3.0])
    y_pred = np.array([2.5, 2.5, 2.5])

    ci = concordance_index(y_true, y_pred)
    assert np.isclose(ci, 0.5), f"Expected CI 0.5 for tied predictions, got {ci}"
    print("[PASS] test_tied_prediction")


def test_results_matrix_formatting():
    mock_results = {
        "Random Split": {
            "Random Forest": {"mse": 0.20, "rmse": 0.447, "pearson_r": 0.85, "ci": 0.78},
            "XGBoost": {"mse": 0.18, "rmse": 0.424, "pearson_r": 0.87, "ci": 0.81},
        },
        "Cold-Drug Split": {
            "Random Forest": {"mse": 0.35, "rmse": 0.591, "pearson_r": 0.65, "ci": 0.68},
            "XGBoost": {"mse": 0.32, "rmse": 0.565, "pearson_r": 0.68, "ci": 0.70},
        },
        "Cold-Protein Split": {
            "Random Forest": {"mse": 0.40, "rmse": 0.632, "pearson_r": 0.60, "ci": 0.64},
            "XGBoost": {"mse": 0.38, "rmse": 0.616, "pearson_r": 0.62, "ci": 0.66},
        },
    }

    df = format_results_matrix(mock_results)
    assert len(df) == 6, f"Expected 6 rows in matrix, got {len(df)}"
    assert "Split" in df.columns and "Algorithm" in df.columns and "CI" in df.columns
    print("[PASS] test_results_matrix_formatting")


if __name__ == "__main__":
    print("=== Running Evaluation Metrics Validation Suite ===")
    test_perfect_prediction()
    test_inverted_prediction()
    test_tied_prediction()
    test_results_matrix_formatting()
    print("=== All Validation Tests Passed Successfully ===")
