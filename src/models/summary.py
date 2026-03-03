import os
import numpy as np


def log_betting_summary(
    *,
    target_col: str,
    best_thr: float,
    best: dict,  # expects keys: roi, precision, pred_perc, median_odds
    p_val: np.ndarray,
    p_test: np.ndarray,
    test_prec: float,
    test_pred_perc: float,
    test_roi: float,
    ci_low_test_roi: float,
    ci_high_test_roi: float,
    test_median_odds: float,
    test_prec_onnx: float,
    test_pred_perc_onnx: float,
    test_roi_onnx: float,
    test_median_odds_onnx: float,
    median_bets_per_week: int,
    n_perc_min: float,
    n_perc_max: float | None,
    mlflow,
    artifacts_root: str = "../artifacts",
    end_run: bool = True,
    ) -> tuple[str, str]:
    """
    Builds a text summary, saves it to ../artifacts/<target_col>/summary.txt,
    logs it + metrics/params to mlflow, prints the summary, and optionally ends the run.

    Returns:
        (output_str, summary_filepath)
    """
    # ---- paths ----
    summary_dir = os.path.join(artifacts_root, str(target_col))
    os.makedirs(summary_dir, exist_ok=True)
    summary_filepath = os.path.join(summary_dir, "summary.txt")

    # ---- Validation metrics (best threshold selected by ROI) ----
    best_roi = float(best.get("roi", np.nan))
    ci_low_best_roi = float(best.get("ci_low", np.nan))
    ci_high_best_roi = float(best.get("ci_high", np.nan))
    best_prec = float(best.get("precision", np.nan))
    best_pred_perc = float(best.get("pred_perc", np.nan))
    best_median_odd = float(best.get("median_odds", np.nan))

    val_size = int(len(p_val))
    predicted_val_records = int(round(best_pred_perc * val_size)) if np.isfinite(best_pred_perc) else 0

    # ---- Test metrics (sklearn) ----
    test_size = int(len(p_test))
    predicted_test_records = int(round(test_pred_perc * test_size)) if np.isfinite(test_pred_perc) else 0

    median_odd = float(test_median_odds)
    roi = float(test_roi)

    # ---- ONNX metrics (test) ----
    predicted_test_records_onnx = (
        int(round(test_pred_perc_onnx * test_size)) if np.isfinite(test_pred_perc_onnx) else 0
    )

    # ---- report text ----
    output_str = ""
    output_str += "==============================\n"
    output_str += "📊 MODEL PERFORMANCE SUMMARY\n"
    output_str += "==============================\n\n"

    output_str += "🔎 VALIDATION SET (threshold tuned to maximize ROI)\n"
    output_str += f"🎯 Best Threshold: {best_thr:.3f}\n"
    output_str += f"💰 ROI: {best_roi:.4f}. CI (90%) = [{ci_low_best_roi:.4f}, {ci_high_best_roi:.4f}] \n"
    output_str += f"✅ Precision: {best_prec:.4f}\n"
    output_str += (
        f"📈 Predicted Bets: {best_pred_perc*100:.2f}% "
        f"[{predicted_val_records}/{val_size}]\n"
    )
    output_str += f"Median odd: {best_median_odd:.4f}\n\n"

    output_str += "🧪 TEST SET\n"
    output_str += f"💰 ROI: {roi:.4f}. CI (90%) = [{ci_low_test_roi:.4f}, {ci_high_test_roi:.4f}] \n"
    output_str += f"✅ Precision: {test_prec:.4f}\n"
    output_str += (
        f"📈 Predicted Bets: {test_pred_perc*100:.2f}% "
        f"[{predicted_test_records}/{test_size}]\n"
    )
    output_str += f"Median odd: {median_odd:.4f}\n\n"

    output_str += "⚙️ ONNX MODEL (TEST SET)\n"
    output_str += f"💰 ROI: {float(test_roi_onnx):.4f}\n"
    output_str += f"✅ Precision: {float(test_prec_onnx):.4f}\n"
    output_str += (
        f"📈 Predicted Bets: {float(test_pred_perc_onnx)*100:.2f}% "
        f"[{predicted_test_records_onnx}/{test_size}]\n"
    )
    output_str += f"Median odd: {float(test_median_odds_onnx):.4f}\n\n"

    output_str += "💰 BUSINESS METRICS\n"
    exp_bets_week = int(round(int(median_bets_per_week) * float(test_pred_perc))) if np.isfinite(test_pred_perc) else -1
    output_str += f"📅 Expected Bets per Week: {exp_bets_week}\n"
    output_str += f"📊 ROI (Test Set): {roi:.4f}\n"

    # ---- write file ----
    with open(summary_filepath, "w", encoding="utf-8") as f:
        f.write(output_str)

    # ---- mlflow logging ----
    mlflow.log_artifact(summary_filepath)
    mlflow.log_metrics({
        "roi_val": best_roi if np.isfinite(best_roi) else -1.0,
        "ci_low_val": ci_low_best_roi if np.isfinite(ci_low_best_roi) else -1.0,
        "ci_high_val": ci_high_best_roi if np.isfinite(ci_high_best_roi) else -1.0,
        "precision_val": best_prec if np.isfinite(best_prec) else -1.0,
        "median_odd_val": best_median_odd if np.isfinite(best_median_odd) else -1.0,
        "pred_perc_val": best_pred_perc if np.isfinite(best_pred_perc) else -1.0,

        "roi_test": roi if np.isfinite(roi) else -1.0,
        "ci_low_test": ci_low_test_roi if np.isfinite(ci_low_test_roi) else -1.0,
        "ci_high_test": ci_high_test_roi if np.isfinite(ci_high_test_roi) else -1.0,
        "precision_test": float(test_prec) if np.isfinite(test_prec) else -1.0,
        "median_odd_test": median_odd if np.isfinite(median_odd) else -1.0,
        "pred_perc_test": float(test_pred_perc) if np.isfinite(test_pred_perc) else -1.0,

        "roi_test_onnx": float(test_roi_onnx) if np.isfinite(test_roi_onnx) else -1.0,
        "precision_test_onnx": float(test_prec_onnx) if np.isfinite(test_prec_onnx) else -1.0,
        "median_odd_test_onnx": float(test_median_odds_onnx) if np.isfinite(test_median_odds_onnx) else -1.0,
        "pred_perc_test_onnx": float(test_pred_perc_onnx) if np.isfinite(test_pred_perc_onnx) else -1.0,

        "bets_per_week": exp_bets_week,
    })

    mlflow.log_params({
        "best_threshold": float(best_thr),
        "n_perc_min": float(n_perc_min),
        "n_perc_max": None if n_perc_max is None else float(n_perc_max),
    })

    print(output_str)

    if end_run:
        mlflow.end_run()

    return None


 