import json
import optuna
import numpy as np
import pandas as pd
from onnxruntime import InferenceSession
from sklearn.compose import ColumnTransformer


def get_onnx_prediction(sess: InferenceSession, preprocessor: ColumnTransformer, input: pd.DataFrame, threshold: float = 0.5):
    # Preprocess input
    # trasformazione del preprocessor salvato
    x_trans = preprocessor.transform(input)
    x_trans = x_trans.astype(np.float32)        # ONNX richiede float32
    # ONNX Inference
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: x_trans})

    # Controlla quanti output ci sono e seleziona probabilità
    # se ONNX produce [label, probability]
    y_prob = np.array([x[1] for x in outputs[1]])
    y_pred = np.array([int(x > threshold) for x in y_prob])

    return y_prob, y_pred


def check_onnx_predictions(y_prob: np.array, p_test: np.array, round_digits: int = 3):
    assert sum(np.round(y_prob, round_digits) ==
               np.round(p_test, round_digits)) == len(y_prob)


def check_overfitting(best_pred_perc, best_prec, best_roi, test_roi, test_prec, test_pred_perc):
    roi_floor = best_roi*0.80
    roi_ceil = best_roi*1.20

    prec_floor = best_prec*0.97
    prec_ceil = best_prec*1.03

    pred_perc_floor = best_pred_perc*0.95
    pred_perc_ceil = best_pred_perc*1.05

    try:
        assert ((test_roi > roi_floor) and (test_roi < roi_ceil))
        print(
            f"ROI OK! Val: {best_roi:.4f}, Test: {test_roi:.4f}, Limiti: [{roi_floor:.4f}, {roi_ceil:.4f}]")
    except AssertionError:
        print(
            f"WARNING: ROI fuori dai limiti di overfitting! Val: {best_roi:.4f}, Test: {test_roi:.4f}, Limiti: [{roi_floor:.4f}, {roi_ceil:.4f}]")
    try:
        assert ((test_prec > prec_floor) and (test_prec < prec_ceil))
        print(
            f"Precision OK! Val: {best_prec:.4f}, Test: {test_prec:.4f}, Limiti: [{prec_floor:.4f}, {prec_ceil:.4f}]")
    except AssertionError:
        print(
            f"WARNING: Precision fuori dai limiti di overfitting! Val: {best_prec:.4f}, Test: {test_prec:.4f}, Limiti: [{prec_floor:.4f}, {prec_ceil:.4f}]")
    try:
        assert ((test_pred_perc > pred_perc_floor)
                and (test_pred_perc < pred_perc_ceil))
        print(
            f"Pred perc OK! Val: {best_pred_perc:.4f}, Test: {test_pred_perc:.4f}, Limiti: [{pred_perc_floor:.4f}, {pred_perc_ceil:.4f}]")
    except AssertionError:
        print(
            f"WARNING: Pred perc fuori dai limiti di overfitting! Val: {best_pred_perc:.4f}, Test: {test_pred_perc:.4f}, Limiti: [{pred_perc_floor:.4f}, {pred_perc_ceil:.4f}]")

    return None


def trial_key(t: optuna.trial.FrozenTrial) -> str:
    # chiave semplice per evitare di aggiungere due volte lo stesso best trial
    return json.dumps(
        {
            "params": t.params,
            "value": t.value,
        },
        sort_keys=True,
        default=str,
    )


def run_one_study(seed, objective, objective_hyperparameters, feature_bins_map, df_binned):
    sampler = optuna.samplers.TPESampler(
        seed=seed,
        multivariate=False,
        group=False,
        n_startup_trials=objective_hyperparameters["tpe_sampler"]["n_startup_trials"], 
        n_ei_candidates=objective_hyperparameters["tpe_sampler"]["n_ei_candidates"], 
    )
    study = optuna.create_study(direction="maximize", sampler=sampler,) # gc is garbage collector, when set to True it prevents running otu of memory
    study.optimize(
        lambda trial: objective(
            trial=trial, 
            min_obs=objective_hyperparameters["min_obs"], 
            _lambda=objective_hyperparameters["lambda"], 
            feature_bins_map=feature_bins_map,
            df_binned=df_binned),
        n_trials=objective_hyperparameters["n_trials"]
        )
    return study