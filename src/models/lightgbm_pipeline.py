import os
import json
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, FunctionTransformer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterGrid

from lightgbm import LGBMClassifier

import mlflow
import mlflow.lightgbm

# ONNX
# pip install onnx onnxruntime skl2onnx onnxmltools
import onnxmltools
from onnxmltools.convert.common.data_types import FloatTensorType
import onnxruntime as ort


# -----------------------------
# 1) Utility: time-based split
# -----------------------------
def time_based_split(df: pd.DataFrame, time_col: str, train_frac=0.7, val_frac=0.15):
    """
    Split per tempo: train (più vecchio), val (intermedio), test (più recente).
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)

    n_train = int(n * train_frac)
    n_val = int(n * val_frac)

    train_df = df_sorted.iloc[:n_train]
    val_df = df_sorted.iloc[n_train:n_train + n_val]
    test_df = df_sorted.iloc[n_train + n_val:]

    return train_df, val_df, test_df


# ---------------------------------------------
# 2) Utility: detect columns by dtype/semantics
# ---------------------------------------------
def infer_column_types(df: pd.DataFrame, target_col: str, time_col: str):
    X = df.drop(columns=[target_col])
    # datetime columns (including time_col) will be converted to numeric
    datetime_cols = []
    for c in X.columns:
        if c == time_col:
            datetime_cols.append(c)
        elif np.issubdtype(X[c].dtype, np.datetime64):
            datetime_cols.append(c)

    bool_cols = [c for c in X.columns if X[c].dtype == "bool"]
    # string/object/category
    cat_cols = [c for c in X.columns if (X[c].dtype == "object" or str(X[c].dtype) == "category")]
    # numeric (exclude bool which is technically numeric sometimes)
    num_cols = [c for c in X.columns if c not in set(datetime_cols + bool_cols + cat_cols)]

    return num_cols, bool_cols, datetime_cols, cat_cols


# ------------------------------------
# 3) Preprocessing transformers
# ------------------------------------
def datetime_to_int64(X: pd.DataFrame):
    """
    Converte datetime -> int64 (nanosecondi dal 1970). Evita feature engineering extra.
    """
    Xc = X.copy()
    for col in Xc.columns:
        # pandas datetime64[ns]
        Xc[col] = pd.to_datetime(Xc[col], errors="coerce").astype("int64")
    return Xc

def bool_to_int(X: pd.DataFrame):
    Xc = X.copy()
    for col in Xc.columns:
        Xc[col] = Xc[col].astype("int8")
    return Xc


# ------------------------------------
# 4) Feature names post-transform
# ------------------------------------
def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer):
    """
    Estrae i nomi feature dopo ColumnTransformer (incluse one-hot).
    """
    output_features = []
    for name, trans, cols in preprocessor.transformers_:
        if name == "remainder" and trans == "drop":
            continue
        if trans == "passthrough":
            # cols è lista di colonne
            output_features.extend(list(cols))
        else:
            # pipeline o transformer singolo
            if isinstance(trans, Pipeline):
                last = trans.steps[-1][1]
            else:
                last = trans

            if hasattr(last, "get_feature_names_out"):
                # OneHotEncoder / ecc.
                fn = last.get_feature_names_out(cols)
                output_features.extend(list(fn))
            else:
                # fallback: usa i nomi originali
                output_features.extend(list(cols))
    return output_features


# ------------------------------------
# 5) Training + MLflow + importance/drop
# ------------------------------------
def train_eval_lgbm_with_mlflow(
    df: pd.DataFrame,
    target_col: str = "target",
    time_col: str = "time",
    experiment_name: str = "lgbm_binary_time_split",
    run_name: str = "baseline",
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    drop_importance_below: float = 0.0,  # es. 0.0 = niente drop, oppure 1e-6 / 0.0001
    params: dict | None = None,
    onnx_export_path: str = "model.onnx",
):
    df = df.copy()

    # Parse time col
    df[time_col] = pd.to_datetime(df[time_col], errors="coerce")

    # Basic sanity
    df = df.dropna(subset=[time_col, target_col])
    # target must be 0/1
    df[target_col] = df[target_col].astype(int)

    train_df, val_df, test_df = time_based_split(df, time_col, train_frac, val_frac)

    num_cols, bool_cols, datetime_cols, cat_cols = infer_column_types(train_df, target_col, time_col)

    X_train = train_df.drop(columns=[target_col])
    y_train = train_df[target_col].values

    X_val = val_df.drop(columns=[target_col])
    y_val = val_df[target_col].values

    X_test = test_df.drop(columns=[target_col])
    y_test = test_df[target_col].values

    # Preprocess
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_cols),
            ("bool", Pipeline(steps=[
                ("to_df", FunctionTransformer(lambda x: pd.DataFrame(x, columns=bool_cols), validate=False)),
                ("cast", FunctionTransformer(bool_to_int, validate=False)),
            ]), bool_cols),
            ("dt", Pipeline(steps=[
                ("to_df", FunctionTransformer(lambda x: pd.DataFrame(x, columns=datetime_cols), validate=False)),
                ("cast", FunctionTransformer(datetime_to_int64, validate=False)),
            ]), datetime_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse=True), cat_cols),
        ],
        remainder="drop",
        sparse_threshold=0.3,
        verbose_feature_names_out=False,
    )

    # Default LGBM params (puoi modificarli)
    lgbm_params = {
        "n_estimators": 2000,
        "learning_rate": 0.03,
        "num_leaves": 64,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "random_state": 42,
        "n_jobs": -1,
        "objective": "binary",
        "metric": "auc",
    }
    if params:
        lgbm_params.update(params)

    model = LGBMClassifier(**lgbm_params)

    # pipeline sklearn
    pipe = Pipeline(steps=[
        ("prep", preprocessor),
        ("clf", model),
    ])

    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        # Log split info
        mlflow.log_params({
            "train_frac": train_frac,
            "val_frac": val_frac,
            "n_train": len(train_df),
            "n_val": len(val_df),
            "n_test": len(test_df),
            **{f"lgbm__{k}": v for k, v in lgbm_params.items()},
        })

        # Fit with early stopping using validation
        # NB: early_stopping via fit params (LightGBM sklearn API)
        pipe.fit(
            X_train, y_train,
            clf__eval_set=[(preprocessor.fit_transform(X_val), y_val)],  # val transformed
            clf__eval_metric="auc",
            clf__callbacks=[],
        )

        # Predict proba
        p_train = pipe.predict_proba(X_train)[:, 1]
        p_val = pipe.predict_proba(X_val)[:, 1]
        p_test = pipe.predict_proba(X_test)[:, 1]

        auc_train = roc_auc_score(y_train, p_train) if len(np.unique(y_train)) > 1 else np.nan
        auc_val = roc_auc_score(y_val, p_val) if len(np.unique(y_val)) > 1 else np.nan
        auc_test = roc_auc_score(y_test, p_test) if len(np.unique(y_test)) > 1 else np.nan

        mlflow.log_metrics({
            "auc_train": float(auc_train) if np.isfinite(auc_train) else -1.0,
            "auc_val": float(auc_val) if np.isfinite(auc_val) else -1.0,
            "auc_test": float(auc_test) if np.isfinite(auc_test) else -1.0,
        })

        # Feature importance
        prep_fitted = pipe.named_steps["prep"]
        feature_names = get_feature_names_from_preprocessor(prep_fitted)

        booster = pipe.named_steps["clf"].booster_
        importances = booster.feature_importance(importance_type="gain")
        imp_df = pd.DataFrame({
            "feature": feature_names,
            "importance_gain": importances
        }).sort_values("importance_gain", ascending=False)

        imp_csv = "feature_importance_gain.csv"
        imp_df.to_csv(imp_csv, index=False)
        mlflow.log_artifact(imp_csv)

        # Optional drop features under threshold & retrain
        if drop_importance_below > 0.0:
            keep_mask = imp_df["importance_gain"].values > drop_importance_below
            kept_features = imp_df.loc[keep_mask, "feature"].tolist()
            dropped = int((~keep_mask).sum())
            mlflow.log_params({
                "drop_importance_below": drop_importance_below,
                "dropped_features_count": dropped,
                "kept_features_count": len(kept_features),
            })

            # Per droppare in modo robusto con one-hot: selezioniamo colonne DOPO il preprocessor
            # Strategy: trasformiamo X_* e poi addestriamo un secondo LGBM su matrice ridotta.
            Xtr = prep_fitted.transform(X_train)
            Xva = prep_fitted.transform(X_val)
            Xte = prep_fitted.transform(X_test)

            keep_idx = np.where(keep_mask)[0]
            Xtr_k = Xtr[:, keep_idx]
            Xva_k = Xva[:, keep_idx]
            Xte_k = Xte[:, keep_idx]

            model2 = LGBMClassifier(**lgbm_params)
            model2.fit(
                Xtr_k, y_train,
                eval_set=[(Xva_k, y_val)],
                eval_metric="auc",
            )

            p_val2 = model2.predict_proba(Xva_k)[:, 1]
            p_test2 = model2.predict_proba(Xte_k)[:, 1]
            auc_val2 = roc_auc_score(y_val, p_val2) if len(np.unique(y_val)) > 1 else np.nan
            auc_test2 = roc_auc_score(y_test, p_test2) if len(np.unique(y_test)) > 1 else np.nan

            mlflow.log_metrics({
                "auc_val_dropped": float(auc_val2) if np.isfinite(auc_val2) else -1.0,
                "auc_test_dropped": float(auc_test2) if np.isfinite(auc_test2) else -1.0,
            })

            # Log modello ridotto come artifact “secondario”
            mlflow.lightgbm.log_model(model2, artifact_path="lgbm_model_retrained_after_drop")
            # Salviamo anche gli indici keep per riprodurre a runtime
            with open("kept_feature_indices.json", "w") as f:
                json.dump(keep_idx.tolist(), f)
            mlflow.log_artifact("kept_feature_indices.json")

        # Log modello pipeline (preprocess + lgbm)
        mlflow.sklearn.log_model(pipe, artifact_path="sklearn_pipeline_lgbm")

        # ------------------------------------
        # ONNX export (modello puro LightGBM)
        # ------------------------------------
        # Per ONNX più compatto: esportiamo il Booster e a runtime replichi il preprocessing.
        # Qui esportiamo il modello *addestrato sullo spazio trasformato*:
        Xtr_trans = prep_fitted.transform(X_train)
        n_features_trans = Xtr_trans.shape[1]

        # Convert LightGBM booster to ONNX
        # NOTE: output probabilità: dipende dal converter; spesso produce label+probabilities
        initial_types = [("input", FloatTensorType([None, n_features_trans]))]
        onnx_model = onnxmltools.convert_lightgbm(
            booster,
            initial_types=initial_types,
            target_opset=15,
        )

        with open(onnx_export_path, "wb") as f:
            f.write(onnx_model.SerializeToString())

        mlflow.log_artifact(onnx_export_path)

        return {
            "auc_train": auc_train,
            "auc_val": auc_val,
            "auc_test": auc_test,
            "importance_top10": imp_df.head(10),
            "onnx_path": onnx_export_path,
        }


# ------------------------------------
# 6) ONNX runtime inference example
# ------------------------------------
def onnx_predict_proba(onnx_path: str, X_transformed: np.ndarray) -> np.ndarray:
    """
    Esegue inferenza su input già preprocessato (float32).
    """
    if not isinstance(X_transformed, np.ndarray):
        X_transformed = X_transformed.toarray()  # se sparse -> dense (attenzione RAM!)
    X_transformed = X_transformed.astype(np.float32)

    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    input_name = sess.get_inputs()[0].name
    outputs = sess.run(None, {input_name: X_transformed})

    # Converter LightGBM spesso ritorna [label, probabilities]
    # Probabilities può essere shape (N, 2). Prendiamo classe 1.
    probs = outputs[-1]
    if probs.ndim == 2 and probs.shape[1] == 2:
        return probs[:, 1]
    return probs.squeeze()


# -----------------------------
# ESEMPIO USO
# -----------------------------
if __name__ == "__main__":
    # df = pd.read_parquet("data.parquet")  # o csv, ecc.
    # Deve contenere: "time" + "target" + features
    # result = train_eval_lgbm_with_mlflow(df, target_col="target", time_col="time",
    #                                     drop_importance_below=0.0,
    #                                     experiment_name="my_exp",
    #                                     run_name="run_001",
    #                                     onnx_export_path="lgbm.onnx")
    pass
