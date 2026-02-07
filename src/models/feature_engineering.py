import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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


def infer_column_types(df: pd.DataFrame, target_col: str):
    """
    Infer column types from a pandas DataFrame.

    Returns
    -------
    num_cols : list
        Numeric columns (int, float)
    bool_cols : list
        Boolean columns
    datetime_cols : list
        Datetime columns
    cat_cols : list
        Categorical / object columns
    """
    df = df.drop(columns=[target_col])

    num_cols = []
    bool_cols = []
    datetime_cols = []
    cat_cols = []

    for col in df.columns:
        dtype = df[col].dtype

        if pd.api.types.is_bool_dtype(dtype):
            bool_cols.append(col)

        elif pd.api.types.is_numeric_dtype(dtype):
            num_cols.append(col)

        elif pd.api.types.is_datetime64_any_dtype(dtype):
            datetime_cols.append(col)

        elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
            cat_cols.append(col)

        else:
            # fallback (rare types)
            cat_cols.append(col)

    return num_cols, bool_cols, datetime_cols, cat_cols


def bool_to_int(X: pd.DataFrame):
    Xc = X.copy()
    for col in Xc.columns:
        Xc[col] = Xc[col].astype("int8")
    return Xc


def datetime_to_int64(X: pd.DataFrame):
    """
    Converte datetime -> int64 (nanosecondi dal 1970). Evita feature engineering extra.
    """
    Xc = X.copy()
    for col in Xc.columns:
        # pandas datetime64[ns]
        Xc[col] = pd.to_datetime(Xc[col], errors="coerce").astype("int64")
    return Xc


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
