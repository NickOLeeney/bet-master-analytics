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


def bool_to_int(x: pd.DataFrame):
    xc = x.copy()
    for col in xc.columns:
        xc[col] = xc[col].astype("int8")
    return xc


def datetime_to_int64(x: pd.DataFrame):
    """
    Converte datetime -> int64 (nanosecondi dal 1970). Evita feature engineering extra.
    """
    xc = x.copy()
    for col in xc.columns:
        # pandas datetime64[ns]
        xc[col] = pd.to_datetime(xc[col], errors="coerce").astype("int64")
    return xc


def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer):
    """
    Estrae i nomi feature dopo ColumnTransformer (include one-hot).
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


def get_chance_1x2_comparison_affini_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df['chance1x2_comparison_affini_3rd_cluster'] = df['chance1x2_comparison_affini'] <= 200
    df['chance1x2_comparison_affini_2nd_cluster'] = (
        df['chance1x2_comparison_affini'] > 200) & (df['chance1x2_comparison_affini'] <= 500)
    df['chance1x2_comparison_affini_1st_cluster'] = df['chance1x2_comparison_affini'] > 500
    return df


def get_under_over_comparison_affini_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df['underOver_comparison_affini_3rd_cluster'] = df['underOver_comparison_affini'] <= 2000
    df['underOver_comparison_affini_2nd_cluster'] = (
        df['underOver_comparison_affini'] > 2000) & (df['underOver_comparison_affini'] <= 5000)
    df['underOver_comparison_affini_1st_cluster'] = df['underOver_comparison_affini'] > 5000
    return df


def get_goal_no_goal_comparison_affini_clusters(df: pd.DataFrame) -> pd.DataFrame:
    df['goalNoGoal_comparison_affini_3rd_cluster'] = df['goalNoGoal_comparison_affini'] <= 2000
    df['goalNoGoal_comparison_affini_2nd_cluster'] = (
        df['goalNoGoal_comparison_affini'] > 2000) & (df['goalNoGoal_comparison_affini'] <= 4500)
    df['goalNoGoal_comparison_affini_1st_cluster'] = df['goalNoGoal_comparison_affini'] > 4500
    return df
