import optuna
import random
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from itertools import combinations
# from sklearn.pipeline import Pipeline
from pandas.api.types import is_numeric_dtype
# from sklearn.compose import ColumnTransformer

from utils import run_one_study

# def time_based_split(df: pd.DataFrame, time_col: str, train_frac=0.7, val_frac=0.15):
#     """
#     Split per tempo: train (più vecchio), val (intermedio), test (più recente).
#     """
#     df_sorted = df.sort_values(time_col).reset_index(drop=True)
#     n = len(df_sorted)

#     n_train = int(n * train_frac)
#     n_val = int(n * val_frac)

#     train_df = df_sorted.iloc[:n_train]
#     val_df = df_sorted.iloc[n_train:n_train + n_val]
#     test_df = df_sorted.iloc[n_train + n_val:]
#     return train_df, val_df, test_df


# def infer_column_types(df: pd.DataFrame, target_col: str):
#     """
#     Infer column types from a pandas DataFrame.

#     Returns
#     -------
#     num_cols : list
#         Numeric columns (int, float)
#     bool_cols : list
#         Boolean columns
#     datetime_cols : list
#         Datetime columns
#     cat_cols : list
#         Categorical / object columns
#     """
#     df = df.drop(columns=[target_col])

#     num_cols = []
#     bool_cols = []
#     datetime_cols = []
#     cat_cols = []

#     for col in df.columns:
#         dtype = df[col].dtype

#         if pd.api.types.is_bool_dtype(dtype):
#             bool_cols.append(col)

#         elif pd.api.types.is_numeric_dtype(dtype):
#             num_cols.append(col)

#         elif pd.api.types.is_datetime64_any_dtype(dtype):
#             datetime_cols.append(col)

#         elif pd.api.types.is_categorical_dtype(dtype) or pd.api.types.is_object_dtype(dtype):
#             cat_cols.append(col)

#         else:
#             # fallback (rare types)
#             cat_cols.append(col)

#     return num_cols, bool_cols, datetime_cols, cat_cols


# def bool_to_int(x: pd.DataFrame):
#     xc = x.copy()
#     for col in xc.columns:
#         xc[col] = xc[col].astype("int8")
#     return xc


# def datetime_to_int64(x: pd.DataFrame):
#     """
#     Converte datetime -> int64 (nanosecondi dal 1970). Evita feature engineering extra.
#     """
#     xc = x.copy()
#     for col in xc.columns:
#         # pandas datetime64[ns]
#         xc[col] = pd.to_datetime(xc[col], errors="coerce").astype("int64")
#     return xc


# def get_feature_names_from_preprocessor(preprocessor: ColumnTransformer):
#     """
#     Estrae i nomi feature dopo ColumnTransformer (include one-hot).
#     """
#     output_features = []
#     for name, trans, cols in preprocessor.transformers_:
#         if name == "remainder" and trans == "drop":
#             continue
#         if trans == "passthrough":
#             # cols è lista di colonne
#             output_features.extend(list(cols))
#         else:
#             # pipeline o transformer singolo
#             if isinstance(trans, Pipeline):
#                 last = trans.steps[-1][1]
#             else:
#                 last = trans

#             if hasattr(last, "get_feature_names_out"):
#                 # OneHotEncoder / ecc.
#                 fn = last.get_feature_names_out(cols)
#                 output_features.extend(list(fn))
#             else:
#                 # fallback: usa i nomi originali
#                 output_features.extend(list(cols))
#     return output_features


# def get_chance_1x2_comparison_affini_clusters(df: pd.DataFrame) -> pd.DataFrame:
#     df['chance1x2_comparison_affini_3rd_cluster'] = df['chance1x2_comparison_affini'] <= 200
#     df['chance1x2_comparison_affini_2nd_cluster'] = (
#         df['chance1x2_comparison_affini'] > 200) & (df['chance1x2_comparison_affini'] <= 500)
#     df['chance1x2_comparison_affini_1st_cluster'] = df['chance1x2_comparison_affini'] > 500
#     return df


# def get_under_over_comparison_affini_clusters(df: pd.DataFrame) -> pd.DataFrame:
#     df['underOver_comparison_affini_3rd_cluster'] = df['underOver_comparison_affini'] <= 2000
#     df['underOver_comparison_affini_2nd_cluster'] = (
#         df['underOver_comparison_affini'] > 2000) & (df['underOver_comparison_affini'] <= 5000)
#     df['underOver_comparison_affini_1st_cluster'] = df['underOver_comparison_affini'] > 5000
#     return df


# def get_goal_no_goal_comparison_affini_clusters(df: pd.DataFrame) -> pd.DataFrame:
#     df['goalNoGoal_comparison_affini_3rd_cluster'] = df['goalNoGoal_comparison_affini'] <= 2000
#     df['goalNoGoal_comparison_affini_2nd_cluster'] = (
#         df['goalNoGoal_comparison_affini'] > 2000) & (df['goalNoGoal_comparison_affini'] <= 4500)
#     df['goalNoGoal_comparison_affini_1st_cluster'] = df['goalNoGoal_comparison_affini'] > 4500
#     return df


def time_based_train_test_split(df: pd.DataFrame, time_col: str, train_frac=0.7):
    """
    Split per tempo: train (più vecchio), test (più recente).
    """
    df_sorted = df.sort_values(time_col).reset_index(drop=True)
    n = len(df_sorted)

    n_train = int(n * train_frac)

    train_df = df_sorted.iloc[:n_train]
    test_df = df_sorted.iloc[n_train:]
    return train_df, test_df


def round_to_step(x, step):
    """
    Arrotonda x a un multiplo di 'step'.
    """
    try:
        if not step:
            return str(x)
        elif step <= 0:
            raise ValueError("step deve essere > 0")
        else:
            ratio = x / step
            return round(ratio) * step
    
    except Exception as e: 
        return None


def get_return(strategy: str, odds: str):
    if strategy:
        gain = odds - 1
    else:
        gain = -1
    return gain


def get_mask(df, params_dict):
    mask = pd.Series(True, index=df.index)
    features = set()

    for key in params_dict:
        for suffix in ["_cat", "_use_min", "_use_max", "_min", "_max", "_include_missing"]: # ,   "_min_idx", "_max_idx", 
             if key.endswith(suffix):
                features.add(key[:-len(suffix)])
                break

    for feat in features:
        s = df[feat]

        include_missing = params_dict.get(f"{feat}_include_missing", False)
        use_min = params_dict.get(f"{feat}_use_min", True)
        use_max = params_dict.get(f"{feat}_use_max", True)

        feat_mask = pd.Series(True, index=df.index)

        # Categorical filtering
        if f"{feat}_cat" in params_dict:
            feat_mask &= s.isin(params_dict[f"{feat}_cat"])

        # Numerical filtering
        if f"{feat}_min" in params_dict and use_min:
            feat_mask &= s >= params_dict[f"{feat}_min"]

        if f"{feat}_max" in params_dict and use_max:
            feat_mask &= s <= params_dict[f"{feat}_max"]

        if include_missing:
            feat_mask = feat_mask | s.isna()
        else:
            feat_mask = feat_mask & s.notna()

        if params_dict[f"use_{feat}"]:
            mask &= feat_mask

    return mask


def add_week_column(df):
    df["week"] = df["time"].dt.to_period("W")
    monday_start = df["time"].min() - pd.to_timedelta(df["time"].min().weekday(), unit="D")
    week_series = ((df["time"] - monday_start).dt.days // 7) + 1
    return week_series


def objective(trial, min_obs:int, _lambda: float, feature_bins_map: dict, df_binned: pd.DataFrame, suggest_feature: bool = True):
    mask = pd.Series(True, index=df_binned.index)
    params_dict_item = dict()

    for feat, step in feature_bins_map.items():

        s = df_binned[feat]
        non_null = s.dropna()

        if non_null.empty:
            continue

        # ---------------------------------
        # CASO 1: variabile categorica
        # ---------------------------------
        if not step:
            # ordine stabile e deterministico
            unique_vals = sorted(map(str, non_null.unique()))

            categorical_set = [
                list(c)
                for r in range(1, len(unique_vals) + 1)
                for c in combinations(unique_vals, r)
            ]

            idx = trial.suggest_categorical(
                f"{feat}_cat_idx",
                list(range(len(categorical_set)))
            )

            categorical_feat = categorical_set[idx]
            # trial.set_user_attr(f"{feat}_cat", categorical_feat)
            params_dict_item[f"{feat}_cat"] = categorical_feat
            
            if suggest_feature:
                use_feat = trial.suggest_categorical(f"use_{feat}", [True, False])
                params_dict_item[f"use_{feat}"] = use_feat


        # ---------------------------------
        # CASO 2: numerica continua/intera
        # ---------------------------------
        else:
            # # TODO: se la feature ha valori nulli, introduco include_missing feature, valutare rimozione  
            # if s.isna().sum() > 0:
            #     include_missing = trial.suggest_categorical(
            #         f"{feat}_include_missing",
            #         [True, False]
            #     )
            #     # trial.set_user_attr(f"{feat}_include_missing", include_missing)
            #     params_dict_item[f"{feat}_include_missing"] = include_missing

            # TODO: vincolo per >=, <=, <,>, da studiare
            use_min = trial.suggest_categorical(f"{feat}_use_min", [True, False])
            use_max = trial.suggest_categorical(f"{feat}_use_max", [True, False])

            # trial.set_user_attr(f"{feat}_use_min", use_min)
            # trial.set_user_attr(f"{feat}_use_max", use_max)
            params_dict_item[f"{feat}_use_min"] = use_min
            params_dict_item[f"{feat}_use_max"] = use_max

            if suggest_feature:
                use_feat = trial.suggest_categorical(f"use_{feat}", [True, False])
                params_dict_item[f"use_{feat}"] = use_feat
            

            # TODO: questo dovrebbe forzare l'utilizzo della feature, da capire se forzarlo è utile o meno
            # # almeno un vincolo deve essere attivo
            if not use_min and not use_max:
                raise optuna.TrialPruned()
            
            if pd.api.types.is_integer_dtype(s):
                feat_min = trial.suggest_int(
                    name=f"{feat}_min",
                    low=int(non_null.min()),
                    high=int(non_null.max()) - step, # considero upper bound meno step
                    step=step
                )
                feat_max = trial.suggest_int(
                    name=f"{feat}_max",
                    low=int(non_null.min()), # feat_min + step, # considero lower bound più step
                    high=int(non_null.max()), 
                    step=step
                )
                # trial.set_user_attr(f"{feat}_min", feat_min)
                # trial.set_user_attr(f"{feat}_max", feat_max)
                if feat_min >= feat_max:
                    raise optuna.TrialPruned()
                params_dict_item[f"{feat}_min"] = feat_min
                params_dict_item[f"{feat}_max"] = feat_max
            else:
                feat_min = trial.suggest_float(
                    name=f"{feat}_min",
                    low=float(non_null.min()),
                    high=float(non_null.max()) - step, # considero upper bound meno step
                    step=step
                )
                feat_max = trial.suggest_float(
                    name=f"{feat}_max",
                    low=float(non_null.min()), # feat_min + step, # considero lower bound più step
                    high=float(non_null.max()),
                    step=step
                )
                # trial.set_user_attr(f"{feat}_min", feat_min)
                # trial.set_user_attr(f"{feat}_max", feat_max)
                if feat_min >= feat_max:
                    raise optuna.TrialPruned()
                params_dict_item[f"{feat}_min"] = feat_min
                params_dict_item[f"{feat}_max"] = feat_max

        trial.set_user_attr("params_dict", params_dict_item)

        # costruzione maschera feature
        feat_mask = pd.Series(True, index=s.index)

    
        if not step:
            # Categorical filtering
            feat_mask &= s.isin(categorical_feat)
        else:
            # Numerical filtering
            if use_min:
                feat_mask &= s >= feat_min

            if use_max:
                feat_mask &= s <= feat_max

            # Filtering for feature with nan
            if s.isna().sum() > 0:
                if include_missing:
                    feat_mask = feat_mask | s.isna()
                else:
                    feat_mask = feat_mask & s.notna()

        if use_feat:
            mask &= feat_mask


    selected = df_binned.loc[mask]

    if len(selected) <= 0:
        return -1e9
    
    mean_weekly_return = selected.groupby("week")["return"].sum().mean() # massimizzo ritorno settimanale
    std_weekly_return = selected.groupby("week")["return"].sum().std(ddof=0) # minimizzo volatilità ritorno settimanale


    if len(selected) < min_obs:
        return -1e9
    
    score = mean_weekly_return - _lambda * std_weekly_return 

    return float(score)



# ---

from itertools import combinations

def get_feature_bins_map(df, features):

    feature_bins_map = dict()

    for feat in features:
        if is_numeric_dtype(df[feat]):
            feature_bins_map[feat] = _get_step(df, feat)
        else:
            feature_bins_map[feat] = None
    return feature_bins_map


def _get_step(df, feature):
    q1 = df[[feature]].describe().loc["25%"].values[0]
    q3 = df[[feature]].describe().loc["75%"].values[0]

    diff = q3 - q1
    
    if df[feature].nunique() <= 20:
        return None

    if 0 < diff <= 1:
        step = 0.2

    elif 1 < diff <= 5:
        step = 1

    elif 5 < diff <= 100:
        step = 5
    elif 100 < diff <= 500:
        step = 100
    elif diff >= 500:
        step = 500
    else: 
        raise Exception
    return step


def _objective_features(trial, min_obs:int, _lambda: float, feature_bins_map: dict, df_binned: pd.DataFrame):
    mask = pd.Series(True, index=df_binned.index)
    params_dict_item = dict()

    for feat, step in feature_bins_map.items():
        s = df_binned[feat]
        non_null = s.dropna()

        if non_null.empty:
            continue

        # ---------------------------------
        # CASO 1: variabile categorica
        # ---------------------------------
        if not step:
            # ordine stabile e deterministico
            unique_vals = sorted(map(str, non_null.unique()))

            categorical_set = [
                list(c)
                for r in range(1, len(unique_vals) + 1)
                for c in combinations(unique_vals, r)
            ]

            idx = trial.suggest_categorical(
                f"{feat}_cat_idx",
                list(range(len(categorical_set)))
            )

        
            categorical_feat = categorical_set[idx]
            params_dict_item[f"{feat}_cat"] = categorical_feat


        # ---------------------------------
        # CASO 2: numerica continua/intera
        # ---------------------------------
        else:
            # # TODO: se la feature ha valori nulli, introduco include_missing feature, valutare rimozione  
            # if s.isna().sum() > 0:
            #     include_missing = trial.suggest_categorical(
            #         f"{feat}_include_missing",
            #         [True, False]
            #     )
            #     params_dict_item[f"{feat}_include_missing"] = include_missing

            # TODO: vincolo per >=, <=, <,>, da studiare
            use_min = trial.suggest_categorical(f"{feat}_use_min", [True, False])
            use_max = trial.suggest_categorical(f"{feat}_use_max", [True, False])


            params_dict_item[f"{feat}_use_min"] = use_min
            params_dict_item[f"{feat}_use_max"] = use_max

            
            if pd.api.types.is_integer_dtype(s):
                feat_min = trial.suggest_int(
                    name=f"{feat}_min",
                    low=int(non_null.min()),
                    high=int(non_null.max()) - step, # considero upper bound meno step
                    step=step
                )
                feat_max = trial.suggest_int(
                    name=f"{feat}_max",
                    low=int(non_null.min()), # feat_min + step, # considero lower bound più step
                    high=int(non_null.max()), 
                    step=step
                )

                if feat_min >= feat_max:
                    raise optuna.TrialPruned()
                
                params_dict_item[f"{feat}_min"] = feat_min
                params_dict_item[f"{feat}_max"] = feat_max
            else:
                feat_min = trial.suggest_float(
                    name=f"{feat}_min",
                    low=float(non_null.min()),
                    high=float(non_null.max()) - step, # considero upper bound meno step
                    step=step
                )
                feat_max = trial.suggest_float(
                    name=f"{feat}_max",
                    low=float(non_null.min()), # feat_min + step, # considero lower bound più step
                    high=float(non_null.max()),
                    step=step
                )

                if feat_min >= feat_max:
                    raise optuna.TrialPruned()
                
                params_dict_item[f"{feat}_min"] = feat_min
                params_dict_item[f"{feat}_max"] = feat_max

        trial.set_user_attr("params_dict", params_dict_item)

        # costruzione maschera feature
        feat_mask = pd.Series(True, index=s.index)

        if not step:
            # Categorical filtering
            feat_mask &= s.isin(categorical_feat)
        else:
            # Numerical filtering
            if use_min:
                feat_mask &= s >= feat_min

            if use_max:
                feat_mask &= s <= feat_max

            # Filtering for feature with nan
            if s.isna().sum() > 0:
                if include_missing:
                    feat_mask = feat_mask | s.isna()
                else:
                    feat_mask = feat_mask & s.notna()

        mask &= feat_mask

    selected = df_binned.loc[mask]
    
    mean_weekly_return = selected.groupby("week")["return"].sum().mean() # massimizzo ritorno settimanale
    std_weekly_return = selected.groupby("week")["return"].sum().std(ddof=0) # minimizzo volatilità ritorno settimanale


    if len(selected) < min_obs:
        return -1e9
    
    score = mean_weekly_return - _lambda * std_weekly_return 

    return float(score)

def get_best_features(df=None, features=None, objective_hyperparameters=None):

    path = Path("./feature_rank.csv")
    if path.exists():
        df_feat = pd.read_csv("./feature_rank.csv", index_col=0)
        return df_feat

    feature_rank = dict()
    feature_bins_map = get_feature_bins_map(df, features)

    for key, value in tqdm(feature_bins_map.items()):
        feature_bins_map_item = {key: value}
        
        # Create the binned dataframe
        df_train_binned = df.copy()

        for feat, step in feature_bins_map_item.items():
            df_train_binned[feat] = [round_to_step(x, step) for x in df_train_binned[feat]]

            if isinstance(step, int):
                df_train_binned[feat] = df_train_binned[feat].astype("Int64")

        # Generate random seeds
        seeds = [random.randint(0, 2**32 - 1) for _ in range(objective_hyperparameters["n_studies"])]
        studies = [
            run_one_study(
                seed=s, 
                objective_hyperparameters=objective_hyperparameters, 
                objective=_objective_features,
                feature_bins_map=feature_bins_map_item, 
                df_binned=df_train_binned
                ) 
                for s in seeds]

        all_trials = []
        for study in studies:
            all_trials.extend(
                [t for t in study.trials if t.value is not None and t.state.name == "COMPLETE"]
            )
        if all_trials:
            best_trial = sorted(all_trials, key=lambda t: -t.value)[0]
            feature_rank[feat] = best_trial.value
    
    sorted_feat = dict(sorted(feature_rank.items(), key=lambda item: item[1], reverse=True))
    df_feat = pd.DataFrame(list(sorted_feat.items()), columns=['feature', 'score'])

    df_feat.to_csv("./feature_rank.csv")

    return df_feat