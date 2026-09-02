import os
import pandas as pd
from pandas import json_normalize
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()

def get_match_data():
    client = MongoClient(os.getenv("MONGO_CONNECTION_STRING"))
    db = client["betmaster"]
    matches = db["matches"]
    cursor = matches.find()

    # Convert to DataFrame
    df = pd.DataFrame(list(cursor))
    # Convert dataframe rows to dicts, then normalize
    df_flat = json_normalize(df.to_dict(orient="records"), sep=".")
 
    # Optional cleanup
    df_flat = df_flat.drop(columns=[c for c in df_flat.columns if c.strip() == ""], errors="ignore")
    df_flat = df_flat.drop(columns=["chance1x2.results", "chance1x2", "evaluation", "goalNoGoal", "underOver", "timestamp"])

    # STRING
    string_cols = ["_id", "matchId", "team.league", "team.home", "team.away", "chance1x2.bookkeeping.arrow", "goalNoGoal.bookkeeping.arrow", "underOver.bookkeeping.arrow"]
    for col in string_cols:
        df_flat[col] = df_flat[col].astype('string')

    # TIMESTAMP
    timestamp_cols = ["time"]
    for col in timestamp_cols:
        df_flat[col] = pd.to_datetime(df_flat[col], errors="coerce", format="%d-%m-%y %H:%M")
     
    # FLOAT / DOUBLE
    float_cols = ["chance1x2.chance.p1", "chance1x2.chance.px", "chance1x2.chance.p2", "chance1x2.chance.p1x", "chance1x2.chance.p2x", "chance1x2.chance.p12", "chance1x2.chance.pHt1", "chance1x2.chance.pHtx", "chance1x2.chance.pHt2", "chance1x2.chance.p2Ht1x", "chance1x2.chance.p2Ht2x", "chance1x2.chance.p2Ht12", "chance1x2.quote.real1", "chance1x2.quote.realx", "chance1x2.quote.real2", "chance1x2.quote.initial1", "chance1x2.quote.initialx", "chance1x2.quote.initial2", "chance1x2.quote.current1", "chance1x2.quote.currentx", "chance1x2.quote.current2", "chance1x2.quote.diffRealCurr1", "chance1x2.quote.diffRealCurrx", "chance1x2.quote.diffRealCurr2", "chance1x2.quote.diffInitialCurr1", "chance1x2.quote.diffInitialCurrx", "chance1x2.quote.diffInitialCurr2", "chance1x2.bookkeeping.actual", "chance1x2.bookkeeping.p1", "chance1x2.bookkeeping.px", "chance1x2.bookkeeping.p2", "chance1x2.bookkeeping.avg", "chance1x2.flashback.p1", "chance1x2.flashback.px", "chance1x2.flashback.p2", "chance1x2.flashback.pHt1", "chance1x2.flashback.pHtx", "chance1x2.flashback.pHt2", "goalNoGoal.chance.goal", "goalNoGoal.chance.noGoal", "goalNoGoal.chance.even", "goalNoGoal.chance.odd", "goalNoGoal.chance.goalHome", "goalNoGoal.chance.goalAway", "goalNoGoal.multigoal.m13", "goalNoGoal.multigoal.m14", "goalNoGoal.multigoal.m24", "goalNoGoal.multigoal.m35", "goalNoGoal.multigoal.m13Home", "goalNoGoal.multigoal.m13Away", "goalNoGoal.multigoal.m24Home", "goalNoGoal.multigoal.m24Away", "goalNoGoal.quote.realGG", "goalNoGoal.quote.realNG", "goalNoGoal.quote.initialGG", "goalNoGoal.quote.initialNG", "goalNoGoal.quote.currentGG", "goalNoGoal.quote.currentNG", "goalNoGoal.quote.diffRealCurrGG", "goalNoGoal.quote.diffRealCurrNG", "goalNoGoal.quote.diffInitialCurrGG", "goalNoGoal.quote.diffInitialCurrNG", "goalNoGoal.bookkeeping.actual", "goalNoGoal.bookkeeping.gg", "goalNoGoal.bookkeeping.ng", "goalNoGoal.bookkeeping.avg", "goalNoGoal.stats.avgGoalHome", "goalNoGoal.stats.avgGoalTakenHome", "goalNoGoal.stats.avgGoalAway", "goalNoGoal.stats.avgGoalTakenAway", "goalNoGoal.flashback.goal", "goalNoGoal.flashback.noGoal", "goalNoGoal.flashback.m13", "goalNoGoal.flashback.m24", "goalNoGoal.flashback.m35", "underOver.chance.under05HT", "underOver.chance.over05HT", "underOver.chance.under052HT", "underOver.chance.over052HT", "underOver.chance.under15HT", "underOver.chance.over15HT", "underOver.chance.under15", "underOver.chance.over15", "underOver.chance.under25", "underOver.chance.over25", "underOver.chance.under35", "underOver.chance.over35", "underOver.chance.under45", "underOver.chance.over45", "underOver.quote.realU", "underOver.quote.realO", "underOver.quote.initialU", "underOver.quote.initialO", "underOver.quote.currentU", "underOver.quote.currentO", "underOver.quote.diffRealCurrU", "underOver.quote.diffRealCurrO", "underOver.quote.diffInitialCurrU", "underOver.quote.diffInitialCurrO", "underOver.bookkeeping.actual", "underOver.bookkeeping.u", "underOver.bookkeeping.o", "underOver.bookkeeping.avg", "underOver.flashback.under05HT", "underOver.flashback.over05HT", "underOver.flashback.under15", "underOver.flashback.over15", "underOver.flashback.under25", "underOver.flashback.over25", "underOver.flashback.under35", "underOver.flashback.over35", "chance1x2.xg.home", "chance1x2.xg.away", "chance1x2.xg.total"] 
    
    for col in float_cols:
        df_flat[col] = pd.to_numeric(df_flat[col], errors="coerce")
 
    # INTEGER
    integer_cols = ["chance1x2.bookkeeping.status", "chance1x2.comparison.affini", "chance1x2.comparison.flashback", "evaluation.val1x2", "evaluation.valUnderOver", "evaluation.valMetrica", "evaluation.valScala", "goalNoGoal.bookkeeping.status", "goalNoGoal.comparison.affini", "goalNoGoal.comparison.flashback", "underOver.bookkeeping.status", "underOver.comparison.affini", "underOver.comparison.flashback", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    for col in integer_cols:
        df_flat[col] = pd.to_numeric(df_flat[col], errors="coerce").astype("Int64")
    
    df_final = df_flat.sort_values(by='time', ascending=True).copy()

    return df_final


def get_win_1_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goal.home", "team.goal.away"])
    df["win_1"] = df["team.goal.home"] > df["team.goal.away"]
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)

    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_win_1x_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goal.home", "team.goal.away"])
    df["win_1x"] = df["team.goal.home"] >= df["team.goal.away"]

    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]

    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_win_2_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goal.home", "team.goal.away"])
    df["win_2"] = df["team.goal.away"] > df["team.goal.home"]

    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]

    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_win_x2_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goal.home", "team.goal.away"])
    df["win_x2"] = df["team.goal.away"] >= df["team.goal.home"]

    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_corner_home_table():
    df = get_match_data()

    df = df.dropna(subset=["team.corner.home", "team.corner.away"])
    df["cn_home"] = df["team.corner.home"] > df["team.corner.away"]
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_multigoal_24_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goal.home", "team.goal.away"])
    df["multigoal_24"] = ((df["team.goal.home"] + df["team.goal.away"])>= 2) & ((df["team.goal.home"] + df["team.goal.away"]) <= 4)

    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_multigoal_12_home_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goal.home"])
    df["multigoal_12_home"] = (1<= df["team.goal.home"]) & (df["team.goal.home"] <=2)
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_multigoal_13_home_table():
    df = get_match_data()

    df = df.dropna(subset=["team.goal.home"])
    df["multigoal_13_home"] = (1<= df["team.goal.home"]) & (df["team.goal.home"] <=3)
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_multigoal_13_away_table():
    df = get_match_data()

    df = df.dropna(subset=["team.goal.away"])
    df["multigoal_13_away"] = (1<= df["team.goal.away"]) & (df["team.goal.away"] <=3)
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_goal_home_ht_table():
    df = get_match_data()

    df = df.dropna(subset=["team.goalHt.home"])
    df["goal_home_ht"] = df["team.goalHt.home"] > 0
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_goal_ht_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goalHt.home", "team.goalHt.away"])
    df["goal_ht"] = df["team.goalHt.home"] + df["team.goalHt.away"] > 0
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_btts_table():
    df = get_match_data()

    df = df.dropna(subset=["team.goal.home", "team.goal.away"]).copy()
   
    df["btts"] = (df["team.goal.home"] > 0) & (df["team.goal.away"] > 0)
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_ntts_table():
    df = get_match_data()

    df = df.dropna(subset=["team.goal.home", "team.goal.away"]).copy()
   
    df["ntts"] = (df["team.goal.home"] == 0) | (df["team.goal.away"] == 0)
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_win_x2_ht_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goal.home", "team.goal.away"])
    df["win_x2_ht"] = df["team.goalHt.away"] >= df["team.goalHt.home"]
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_over_15_table():
    df = get_match_data()

    df = df.dropna(subset=["team.goal.home", "team.goal.away"])
    df["over_15"] = (df["team.goal.home"] + df["team.goal.away"] > 1.5)
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_over_25_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goal.home", "team.goal.away"])
    df["over_25"] = (df["team.goal.home"] + df["team.goal.away"] > 2.5)
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_under_25_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goal.home", "team.goal.away"])
    df["under_25"] = (df["team.goal.home"] + df["team.goal.away"] < 2.5)
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy


def get_x_table():
    df = get_match_data()
    df = df.dropna(subset=["team.goal.home", "team.goal.away"])
    df["tie"] = df["team.goal.home"] == df["team.goal.away"]
    to_drop = ["_id", "matchId", "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home", "team.corner.away"]
    df_strategy = df.drop(columns=to_drop)
    df_strategy = _cast_df(df_strategy)
    return df_strategy

def _cast_df(df):
    df.columns = [x.replace(".", "_") for x in df.columns]
    return df
    