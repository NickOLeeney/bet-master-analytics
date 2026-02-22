import os
import mlflow
import pandas as pd
from dotenv import load_dotenv
from databricks.sdk import WorkspaceClient
import warnings

warnings.filterwarnings("ignore")
load_dotenv()

w = WorkspaceClient(
    host=os.getenv("DATABRICKS_HOST"),
    token=os.getenv("DATABRICKS_TOKEN"),
)

def get_match_table():
    statement = w.statement_execution.execute_statement(
        warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),
        statement="""
            SELECT *
            FROM bet_master_analytics.matches.match_table
        """,
        wait_timeout="30s"
    )

    # Extract column names
    columns = [c.name for c in statement.manifest.schema.columns]

    # Extract row data
    rows = [row for row in statement.result.data_array]

    # Create pandas DataFrame
    df = pd.DataFrame(rows, columns=columns)

    df_cast = _cast_df(df)
    return df_cast


def get_table(query):
    statement = w.statement_execution.execute_statement(
        warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),
        statement=query,
        wait_timeout="30s"
    )

    # Extract column names
    columns = [c.name for c in statement.manifest.schema.columns]

    # Extract row data
    rows = [row for row in statement.result.data_array]

    # Create pandas DataFrame
    df = pd.DataFrame(rows, columns=columns)
    df_cast = _cast_df(df)
    df_cast.columns = [x.replace(".", "_") for x in df_cast.columns]
    return df_cast


def _cast_df(df):
    # STRING
    string_cols = ["_id", "matchId", "team.league", "team.home", "team.away", "chance1x2.bookkeeping.arrow",
                   "goalNoGoal.bookkeeping.arrow", "underOver.bookkeeping.arrow"]
    for col in string_cols:
        if col in df.columns:
            df[col] = df[col].astype('string')

    # TIMESTAMP
    timestamp_cols = ["time", "timestamp"]
    for col in timestamp_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # FLOAT / DOUBLE
    float_cols = ["chance1x2.chance.p1", "chance1x2.chance.px", "chance1x2.chance.p2", "chance1x2.chance.p1x",
                  "chance1x2.chance.p2x", "chance1x2.chance.p12", "chance1x2.chance.pHt1", "chance1x2.chance.pHtx",
                  "chance1x2.chance.pHt2", "chance1x2.chance.p2Ht1x", "chance1x2.chance.p2Ht2x",
                  "chance1x2.chance.p2Ht12", "chance1x2.quote.real1", "chance1x2.quote.realx", "chance1x2.quote.real2",
                  "chance1x2.quote.initial1", "chance1x2.quote.initialx", "chance1x2.quote.initial2",
                  "chance1x2.quote.current1", "chance1x2.quote.currentx", "chance1x2.quote.current2",
                  "chance1x2.quote.diffRealCurr1", "chance1x2.quote.diffRealCurrx", "chance1x2.quote.diffRealCurr2",
                  "chance1x2.quote.diffInitialCurr1", "chance1x2.quote.diffInitialCurrx",
                  "chance1x2.quote.diffInitialCurr2", "chance1x2.bookkeeping.actual", "chance1x2.bookkeeping.p1",
                  "chance1x2.bookkeeping.px", "chance1x2.bookkeeping.p2", "chance1x2.bookkeeping.avg",
                  "chance1x2.flashback.p1", "chance1x2.flashback.px", "chance1x2.flashback.p2",
                  "chance1x2.flashback.pHt1", "chance1x2.flashback.pHtx", "chance1x2.flashback.pHt2",
                  "goalNoGoal.chance.goal", "goalNoGoal.chance.noGoal", "goalNoGoal.chance.even",
                  "goalNoGoal.chance.odd", "goalNoGoal.chance.goalHome", "goalNoGoal.chance.goalAway",
                  "goalNoGoal.multigoal.m13", "goalNoGoal.multigoal.m14", "goalNoGoal.multigoal.m24",
                  "goalNoGoal.multigoal.m35", "goalNoGoal.multigoal.m13Home", "goalNoGoal.multigoal.m13Away",
                  "goalNoGoal.multigoal.m24Home", "goalNoGoal.multigoal.m24Away", "goalNoGoal.quote.realGG",
                  "goalNoGoal.quote.realNG", "goalNoGoal.quote.initialGG", "goalNoGoal.quote.initialNG",
                  "goalNoGoal.quote.currentGG", "goalNoGoal.quote.currentNG", "goalNoGoal.quote.diffRealCurrGG",
                  "goalNoGoal.quote.diffRealCurrNG", "goalNoGoal.quote.diffInitialCurrGG",
                  "goalNoGoal.quote.diffInitialCurrNG", "goalNoGoal.bookkeeping.actual", "goalNoGoal.bookkeeping.gg",
                  "goalNoGoal.bookkeeping.ng", "goalNoGoal.bookkeeping.avg", "goalNoGoal.stats.avgGoalHome",
                  "goalNoGoal.stats.avgGoalTakenHome", "goalNoGoal.stats.avgGoalAway",
                  "goalNoGoal.stats.avgGoalTakenAway", "goalNoGoal.flashback.goal", "goalNoGoal.flashback.noGoal",
                  "goalNoGoal.flashback.m13", "goalNoGoal.flashback.m24", "goalNoGoal.flashback.m35",
                  "underOver.chance.under05HT", "underOver.chance.over05HT", "underOver.chance.under052HT",
                  "underOver.chance.over052HT", "underOver.chance.under15HT", "underOver.chance.over15HT",
                  "underOver.chance.under15", "underOver.chance.over15", "underOver.chance.under25",
                  "underOver.chance.over25", "underOver.chance.under35", "underOver.chance.over35",
                  "underOver.chance.under45", "underOver.chance.over45", "underOver.quote.realU",
                  "underOver.quote.realO", "underOver.quote.initialU", "underOver.quote.initialO",
                  "underOver.quote.currentU", "underOver.quote.currentO", "underOver.quote.diffRealCurrU",
                  "underOver.quote.diffRealCurrO", "underOver.quote.diffInitialCurrU",
                  "underOver.quote.diffInitialCurrO", "underOver.bookkeeping.actual", "underOver.bookkeeping.u",
                  "underOver.bookkeeping.o", "underOver.bookkeeping.avg", "underOver.flashback.under05HT",
                  "underOver.flashback.over05HT", "underOver.flashback.under15", "underOver.flashback.over15",
                  "underOver.flashback.under25", "underOver.flashback.over25", "underOver.flashback.under35",
                  "underOver.flashback.over35"]

    for col in float_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # INTEGER
    integer_cols = ["chance1x2.bookkeeping.status", "chance1x2.comparison.affini", "chance1x2.comparison.flashback",
                    "evaluation.val1x2", "evaluation.valUnderOver", "evaluation.valMetrica", "evaluation.valScala",
                    "goalNoGoal.bookkeeping.status", "goalNoGoal.comparison.affini", "goalNoGoal.comparison.flashback",
                    "underOver.bookkeeping.status", "underOver.comparison.affini", "underOver.comparison.flashback",
                    "team.goal.home", "team.goal.away", "team.goalHt.home", "team.goalHt.away", "team.corner.home",
                    "team.corner.away"]
    for col in integer_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")


    boolean_cols = ["win_1x", "win_1", "multigoal_24"]
    for col in boolean_cols:
        if col in df.columns:
            df[col] = (df[col] == "true")

    return df

def set_mlflow_experiment(experiment_name: str) -> None:
    mlflow.set_tracking_uri("databricks")
    mlflow.set_registry_uri("databricks-uc")
    mlflow.set_experiment(f"/Users/maicolnicolini96@gmail.com/bet_analytics_experiments/{experiment_name}")
    return None