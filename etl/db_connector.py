import os

from databricks.sdk import WorkspaceClient
from dotenv import load_dotenv

load_dotenv()

w = WorkspaceClient(
    host=os.getenv("DATABRICKS_HOST"),
    token=os.getenv("DATABRICKS_TOKEN"),
)


statement = w.statement_execution.execute_statement(
    warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),
    statement="""
        SELECT *
        FROM bet_master_analytics.matches.match_table
        LIMIT 10
    """,
    wait_timeout="30s"
)

print(statement.result)
