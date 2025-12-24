import os
import pandas as pd
from pandas import json_normalize
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()
# Local MongoDB

def get_data():
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
    return df_flat