import pandas as pd
import os

path = "./crimestats/source/"
for f in os.listdir(path):
    if os.path.isfile(os.path.join(path, f)):
        df = pd.read_csv(os.path.join(path, f))

        df["START_DATE"] = pd.to_datetime(df["START_DATE"])

        result = (
            df.assign(start_date=df["START_DATE"].dt.strftime("%m/%d/%Y"))
            .groupby("START_DATE")
            .size()
            .reset_index(name="count")
        )

        result["START_DATE"] = pd.to_datetime(result["START_DATE"]).dt.strftime("%m/%d/%Y")

        result.to_csv(os.path.join("./crimestats/", f), index=False)