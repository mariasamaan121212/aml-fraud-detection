from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"


def load_split(protocol: str, split: str) -> pd.DataFrame:
    """Load one parquet file, e.g. load_split("benchmark", "train")."""
    path = RAW / protocol / f"{split}.parquet"
    table = pq.read_table(path, read_dictionary=["payment_channel"])
    df = table.to_pandas()
    df = df.drop(columns=["month"], errors="ignore")
    for col in df.select_dtypes("int64").columns:
        df[col] = pd.to_numeric(df[col], downcast="integer")
    return df