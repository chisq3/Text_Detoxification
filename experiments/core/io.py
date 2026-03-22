from typing import Dict, Iterable

import pandas as pd
from datasets import Dataset


def load_tsv_dataset(tsv_path: str, required_columns: Iterable[str], column_types: Dict[str, type]) -> Dataset:
    df = pd.read_csv(tsv_path, sep="\t")

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        cols = ", ".join(missing)
        raise ValueError(f"Invalid schema in {tsv_path}. Missing columns: {cols}")

    for col, typ in column_types.items():
        df[col] = df[col].astype(typ)

    columns = list(required_columns)
    return Dataset.from_pandas(df[columns], preserve_index=False)
