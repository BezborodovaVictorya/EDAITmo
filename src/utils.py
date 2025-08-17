import pandas as pd

def non_empty_unique(df: pd.DataFrame, col: str) -> int:
    return df[col].dropna().nunique()
