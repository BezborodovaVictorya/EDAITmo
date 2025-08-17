import logging
import pandas as pd
from pathlib import Path
from typing import Optional, List
from .config import PATHS, PANDAS_OPTS
from .exceptions import DataFileNotFoundError, ParquetReadError
from .validators import validate_transactions, validate_fx

logger = logging.getLogger(__name__)

def _read_parquet(path: Path, engines: Optional[List[str]] = None) -> pd.DataFrame:
    if not path.exists():
        raise DataFileNotFoundError(f"Файл не найден: {path}")

    if engines is None:
        engines = ["pyarrow", "fastparquet"]

    last_err = None
    for eng in engines:
        try:
            return pd.read_parquet(path, engine=eng)
        except Exception as e:  # noqa: BLE001
            last_err = e
            logger.warning(f"Не удалось прочитать {path.name} с движком {eng}: {e}")
    raise ParquetReadError(f"Ошибка чтения parquet {path}: {last_err}")

def configure_pandas():
    for k, v in PANDAS_OPTS.items():
        pd.set_option(k, v)

def load_transactions() -> pd.DataFrame:
    df = _read_parquet(PATHS.tx_file)
    # мягкое приведение типов
    if not pd.api.types.is_datetime64_any_dtype(df.get("timestamp")):
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
    validate_transactions(df)
    return df

def load_fx() -> pd.DataFrame:
    df = _read_parquet(PATHS.fx_file)
    # приведение типов
    df["date"] = pd.to_datetime(df["date"], errors="coerce", utc=True).dt.date
    validate_fx(df)
    return df
