from pathlib import Path
from pydantic import BaseModel, Field, ValidationError
import os

class Paths(BaseModel):
    base_dir: Path = Field(default_factory=lambda: Path(__file__).resolve().parent.parent)
    data_dir: Path = Field(default=None)
    reports_dir: Path = Field(default=None)
    tx_file: Path = Field(default=None)
    fx_file: Path = Field(default=None)

    def __init__(self, **data):
        super().__init__(**data)
        object.__setattr__(self, "data_dir", self.base_dir / "data")
        object.__setattr__(self, "reports_dir", self.base_dir / "reports")
        object.__setattr__(self, "tx_file", self.data_dir / "transaction_fraud_data.parquet")
        object.__setattr__(self, "fx_file", self.data_dir / "historical_currency_exchange.parquet")

PATHS = Paths()

# Настройки pandas 
PANDAS_OPTS = {
    "display.max_columns": 200,
    "display.width": 200,
    "mode.copy_on_write": True
}

