import pandas as pd
from typing import Iterable
from .exceptions import SchemaValidationError

TX_REQUIRED_COLS = {
    "transaction_id", "customer_id", "card_number", "timestamp",
    "vendor_category", "vendor_type", "vendor", "amount", "currency",
    "country", "city", "city_size", "card_type", "is_card_present",
    "device", "channel", "device_fingerprint", "ip_address",
    "is_outside_home_country", "is_high_risk_vendor", "is_weekend",
    "last_hour_activity", "is_fraud"
}

FX_REQUIRED_COLS = {"date", "USD"} 
def ensure_columns(df: pd.DataFrame, required: Iterable[str], df_name: str):
    missing = set(required) - set(df.columns)
    if missing:
        raise SchemaValidationError(f"{df_name}: отсутствуют колонки: {sorted(missing)}")

def ensure_not_empty(df: pd.DataFrame, df_name: str):
    if df.empty:
        raise SchemaValidationError(f"{df_name}: пустой DataFrame")

def validate_transactions(df: pd.DataFrame):
    ensure_not_empty(df, "transactions")
    ensure_columns(df, TX_REQUIRED_COLS, "transactions")
    if not pd.api.types.is_datetime64_any_dtype(df["timestamp"]):
        raise SchemaValidationError("transactions: 'timestamp' должен быть datetime")
    if df["amount"].lt(0).any():
       
        raise SchemaValidationError("transactions: обнаружены отрицательные суммы 'amount'")

def validate_fx(df: pd.DataFrame):
    ensure_not_empty(df, "fx")
    ensure_columns(df, FX_REQUIRED_COLS, "fx")
    if not pd.api.types.is_datetime64_any_dtype(pd.to_datetime(df["date"], errors="coerce")):
        raise SchemaValidationError("fx: 'date' должен парситься как дата")
