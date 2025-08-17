import logging
from typing import Optional, Set
import numpy as np
import pandas as pd
from .exceptions import CurrencyConversionError

logger = logging.getLogger(__name__)

def _infer_currency_columns(fx: pd.DataFrame) -> Set[str]:
   
    return set(col for col in fx.columns if col != "date")

def convert_to_usd(trans: pd.DataFrame, fx: pd.DataFrame, strict: bool = False) -> pd.DataFrame:
    """
   
    strict=True -> падаем, если курс для валюты/даты отсутствует.
    """
    df = trans.copy()
    df["date_only"] = df["timestamp"].dt.date

    fx_cols = _infer_currency_columns(fx)
    # join по дате
    df = df.merge(fx, left_on="date_only", right_on="date", how="left", suffixes=("", "_fx"))

    def safe_convert(row) -> Optional[float]:
        cur = row.get("currency")
        if pd.isna(cur) or cur not in fx_cols:
            return np.nan
        rate = row.get(cur)  # курс этой валюты относительно USD
        amt = row.get("amount")
        if pd.isna(rate) or pd.isna(amt) or rate == 0:
            return np.nan
        # Если в fx задано "сколько валюты за 1 USD", то перевод в USD = amount / rate
        try:
            return float(amt) / float(rate)
        except Exception:  # noqa: BLE001
            return np.nan

    df["amount_usd"] = df.apply(safe_convert, axis=1)

    missing_mask = df["amount_usd"].isna()
    missing_count = int(missing_mask.sum())
    if missing_count > 0:
        msg = f"amount_usd не удалось вычислить для {missing_count} строк (нет курса/валюты/даты)."
        if strict:
            raise CurrencyConversionError(msg)
        logger.warning(msg)

    # подчистим fx-колонки после merge
    drop_cols = ["date_only", "date"] + list(fx_cols)
    df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
    return df

def build_features(df: pd.DataFrame) -> pd.DataFrame:
   
    out = df.copy()

    # Временные признаки
    out["hour"] = out["timestamp"].dt.hour
    out["dayofweek"] = out["timestamp"].dt.dayofweek  # 0=Mon
    out["is_night"] = out["hour"].between(0, 6).astype("int8")

    # Клиентские агрегаты: медиана и ст.откл. по USD
    grp = out.groupby("customer_id")["amount_usd"]
    out["cust_amount_median"] = grp.transform("median").fillna(0.0)
    out["cust_amount_std"] = grp.transform("std").fillna(0.0)

    # Отношение суммы к привычным тратам клиента
    denom = (out["cust_amount_median"].abs() + 1e-6)
    out["txn_amount_ratio"] = (out["amount_usd"].abs() / denom).clip(upper=1e4)

    # Вспомогательные признаки из last_hour_activity (если struct уже сериализован в dict/строку)
    # Попытаемся аккуратно распаковать, не упав.
    for k in ["num_transactions", "total_amount", "unique_merchants", "unique_countries", "max_single_amount"]:
        out[f"lha_{k}"] = np.nan

    def _extract_lha(row):
        lha = row.get("last_hour_activity")
        if isinstance(lha, dict):
            return (
                lha.get("num_transactions"),
                lha.get("total_amount"),
                lha.get("unique_merchants"),
                lha.get("unique_countries"),
                lha.get("max_single_amount"),
            )
        return (np.nan, np.nan, np.nan, np.nan, np.nan)

    lha_vals = out.apply(_extract_lha, axis=1, result_type="expand")
    lha_cols = ["lha_num_transactions", "lha_total_amount", "lha_unique_merchants",
                "lha_unique_countries", "lha_max_single_amount"]
    out[lha_cols] = lha_vals

    # Робастные clip’ы от выбросов
    for col in ["amount_usd", "lha_total_amount", "lha_max_single_amount"]:
        if col in out.columns:
            q99 = out[col].quantile(0.99)
            out[col] = out[col].clip(upper=q99)

    # Готовые бинарные удобные флаги
    out["is_card_not_present"] = (~out["is_card_present"]).astype("int8")
    out["is_high_risk_vendor_f"] = out["is_high_risk_vendor"].astype("int8")
    out["is_outside_home_country_f"] = out["is_outside_home_country"].astype("int8")

 
    return out

