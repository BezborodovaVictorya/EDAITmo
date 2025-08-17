import pandas as pd
import pytest
from datetime import datetime, timezone

@pytest.fixture
def tx_sample():
    ts = pd.to_datetime(["2024-10-01T10:00:00Z","2024-10-01T11:00:00Z"]).tz_convert(None)
    return pd.DataFrame({
        "transaction_id": ["t1","t2"],
        "customer_id": ["c1","c1"],
        "card_number": [1234, 1234],
        "timestamp": ts,
        "vendor_category": ["Retail","Retail"],
        "vendor_type": ["online","online"],
        "vendor": ["ShopA","ShopB"],
        "amount": [100.0, 200.0],
        "currency": ["EUR","USD"],
        "country": ["DE","US"], "city": ["Berlin","NYC"], "city_size": ["Large","Large"],
        "card_type": ["Basic Credit","Basic Credit"],
        "is_card_present": [False, False],
        "device": ["Chrome","iOS App"], "channel": ["web","mobile"],
        "device_fingerprint": ["df1","df2"], "ip_address": ["1.1.1.1","2.2.2.2"],
        "is_outside_home_country": [True, False],
        "is_high_risk_vendor": [False, True], "is_weekend": [False, False],
        "last_hour_activity": [{"num_transactions":1,"total_amount":100.0,"unique_merchants":1,"unique_countries":1,"max_single_amount":100.0}, {}],
        "is_fraud": [0, 1]
    })

@pytest.fixture
def fx_sample():
    return pd.DataFrame({
        "date": pd.to_datetime(["2024-10-01"]).date,
        "USD": [1.0],
        "EUR": [0.9],
    })
