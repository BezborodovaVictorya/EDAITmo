import numpy as np
from src.feature_engineering import convert_to_usd, build_features

def test_convert_to_usd(tx_sample, fx_sample):
    out = convert_to_usd(tx_sample, fx_sample, strict=False)
    # EUR: 100 / 0.9 ~= 111.11 ; USD: 200 / 1 = 200
    assert np.isclose(out.loc[0,"amount_usd"], 111.111, rtol=1e-2)
    assert np.isclose(out.loc[1,"amount_usd"], 200.0, rtol=1e-6)

def test_build_features(tx_sample, fx_sample):
    out = convert_to_usd(tx_sample, fx_sample, strict=False)
    out = build_features(out)
    for col in ["hour","dayofweek","is_night","txn_amount_ratio","lha_num_transactions"]:
        assert col in out.columns

