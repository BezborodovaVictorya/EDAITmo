import pandas as pd
import pytest
from src.validators import validate_transactions, validate_fx, SchemaValidationError

def test_validate_fx_ok(fx_sample):
    validate_fx(fx_sample)

def test_validate_tx_ok(tx_sample):
    validate_transactions(tx_sample)

def test_validate_tx_negative_amount(tx_sample):
    tx_sample.loc[0,"amount"] = -1
    with pytest.raises(SchemaValidationError):
        validate_transactions(tx_sample)
