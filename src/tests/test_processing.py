from pathlib import Path

import pandas as pd
import pytest

from src.pipeline.processing import remove_spaces

RAW_DATA = """
col1, col2,col3 , col4, col5
xxxx, xxxx,xxxx , xxxx, xxxx
xxxx, xxxx,xxxx , xxxx, xxxx
"""


def test_remove_spaces(tmpdir):
    """Test data processing - remove spaces."""

    TEMP_DIR = Path(tmpdir)
    RAW_DATA_PATH = TEMP_DIR / "raw_data.csv"

    RAW_DATA_PATH.write_text(RAW_DATA)
    PRIMARY_DATA_PATH = TEMP_DIR / "primary_data.csv"

    raw_data = pd.read_csv(RAW_DATA_PATH)
    assert raw_data.columns[0] == "col1"
    assert raw_data.columns[1] == " col2"
    assert raw_data.columns[2] == "col3 "
    assert raw_data.iloc[1, 3] == " xxxx"

    remove_spaces(RAW_DATA_PATH, PRIMARY_DATA_PATH)

    primary_data = pd.read_csv(PRIMARY_DATA_PATH)
    assert primary_data.columns[0] == "col1"
    assert primary_data.columns[1] == "col2"
    assert primary_data.columns[2] == "col3"
    assert primary_data.iloc[1, 3] == "xxxx"
