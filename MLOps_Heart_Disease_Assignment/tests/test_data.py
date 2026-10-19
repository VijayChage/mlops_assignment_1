import pandas as pd
import pytest
from src.heartml.data import load_dataset, make_preprocessor, validate_features
from src.heartml.schema import FEATURES


def test_load_dataset_binarizes_target_and_keeps_text_categories(tmp_path):
    row = {c: 1 for c in FEATURES}
    row["cp"] = "asymptomatic"
    df = pd.DataFrame([{**row, "target": 0}, {**row, "target": 3}])
    path = tmp_path / "tiny.csv"
    df.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert loaded["target"].tolist() == [0, 1]
    assert loaded["cp"].tolist() == ["asymptomatic", "asymptomatic"]


def test_load_dataset_maps_source_num_target(tmp_path):
    row = {c: 1 for c in FEATURES}
    df = pd.DataFrame([{**row, "num": 0}, {**row, "num": 1}])
    path = tmp_path / "num_target.csv"
    df.to_csv(path, index=False)
    loaded = load_dataset(path)
    assert loaded["target"].tolist() == [0, 1]


def test_missing_column_raises(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"age": [40], "target": [0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Dataset is missing required columns"):
        load_dataset(path)


def test_validate_features_requires_all_features():
    with pytest.raises(ValueError, match="Missing features"):
        validate_features({"age": 50})


def test_preprocessor_can_be_created():
    assert make_preprocessor() is not None
