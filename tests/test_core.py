import numpy as np
import pytest
from modelaudit.core import *


def test_high_accuracy_leaky_model_rejected():
    r = audit(fixture(flawed=True))
    codes = {f["code"] for f in r["findings"]}
    assert r["metrics"]["auc"] > 0.99 and r["decision"] == "REJECT"
    assert {"POST_OUTCOME_FEATURE", "TEMPORAL_OVERLAP"} <= codes


def test_clean_model_no_structural_failures():
    r = audit(fixture())
    assert not any(f["severity"] == "FAIL" for f in r["findings"])


def test_shared_identifiers_rejected():
    b = fixture()
    b["test"]["row_id"][0] = b["train"]["row_id"][0]
    assert "OVERLAPPING_ROW_IDS" in {f["code"] for f in audit(b)["findings"]}


def test_bad_scores_rejected():
    b = fixture()
    b["test"]["scores"][0] = float("nan")
    with pytest.raises(ValueError):
        audit(b)


def test_perfect_calibration_and_bh():
    assert ece(np.array([0, 1]), np.array([0.0, 1.0])) == 0
    assert np.allclose(bh([0.01, 0.04, 0.03]), [0.03, 0.04, 0.04])


def test_drift_detected():
    b = fixture()
    b["test"]["x"] = (np.array(b["test"]["x"]) + 10).tolist()
    assert "FEATURE_DRIFT" in {f["code"] for f in audit(b)["findings"]}


def test_fractional_labels_not_silently_truncated():
    b = fixture()
    b["train"]["y"][0] = 0.5
    with pytest.raises(ValueError):
        audit(b)
