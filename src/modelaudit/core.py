import hashlib
import json
import numpy as np
from scipy import stats
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, brier_score_loss


def ece(y, p, bins=10):
    total = 0.0
    for left in np.linspace(0, 1, bins, endpoint=False):
        mask = (p >= left) & (
            p < left + 1 / bins if left + 1 / bins < 0.999999 else p <= 1
        )
        if mask.any():
            total += mask.mean() * abs(y[mask].mean() - p[mask].mean())
    return float(total)


def bh(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.minimum.accumulate(
        (p[order] * len(p) / np.arange(1, len(p) + 1))[::-1]
    )[::-1]
    output = np.empty(len(p))
    output[order] = np.minimum(adjusted, 1)
    return output


def validate(bundle):
    for key in ["train", "test"]:
        block = bundle[key]
        x = np.array(block["x"], float)
        y = np.array(block["y"])
        if (
            x.ndim != 2
            or len(y) != len(x)
            or x.shape[1] != len(bundle["features"])
            or len(x) < 2
        ):
            raise ValueError("Invalid feature shape or row counts")
        if (
            not np.isfinite(x).all()
            or not np.isin(y, [0, 1]).all()
            or len(np.unique(y)) < 2
        ):
            raise ValueError("Need finite features and both binary classes")
        if (
            len(block["time"]) != len(y)
            or len(block["row_id"]) != len(y)
            or len(block["group"]) != len(y)
        ):
            raise ValueError("Metadata length mismatch")
        if not np.isfinite(np.asarray(block["time"], float)).all():
            raise ValueError("Invalid times")
    p = np.array(bundle["test"]["scores"], float)
    if (
        len(p) != len(bundle["test"]["y"])
        or not np.isfinite(p).all()
        or np.any((p < 0) | (p > 1))
    ):
        raise ValueError("Invalid probability scores")


def audit(bundle):
    validate(bundle)
    train = bundle["train"]
    test = bundle["test"]
    a = np.array(train["x"])
    b = np.array(test["x"])
    y = np.array(test["y"])
    p = np.array(test["scores"])
    findings = []

    def finding(code, severity, detail):
        findings.append(dict(code=code, severity=severity, detail=detail))

    future = [f["name"] for f in bundle["features"] if f["available_offset"] > 0]
    if future:
        finding("POST_OUTCOME_FEATURE", "FAIL", future)
    shared = set(train["row_id"]) & set(test["row_id"])
    if shared:
        finding("OVERLAPPING_ROW_IDS", "FAIL", len(shared))
    if np.max(np.asarray(train["time"], float)) >= np.min(
        np.asarray(test["time"], float)
    ):
        finding(
            "TEMPORAL_OVERLAP",
            "FAIL",
            "Training observations are not all before test observations",
        )
    hashes = {
        hashlib.sha256(
            np.ascontiguousarray(row, dtype=np.float64).tobytes()
        ).hexdigest()
        for row in a
    }
    duplicates = sum(
        hashlib.sha256(
            np.ascontiguousarray(row, dtype=np.float64).tobytes()
        ).hexdigest()
        in hashes
        for row in b
    )
    if duplicates:
        finding(
            "EXACT_FEATURE_DUPLICATES",
            "WARN",
            dict(
                test_rows=duplicates,
                note="Identical feature vectors can be legitimate; inspect provenance",
            ),
        )
    drift = []
    for j, feature in enumerate(bundle["features"]):
        statistic, pvalue = stats.ks_2samp(a[:, j], b[:, j])
        drift.append(
            dict(feature=feature["name"], ks=float(statistic), p=float(pvalue))
        )
    for row, q in zip(drift, bh([r["p"] for r in drift])):
        row["bh_q"] = float(q)
        if q < 0.05 and row["ks"] > 0.15:
            finding("FEATURE_DRIFT", "WARN", row)
    metrics = dict(
        auc=float(roc_auc_score(y, p)),
        brier=float(brier_score_loss(y, p)),
        ece=ece(y, p),
        prevalence=float(y.mean()),
        baseline_brier=float(brier_score_loss(y, np.full(len(y), np.mean(train["y"])))),
    )
    if metrics["auc"] < 0.65:
        finding("WEAK_DISCRIMINATION", "FAIL", metrics["auc"])
    if metrics["brier"] >= metrics["baseline_brier"]:
        finding("NO_BRIER_IMPROVEMENT", "FAIL", metrics)
    if metrics["ece"] > 0.10:
        finding("POOR_CALIBRATION", "WARN", metrics["ece"])
    if metrics["auc"] > 0.995:
        finding(
            "SUSPICIOUSLY_HIGH_AUC",
            "WARN",
            "Investigate leakage; high AUC alone does not prove it",
        )
    groups = []
    g = np.array(test["group"])
    for name in np.unique(g):
        mask = g == name
        yg = y[mask]
        pg = p[mask]
        record = dict(
            group=str(name),
            n=int(mask.sum()),
            prevalence=float(yg.mean()),
            brier=float(brier_score_loss(yg, pg)),
        )
        if len(np.unique(yg)) < 2:
            record["auc"] = None
            finding("SUBGROUP_SUPPORT", "WARN", str(name))
        else:
            record["auc"] = float(roc_auc_score(yg, pg))
        record["tpr"] = (
            float((pg[yg == 1] >= 0.5).mean()) if (yg == 1).sum() >= 30 else None
        )
        groups.append(record)
    tprs = [r["tpr"] for r in groups if r["tpr"] is not None]
    if len(tprs) > 1 and max(tprs) - min(tprs) > 0.2:
        finding("SUBGROUP_TPR_GAP", "WARN", max(tprs) - min(tprs))
    decision = (
        "REJECT"
        if any(f["severity"] == "FAIL" for f in findings)
        else ("REVIEW" if findings else "PASS_WITHIN_SCOPE")
    )
    return dict(
        decision=decision,
        metrics=metrics,
        findings=findings,
        drift=drift,
        subgroups=groups,
        scope="Heuristic independent checks; metadata must be trustworthy. No legal compliance certification or proof of fairness.",
    )


def fixture(seed=17, flawed=False):
    rng = np.random.default_rng(seed)
    n = 4000
    x = rng.normal(size=(n, 3))
    prob = 1 / (1 + np.exp(-(1.8 * x[:, 0] - x[:, 1])))
    y = rng.binomial(1, prob)
    group = rng.choice(["A", "B"], n)
    if flawed:
        x = np.c_[x, y]
        # A real-looking random split violates the requested future-deployment protocol.
        order = rng.permutation(n)
        tr = order[:2800]
        te = order[2800:]
    else:
        tr = np.arange(2800)
        te = np.arange(2800, n)
    model = LogisticRegression(C=100 if flawed else 1, max_iter=500).fit(x[tr], y[tr])

    def block(indices):
        return dict(
            x=x[indices].tolist(),
            y=y[indices].tolist(),
            time=indices.tolist(),
            row_id=[f"row-{i}" for i in indices],
            group=group[indices].tolist(),
        )

    train = block(tr)
    test = block(te)
    test["scores"] = model.predict_proba(x[te])[:, 1].tolist()
    return dict(
        data="SYNTHETIC binary risk task",
        features=[
            dict(name=f"x{j}", available_offset=1 if j == 3 else -1)
            for j in range(x.shape[1])
        ],
        train=train,
        test=test,
    )
