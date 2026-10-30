"""Optional legacy cal_metrics.py JSONL augmentation from score_properties CSV."""
from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path

from evaluation.score_properties import METRICS


def _finite(val):
    if val is None or val == "":
        return None
    try:
        value = float(val)
        return value if math.isfinite(value) else None
    except (ValueError, TypeError):
        return None


def _score_summary(values, pmets):
    valid = [(i, v) for i, v in enumerate(values) if v is not None]
    if not valid:
        return None
    corr = 0.0
    pmet_pairs = [(values[i], _finite(pmets[i])) for i, _ in valid]
    pmet_pairs = [(v, p) for v, p in pmet_pairs if p is not None]
    if len(pmet_pairs) >= 2:
        from scipy.stats import spearmanr
        tmp = spearmanr([p[0] for p in pmet_pairs], [p[1] for p in pmet_pairs]).statistic
        if math.isfinite(tmp):
            corr = float(tmp)
    chosen = [(i, v) for i, v in valid if _finite(pmets[i]) is not None]
    if chosen:
        highest = max(chosen, key=lambda pair: float(pmets[pair[0]]))[1]
        lowest = min(chosen, key=lambda pair: float(pmets[pair[0]]))[1]
    else:
        highest = lowest = None
    return {"max": max(v for _, v in valid), "min": min(v for _, v in valid),
            "mean": statistics.mean(v for _, v in valid), "random": values[0],
            "max*": highest if corr > 0 else lowest,
            "min*": lowest if corr > 0 else highest, "pmet_corr": corr,
            "individual": values, "individual_pmet": pmets,
            "n_scored": len(valid)}


def append_properties_to_eval(report_jsonl, id2items, csv_path, valid_only=False):
    """Preserve existing keys; add only metrics with available candidate scores.

    Alignment uses (id, number) not position in predictor CSV. Missing predictions
    are represented as nulls in `individual`, never silently treated as zeros.
    """
    with open(csv_path, newline="") as handle:
        table = list(csv.DictReader(handle))
    lookup = {}
    for rec in table:
        key = rec["id"], str(int(rec["number"]))
        if key in lookup:
            raise ValueError(f"Duplicate candidate in property CSV: {key}")
        lookup[key] = rec
    new_lines = []
    with open(report_jsonl) as handle:
        reports = [json.loads(line) for line in handle if line.strip()]
    added = 0
    for report in reports:
        target = report["id"]
        items = id2items[target]
        pmets = [item.get("pmetric") for item in items]
        candidates = [lookup.get((target, str(int(item["number"])))) for item in items]
        for metric in METRICS:
            values = [_finite(c.get(metric)) if c and
                      (not valid_only or str(c.get("valid", "")).lower() == "true")
                      else None for c in candidates]
            summary = _score_summary(values, pmets)
            if summary is not None:
                report[metric] = summary
                added += 1
        new_lines.append(json.dumps(report, allow_nan=False))
    Path(report_jsonl).write_text("\n".join(new_lines) + "\n")
    print(f"Augmented {report_jsonl} with {added} target/metric combinations")
