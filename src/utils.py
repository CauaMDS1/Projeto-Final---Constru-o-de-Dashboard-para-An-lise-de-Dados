from __future__ import annotations

import re
from collections import Counter
from typing import Iterable

import numpy as np
import pandas as pd


def clean_column_name(value: str) -> str:
    value = value.strip().lower()
    value = re.sub(r"[^0-9a-zA-Z]+", "_", value)
    value = re.sub(r"_+", "_", value).strip("_")
    return value


def parse_owner_range(series: pd.Series) -> pd.DataFrame:
    owners = series.fillna("").astype(str).str.extract(
        r"(?P<owners_low>\d+)\s*-\s*(?P<owners_high>\d+)"
    )
    owners["owners_low"] = pd.to_numeric(owners["owners_low"], errors="coerce")
    owners["owners_high"] = pd.to_numeric(owners["owners_high"], errors="coerce")
    owners["owners_mid"] = (owners["owners_low"] + owners["owners_high"]) / 2
    return owners


def split_multi_value(value: object) -> list[str]:
    if pd.isna(value):
        return []
    text = str(value).strip()
    if not text or text == "[]":
        return []
    return [part.strip() for part in text.split(",") if part.strip()]


def top_values(values: Iterable[object], top_n: int = 12) -> list[str]:
    counts: Counter[str] = Counter()
    for value in values:
        counts.update(split_multi_value(value))
    return [name for name, _ in counts.most_common(top_n)]


def minmax(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    min_value = values.min()
    max_value = values.max()
    if pd.isna(min_value) or pd.isna(max_value) or max_value == min_value:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - min_value) / (max_value - min_value)


def standardize(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").astype(float)
    mean = values.mean()
    std = values.std(ddof=0)
    if pd.isna(std) or std == 0:
        return pd.Series(np.zeros(len(values)), index=series.index)
    return (values - mean) / std


def safe_divide(numerator: pd.Series, denominator: pd.Series) -> pd.Series:
    numerator = pd.to_numeric(numerator, errors="coerce").fillna(0)
    denominator = pd.to_numeric(denominator, errors="coerce").fillna(0)
    return np.where(denominator > 0, numerator / denominator, np.nan)


def oversample_minority(
    frame: pd.DataFrame,
    target_col: str,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    counts = frame[target_col].value_counts(dropna=False).to_dict()
    if len(counts) < 2:
        return frame.copy(), {
            "applied": False,
            "reason": "Only one class was available.",
            "before": counts,
            "after": counts,
        }

    majority_class = max(counts, key=counts.get)
    majority_count = counts[majority_class]
    pieces = [frame]

    for class_value, class_count in counts.items():
        if class_count == majority_count:
            continue
        sample = frame[frame[target_col] == class_value].sample(
            n=majority_count - class_count,
            replace=True,
            random_state=random_state,
        )
        pieces.append(sample)

    balanced = pd.concat(pieces, ignore_index=True)
    after = balanced[target_col].value_counts(dropna=False).to_dict()
    return balanced, {
        "applied": True,
        "method": "Random oversampling with replacement using pandas.sample",
        "before": counts,
        "after": after,
    }
