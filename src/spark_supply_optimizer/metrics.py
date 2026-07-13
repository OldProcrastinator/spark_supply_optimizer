"""Metrics used by training and benchmarking commands."""

from __future__ import annotations

from pyspark.sql import DataFrame
from pyspark.sql import functions as F


def regression_metrics(
    frame: DataFrame,
    label_col: str = "sales",
    prediction_col: str = "prediction",
) -> dict[str, float]:
    """Compute simple regression metrics without collecting the full dataset."""

    row = (
        frame.select(
            F.sqrt(F.avg(F.pow(F.col(label_col) - F.col(prediction_col), 2))).alias("rmse"),
            F.avg(F.abs(F.col(label_col) - F.col(prediction_col))).alias("mae"),
        )
        .first()
        .asDict()
    )
    return {key: float(value) for key, value in row.items()}
