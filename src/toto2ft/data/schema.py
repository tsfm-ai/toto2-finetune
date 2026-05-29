from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import pandas as pd


class CovariateType(str, Enum):
    TARGET = "target"
    PAST_OBSERVED = "past_observed"
    KNOWN_FUTURE = "known_future"
    STATIC = "static"


@dataclass
class DataSchema:
    timestamp_col: str
    group_cols: list[str]
    target_cols: list[str]
    past_covariate_cols: list[str] = field(default_factory=list)
    future_covariate_cols: list[str] = field(default_factory=list)
    static_covariate_cols: list[str] = field(default_factory=list)
    # Column that holds the availability timestamp for each row's features.
    # If set, leakage.audit_availability() will use it.
    availability_col: str | None = None
    frequency: str = "1h"

    @property
    def all_value_cols(self) -> list[str]:
        """All time-varying columns in variate order: [targets, past_covs, future_covs]."""
        return self.target_cols + self.past_covariate_cols + self.future_covariate_cols

    @property
    def n_variates(self) -> int:
        return len(self.all_value_cols)

    @property
    def n_targets(self) -> int:
        return len(self.target_cols)

    @property
    def target_variate_indices(self) -> list[int]:
        return list(range(self.n_targets))

    @property
    def future_covariate_indices(self) -> list[int]:
        start = len(self.target_cols) + len(self.past_covariate_cols)
        return list(range(start, start + len(self.future_covariate_cols)))

    def validate(self, df: pd.DataFrame) -> None:
        required = [self.timestamp_col] + self.group_cols + self.all_value_cols
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")
        if self.availability_col and self.availability_col not in df.columns:
            raise ValueError(
                f"availability_col '{self.availability_col}' not in DataFrame"
            )
