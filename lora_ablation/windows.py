"""Device-resident series storage and on-the-fly window/batch construction.

Windows are never materialised: a batch is gathered by index from the stored
series, which keeps memory flat regardless of how many windows a split has.

Per-timestep input features (N_FEATURES = 10):
  0 demand, instance-normalised by the history window (0 for horizon steps)
  1 horizon mask (1 = future step whose demand is unknown)
  2 temperature: observed for history, day-ahead forecast for horizon
  3-8 sin/cos of hour-of-day, day-of-week, day-of-year
  9 holiday flag
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

N_FEATURES = 10
TEMP_CENTER, TEMP_SCALE = 10.0, 12.0


def load_frames(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    source = pd.read_csv(data_dir / "source_coops.csv.gz", parse_dates=["timestamp"])
    target = pd.read_csv(data_dir / "target_coop.csv.gz", parse_dates=["timestamp"])
    return source, target


class SeriesBank:
    def __init__(self, df: pd.DataFrame, device: torch.device):
        self.coop_ids = list(dict.fromkeys(df["coop_id"]))
        groups = [df[df["coop_id"] == c].sort_values("timestamp") for c in self.coop_ids]
        self.timestamps = pd.DatetimeIndex(groups[0]["timestamp"].values)
        ts = self.timestamps
        hour, dow, doy = ts.hour.values, ts.dayofweek.values, ts.dayofyear.values
        cal = np.stack([
            np.sin(2 * np.pi * hour / 24), np.cos(2 * np.pi * hour / 24),
            np.sin(2 * np.pi * dow / 7), np.cos(2 * np.pi * dow / 7),
            np.sin(2 * np.pi * doy / 365.25), np.cos(2 * np.pi * doy / 365.25),
        ], axis=-1).astype(np.float32)

        def stack(col):
            return torch.tensor(np.stack([g[col].to_numpy(np.float32) for g in groups]), device=device)

        self.demand = stack("demand_mw")                                    # [S, T]
        self.temp_obs = (stack("temperature_c") - TEMP_CENTER) / TEMP_SCALE
        self.temp_fc = (stack("temperature_forecast_c") - TEMP_CENTER) / TEMP_SCALE
        self.holiday = stack("is_holiday")
        self.calendar = torch.tensor(cal, device=device)                    # [T, 6]
        self.device = device

    def index_of(self, when) -> int:
        return int(self.timestamps.searchsorted(pd.Timestamp(when)))

    def window_starts(self, start, end, context: int, horizon: int,
                      context_inside: bool, stride: int = 1) -> np.ndarray:
        """Window start indices whose forecast horizon lies in [start, end).

        context_inside=True also keeps the history inside [start, end), used for
        the target training split so the model only sees the co-op's 90 days.
        """
        a, b = self.index_of(start), self.index_of(end)
        first_origin = max(a + (context if context_inside else 0), context)
        origins = np.arange(first_origin, b - horizon + 1, stride)
        return origins - context

    def all_pairs(self, starts: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
        """Cartesian product (series x starts) as flat index tensors."""
        s = np.repeat(np.arange(len(self.coop_ids)), len(starts))
        t = np.tile(starts, len(self.coop_ids))
        return torch.as_tensor(s), torch.as_tensor(t)

    def batch(self, sidx: torch.Tensor, starts: torch.Tensor, context: int, horizon: int):
        """Return (x [B,L+H,F], y_norm [B,H], mu [B,1], sd [B,1], y_raw [B,H])."""
        sidx, starts = sidx.to(self.device), starts.to(self.device)
        idx = starts[:, None] + torch.arange(context + horizon, device=self.device)
        rows = sidx[:, None]
        y = self.demand[rows, idx]
        hist = y[:, :context]
        mu = hist.mean(1, keepdim=True)
        sd = hist.std(1, keepdim=True) + 1e-3
        yn = (y - mu) / sd
        is_future = torch.zeros_like(y)
        is_future[:, context:] = 1.0
        demand_in = yn * (1.0 - is_future)
        temp = torch.where(is_future.bool(), self.temp_fc[rows, idx], self.temp_obs[rows, idx])
        x = torch.cat([
            demand_in[..., None], is_future[..., None], temp[..., None],
            self.calendar[idx], self.holiday[rows, idx][..., None],
        ], dim=-1)
        return x, yn[:, context:], mu, sd, y[:, context:]
