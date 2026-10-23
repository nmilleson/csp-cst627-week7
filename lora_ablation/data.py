"""Seeded synthetic generator for rural electricity-co-op demand traces.

Each co-op gets its own weather (annual + diurnal cycle + multi-day AR(1)
anomalies) and a demand model built from interpretable components:

    demand = base * trend * (diurnal profile + heating + cooling
                             + irrigation + EV charging) * exp(AR(1) noise)

Source co-ops (US + Central EU) are used for pretraining. The target co-op is a
Nordic EU co-op with heat-pump heating and EV evening charging: its demand
responds to temperature and time-of-day in ways the source population does not,
which is the distribution shift the LoRA adapters must learn from 90 days of data.
Values are rounded before writing so file hashes are stable across platforms.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd
from dateutil.easter import easter
from scipy.signal import lfilter, lfilter_zi


@dataclass
class CoopParams:
    coop_id: str
    region: str               # "US" or "EU"
    climate: str
    base_mw: float
    growth_per_year: float
    t_mean: float             # deg C
    t_annual_amp: float
    t_daily_amp: float
    t_anomaly_sd: float
    heat_slope: float         # fraction of base per deg C below 15 C
    cool_slope: float         # fraction of base per deg C above 22 C
    irrigation: float         # summer midday pumping load (fraction of base)
    commercial: float
    morning_peak: float
    evening_peak: float
    ev_charging: float        # late-evening EV load (fraction of base)
    forecast_error_sd: float  # day-ahead temperature-forecast error, deg C


def _us_holidays(year: int) -> list[date]:
    def nth_weekday(month, weekday, n):
        d = date(year, month, 1)
        d += timedelta(days=(weekday - d.weekday()) % 7)
        return d + timedelta(weeks=n - 1)

    def last_weekday(month, weekday):
        d = date(year, month + 1, 1) - timedelta(days=1)
        return d - timedelta(days=(d.weekday() - weekday) % 7)

    return [date(year, 1, 1), last_weekday(5, 0), date(year, 7, 4),
            nth_weekday(9, 0, 1), nth_weekday(11, 3, 4), date(year, 12, 25)]


def _eu_holidays(year: int) -> list[date]:
    e = easter(year)
    return [date(year, 1, 1), e - timedelta(days=2), e + timedelta(days=1),
            date(year, 5, 1), e + timedelta(days=39), date(year, 12, 24),
            date(year, 12, 25), date(year, 12, 26)]


def holiday_mask(ts: pd.DatetimeIndex, region: str) -> np.ndarray:
    fn = _us_holidays if region == "US" else _eu_holidays
    days = {d for y in range(ts.year.min(), ts.year.max() + 1) for d in fn(y)}
    return np.isin(ts.date, list(days))


def _ar1(rng, n, phi, sd):
    eps = rng.normal(0.0, sd * np.sqrt(1 - phi**2), n)
    return lfilter([1.0], [1.0, -phi], eps)


def _circ_gauss(hour, mu, sigma):
    d = (hour - mu + 12.0) % 24.0 - 12.0
    return np.exp(-0.5 * (d / sigma) ** 2)


def sample_source_params(rng: np.random.Generator, n: int) -> list[CoopParams]:
    out = []
    for i in range(n):
        if i % 2 == 0:  # US Midwest / South
            p = CoopParams(
                coop_id=f"US_{i:02d}", region="US", climate="continental_or_subtropical",
                base_mw=rng.uniform(15, 80), growth_per_year=rng.uniform(0.0, 0.03),
                t_mean=rng.uniform(8, 17), t_annual_amp=rng.uniform(10, 15),
                t_daily_amp=rng.uniform(4, 7), t_anomaly_sd=rng.uniform(2.5, 4.0),
                heat_slope=rng.uniform(0.008, 0.02), cool_slope=rng.uniform(0.02, 0.05),
                irrigation=rng.uniform(0.0, 0.25), commercial=rng.uniform(0.1, 0.25),
                morning_peak=rng.uniform(0.15, 0.3), evening_peak=rng.uniform(0.3, 0.5),
                ev_charging=0.0, forecast_error_sd=rng.uniform(1.0, 1.8),
            )
        else:  # Central EU
            p = CoopParams(
                coop_id=f"EU_{i:02d}", region="EU", climate="temperate",
                base_mw=rng.uniform(15, 60), growth_per_year=rng.uniform(0.0, 0.02),
                t_mean=rng.uniform(7, 12), t_annual_amp=rng.uniform(8, 11),
                t_daily_amp=rng.uniform(3, 6), t_anomaly_sd=rng.uniform(2.0, 3.5),
                heat_slope=rng.uniform(0.01, 0.025), cool_slope=rng.uniform(0.0, 0.015),
                irrigation=rng.uniform(0.0, 0.08), commercial=rng.uniform(0.1, 0.25),
                morning_peak=rng.uniform(0.15, 0.3), evening_peak=rng.uniform(0.3, 0.45),
                ev_charging=rng.uniform(0.0, 0.04), forecast_error_sd=rng.uniform(1.0, 1.5),
            )
        out.append(p)
    return out


def target_params() -> CoopParams:
    """Fixed, documented target: outside the source parameter ranges on purpose."""
    return CoopParams(
        coop_id="EU_NORDIC_TARGET", region="EU", climate="subarctic_heat_pump",
        base_mw=40.0, growth_per_year=0.04,
        t_mean=3.5, t_annual_amp=12.0, t_daily_amp=3.5, t_anomaly_sd=4.0,
        heat_slope=0.05, cool_slope=0.0, irrigation=0.0, commercial=0.15,
        morning_peak=0.3, evening_peak=0.35, ev_charging=0.25, forecast_error_sd=1.5,
    )


def simulate_coop(rng: np.random.Generator, p: CoopParams, ts: pd.DatetimeIndex) -> pd.DataFrame:
    n = len(ts)
    n_days = n // 24
    hour = ts.hour.values.astype(float)
    doy = ts.dayofyear.values.astype(float)
    dow = ts.dayofweek.values

    # Weather: seasonal + diurnal + multi-day anomaly + hourly jitter.
    anomaly = np.interp(np.arange(n) / 24.0, np.arange(n_days), _ar1(rng, n_days, 0.8, p.t_anomaly_sd))
    temp = (p.t_mean - p.t_annual_amp * np.cos(2 * np.pi * (doy - 15) / 365.25)
            - p.t_daily_amp * np.cos(2 * np.pi * (hour - 15) / 24) + anomaly
            + rng.normal(0.0, 0.6, n))
    # Day-ahead forecast = truth + autocorrelated error (what the model sees for the horizon).
    temp_fc = temp + _ar1(rng, n, 0.97, p.forecast_error_sd)
    # Buildings respond to a smoothed temperature (thermal inertia).
    b, a = [0.1], [1.0, -0.9]
    temp_eff = lfilter(b, a, temp, zi=lfilter_zi(b, a) * temp[0])[0]

    holiday = holiday_mask(ts, p.region)
    offday = (dow >= 5) | holiday
    shift = np.where(offday, 1.5, 0.0)
    profile = (0.55 + p.morning_peak * _circ_gauss(hour, 7.5 + shift, 1.5)
               + p.evening_peak * _circ_gauss(hour, 19.0, 2.5)
               + np.where(offday, 0.3, 1.0) * p.commercial * _circ_gauss(hour, 13.0, 3.5))
    heating = p.heat_slope * np.clip(15.0 - temp_eff, 0, None)
    cooling = p.cool_slope * np.clip(temp_eff - 22.0, 0, None)
    season = np.clip(np.sin(np.pi * (doy - 120) / 150), 0, None)
    irrigation = p.irrigation * season * _circ_gauss(hour, 13.0, 4.0) * np.where(offday, 0.7, 1.0)
    ev = p.ev_charging * _circ_gauss(hour, 21.5, 1.3)
    years = (ts - ts[0]).total_seconds().values / (365.25 * 86400)
    trend = 1.0 + p.growth_per_year * years
    noise = np.exp(_ar1(rng, n, 0.95, 0.015))

    demand = p.base_mw * trend * (profile + heating + cooling + irrigation + ev) * noise
    return pd.DataFrame({
        "timestamp": ts,
        "coop_id": p.coop_id,
        "region": p.region,
        "demand_mw": np.round(demand, 3),
        "temperature_c": np.round(temp, 2),
        "temperature_forecast_c": np.round(temp_fc, 2),
        "is_holiday": holiday.astype(np.int8),
    })


def generate(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    dcfg = cfg["data"]
    ts = pd.date_range(dcfg["start"], pd.Timestamp(dcfg["end"]) + pd.Timedelta(hours=23), freq="h")
    root = np.random.SeedSequence(dcfg["seed"])
    param_seed, *series_seeds = root.spawn(dcfg["n_source_coops"] + 2)
    params = sample_source_params(np.random.default_rng(param_seed), dcfg["n_source_coops"])
    source = pd.concat(
        [simulate_coop(np.random.default_rng(s), p, ts) for p, s in zip(params, series_seeds)],
        ignore_index=True,
    )
    tp = target_params()
    target = simulate_coop(np.random.default_rng(series_seeds[-1]), tp, ts)
    card = [asdict(p) | {"role": "source"} for p in params] + [asdict(tp) | {"role": "target"}]
    return source, target, card
