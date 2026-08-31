"""
Data pipeline: acquisition, integrity verification, transformation and
partitioning with purging and embargoing.

Implements Section 4.3. The pipeline is executed once and cached, so that
every training run consumes a byte-identical dataset.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as cfg

log = logging.getLogger(__name__)


class DataIntegrityError(RuntimeError):
    """Raised when the retrieved panel fails a pre-transformation check.

    Failures raise rather than being silently repaired. Silent repair would
    conceal a data problem behind plausible-looking output, which is the
    failure mode hardest to detect downstream.
    """


# ---------------------------------------------------------------------------
# Stage 1: acquisition
# ---------------------------------------------------------------------------
def download_panel(
    assets: tuple[str, ...] = cfg.ASSETS,
    start: str = cfg.DATA_START,
    end: str = cfg.DATA_END,
    cache_dir: Path = cfg.DATA_DIR,
    force: bool = False,
) -> pd.DataFrame:
    """Retrieve adjusted OHLCV data and cache it with its retrieval date.

    ``auto_adjust=True`` is set explicitly rather than relied upon as a
    library default. With it enabled the returned Close series incorporates
    dividends and splits, which is the adjusted close specified in Section
    3.3. Without it the series would contain artificial discontinuities on
    ex-dividend dates -- a material concern given that TLT and LQD
    distribute monthly.
    """
    cache = cache_dir / f"panel_{start}_{end}.parquet"
    if cache.exists() and not force:
        log.info("Loading cached panel from %s", cache)
        return pd.read_parquet(cache)

    import yfinance as yf  # imported lazily so tests need no network

    log.info("Downloading %d tickers, %s to %s", len(assets), start, end)
    raw = yf.download(
        list(assets), start=start, end=end,
        auto_adjust=True, progress=False, group_by="column",
    )
    if raw.empty:
        raise DataIntegrityError("yfinance returned an empty frame")

    raw.to_parquet(cache)
    (cache_dir / "RETRIEVED.txt").write_text(
        f"retrieved={date.today().isoformat()}\n"
        f"start={start}\nend={end}\nassets={','.join(assets)}\n"
    )
    log.info("Cached %d rows to %s", len(raw), cache)
    return raw


# ---------------------------------------------------------------------------
# Stage 2: integrity checks
# ---------------------------------------------------------------------------
def extract_close(raw: pd.DataFrame, assets: tuple[str, ...] = cfg.ASSETS) -> pd.DataFrame:
    """Select the adjusted close columns in canonical asset order."""
    close = raw["Close"][list(assets)].copy()
    close.index = pd.to_datetime(close.index)
    return close


def trim_to_common_start(close: pd.DataFrame) -> pd.DataFrame:
    """Drop rows before every asset has a valid adjusted close.

    ``DATA_START`` may precede the latest inception in the universe (USO,
    April 2006). yfinance returns NaN for pre-inception dates; this trims
    to the common start rather than failing or silently forward-filling.
    """
    first_valid = close.apply(lambda s: s.first_valid_index())
    latest = first_valid.max()
    if latest > close.index[0]:
        late_asset = first_valid.idxmax()
        log.info(
            "Trimming panel from %s to %s (%s inception; %d pre-inception rows dropped)",
            close.index[0].date(), latest.date(), late_asset,
            close.index.get_loc(latest),
        )
        close = close.loc[latest:]
    return close


def verify_panel(close: pd.DataFrame, assets: tuple[str, ...] = cfg.ASSETS) -> None:
    """Three pre-transformation checks; any failure raises.

    1. No nulls in the adjusted close series (after common-start trim).
    2. Index strictly increasing with no duplicated dates.
    3. Every asset has data from the first row of the trimmed panel.
    """
    missing = set(assets) - set(close.columns)
    if missing:
        raise DataIntegrityError(f"absent columns: {sorted(missing)}")

    nulls = close.isna().sum()
    if nulls.any():
        offenders = nulls[nulls > 0].to_dict()
        raise DataIntegrityError(f"null adjusted closes: {offenders}")

    if not close.index.is_monotonic_increasing:
        raise DataIntegrityError("index is not monotonic increasing")
    if close.index.has_duplicates:
        dupes = close.index[close.index.duplicated()].tolist()[:5]
        raise DataIntegrityError(f"duplicated dates, first few: {dupes}")

    first_valid = close.apply(lambda s: s.first_valid_index())
    if (first_valid > close.index[0]).any():
        late = first_valid[first_valid > close.index[0]].index[0]
        raise DataIntegrityError(
            f"asset '{late}' has no data until {first_valid[late].date()}, "
            f"after the panel start {close.index[0].date()}. "
            "Call trim_to_common_start() first."
        )
    log.info("Integrity checks passed: %d rows, %d assets", *close.shape)


# ---------------------------------------------------------------------------
# Stage 3: transformation
# ---------------------------------------------------------------------------
def to_price_relatives(close: pd.DataFrame) -> pd.DataFrame:
    """Convert adjusted closes to price relatives y_t = P_t / P_(t-1).

    Price relatives rather than raw prices form the model input: price
    levels are non-stationary and incomparable across assets of differing
    scale, whereas relatives are approximately stationary and cluster near
    unity. The first row is undefined and is dropped.
    """
    rel = (close / close.shift(1)).dropna(how="any")
    if (rel <= 0).any().any():
        bad = rel.columns[(rel <= 0).any()].tolist()
        raise DataIntegrityError(f"non-positive price relative in {bad}")
    return rel


# ---------------------------------------------------------------------------
# Stage 4: partitioning with purge and embargo
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Partitions:
    """Three chronological partitions with their realised boundary dates."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame

    def summary(self) -> pd.DataFrame:
        rows = []
        for name in ("train", "val", "test"):
            d = getattr(self, name)
            rows.append({
                "partition": name,
                "rows": len(d),
                "first": d.index[0].date(),
                "last": d.index[-1].date(),
            })
        return pd.DataFrame(rows)


def partition(
    rel: pd.DataFrame,
    train_end: str = cfg.TRAIN_END,
    val_end: str = cfg.VAL_END,
    lookback: int = cfg.LOOKBACK,
    embargo: int = cfg.EMBARGO,
) -> Partitions:
    """Split chronologically, purging and embargoing at each boundary.

    Strict chronological ordering prevents the obvious leakage in which a
    model is evaluated on data preceding its training period. A subtler form
    remains: an observation shortly after a boundary has a lookback window
    extending backwards across it, so its state is built largely from prices
    the agent has already seen.

    Purging removes observations whose window would span a boundary: the
    final ``lookback`` rows of a partition and the first ``lookback`` rows of
    the next. Embargoing removes a further ``embargo`` rows, guarding against
    residual serial dependence beyond the mechanical window length
    (Lopez de Prado, 2018).

    Cost: roughly ``lookback + embargo`` observations at each of two
    boundaries, about 100 out of ~4,700.
    """
    tr = rel.loc[:train_end]
    va = rel.loc[train_end:val_end]
    te = rel.loc[val_end:]

    drop = lookback + embargo
    if min(len(tr), len(va), len(te)) <= drop:
        raise DataIntegrityError(
            f"a partition is shorter than the {drop}-row purge+embargo"
        )

    tr = tr.iloc[:-lookback]        # windows would extend into validation
    va = va.iloc[drop:]             # purge then embargo
    te = te.iloc[drop:]

    parts = Partitions(train=tr, val=va, test=te)
    log.info("Partition boundaries:\n%s", parts.summary().to_string(index=False))
    return parts


# ---------------------------------------------------------------------------
# Convenience entry point
# ---------------------------------------------------------------------------
def build(force_download: bool = False) -> Partitions:
    """Run the full pipeline and return the three partitions."""
    raw = download_panel(force=force_download)
    close = trim_to_common_start(extract_close(raw))
    verify_panel(close)
    rel = to_price_relatives(close)
    parts = partition(rel)
    parts.summary().to_csv(cfg.RESULTS_DIR / "partition_boundaries.csv", index=False)
    return parts
