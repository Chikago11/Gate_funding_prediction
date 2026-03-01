import argparse
import base64
import datetime as dt
import html
import hmac
import json
import sqlite3
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

import matplotlib.pyplot as plt
import pandas as pd
import requests

BASE_URL = "https://api.gateio.ws/api/v4"
SETTLE = "usdt"
REQUEST_TIMEOUT = 20
MAX_RETRIES = 3
BACKOFF_SECONDS = (1, 2, 4)

ACTIVE_DIVERGENCE_THRESHOLD = 0.005
ADJUSTMENT_CLAMP = 0.0005
ACTIVATION_GAP_TARGET_RATIO = 0.1
DEFAULT_FMAX = 0.02

CSV_COLUMNS = ["timestamp", "datetime", "premium_index"]

# Web UI credentials (HTTP Basic Auth)
WEB_USERNAME = "meta"
WEB_PASSWORD = "mors"


def utc_now() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def configure_stdout() -> None:
    try:
        if hasattr(sys.stdout, "reconfigure"):
            sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def as_int(value: Any) -> Optional[int]:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def format_percent(value: Optional[float], decimals: int = 2) -> str:
    if value is None:
        return "n/a"
    return f"{float(value) * 100:.{decimals}f}%"


def ts_to_iso_utc(ts: int) -> str:
    return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )


def gate_get_json(path: str, params: Optional[Dict[str, Any]] = None) -> Any:
    url = f"{BASE_URL}{path}"
    last_error: Optional[Exception] = None

    for idx in range(MAX_RETRIES):
        try:
            response = requests.get(url, params=params, timeout=REQUEST_TIMEOUT)
            response.raise_for_status()
            return response.json()
        except (requests.RequestException, ValueError) as exc:
            last_error = exc
            if idx < MAX_RETRIES - 1:
                time.sleep(BACKOFF_SECONDS[idx])

    raise RuntimeError(f"Gate API request failed for {url}: {last_error}")


def normalize_interval_hours(raw: Any) -> Optional[int]:
    if raw is None:
        return None

    if isinstance(raw, str):
        text = raw.strip().lower()
        if text.endswith("h"):
            val = as_float(text[:-1])
            if val is not None and val in (1.0, 4.0, 8.0):
                return int(val)
            return None
        if text.endswith("m"):
            val = as_float(text[:-1])
            if val is None:
                return None
            hours = val / 60.0
            if hours in (1.0, 4.0, 8.0):
                return int(hours)
            return None
        parsed = as_float(text)
        if parsed is None:
            return None
        raw = parsed

    if isinstance(raw, (int, float)):
        val = float(raw)
        if val in (1.0, 4.0, 8.0):
            return int(val)
        if val in (3600.0, 14400.0, 28800.0):
            return int(val / 3600)
        if val in (60.0, 240.0, 480.0):
            return int(val / 60)
    return None


def parse_funding_interval_hours(item: Dict[str, Any]) -> Optional[int]:
    candidate_keys = [
        "funding_interval",
        "funding_interval_hours",
        "funding_interval_hour",
        "funding_interval_seconds",
        "funding_interval_secs",
        "funding_rate_interval",
    ]
    for key in candidate_keys:
        if key in item:
            parsed = normalize_interval_hours(item.get(key))
            if parsed is not None:
                return parsed
    return None


def compute_divergence(mark_price: float, index_price: float) -> Optional[float]:
    if index_price == 0:
        return None
    return abs(mark_price - index_price) / abs(index_price)


def fetch_contracts_snapshot(settle: str) -> List[Dict[str, Any]]:
    raw = gate_get_json(f"/futures/{settle}/contracts")
    if not isinstance(raw, list):
        raise RuntimeError("Unexpected contracts response format")

    snapshot: List[Dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue

        contract = item.get("name") or item.get("contract")
        mark_price = as_float(item.get("mark_price"))
        index_price = as_float(item.get("index_price"))
        interval_hours = parse_funding_interval_hours(item)

        if (
            not contract
            or mark_price is None
            or index_price is None
            or interval_hours is None
        ):
            continue

        divergence = compute_divergence(mark_price, index_price)
        if divergence is None:
            continue

        snapshot.append(
            {
                "contract": contract,
                "settle": settle,
                "mark_price": mark_price,
                "index_price": index_price,
                "divergence": divergence,
                "funding_interval_hours": interval_hours,
            }
        )
    return snapshot


def parse_premium_row(item: Any) -> Optional[Tuple[int, float]]:
    if isinstance(item, dict):
        ts_val = item.get("t") or item.get("time") or item.get("timestamp")
        premium_val = (
            item.get("close")
            or item.get("c")
            or item.get("premium_index")
            or item.get("value")
            or item.get("v")
        )
        ts = as_int(ts_val)
        premium = as_float(premium_val)
        if ts is not None and premium is not None:
            return ts, premium
        return None

    if isinstance(item, (list, tuple)) and len(item) >= 3:
        ts = as_int(item[0])
        premium: Optional[float] = None
        for idx in (2, 4, 1):
            if idx < len(item):
                premium = as_float(item[idx])
                if premium is not None:
                    break
        if ts is not None and premium is not None:
            return ts, premium
    return None


def empty_premium_df() -> pd.DataFrame:
    return pd.DataFrame(columns=CSV_COLUMNS)


def fetch_premium_index(
    settle: str, contract: str, ts_from: int, ts_to: int, interval: str = "1m"
) -> pd.DataFrame:
    if ts_from > ts_to:
        return empty_premium_df()

    raw = gate_get_json(
        f"/futures/{settle}/premium_index",
        params={
            "contract": contract,
            "interval": interval,
            "from": ts_from,
            "to": ts_to,
        },
    )

    if not isinstance(raw, list):
        raise RuntimeError(f"Unexpected premium_index response for {contract}")

    rows: List[Tuple[int, float]] = []
    for item in raw:
        parsed = parse_premium_row(item)
        if parsed is not None:
            rows.append(parsed)

    if not rows:
        return empty_premium_df()

    df = pd.DataFrame(rows, columns=["timestamp", "premium_index"])
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["premium_index"] = pd.to_numeric(df["premium_index"], errors="coerce")
    df = df.dropna(subset=["timestamp", "premium_index"])
    if df.empty:
        return empty_premium_df()
    df["timestamp"] = df["timestamp"].astype(int)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    df["datetime"] = df["timestamp"].map(ts_to_iso_utc)
    return df[CSV_COLUMNS].reset_index(drop=True)


def load_contract_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        return empty_premium_df()

    df = pd.read_csv(path)
    missing_cols = [c for c in CSV_COLUMNS if c not in df.columns]
    if missing_cols:
        raise RuntimeError(f"{path}: missing columns {missing_cols}")

    df = df[CSV_COLUMNS].copy()
    df["timestamp"] = pd.to_numeric(df["timestamp"], errors="coerce")
    df["premium_index"] = pd.to_numeric(df["premium_index"], errors="coerce")
    df = df.dropna(subset=["timestamp", "premium_index"])
    if df.empty:
        return empty_premium_df()

    df["timestamp"] = df["timestamp"].astype(int)
    df["datetime"] = df["timestamp"].map(ts_to_iso_utc)
    df = df.sort_values("timestamp").drop_duplicates(subset=["timestamp"], keep="last")
    return df[CSV_COLUMNS].reset_index(drop=True)


def merge_dedupe(old_df: pd.DataFrame, new_df: pd.DataFrame) -> pd.DataFrame:
    if old_df.empty and new_df.empty:
        return empty_premium_df()
    merged = pd.concat([old_df, new_df], ignore_index=True)
    merged["timestamp"] = pd.to_numeric(merged["timestamp"], errors="coerce")
    merged["premium_index"] = pd.to_numeric(merged["premium_index"], errors="coerce")
    merged = merged.dropna(subset=["timestamp", "premium_index"])
    if merged.empty:
        return empty_premium_df()
    merged["timestamp"] = merged["timestamp"].astype(int)
    merged = merged.sort_values("timestamp").drop_duplicates(
        subset=["timestamp"], keep="last"
    )
    merged["datetime"] = merged["timestamp"].map(ts_to_iso_utc)
    return merged[CSV_COLUMNS].reset_index(drop=True)


def save_contract_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.to_csv(tmp_path, index=False)
    tmp_path.replace(path)


def get_interval_bounds_utc(now: dt.datetime, interval_hours: int) -> Tuple[int, int]:
    if interval_hours not in (1, 4, 8):
        raise ValueError(f"Unsupported interval_hours: {interval_hours}")
    aligned_hour = (now.hour // interval_hours) * interval_hours
    start = now.replace(hour=aligned_hour, minute=0, second=0, microsecond=0)
    end = start + dt.timedelta(hours=interval_hours)
    return int(start.timestamp()), int(end.timestamp())


def init_db(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS contracts_state (
            contract TEXT NOT NULL,
            settle TEXT NOT NULL,
            is_active INTEGER NOT NULL DEFAULT 0,
            funding_interval_hours INTEGER,
            last_seen_ts INTEGER,
            last_divergence REAL,
            below_start_ts INTEGER,
            activated_ts INTEGER,
            deactivated_ts INTEGER,
            deactivate_reason TEXT,
            last_valid_interval_hours INTEGER,
            PRIMARY KEY (contract, settle)
        )
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS events_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ts INTEGER NOT NULL,
            contract TEXT NOT NULL,
            event_type TEXT NOT NULL,
            details TEXT
        )
        """
    )
    conn.commit()


def log_event(
    conn: sqlite3.Connection,
    ts: int,
    contract: str,
    event_type: str,
    details: Dict[str, Any],
) -> None:
    conn.execute(
        """
        INSERT INTO events_log (ts, contract, event_type, details)
        VALUES (?, ?, ?, ?)
        """,
        (ts, contract, event_type, json.dumps(details, ensure_ascii=True)),
    )


def update_active_state(
    conn: sqlite3.Connection, snapshot: List[Dict[str, Any]], now_ts: int
) -> List[Dict[str, Any]]:
    existing_rows = conn.execute(
        "SELECT * FROM contracts_state WHERE settle = ?", (SETTLE,)
    ).fetchall()
    existing = {row["contract"]: row for row in existing_rows}
    active_now: List[Dict[str, Any]] = []

    for row in snapshot:
        contract = row["contract"]
        prev = existing.get(contract)
        prev_active = bool(prev["is_active"]) if prev else False
        prev_interval = prev["funding_interval_hours"] if prev else None
        prev_below_start = prev["below_start_ts"] if prev else None
        prev_activated = prev["activated_ts"] if prev else None
        prev_last_valid = prev["last_valid_interval_hours"] if prev else None

        interval_hours = int(row["funding_interval_hours"])
        divergence = float(row["divergence"])
        interval_valid = interval_hours in (4, 8)
        deactivate_below_seconds = interval_hours * 3600

        is_active = prev_active
        below_start_ts = prev_below_start
        activated_ts = prev_activated
        deactivated_ts = prev["deactivated_ts"] if prev else None
        deactivate_reason = prev["deactivate_reason"] if prev else None
        last_valid_interval_hours = prev_last_valid
        just_activated = False

        if prev_interval is not None and prev_interval != interval_hours:
            log_event(
                conn,
                now_ts,
                contract,
                "interval_changed",
                {"from": prev_interval, "to": interval_hours},
            )

        if interval_valid:
            last_valid_interval_hours = interval_hours
            if divergence >= ACTIVE_DIVERGENCE_THRESHOLD:
                below_start_ts = None
                if not prev_active:
                    is_active = True
                    just_activated = True
                    activated_ts = now_ts
                    deactivated_ts = None
                    deactivate_reason = None
                    log_event(
                        conn,
                        now_ts,
                        contract,
                        "activated",
                        {"divergence": divergence, "interval_hours": interval_hours},
                    )
            else:
                if prev_active:
                    if below_start_ts is None:
                        below_start_ts = now_ts
                    elif now_ts - below_start_ts >= deactivate_below_seconds:
                        is_active = False
                        below_start_ts = None
                        deactivated_ts = now_ts
                        deactivate_reason = "deactivated_below_interval"
                        log_event(
                            conn,
                            now_ts,
                            contract,
                            "deactivated_below_interval",
                            {
                                "interval_hours": interval_hours,
                                "below_seconds": deactivate_below_seconds,
                            },
                        )
                else:
                    is_active = False
                    below_start_ts = None
        else:
            below_start_ts = None
            is_active = False
            if prev_active:
                deactivated_ts = now_ts
                deactivate_reason = (
                    "deactivated_interval_1h"
                    if interval_hours == 1
                    else "deactivated_interval_invalid"
                )
                event_type = (
                    "deactivated_interval_1h"
                    if interval_hours == 1
                    else "deactivated_interval_invalid"
                )
                log_event(
                    conn,
                    now_ts,
                    contract,
                    event_type,
                    {"interval_hours": interval_hours},
                )

        conn.execute(
            """
            INSERT INTO contracts_state (
                contract, settle, is_active, funding_interval_hours, last_seen_ts,
                last_divergence, below_start_ts, activated_ts, deactivated_ts,
                deactivate_reason, last_valid_interval_hours
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(contract, settle) DO UPDATE SET
                is_active = excluded.is_active,
                funding_interval_hours = excluded.funding_interval_hours,
                last_seen_ts = excluded.last_seen_ts,
                last_divergence = excluded.last_divergence,
                below_start_ts = excluded.below_start_ts,
                activated_ts = excluded.activated_ts,
                deactivated_ts = excluded.deactivated_ts,
                deactivate_reason = excluded.deactivate_reason,
                last_valid_interval_hours = excluded.last_valid_interval_hours
            """,
            (
                contract,
                row["settle"],
                int(is_active),
                interval_hours,
                now_ts,
                divergence,
                below_start_ts,
                activated_ts,
                deactivated_ts,
                deactivate_reason,
                last_valid_interval_hours,
            ),
        )

        if is_active and interval_valid:
            active_now.append(
                {
                    "contract": contract,
                    "settle": row["settle"],
                    "funding_interval_hours": interval_hours,
                    "divergence": divergence,
                    "just_activated": just_activated,
                    "activated_ts": activated_ts,
                }
            )

    conn.commit()
    return active_now


def compute_funding_metrics(
    df_interval: pd.DataFrame,
    interval_hours: int,
    k_last: int,
    interest_daily: float,
    fmax: float,
) -> Optional[Dict[str, Any]]:
    if df_interval.empty:
        return None

    premiums = df_interval["premium_index"]
    mins_total = interval_hours * 60

    start_ts = as_int(df_interval["timestamp"].min())
    now_ts = as_int(df_interval["timestamp"].max())
    if start_ts is None or now_ts is None:
        return None

    mins_elapsed = int((now_ts - start_ts) // 60) + 1
    mins_elapsed = min(max(1, mins_elapsed), mins_total)
    mins_left = max(0, mins_total - mins_elapsed)

    current_premium = float(premiums.iloc[-1])
    mean_so_far = float(premiums.mean())
    k = min(max(1, k_last), len(premiums))
    avg_k = float(premiums.tail(k).mean())

    avg_premium_forecast = (mean_so_far * mins_elapsed + avg_k * mins_left) / mins_total
    interest_interval = interest_daily * (interval_hours / 24.0)
    adj = clamp(
        interest_interval - current_premium, -ADJUSTMENT_CLAMP, ADJUSTMENT_CLAMP
    )
    funding_raw = avg_premium_forecast + adj
    funding_clamped = clamp(funding_raw, -fmax, fmax)
    hit_eps = 1e-15
    hit_fmax = abs(funding_clamped) >= fmax - hit_eps

    max_remaining_avg_before_pos_fmax: Optional[float] = None
    min_remaining_avg_before_neg_fmax: Optional[float] = None
    remaining_avg_limit_current_side: Optional[float] = None
    remaining_avg_limit_rule = "n/a"
    remaining_avg_limit_side = "n/a"
    if mins_left > 0:
        max_remaining_avg_before_pos_fmax = (
            ((fmax - hit_eps) - adj) * mins_total - mean_so_far * mins_elapsed
        ) / mins_left
        min_remaining_avg_before_neg_fmax = (
            ((-fmax + hit_eps) - adj) * mins_total - mean_so_far * mins_elapsed
        ) / mins_left

        if funding_raw >= 0:
            remaining_avg_limit_current_side = max_remaining_avg_before_pos_fmax
            remaining_avg_limit_rule = "<="
            remaining_avg_limit_side = "+fmax"
        else:
            remaining_avg_limit_current_side = min_remaining_avg_before_neg_fmax
            remaining_avg_limit_rule = ">="
            remaining_avg_limit_side = "-fmax"

    return {
        "points": len(premiums),
        "interval_hours": interval_hours,
        "mins_total": mins_total,
        "mins_elapsed": mins_elapsed,
        "mins_left": mins_left,
        "current_premium": current_premium,
        "mean_so_far": mean_so_far,
        "avg_k": avg_k,
        "avg_premium_forecast": avg_premium_forecast,
        "interest_interval": interest_interval,
        "adj": adj,
        "fmax": fmax,
        "funding_raw": funding_raw,
        "funding_clamped": funding_clamped,
        "hit_fmax": hit_fmax,
        "max_remaining_avg_before_pos_fmax": max_remaining_avg_before_pos_fmax,
        "min_remaining_avg_before_neg_fmax": min_remaining_avg_before_neg_fmax,
        "remaining_avg_limit_current_side": remaining_avg_limit_current_side,
        "remaining_avg_limit_rule": remaining_avg_limit_rule,
        "remaining_avg_limit_side": remaining_avg_limit_side,
    }


def render_report(
    contract: str, state_row: Optional[sqlite3.Row], metrics: Dict[str, Any]
) -> None:
    status = "inactive"
    divergence = None
    if state_row is not None:
        status = "active" if int(state_row["is_active"]) == 1 else "inactive"
        divergence = state_row["last_divergence"]

    div_text = format_percent(as_float(divergence))

    print("==== FUNDING REPORT ====")
    print(f"contract: {contract}")
    print(f"status: {status}")
    print(f"interval_hours: {metrics['interval_hours']}")
    print(f"current_divergence: {div_text}")
    print(f"minutes_elapsed: {metrics['mins_elapsed']}/{metrics['mins_total']}")
    print(f"minutes_left: {metrics['mins_left']}")
    print(f"current_premium: {format_percent(metrics['current_premium'])}")
    print(f"mean_premium_so_far: {format_percent(metrics['mean_so_far'])}")
    print(f"forecast_avg_premium: {format_percent(metrics['avg_premium_forecast'])}")
    print(f"fmax: {format_percent(metrics['fmax'])}")
    limit_current_side = metrics["remaining_avg_limit_current_side"]
    if limit_current_side is None:
        threshold_text = "n/a"
    else:
        threshold_text = (
            f"{metrics['remaining_avg_limit_rule']} "
            f"{format_percent(limit_current_side)} "
            f"(avoid {metrics['remaining_avg_limit_side']})"
        )
    print(f"avg_remaining_threshold_current_side: {threshold_text}")
    print(f"funding_raw: {format_percent(metrics['funding_raw'])}")
    print(f"funding_clamped: {format_percent(metrics['funding_clamped'])}")
    print(f"hit_fmax: {'yes' if metrics['hit_fmax'] else 'no'}")


def plot_interval(
    df_interval: pd.DataFrame, forecast_avg: float, out_png: Path, contract: str
) -> None:
    if df_interval.empty:
        return

    out_png.parent.mkdir(parents=True, exist_ok=True)
    times = pd.to_datetime(df_interval["timestamp"], unit="s", utc=True)

    plt.figure(figsize=(11, 4))
    plt.plot(times, df_interval["premium_index"], label="premium_index")
    plt.axhline(forecast_avg, color="red", linestyle="--", label="forecast_avg")
    plt.title(f"{contract} premium index (1m)")
    plt.xlabel("UTC")
    plt.ylabel("Premium index")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_png, dpi=140)
    plt.close()


def get_contract_csv_path(data_dir: Path, contract: str) -> Path:
    return data_dir / f"{contract}_premium.csv"


def get_contract_plot_path(data_dir: Path, contract: str) -> Path:
    return data_dir / f"{contract}_forecast.png"


def load_interval_slice(
    df: pd.DataFrame, now_ts: int, interval_hours: int
) -> Tuple[pd.DataFrame, int, int]:
    start_ts, end_ts = get_interval_bounds_utc(
        dt.datetime.fromtimestamp(now_ts, tz=dt.timezone.utc), interval_hours
    )
    sliced = df[
        (df["timestamp"] >= start_ts)
        & (df["timestamp"] < end_ts)
        & (df["timestamp"] <= now_ts)
    ].copy()
    sliced = sliced.sort_values("timestamp")
    return sliced, start_ts, end_ts


def build_activation_gap_fill_df(
    start_ts: int, activated_ts: int, current_divergence: float
) -> pd.DataFrame:
    if activated_ts <= start_ts:
        return empty_premium_df()

    minute_start = start_ts - (start_ts % 60)
    minute_end_exclusive = activated_ts - (activated_ts % 60)
    timestamps = list(range(minute_start, minute_end_exclusive, 60))
    if not timestamps:
        return empty_premium_df()

    target = max(0.0, float(current_divergence)) * ACTIVATION_GAP_TARGET_RATIO
    points = len(timestamps)

    if points == 1:
        values = [target]
    else:
        values = [target * (idx / (points - 1)) for idx in range(points)]

    df = pd.DataFrame({"timestamp": timestamps, "premium_index": values})
    df["datetime"] = df["timestamp"].map(ts_to_iso_utc)
    return df[CSV_COLUMNS]


def collect_contract_data(
    active_row: Dict[str, Any],
    now: dt.datetime,
    data_dir: Path,
    k_last: int,
    interest_daily: float,
    fmax: float,
    do_plot: bool,
) -> Optional[Dict[str, Any]]:
    contract = active_row["contract"]
    settle = active_row["settle"]
    interval_hours = int(active_row["funding_interval_hours"])
    now_ts = int(now.timestamp())
    start_ts, _ = get_interval_bounds_utc(now, interval_hours)

    csv_path = get_contract_csv_path(data_dir, contract)
    old_df = load_contract_csv(csv_path)
    is_just_activated = bool(active_row.get("just_activated", False))

    if is_just_activated:
        activated_ts = as_int(active_row.get("activated_ts"))
        if activated_ts is None:
            activated_ts = now_ts
        divergence_now = as_float(active_row.get("divergence")) or 0.0
        gap_df = build_activation_gap_fill_df(start_ts, activated_ts, divergence_now)
        if not gap_df.empty:
            old_df = merge_dedupe(old_df, gap_df)

    if old_df.empty or is_just_activated:
        fetch_from = start_ts
    else:
        last_ts = int(old_df["timestamp"].max())
        fetch_from = max(start_ts, last_ts + 60)

    new_df = empty_premium_df()
    if fetch_from <= now_ts:
        try:
            new_df = fetch_premium_index(
                settle, contract, fetch_from, now_ts, interval="1m"
            )
        except Exception as exc:
            print(f"[WARN] premium fetch failed for {contract}: {exc}")

    merged = merge_dedupe(old_df, new_df)
    save_contract_csv(csv_path, merged)

    df_interval, _, _ = load_interval_slice(merged, now_ts, interval_hours)
    metrics = compute_funding_metrics(
        df_interval, interval_hours, k_last, interest_daily, fmax
    )

    if do_plot and metrics is not None:
        out_png = get_contract_plot_path(data_dir, contract)
        plot_interval(df_interval, metrics["avg_premium_forecast"], out_png, contract)

    return metrics


def connect_db(path: Path) -> sqlite3.Connection:
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    init_db(conn)
    return conn


def run_collect_iteration(args: argparse.Namespace, conn: sqlite3.Connection) -> None:
    now = utc_now()
    now_ts = int(now.timestamp())

    try:
        snapshot = fetch_contracts_snapshot(SETTLE)
    except Exception as exc:
        print(f"[ERROR] snapshot fetch failed: {exc}")
        return

    active_rows = update_active_state(conn, snapshot, now_ts)
    print(
        f"[{now.strftime('%Y-%m-%d %H:%M:%S')} UTC] contracts={len(snapshot)} active={len(active_rows)}"
    )

    data_dir = Path(args.data_dir)
    for active_row in active_rows:
        contract = active_row["contract"]
        metrics = collect_contract_data(
            active_row=active_row,
            now=now,
            data_dir=data_dir,
            k_last=args.k_last,
            interest_daily=args.interest_daily,
            fmax=args.fmax,
            do_plot=args.plot,
        )
        if metrics is None:
            print(f"  - {contract}: updated (insufficient data in current interval)")
            continue
        print(
            f"  - {contract}: points={metrics['points']} "
            f"raw={format_percent(metrics['funding_raw'])} "
            f"clamped={format_percent(metrics['funding_clamped'])}"
        )


def sleep_to_next_minute() -> None:
    now = time.time()
    wait = 60.0 - (now % 60.0)
    if wait < 0.5:
        wait += 60.0
    time.sleep(wait)


def run_collect_once(args: argparse.Namespace) -> int:
    conn = connect_db(Path(args.db))
    try:
        run_collect_iteration(args, conn)
    finally:
        conn.close()
    return 0


def run_collect_loop(args: argparse.Namespace) -> int:
    conn = connect_db(Path(args.db))
    try:
        while True:
            run_collect_iteration(args, conn)
            sleep_to_next_minute()
    except KeyboardInterrupt:
        print("Stopped by user")
    finally:
        conn.close()
    return 0


def resolve_interval_hours(state_row: Optional[sqlite3.Row]) -> Optional[int]:
    if state_row is None:
        return None
    current = state_row["funding_interval_hours"]
    if current in (4, 8):
        return int(current)
    previous = state_row["last_valid_interval_hours"]
    if previous in (4, 8):
        return int(previous)
    return None


def build_contract_report(
    token: str,
    db_path: Path,
    data_dir: Path,
    k_last: int,
    interest_daily: float,
    fmax: float,
) -> Tuple[Optional[Dict[str, Any]], Optional[str]]:
    token = token.strip().upper()
    if not token:
        return None, "token is empty"

    contract = f"{token}_USDT"
    csv_path = get_contract_csv_path(data_dir, contract)

    conn = connect_db(db_path)
    try:
        state_row = conn.execute(
            "SELECT * FROM contracts_state WHERE contract = ? AND settle = ?",
            (contract, SETTLE),
        ).fetchone()
    finally:
        conn.close()

    if state_row is None and not csv_path.exists():
        return None, f"contract not found: {contract}"

    if not csv_path.exists():
        return None, f"no data for contract: {contract}"

    try:
        full_df = load_contract_csv(csv_path)
    except Exception as exc:
        return None, f"failed to load CSV for {contract}: {exc}"

    if full_df.empty:
        return None, f"insufficient data for contract: {contract}"

    interval_hours = resolve_interval_hours(state_row)
    if interval_hours is None:
        return (
            None,
            "no valid funding interval (4h/8h) in saved state for this contract",
        )

    now = utc_now()
    now_ts = int(now.timestamp())
    df_interval, _, _ = load_interval_slice(full_df, now_ts, interval_hours)
    metrics = compute_funding_metrics(
        df_interval=df_interval,
        interval_hours=interval_hours,
        k_last=k_last,
        interest_daily=interest_daily,
        fmax=fmax,
    )
    if metrics is None:
        return None, "insufficient data in current interval"

    return {
        "token": token,
        "contract": contract,
        "status": (
            "active"
            if (state_row is not None and int(state_row["is_active"]) == 1)
            else "inactive"
        ),
        "divergence": (
            float(state_row["last_divergence"])
            if state_row is not None and state_row["last_divergence"] is not None
            else None
        ),
        "state_row": state_row,
        "metrics": metrics,
        "df_interval": df_interval,
    }, None


def get_active_contracts(db_path: Path, limit: int = 100) -> List[Dict[str, Any]]:
    conn = connect_db(db_path)
    try:
        rows = conn.execute(
            """
            SELECT contract, funding_interval_hours, last_divergence, last_seen_ts
            FROM contracts_state
            WHERE settle = ? AND is_active = 1
            ORDER BY last_divergence DESC
            LIMIT ?
            """,
            (SETTLE, limit),
        ).fetchall()
    finally:
        conn.close()

    result: List[Dict[str, Any]] = []
    for row in rows:
        result.append(
            {
                "contract": row["contract"],
                "funding_interval_hours": row["funding_interval_hours"],
                "last_divergence": row["last_divergence"],
                "last_seen_ts": row["last_seen_ts"],
            }
        )
    return result


def is_web_request_authorized(headers: Any) -> bool:
    auth_header = headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Basic "):
        return False

    encoded = auth_header[6:].strip()
    if not encoded:
        return False

    try:
        decoded = base64.b64decode(encoded).decode("utf-8")
    except (ValueError, UnicodeDecodeError):
        return False

    if ":" not in decoded:
        return False

    username, password = decoded.split(":", 1)
    return hmac.compare_digest(username, WEB_USERNAME) and hmac.compare_digest(
        password, WEB_PASSWORD
    )


def run_query(args: argparse.Namespace) -> int:
    report, error = build_contract_report(
        token=args.token,
        db_path=Path(args.db),
        data_dir=Path(args.data_dir),
        k_last=args.k_last,
        interest_daily=args.interest_daily,
        fmax=args.fmax,
    )
    if error is not None:
        print(error)
        return 1

    assert report is not None
    render_report(report["contract"], report["state_row"], report["metrics"])
    if args.plot:
        out_png = get_contract_plot_path(Path(args.data_dir), report["contract"])
        plot_interval(
            report["df_interval"],
            report["metrics"]["avg_premium_forecast"],
            out_png,
            report["contract"],
        )
        print(f"plot saved: {out_png}")

    return 0


def render_web_html(
    token: str,
    report: Optional[Dict[str, Any]],
    error: Optional[str],
    active_rows: List[Dict[str, Any]],
) -> str:
    safe_token = html.escape(token)
    rows_html = ""
    for row in active_rows:
        divergence = row["last_divergence"]
        divergence_text = format_percent(as_float(divergence))
        seen = ts_to_iso_utc(int(row["last_seen_ts"])) if row["last_seen_ts"] else "n/a"
        rows_html += (
            f"<tr><td>{html.escape(str(row['contract']))}</td>"
            f"<td>{html.escape(str(row['funding_interval_hours']))}</td>"
            f"<td>{html.escape(divergence_text)}</td>"
            f"<td>{html.escape(seen)}</td></tr>"
        )
    if not rows_html:
        rows_html = "<tr><td colspan='4'>No active contracts in DB.</td></tr>"

    report_html = ""
    if error:
        report_html = f"<div class='error'>{html.escape(error)}</div>"
    elif report is not None:
        metrics = report["metrics"]
        divergence_text = format_percent(as_float(report["divergence"]))
        limit_current_side = metrics["remaining_avg_limit_current_side"]
        if limit_current_side is None:
            threshold_text = "n/a"
        else:
            threshold_text = (
                f'{metrics["remaining_avg_limit_rule"]} '
                f'{format_percent(limit_current_side)} '
                f'(avoid {metrics["remaining_avg_limit_side"]})'
            )
        report_html = f"""
        <h2>Funding Report: {html.escape(report["contract"])}</h2>
        <table class="metrics">
          <tr><th>Status</th><td>{html.escape(report["status"])}</td></tr>
          <tr><th>Current divergence</th><td>{html.escape(divergence_text)}</td></tr>
          <tr><th>Interval</th><td>{metrics["interval_hours"]}h</td></tr>
          <tr><th>Minutes elapsed</th><td>{metrics["mins_elapsed"]}/{metrics["mins_total"]}</td></tr>
          <tr><th>Minutes left</th><td>{metrics["mins_left"]}</td></tr>
          <tr><th>Current premium</th><td>{format_percent(metrics["current_premium"])}</td></tr>
          <tr><th>Mean premium so far</th><td>{format_percent(metrics["mean_so_far"])}</td></tr>
          <tr><th>Forecast avg premium</th><td>{format_percent(metrics["avg_premium_forecast"])}</td></tr>
          <tr><th>Funding cap (fmax)</th><td>{format_percent(metrics["fmax"])}</td></tr>
          <tr><th>Avg left threshold (current side)</th><td>{html.escape(threshold_text)}</td></tr>
          <tr><th>Funding raw</th><td>{format_percent(metrics["funding_raw"])}</td></tr>
          <tr><th>Funding clamped</th><td>{format_percent(metrics["funding_clamped"])}</td></tr>
          <tr><th>Hit fmax</th><td>{"yes" if metrics["hit_fmax"] else "no"}</td></tr>
        </table>
        """
    legend_html = """
    <h3>Легенда (рус)</h3>
    <table class="legend">
      <tr><th>Status</th><td>Статус контракта в локальной базе: active/inactive.</td></tr>
      <tr><th>Current divergence</th><td>Текущее расхождение между mark и index: |mark - index| / |index|.</td></tr>
      <tr><th>Interval</th><td>Длина текущего funding-окна (обычно 4ч или 8ч).</td></tr>
      <tr><th>Minutes elapsed</th><td>Сколько минут уже прошло в текущем funding-окне.</td></tr>
      <tr><th>Minutes left</th><td>Сколько минут осталось до конца funding-окна.</td></tr>
      <tr><th>Current premium</th><td>Последнее значение premium index (за последнюю минуту).</td></tr>
      <tr><th>Mean premium so far</th><td>Средний premium index с начала текущего funding-окна до сейчас.</td></tr>
      <tr><th>Forecast avg premium</th><td>Прогноз среднего premium index к закрытию окна.</td></tr>
      <tr><th>Funding cap (fmax)</th><td>Лимит funding в модели (ограничение по модулю).</td></tr>
      <tr><th>Avg left threshold (current side)</th><td>Порог для среднего premium на оставшееся время, чтобы не упереться в cap на текущей стороне.</td></tr>
      <tr><th>Funding raw</th><td>Сырой расчёт funding до применения лимита fmax.</td></tr>
      <tr><th>Funding clamped</th><td>Итог после ограничения Funding raw диапазоном [-fmax, +fmax].</td></tr>
      <tr><th>Hit fmax</th><td>Признак, что итог уткнулся в лимит fmax.</td></tr>
      <tr><th>Active Contracts</th><td>Список активных контрактов (топ-100 по divergence) из локальной базы.</td></tr>
      <tr><th>Last seen (UTC)</th><td>Время (UTC), когда контракт в последний раз обновлялся коллектором.</td></tr>
    </table>
    """

    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width,initial-scale=1" />
  <title>Gate Funding Dashboard</title>
  <style>
    body {{ font-family: Segoe UI, Arial, sans-serif; margin: 24px; background:#f4f6f8; color:#1d2430; }}
    .wrap {{ max-width: 980px; margin: 0 auto; }}
    .card {{ background:#fff; border:1px solid #d8dee8; border-radius:10px; padding:16px; margin-bottom:16px; }}
    h1,h2 {{ margin: 0 0 12px 0; }}
    form {{ display:flex; gap:8px; flex-wrap:wrap; }}
    input[type=text] {{ padding:10px; min-width:220px; border:1px solid #c4ccd8; border-radius:8px; }}
    button {{ padding:10px 14px; border:0; border-radius:8px; background:#1967d2; color:#fff; cursor:pointer; }}
    table {{ width:100%; border-collapse: collapse; }}
    th, td {{ border-bottom:1px solid #e5eaf0; padding:8px; text-align:left; font-size:14px; }}
    .metrics th {{ width:240px; }}
    .legend {{ margin-top:16px; }}
    .legend th {{ width:260px; }}
    .error {{ background:#fdecec; border:1px solid #f5b5b5; color:#8f1d1d; padding:10px; border-radius:8px; }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <h1>Gate Funding Dashboard</h1>
      <form method="GET" action="/">
        <input type="text" name="token" placeholder="Enter token, e.g. BTC" value="{safe_token}" />
        <button type="submit">Show report</button>
      </form>
      <p>Use this page after running collector mode to keep data fresh.</p>
    </div>
    <div class="card">
      {report_html if report_html else "<p>No token selected.</p>"}
      {legend_html}
    </div>
    <div class="card">
      <h2>Active Contracts (top 100 by divergence)</h2>
      <table>
        <thead><tr><th>Contract</th><th>Interval (h)</th><th>Divergence</th><th>Last seen (UTC)</th></tr></thead>
        <tbody>{rows_html}</tbody>
      </table>
    </div>
  </div>
</body>
</html>"""


def run_web(args: argparse.Namespace) -> int:
    db_path = Path(args.db)
    data_dir = Path(args.data_dir)

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if not is_web_request_authorized(self.headers):
                self.send_response(401)
                self.send_header(
                    "WWW-Authenticate",
                    'Basic realm="Gate Funding Dashboard", charset="UTF-8"',
                )
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Unauthorized")
                return

            parsed = urlparse(self.path)
            if parsed.path != "/":
                self.send_response(404)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Not found")
                return

            params = parse_qs(parsed.query)
            token = params.get("token", [""])[0].strip().upper()

            report: Optional[Dict[str, Any]] = None
            error: Optional[str] = None
            if token:
                report, error = build_contract_report(
                    token=token,
                    db_path=db_path,
                    data_dir=data_dir,
                    k_last=args.k_last,
                    interest_daily=args.interest_daily,
                    fmax=args.fmax,
                )

            active_rows = get_active_contracts(db_path=db_path, limit=100)
            content = render_web_html(
                token=token, report=report, error=error, active_rows=active_rows
            )

            payload = content.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def log_message(self, format: str, *args: Any) -> None:  # noqa: A003
            return

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"Web UI: http://{args.host}:{args.port}")
    print("Press Ctrl+C to stop.")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Gate.io funding predictor with active-list collection"
    )
    subparsers = parser.add_subparsers(dest="cmd", required=True)

    collect_parser = subparsers.add_parser(
        "collect", help="Collect data for active contracts"
    )
    mode_group = collect_parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--once", action="store_true", help="Run one collection iteration"
    )
    mode_group.add_argument(
        "--loop", action="store_true", help="Run collection every minute"
    )
    collect_parser.add_argument("--db", default="state.db", help="SQLite database path")
    collect_parser.add_argument(
        "--data_dir", default="data", help="Directory for CSV/PNG files"
    )
    collect_parser.add_argument(
        "--k_last", type=int, default=30, help="Last K minutes for forecast"
    )
    collect_parser.add_argument(
        "--interest_daily",
        type=float,
        default=0.0003,
        help="Interest value per day (e.g. 0.0003)",
    )
    collect_parser.add_argument(
        "--fmax", type=float, default=DEFAULT_FMAX, help="Funding cap"
    )
    collect_parser.add_argument("--plot", action="store_true", help="Save PNG plots")

    query_parser = subparsers.add_parser("query", help="Show funding metrics for token")
    query_parser.add_argument("--token", required=True, help="Token symbol, e.g. BTC")
    query_parser.add_argument("--db", default="state.db", help="SQLite database path")
    query_parser.add_argument(
        "--data_dir", default="data", help="Directory for CSV/PNG files"
    )
    query_parser.add_argument(
        "--k_last", type=int, default=30, help="Last K minutes for forecast"
    )
    query_parser.add_argument(
        "--interest_daily",
        type=float,
        default=0.0003,
        help="Interest value per day (e.g. 0.0003)",
    )
    query_parser.add_argument(
        "--fmax", type=float, default=DEFAULT_FMAX, help="Funding cap"
    )
    query_parser.add_argument("--plot", action="store_true", help="Save PNG plot")

    web_parser = subparsers.add_parser("web", help="Run web dashboard")
    web_parser.add_argument("--host", default="127.0.0.1", help="Web host")
    web_parser.add_argument("--port", type=int, default=8080, help="Web port")
    web_parser.add_argument("--db", default="state.db", help="SQLite database path")
    web_parser.add_argument(
        "--data_dir", default="data", help="Directory for CSV/PNG files"
    )
    web_parser.add_argument(
        "--k_last", type=int, default=30, help="Last K minutes for forecast"
    )
    web_parser.add_argument(
        "--interest_daily",
        type=float,
        default=0.0003,
        help="Interest value per day (e.g. 0.0003)",
    )
    web_parser.add_argument(
        "--fmax", type=float, default=DEFAULT_FMAX, help="Funding cap"
    )

    args = parser.parse_args()
    if getattr(args, "k_last", 1) <= 0:
        parser.error("--k_last must be > 0")
    if getattr(args, "fmax", 1) <= 0:
        parser.error("--fmax must be > 0")
    if getattr(args, "interest_daily", 0.0) < 0:
        parser.error("--interest_daily must be >= 0")
    if getattr(args, "port", 1) <= 0 or getattr(args, "port", 1) > 65535:
        parser.error("--port must be in range 1..65535")
    return args


def main() -> int:
    configure_stdout()
    args = parse_args()
    if args.cmd == "collect":
        if args.once:
            return run_collect_once(args)
        if args.loop:
            return run_collect_loop(args)
        return 1
    if args.cmd == "query":
        return run_query(args)
    if args.cmd == "web":
        return run_web(args)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
