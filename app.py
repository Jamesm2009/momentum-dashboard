"""
Market Momentum Dashboard — Matrix Series Bull/Bear Signal
Computes the Matrix Series indicator (Pine Script port) using yFinance OHLC data.
Signal: up > down = Bull, up < down = Bear
% Bullish = (bull ETFs reading Bull + bear ETFs reading Bear) / 42 * 100
Stores daily history in Upstash Redis. Seed load on first boot (60 days).
Daily refresh via /refresh cron after market close.
"""

from flask import Flask, render_template, jsonify
import yfinance as yf
import pandas as pd
import numpy as np
import threading
import time
import json
import os
import re
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo
from bs4 import BeautifulSoup

app = Flask(__name__)
CT = ZoneInfo("America/Chicago")

REDIS_URL   = os.environ.get("UPSTASH_REDIS_REST_URL", "")
REDIS_TOKEN = os.environ.get("UPSTASH_REDIS_REST_TOKEN", "")
REDIS_KEY   = "breadth_history_v1"
REDIS_KEY_FV = "finviz_breadth_v1"

# ── Ticker lists ──────────────────────────────────────────────────────────────
# For bull ETFs: Bull signal = correct (bullish breadth)
# For bear ETFs: Bear signal = correct (bullish breadth = inverse reading)

BULL_ETFS = [
    ("TQQQ",  "UltraPro QQQ 3x"),
    ("SPXL",  "S&P 500 Bull 3X"),
    ("TNA",   "Small Cap Bull 3X"),
    ("WEBL",  "DJ Internet Bull 3X"),
    ("FAS",   "Financial Bull 3X"),
    ("HIBL",  "S&P High Beta Bull 3X"),
    ("LABU",  "S&P Biotech Bull 3X"),
    ("SOXL",  "Semiconductor Bull 3X"),
    ("TECL",  "Technology Bull 3X"),
    ("SPY",   "SPDR S&P 500"),
    ("QQQ",   "Nasdaq-100"),
    ("ERX",   "Energy Bull 2X"),
    ("GUSH",  "Oil & Gas Bull 2X"),
    ("NUGT",  "Gold Miners Bull 2X"),
    ("TMF",   "20Y Treasury Bull 3X"),
    ("TYD",   "7-10Y Treasury Bull 3X"),
    ("UUP",   "USD Index Bullish"),
    ("DRN",   "Real Estate Bull 3X"),
    ("YINN",  "FTSE China Bull 3X"),
    ("EDC",   "Emerging Mkts Bull 3X"),
    ("BULZ",  "FANG & Innov. 3X"),
]

BEAR_ETFS = [
    ("SQQQ",  "Short QQQ -3X"),
    ("SPXS",  "S&P 500 Bear -3X"),
    ("TZA",   "Small Cap Bear 3X"),
    ("WEBS",  "DJ Internet Bear -3X"),
    ("FAZ",   "Financial Bear 3X"),
    ("HIBS",  "S&P High Beta Bear 3X"),
    ("LABD",  "S&P Biotech Bear 3X"),
    ("SOXS",  "Semiconductor Bear 3X"),
    ("TECS",  "Technology Bear -3X"),
    ("SH",    "S&P500 -1X"),
    ("PSQ",   "Short QQQ -1X"),
    ("ERY",   "Energy Bear -2X"),
    ("DRIP",  "Oil & Gas Bear 2X"),
    ("DUST",  "Gold Miners Bear -2X"),
    ("TMV",   "20Y Treasury Bear -3X"),
    ("TYO",   "7-10Y Treasury Bear -3X"),
    ("UDN",   "USD Index Bearish -1X"),
    ("DRV",   "Real Estate Bear -3X"),
    ("YANG",  "FTSE China Bear -3X"),
    ("EDZ",   "Emerging Mkts Bear -3X"),
    ("BERZ",  "FANG & Innov. 3X Inv."),
]

ALL_TICKERS = [t for t, _ in BULL_ETFS] + [t for t, _ in BEAR_ETFS]
BULL_SET    = {t for t, _ in BULL_ETFS}
BEAR_SET    = {t for t, _ in BEAR_ETFS}
TOTAL       = len(ALL_TICKERS)  # 42

# ── In-memory cache ───────────────────────────────────────────────────────────
cache = {
    "history":      [],   # list of {date, pct_bullish, signals: {ticker: "Bull"/"Bear"}}
    "last_updated": "—",
    "phase":        0,    # 0=idle, 1=loading, 4=ready
    "progress":     "Starting...",
    "error":        None,
}
_lock    = threading.Lock()
_started = False


# ── Redis helpers ─────────────────────────────────────────────────────────────

import requests as _req

def _rget(key):
    if not REDIS_URL or not REDIS_TOKEN:
        return None
    try:
        r = _req.get(f"{REDIS_URL}/get/{key}",
                     headers={"Authorization": f"Bearer {REDIS_TOKEN}"}, timeout=10)
        if r.status_code != 200:
            return None
        result = r.json().get("result")
        return json.loads(result) if result else None
    except Exception as e:
        print(f"  Redis GET error: {e}")
        return None


def _rset(key, value, ex=90000):
    if not REDIS_URL or not REDIS_TOKEN:
        return False
    try:
        r = _req.post(
            f"{REDIS_URL}/pipeline",
            headers={"Authorization": f"Bearer {REDIS_TOKEN}",
                     "Content-Type": "application/json"},
            data=json.dumps([["SET", key, json.dumps(value), "EX", ex]]),
            timeout=15
        )
        return r.status_code == 200
    except Exception as e:
        print(f"  Redis SET error: {e}")
        return False


def save_history():
    ok = _rset(REDIS_KEY, cache["history"], ex=60 * 60 * 24 * 90)  # 90 days
    print(f"  Redis save: {'OK' if ok else 'FAILED'} ({len(cache['history'])} days)")


def load_history():
    print("  Checking Redis for history...")
    data = _rget(REDIS_KEY)
    if not data:
        print("  No history found.")
        return False
    cache["history"] = data
    print(f"  Loaded {len(data)} days from Redis.")
    return True


# ── Finviz market breadth scraper ─────────────────────────────────────────────

def fetch_finviz_breadth():
    """
    Scrape the 4 breadth bars from the Finviz homepage:
    Advancing/Declining, New High/New Low, Above/Below SMA50, Above/Below SMA200.
    Returns dict or None on failure.
    """
    try:
        headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                          "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
            "Accept": "text/html,application/xhtml+xml",
            "Accept-Language": "en-US,en;q=0.9",
        }
        r = _req.get("https://finviz.com/", headers=headers, timeout=15)
        if r.status_code != 200:
            print(f"  Finviz fetch failed: HTTP {r.status_code}")
            return None

        text = r.text
        result = {"date": str(date.today())}

        # Pattern: "Advancing\n41.5% (2323)" / "Declining\n(3009) 53.7%"
        m = re.search(r'Advancing[^0-9]*?([\d.]+)%\s*\((\d+)\)', text)
        if m:
            result["adv_pct"] = float(m.group(1))
            result["adv_count"] = int(m.group(2))

        m = re.search(r'Declining[^0-9]*?\((\d+)\)\s*([\d.]+)%', text)
        if m:
            result["dec_count"] = int(m.group(1))
            result["dec_pct"] = float(m.group(2))

        # Pattern: "New High\n53.6% (173)" / "New Low\n(150) 46.4%"
        m = re.search(r'New High[^0-9]*?([\d.]+)%\s*\((\d+)\)', text)
        if m:
            result["hi_pct"] = float(m.group(1))
            result["hi_count"] = int(m.group(2))

        m = re.search(r'New Low[^0-9]*?\((\d+)\)\s*([\d.]+)%', text)
        if m:
            result["lo_count"] = int(m.group(1))
            result["lo_pct"] = float(m.group(2))

        # Pattern: "Above\n51.2% (2861)...SMA50" / "Below\n(2724) 48.8%"
        # SMA50 block
        sma50_block = re.search(r'(Above[^S]*?SMA50[^B]*?Below[^0-9]*?\(\d+\)\s*[\d.]+%)', text, re.DOTALL)
        if sma50_block:
            block = sma50_block.group(1)
            m = re.search(r'Above[^0-9]*?([\d.]+)%\s*\((\d+)\)', block)
            if m:
                result["sma50_above_pct"] = float(m.group(1))
                result["sma50_above_count"] = int(m.group(2))
            m = re.search(r'Below[^0-9]*?\((\d+)\)\s*([\d.]+)%', block)
            if m:
                result["sma50_below_count"] = int(m.group(1))
                result["sma50_below_pct"] = float(m.group(2))

        # SMA200 block
        sma200_block = re.search(r'(Above[^S]*?SMA200[^B]*?Below[^0-9]*?\(\d+\)\s*[\d.]+%)', text, re.DOTALL)
        if sma200_block:
            block = sma200_block.group(1)
            m = re.search(r'Above[^0-9]*?([\d.]+)%\s*\((\d+)\)', block)
            if m:
                result["sma200_above_pct"] = float(m.group(1))
                result["sma200_above_count"] = int(m.group(2))
            m = re.search(r'Below[^0-9]*?\((\d+)\)\s*([\d.]+)%', block)
            if m:
                result["sma200_below_count"] = int(m.group(1))
                result["sma200_below_pct"] = float(m.group(2))

        # Check we got at least the advancing/declining data
        if "adv_pct" not in result:
            print("  Finviz parse failed: could not find Advancing data")
            return None

        print(f"  Finviz breadth scraped: Adv {result.get('adv_pct')}% / Dec {result.get('dec_pct')}%")
        return result

    except Exception as e:
        print(f"  Finviz scrape error: {e}")
        return None


def save_finviz(data):
    """Save Finviz breadth data to Redis (4-day TTL)."""
    if data:
        _rset(REDIS_KEY_FV, data, ex=345600)


def load_finviz():
    """Load Finviz breadth data from Redis."""
    return _rget(REDIS_KEY_FV)


# ── Matrix Series calculation ─────────────────────────────────────────────────

def matrix_signal_series(h: pd.Series, l: pd.Series, c: pd.Series, smoother=5):
    """
    Port of Pine Script Matrix Series indicator.
    Returns a Series of 'Bull'/'Bear' strings indexed by date.
    ys1 = (H+L+C*2)/4
    rk3 = EMA(ys1, n)
    rk4 = StdDev(ys1, n)
    rk5 = (ys1 - rk3) * 200 / rk4
    rk6 = EMA(rk5, n)
    up   = EMA(rk6, n)
    down = EMA(up, n)
    signal: up > down -> Bull, else Bear
    """
    n    = smoother
    ys1  = (h + l + c * 2) / 4
    rk3  = ys1.ewm(span=n, adjust=False).mean()
    rk4  = ys1.rolling(n).std()
    # Avoid division by zero on early bars
    rk5  = ((ys1 - rk3) * 200 / rk4).fillna(0)
    rk6  = rk5.ewm(span=n, adjust=False).mean()
    up   = rk6.ewm(span=n, adjust=False).mean()
    down = up.ewm(span=n, adjust=False).mean()
    return pd.Series(
        np.where(up > down, "Bull", "Bear"),
        index=c.index
    )


def fetch_signals_for_range(trading_days: list[str]) -> dict:
    """
    Fetch OHLC for all 42 tickers, compute Matrix Series for each day in trading_days.
    Returns: {date_str: {ticker: "Bull"/"Bear", ..., "pct_bullish": float}}
    """
    # Need extra history to warm up the EMA (smoother=5, triple EMA → ~40 bars)
    warmup    = 60
    start_dt  = (datetime.strptime(trading_days[0], "%Y-%m-%d") - timedelta(days=warmup * 2)).strftime("%Y-%m-%d")
    # yfinance 'end' is exclusive — add 1 day so we include the last trading day
    end_dt    = (datetime.strptime(trading_days[-1], "%Y-%m-%d") + timedelta(days=1)).strftime("%Y-%m-%d")

    results   = {d: {} for d in trading_days}
    spy_closes = {}   # date_str -> SPY close price
    spy_vols   = {}   # date_str -> SPY volume
    total     = len(ALL_TICKERS)

    for i, ticker in enumerate(ALL_TICKERS):
        with _lock:
            cache["progress"] = f"Fetching {i+1}/{total}: {ticker}"
        print(f"  [{i+1}/{total}] {ticker}")

        try:
            df = yf.download(ticker, start=start_dt, end=end_dt,
                             interval="1d", auto_adjust=True, progress=False)
            if df.empty or len(df) < 25:
                print(f"    skip — insufficient data")
                continue

            h = df["High"].squeeze()
            l = df["Low"].squeeze()
            c = df["Close"].squeeze()

            signals = matrix_signal_series(h, l, c)
            # Convert index to date strings
            sig_map   = {str(d.date()): s for d, s in signals.items()}
            close_map = {str(d.date()): round(float(v), 2) for d, v in c.items()}

            for day in trading_days:
                if day in sig_map:
                    results[day][ticker] = sig_map[day]

            # Capture SPY closes and volume for chart overlay
            if ticker == "SPY":
                spy_closes = close_map
                spy_vols = {str(d.date()): int(v) for d, v in df["Volume"].squeeze().items()}

        except Exception as e:
            print(f"    ERR {ticker}: {e}")

        time.sleep(0.3)  # polite rate limiting

    # Attach SPY close and volume to each day's results
    for day in trading_days:
        results[day]["spy_close"] = spy_closes.get(day)
        results[day]["spy_volume"] = spy_vols.get(day)

    # Compute % Bullish for each day
    for day in trading_days:
        sigs = results[day]
        # Count only actual ticker signals (exclude metadata keys)
        ticker_sigs = {k: v for k, v in sigs.items() if k in BULL_SET or k in BEAR_SET}
        if len(ticker_sigs) < 10:
            # Not enough data for this day (likely holiday/missing)
            results[day]["pct_bullish"] = None
            continue
        bullish_count = sum(
            1 for t, s in ticker_sigs.items()
            if (t in BULL_SET and s == "Bull") or (t in BEAR_SET and s == "Bear")
        )
        results[day]["pct_bullish"] = round(bullish_count / TOTAL * 100, 1)

    return results


def get_trading_days(start: date, end: date) -> list[str]:
    """Return weekdays (Mon-Fri) between start and end inclusive."""
    days = []
    cur  = start
    while cur <= end:
        if cur.weekday() < 5:
            days.append(str(cur))
        cur += timedelta(days=1)
    return days


# ── Seed load ─────────────────────────────────────────────────────────────────

def run_seed_load():
    """Pull 60 trading days of history and build initial dataset."""
    with _lock:
        cache["phase"]    = 1
        cache["progress"] = "Starting seed load (60 days)..."
        cache["error"]    = None

    try:
        today      = date.today()
        start      = today - timedelta(days=90)  # ~60 trading days
        days       = get_trading_days(start, today)

        print(f"  Seed load: {len(days)} calendar days from {days[0]} to {days[-1]}")

        results = fetch_signals_for_range(days)

        history = []
        for day in days:
            pct = results[day].get("pct_bullish")
            if pct is None:
                continue
            spy = results[day].get("spy_close")
            vol = results[day].get("spy_volume")
            sigs = {k: v for k, v in results[day].items() if k not in ("pct_bullish", "spy_close", "spy_volume")}
            history.append({"date": day, "pct_bullish": pct, "spy_close": spy, "spy_volume": vol, "signals": sigs})

        with _lock:
            cache["history"]      = history
            cache["last_updated"] = datetime.now(CT).strftime("%-m/%-d/%y %H:%M CT")
            cache["phase"]        = 4
            cache["progress"]     = "Complete"

        save_history()
        print(f"  Seed load complete: {len(history)} days stored.")

    except Exception as e:
        import traceback; traceback.print_exc()
        with _lock:
            cache["error"]   = str(e)
            cache["phase"]   = 4


def run_daily_refresh():
    """Add today's reading to history."""
    with _lock:
        cache["phase"]    = 1
        cache["progress"] = "Refreshing today's data..."
        cache["error"]    = None

    try:
        today     = str(date.today())
        # Skip weekends
        if date.today().weekday() >= 5:
            with _lock:
                cache["phase"]    = 4
                cache["progress"] = "Weekend — no update"
            return

        results = fetch_signals_for_range([today])
        pct     = results[today].get("pct_bullish")

        if pct is None:
            # Still scrape Finviz even if market data isn't ready
            fv = fetch_finviz_breadth()
            if fv:
                save_finviz(fv)
            with _lock:
                cache["phase"]    = 4
                cache["progress"] = "No data for today yet"
            return

        spy = results[today].get("spy_close")
        vol = results[today].get("spy_volume")
        sigs = {k: v for k, v in results[today].items() if k not in ("pct_bullish", "spy_close", "spy_volume")}
        entry = {"date": today, "pct_bullish": pct, "spy_close": spy, "spy_volume": vol, "signals": sigs}

        with _lock:
            # Replace today's entry if it exists, else append
            existing = [h for h in cache["history"] if h["date"] != today]
            cache["history"]      = existing + [entry]
            cache["last_updated"] = datetime.now(CT).strftime("%-m/%-d/%y %H:%M CT")
            cache["phase"]        = 4
            cache["progress"]     = "Complete"

        save_history()
        print(f"  Daily refresh complete: {today} = {pct}% bullish")

        # Scrape Finviz breadth bars
        fv = fetch_finviz_breadth()
        if fv:
            save_finviz(fv)
            print(f"  Finviz breadth saved")

    except Exception as e:
        import traceback; traceback.print_exc()
        with _lock:
            cache["error"]   = str(e)
            cache["phase"]   = 4


def _ensure_started():
    global _started
    if not _started:
        _started = True
        loaded = load_history()
        if loaded and len(cache["history"]) > 0:
            with _lock:
                cache["phase"]        = 4
                cache["progress"]     = "Loaded from cache"
                cache["last_updated"] = cache["history"][-1]["date"]
            print(f"  Cache loaded: {len(cache['history'])} days")
        else:
            print("  No cache — starting seed load")
            threading.Thread(target=run_seed_load, daemon=True).start()


# ── Routes ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    _ensure_started()
    with _lock:
        snap = dict(cache)
    is_loading = snap["phase"] < 4 or len(snap["history"]) == 0
    return render_template("index.html",
        history=snap["history"],
        last_updated=snap["last_updated"],
        is_loading=is_loading,
        phase=snap["phase"],
        progress=snap["progress"],
        error=snap["error"],
        bull_etfs=BULL_ETFS,
        bear_etfs=BEAR_ETFS,
        finviz=load_finviz(),
    )


@app.route("/refresh")
def refresh():
    """Daily cron endpoint — call after market close (e.g. 4:30 PM CT)."""
    threading.Thread(target=run_daily_refresh, daemon=True).start()
    return jsonify({"status": "daily refresh started"})


@app.route("/reseed")
def reseed():
    """Force full 60-day seed reload (use if Redis data is lost)."""
    with _lock:
        cache["history"] = []
        cache["phase"]   = 0
    threading.Thread(target=run_seed_load, daemon=True).start()
    return jsonify({"status": "seed reload started"})


@app.route("/status")
def status():
    _ensure_started()
    with _lock:
        return jsonify({
            "phase":        cache["phase"],
            "days":         len(cache["history"]),
            "progress":     cache["progress"],
            "last_updated": cache["last_updated"],
            "error":        cache["error"],
        })


@app.route("/api/data")
def api_data():
    with _lock:
        return jsonify(cache["history"])


@app.route("/test-finviz")
def test_finviz():
    """Debug: scrape Finviz now and show parsed results."""
    data = fetch_finviz_breadth()
    if data:
        save_finviz(data)
    return jsonify({"status": "ok" if data else "failed", "data": data})


@app.route("/refresh-finviz")
def refresh_finviz():
    """Manually refresh just the Finviz breadth bars."""
    data = fetch_finviz_breadth()
    if data:
        save_finviz(data)
        return jsonify({"status": "saved", "data": data})
    return jsonify({"status": "scrape failed", "data": None})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
