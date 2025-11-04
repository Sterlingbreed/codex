"""
Trading Signal Generator using Kraken Public API
================================================

This script fetches order book data and historical price data from Kraken's
public REST API and calculates technical indicators (MACD and RSI) across
multiple time frames. It then combines these indicators with a simple order
book imbalance measure to suggest buy, sell or hold signals. **No orders are
executed by this program.**

Usage example:

```
python trading_app.py --pair XXBTZUSD --intervals 15 60 240
```

**Disclaimer:** This tool is provided for educational purposes only and should
not be relied upon for actual trading without professional advice. Digital
asset trading carries risk, and you are solely responsible for your own
decisions.

The program also offers a **paper trading** mode that simulates buying and
selling based on the generated signals. This allows you to backtest the
strategy on historical data using a virtual balance, without executing any real
orders.
"""

import argparse
from typing import List, Tuple, Dict

import numpy as np
import pandas as pd
import requests


def fetch_ohlc(pair: str, interval: int = 60, since: int | None = None) -> pd.DataFrame:
    """Fetch OHLC data from Kraken for a given trading pair and interval.

    Args:
        pair: Trading pair symbol (e.g., 'XXBTZUSD' for BTC/USD).
        interval: Candle width in minutes (1, 5, 15, 30, 60, 240, 1440, 10080, 21600).
        since: Optional Unix timestamp to fetch data since a specific time.

    Returns:
        DataFrame with time-indexed OHLC data and numeric columns for price/volume.
    """
    url = "https://api.kraken.com/0/public/OHLC"
    params: Dict[str, str | int] = {"pair": pair, "interval": interval}
    if since:
        params["since"] = since
    response = requests.get(url, params=params, timeout=10)
    response.raise_for_status()
    data = response.json()
    # The result dictionary contains a key with the pair name
    pair_key = next(iter(data["result"]))
    rows = data["result"][pair_key]
    df = pd.DataFrame(
        rows,
        columns=[
            "time",
            "open",
            "high",
            "low",
            "close",
            "vwap",
            "volume",
            "count",
        ],
    )
    # Convert timestamp to datetime and numeric columns to floats
    df["time"] = pd.to_datetime(df["time"], unit="s")
    for col in ["open", "high", "low", "close", "vwap", "volume"]:
        df[col] = df[col].astype(float)
    return df


def fetch_order_book(pair: str, count: int = 10) -> float:
    """Fetch the top levels of the order book and compute imbalance.

    The imbalance is defined as (bid_volume - ask_volume) / (bid_volume + ask_volume).
    A positive value indicates stronger bid side; negative indicates stronger ask side.

    Args:
        pair: Trading pair symbol.
        count: Number of price levels to include from each side (default 10).

    Returns:
        A float representing the order book imbalance between -1 and 1.
    """
    url = "https://api.kraken.com/0/public/Depth"
    response = requests.get(url, params={"pair": pair, "count": count}, timeout=10)
    response.raise_for_status()
    data = response.json()
    pair_key = next(iter(data["result"]))
    asks = data["result"][pair_key]["asks"]
    bids = data["result"][pair_key]["bids"]
    ask_vol = sum(float(a[1]) for a in asks)
    bid_vol = sum(float(b[1]) for b in bids)
    # Prevent division by zero
    if ask_vol + bid_vol == 0:
        return 0.0
    imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol)
    return imbalance


def compute_macd(df: pd.DataFrame, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.DataFrame:
    """Compute MACD and signal line for a DataFrame of OHLC data.

    Args:
        df: DataFrame containing a 'close' column.
        fast: Period for the fast EMA.
        slow: Period for the slow EMA.
        signal: Period for the signal line EMA.

    Returns:
        DataFrame with added columns 'ema_fast', 'ema_slow', 'macd', 'signal', and 'hist'.
    """
    df = df.copy()
    df["ema_fast"] = df["close"].ewm(span=fast, adjust=False).mean()
    df["ema_slow"] = df["close"].ewm(span=slow, adjust=False).mean()
    df["macd"] = df["ema_fast"] - df["ema_slow"]
    df["signal"] = df["macd"].ewm(span=signal, adjust=False).mean()
    df["hist"] = df["macd"] - df["signal"]
    return df


def compute_rsi(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Compute the Relative Strength Index (RSI) for a given DataFrame.

    Args:
        df: DataFrame containing a 'close' column.
        period: Window size for the RSI calculation.

    Returns:
        A pandas Series with the RSI values.
    """
    delta = df["close"].diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    roll_up = pd.Series(gain).rolling(period).mean()
    roll_down = pd.Series(loss).rolling(period).mean()
    rs = roll_up / roll_down
    rsi = 100 - (100 / (1 + rs))
    return rsi


def generate_signal(df: pd.DataFrame, ob_imbalance: float, rsi_thresh_low: int = 30, rsi_thresh_high: int = 70) -> Tuple[str, Dict[str, str]]:
    """Generate a buy, sell or hold signal based on MACD, RSI and order book imbalance.

    Args:
        df: DataFrame containing MACD and RSI values.
        ob_imbalance: Order book imbalance metric (-1 to 1).
        rsi_thresh_low: RSI threshold below which asset is considered oversold.
        rsi_thresh_high: RSI threshold above which asset is considered overbought.

    Returns:
        A tuple containing the aggregated signal and a dictionary of component signals.
    """
    # Use the penultimate row to avoid signals on incomplete candle
    if len(df) < 2:
        return "hold", {"macd": "hold", "rsi": "hold", "orderbook": "hold"}
    prev = df.iloc[-2]
    last = df.iloc[-1]
    # MACD component: look for crossovers
    if prev["macd"] < prev["signal"] and last["macd"] > last["signal"]:
        macd_sig = "buy"
    elif prev["macd"] > prev["signal"] and last["macd"] < last["signal"]:
        macd_sig = "sell"
    else:
        macd_sig = "hold"
    # RSI component
    rsi_val = last["rsi"]
    if np.isnan(rsi_val):
        rsi_sig = "hold"
    elif rsi_val < rsi_thresh_low:
        rsi_sig = "buy"
    elif rsi_val > rsi_thresh_high:
        rsi_sig = "sell"
    else:
        rsi_sig = "hold"
    # Order book component
    if ob_imbalance > 0.1:
        ob_sig = "buy"
    elif ob_imbalance < -0.1:
        ob_sig = "sell"
    else:
        ob_sig = "hold"
    # Aggregate signals: majority wins, require at least two matching signals to act
    signals = [macd_sig, rsi_sig, ob_sig]
    if signals.count("buy") > signals.count("sell") and signals.count("buy") >= 2:
        overall = "buy"
    elif signals.count("sell") > signals.count("buy") and signals.count("sell") >= 2:
        overall = "sell"
    else:
        overall = "hold"
    return overall, {"macd": macd_sig, "rsi": rsi_sig, "orderbook": ob_sig}


def paper_backtest(
    df: pd.DataFrame,
    ob_imbalance: float,
    initial_balance: float = 10_000.0,
    rsi_thresh_low: int = 30,
    rsi_thresh_high: int = 70,
) -> float:
    """
    Run a simple paper trading backtest on historical data using the generated signals.

    The strategy buys the asset with the entire cash balance when a buy signal is
    triggered and sells the entire position when a sell signal occurs. If no
    position is held and the signal is hold, the balance remains in cash. This
    function does not execute any real trades and is intended for educational
    purposes.

    Args:
        df: DataFrame containing OHLC data with at least a 'close' column.
        ob_imbalance: Order book imbalance metric used for signal generation.
        initial_balance: The starting cash balance for the simulation.
        rsi_thresh_low: RSI threshold considered oversold.
        rsi_thresh_high: RSI threshold considered overbought.

    Returns:
        The final portfolio value after processing all available data.
    """
    balance = initial_balance
    position = 0.0
    for i in range(1, len(df)):
        # Use data up to the current point for indicator calculation
        sub_df = df.iloc[: i + 1].copy()
        sub_df = compute_macd(sub_df)
        sub_df["rsi"] = compute_rsi(sub_df)
        signal, _ = generate_signal(sub_df, ob_imbalance, rsi_thresh_low, rsi_thresh_high)
        price = float(sub_df["close"].iloc[-1])
        if signal == "buy" and position == 0:
            # Buy with all available capital
            if price > 0:
                position = balance / price
                balance = 0.0
        elif signal == "sell" and position > 0:
            # Sell entire position
            balance = position * price
            position = 0.0
        # Otherwise hold
    # Value remaining position at last close price
    final_price = float(df["close"].iloc[-1])
    final_value = balance + position * final_price
    return final_value


def main(args: List[str] | None = None) -> None:
    """Main function to parse arguments, compute indicators and print signals."""
    parser = argparse.ArgumentParser(
        description="Kraken trading signal generator (educational use only)"
    )
    parser.add_argument(
        "--pair",
        default="XXBTZUSD",
        help="Trading pair symbol, e.g., XXBTZUSD (BTC/USD). See Kraken documentation for codes.",
    )
    parser.add_argument(
        "--intervals",
        nargs="+",
        type=int,
        default=[15, 60, 240],
        help="List of candle intervals in minutes to evaluate (e.g., 15 60 240).",
    )
    parser.add_argument(
        "--since",
        type=int,
        default=None,
        help="Optional Unix timestamp to start fetching OHLC data from.",
    )

    parser.add_argument(
        "--paper",
        action="store_true",
        help=(
            "Run a paper trading simulation using the first interval provided. "
            "The simulation buys and sells based on the generated signals using a virtual balance."
        ),
    )
    parser.add_argument(
        "--initial_balance",
        type=float,
        default=10_000.0,
        help="Initial balance for paper trading simulation (default: 10000).",
    )
    ns = parser.parse_args(args)
    # Fetch order book imbalance once
    try:
        ob_imb = fetch_order_book(ns.pair)
    except Exception as exc:
        print(f"Failed to fetch order book: {exc}")
        ob_imb = 0.0
    print(f"Order book imbalance: {ob_imb:.3f}")
    results: List[str] = []
    for interval in ns.intervals:
        try:
            df = fetch_ohlc(ns.pair, interval, ns.since)
        except Exception as exc:
            print(f"Failed to fetch OHLC for interval {interval}: {exc}")
            results.append("hold")
            continue
        df = compute_macd(df)
        df["rsi"] = compute_rsi(df)
        signal, detail = generate_signal(df, ob_imb)
        print(
            f"Interval {interval}m | MACD: {detail['macd']}, RSI: {detail['rsi']}, OrderBook: {detail['orderbook']} -> Signal: {signal}"
        )
        results.append(signal)
    # Aggregate final recommendation
    if results.count("buy") > results.count("sell") and results.count("buy") >= 2:
        final = "buy"
    elif results.count("sell") > results.count("buy") and results.count("sell") >= 2:
        final = "sell"
    else:
        final = "hold"
    print(f"Overall signal across time frames: {final}")
    print(
        "\nDisclaimer: This script is for educational purposes only. It does not execute trades and "
        "should not be taken as financial advice. Always do your own research or consult a professional before trading."
    )

    # If paper mode is requested, run a backtest on the first interval
    if ns.paper:
        if ns.intervals:
            sim_interval = ns.intervals[0]
            try:
                sim_df = fetch_ohlc(ns.pair, sim_interval, ns.since)
                # Precompute indicators once for simulation
                sim_df = compute_macd(sim_df)
                sim_df["rsi"] = compute_rsi(sim_df)
                final_val = paper_backtest(sim_df, ob_imb, ns.initial_balance)
                print(
                    f"\nPaper trading simulation on {sim_interval}m candles: "
                    f"starting balance {ns.initial_balance:.2f} → final value {final_val:.2f}"
                )
                print(
                    "(No real trades were executed. This backtest is for demonstration purposes only.)"
                )
            except Exception as exc:
                print(f"Failed to run paper trading simulation: {exc}")
        else:
            print("Paper trading mode requested, but no intervals were provided.")


if __name__ == "__main__":
    main()