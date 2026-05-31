import argparse
import json
import math
import os
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

os.environ.setdefault("MPLCONFIGDIR", os.path.abspath(".matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.abspath(".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


FEATURES = [
    "ret_1",
    "intraday_ret",
    "high_low_spread",
    "close_vwap_gap",
    "log_vol",
    "log_amount",
    "turnover_proxy",
    "turnover_rate",
    "volume_ratio",
    "log_circ_mv",
    "pb",
    "pe_ttm_pos",
    "net_mf_ratio",
    "large_order_net_ratio",
    "small_order_net_ratio",
]

METRIC_COLS = [
    "ts_code",
    "trade_date",
    "turnover_rate",
    "volume_ratio",
    "pe_ttm",
    "pb",
    "circ_mv",
]

MONEYFLOW_COLS = [
    "ts_code",
    "trade_date",
    "buy_sm_amount",
    "sell_sm_amount",
    "buy_lg_amount",
    "sell_lg_amount",
    "buy_elg_amount",
    "sell_elg_amount",
    "net_mf_amount",
]


def parse_args():
    parser = argparse.ArgumentParser(description="GRU stock trend prediction and backtest")
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/gru_daily"))
    parser.add_argument("--start-date", type=int, default=20190101)
    parser.add_argument("--end-date", type=int, default=20251231)
    parser.add_argument("--train-end", type=int, default=20231229)
    parser.add_argument("--val-start", type=int, default=20240101)
    parser.add_argument("--val-end", type=int, default=20251231)
    parser.add_argument("--lookback", type=int, default=20)
    parser.add_argument("--horizon", type=int, default=1)
    parser.add_argument(
        "--target",
        choices=["close_to_close", "next_open_to_close"],
        default="next_open_to_close",
        help="Prediction target. next_open_to_close matches after-close signals with next-day execution.",
    )
    parser.add_argument(
        "--label-transform",
        choices=["raw", "rank", "zscore"],
        default="raw",
        help="Transform the training label within each trade date. rank/zscore align training with cross-sectional selection.",
    )
    parser.add_argument("--model", choices=["gru", "lstm", "mlp", "transformer"], default="gru")
    parser.add_argument("--max-stocks", type=int, default=300)
    parser.add_argument("--max-train-samples", type=int, default=80000)
    parser.add_argument("--max-val-samples", type=int, default=40000)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--num-layers", type=int, default=1)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--top-n", type=int, default=10)
    parser.add_argument("--rebalance-k", type=int, default=3)
    parser.add_argument("--transaction-cost-bps", type=float, default=10.0)
    parser.add_argument(
        "--min-trade-amount",
        type=float,
        default=0.0,
        help="Minimum signal-day amount, in the raw data unit, for IC/backtest/latest recommendation eligibility.",
    )
    parser.add_argument("--min-price", type=float, default=0.0, help="Minimum signal-day close price for eligibility.")
    parser.add_argument("--max-price", type=float, default=0.0, help="Maximum signal-day close price for eligibility; 0 disables it.")
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_stock_pool(data_dir, max_stocks):
    basic = pd.read_csv(data_dir / "basic.csv", dtype={"ts_code": str, "list_date": str})
    pool = basic.copy()
    if "market" in pool.columns:
        pool = pool[pool["market"] != "北交所"]
    if "name" in pool.columns:
        pool = pool[~pool["name"].fillna("").str.contains("ST", case=False, regex=False)]
    pool = pool.sort_values(["list_date", "ts_code"])
    if max_stocks and max_stocks > 0:
        pool = pool.head(max_stocks)
    return pool[["ts_code", "name", "industry", "market"]].copy()


def list_daily_files(data_dir, start_date, end_date):
    files = []
    for path in sorted((data_dir / "daily").glob("*.csv")):
        try:
            date = int(path.stem)
        except ValueError:
            continue
        if start_date <= date <= end_date:
            files.append((date, path))
    return files


def load_st_set(data_dir, date):
    path = data_dir / "stock_st" / f"{date}.csv"
    if not path.exists():
        return set()
    st = pd.read_csv(path, usecols=["ts_code"], dtype={"ts_code": str})
    return set(st["ts_code"].dropna())


def load_daily_panel(data_dir, stock_pool, start_date, end_date):
    keep_codes = set(stock_pool["ts_code"])
    frames = []
    cols = [
        "ts_code",
        "trade_date",
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "pct_chg",
        "vol",
        "amount",
        "vwap",
    ]
    files = list_daily_files(data_dir, start_date, end_date)
    if not files:
        raise FileNotFoundError("No daily csv files found in the requested date range.")

    for date, path in files:
        df = pd.read_csv(path, usecols=cols, dtype={"ts_code": str})
        df = df[df["ts_code"].isin(keep_codes)]
        metric_path = data_dir / "metric" / f"{date}.csv"
        if metric_path.exists():
            metric = pd.read_csv(metric_path, usecols=METRIC_COLS, dtype={"ts_code": str})
            df = df.merge(metric, on=["ts_code", "trade_date"], how="left")
        moneyflow_path = data_dir / "moneyflow" / f"{date}.csv"
        if moneyflow_path.exists():
            moneyflow = pd.read_csv(moneyflow_path, usecols=MONEYFLOW_COLS, dtype={"ts_code": str})
            df = df.merge(moneyflow, on=["ts_code", "trade_date"], how="left")
        st_codes = load_st_set(data_dir, date)
        if st_codes:
            df = df[~df["ts_code"].isin(st_codes)]
        frames.append(df)

    panel = pd.concat(frames, ignore_index=True)
    panel["trade_date"] = panel["trade_date"].astype(int)
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    return panel


def add_features_and_label(panel, horizon):
    df = panel.copy()
    numeric_cols = [
        "open",
        "high",
        "low",
        "close",
        "pre_close",
        "vol",
        "amount",
        "vwap",
        "turnover_rate",
        "volume_ratio",
        "pe_ttm",
        "pb",
        "circ_mv",
        "buy_sm_amount",
        "sell_sm_amount",
        "buy_lg_amount",
        "sell_lg_amount",
        "buy_elg_amount",
        "sell_elg_amount",
        "net_mf_amount",
    ]
    for col in numeric_cols:
        if col not in df.columns:
            df[col] = np.nan
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ret_1"] = df["pct_chg"].astype(float) / 100.0
    df["intraday_ret"] = df["close"] / df["open"] - 1.0
    df["high_low_spread"] = df["high"] / df["low"] - 1.0
    df["close_vwap_gap"] = df["close"] / df["vwap"] - 1.0
    df["log_vol"] = np.log1p(df["vol"])
    df["log_amount"] = np.log1p(df["amount"])
    df["turnover_proxy"] = df["amount"] / df["vol"].replace(0, np.nan)
    df["turnover_proxy"] = np.log1p(df["turnover_proxy"])
    df["log_circ_mv"] = np.log1p(df["circ_mv"].clip(lower=0))
    df["pe_ttm_pos"] = np.log1p(df["pe_ttm"].clip(lower=0))
    df["net_mf_ratio"] = df["net_mf_amount"] / df["amount"].replace(0, np.nan)
    df["large_order_net_ratio"] = (
        df["buy_lg_amount"]
        + df["buy_elg_amount"]
        - df["sell_lg_amount"]
        - df["sell_elg_amount"]
    ) / df["amount"].replace(0, np.nan)
    df["small_order_net_ratio"] = (df["buy_sm_amount"] - df["sell_sm_amount"]) / df[
        "amount"
    ].replace(0, np.nan)

    grouped = df.groupby("ts_code")
    future_close = grouped["close"].shift(-horizon)
    next_open = grouped["open"].shift(-1)
    df["close_to_close_return"] = future_close / df["close"] - 1.0
    df["next_open_to_close_return"] = future_close / next_open - 1.0

    df = df.replace([np.inf, -np.inf], np.nan)
    for col in FEATURES:
        df[col] = df[col].fillna(df.groupby("trade_date")[col].transform("median"))
    df[FEATURES] = df[FEATURES].fillna(0.0)
    needed = FEATURES + ["close"]
    df = df.dropna(subset=needed)
    return df


def add_training_label(df, target, label_transform):
    data = df.copy()
    raw_col = f"{target}_return"
    if label_transform == "raw":
        data["label"] = data[raw_col]
    elif label_transform == "rank":
        counts = data.groupby("trade_date")[raw_col].transform("count")
        ranks = data.groupby("trade_date")[raw_col].rank(method="average", pct=True)
        data["label"] = (ranks - 0.5).where(counts > 1)
    elif label_transform == "zscore":
        grouped = data.groupby("trade_date")[raw_col]
        mean = grouped.transform("mean")
        std = grouped.transform("std").replace(0, np.nan)
        data["label"] = ((data[raw_col] - mean) / (std + 1e-6)).clip(-5.0, 5.0)
    else:
        raise ValueError(f"Unknown label transform: {label_transform}")
    return data


def deterministic_cap(indices, max_samples):
    if max_samples is None or max_samples <= 0 or len(indices) <= max_samples:
        return indices
    positions = np.linspace(0, len(indices) - 1, max_samples).astype(int)
    return indices[positions]


def build_windows(df, lookback, train_end, val_start, val_end, max_train, max_val, target):
    xs = []
    raw_returns = []
    dates = []
    codes = []
    closes = []
    amounts = []
    label_values = []
    close_to_close_returns = []
    executable_returns = []

    for code, group in df.groupby("ts_code", sort=False):
        group = group.sort_values("trade_date")
        feat = group[FEATURES].to_numpy(dtype=np.float32)
        label = group["label"].to_numpy(dtype=np.float32)
        raw_return = group[f"{target}_return"].to_numpy(dtype=np.float32)
        close_to_close = group["close_to_close_return"].to_numpy(dtype=np.float32)
        executable_return = group["next_open_to_close_return"].to_numpy(dtype=np.float32)
        date = group["trade_date"].to_numpy(dtype=np.int64)
        close = group["close"].to_numpy(dtype=np.float32)
        amount = group["amount"].to_numpy(dtype=np.float32)

        for i in range(lookback - 1, len(group)):
            if np.isnan(label[i]):
                continue
            if np.isnan(executable_return[i]):
                continue
            window = feat[i - lookback + 1 : i + 1]
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True) + 1e-6
            xs.append((window - mean) / std)
            raw_returns.append(raw_return[i])
            dates.append(date[i])
            codes.append(code)
            closes.append(close[i])
            amounts.append(amount[i])
            label_values.append(label[i])
            close_to_close_returns.append(close_to_close[i])
            executable_returns.append(executable_return[i])

    x = np.stack(xs).astype(np.float32)
    y_raw = np.asarray(raw_returns, dtype=np.float32)
    y_label = np.asarray(label_values, dtype=np.float32)
    dates = np.asarray(dates, dtype=np.int64)
    closes = np.asarray(closes, dtype=np.float32)
    amounts = np.asarray(amounts, dtype=np.float32)
    close_to_close_returns = np.asarray(close_to_close_returns, dtype=np.float32)
    executable_returns = np.asarray(executable_returns, dtype=np.float32)
    codes = np.asarray(codes)

    train_idx = np.where(dates <= train_end)[0]
    eval_idx = np.where((dates >= val_start) & (dates <= val_end))[0]
    train_idx = deterministic_cap(train_idx, max_train)
    val_loss_idx = deterministic_cap(eval_idx, max_val)

    meta = pd.DataFrame(
        {
            "trade_date": dates[eval_idx],
            "ts_code": codes[eval_idx],
            "close": closes[eval_idx],
            "amount": amounts[eval_idx],
            "y_true": y_raw[eval_idx],
            "y_label": y_label[eval_idx],
            "close_to_close_return": close_to_close_returns[eval_idx],
            "strategy_return": executable_returns[eval_idx],
        }
    )
    return (
        x[train_idx],
        y_label[train_idx],
        x[val_loss_idx],
        y_label[val_loss_idx],
        x[eval_idx],
        y_label[eval_idx],
        meta,
    )


def build_latest_windows(df, lookback):
    xs = []
    rows = []
    for code, group in df.groupby("ts_code", sort=False):
        group = group.sort_values("trade_date")
        if len(group) < lookback:
            continue
        window = group[FEATURES].tail(lookback).to_numpy(dtype=np.float32)
        mean = window.mean(axis=0, keepdims=True)
        std = window.std(axis=0, keepdims=True) + 1e-6
        xs.append((window - mean) / std)
        last = group.iloc[-1]
        rows.append(
            {
                "trade_date": int(last["trade_date"]),
                "ts_code": code,
                "close": float(last["close"]),
                "amount": float(last["amount"]),
            }
        )
    if not xs:
        raise ValueError("No stocks have enough history for latest inference.")
    return np.stack(xs).astype(np.float32), pd.DataFrame(rows)


class GRURegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=gru_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.gru(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class LSTMRegressor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout):
        super().__init__()
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.head(out[:, -1, :]).squeeze(-1)


class MLPRegressor(nn.Module):
    def __init__(self, input_size, lookback, hidden_size, dropout):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(input_size * lookback, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


class TransformerRegressor(nn.Module):
    def __init__(self, input_size, lookback, hidden_size, num_layers, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_size)
        self.pos_embed = nn.Parameter(torch.zeros(1, lookback, hidden_size))
        layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=4,
            dim_feedforward=hidden_size * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x):
        x = self.input_proj(x) + self.pos_embed[:, : x.shape[1], :]
        out = self.encoder(x)
        return self.head(out[:, -1, :]).squeeze(-1)


def build_model(args, input_size):
    if args.model == "gru":
        return GRURegressor(input_size, args.hidden_size, args.num_layers, args.dropout)
    if args.model == "lstm":
        return LSTMRegressor(input_size, args.hidden_size, args.num_layers, args.dropout)
    if args.model == "mlp":
        return MLPRegressor(input_size, args.lookback, args.hidden_size, args.dropout)
    if args.model == "transformer":
        return TransformerRegressor(input_size, args.lookback, args.hidden_size, args.num_layers, args.dropout)
    raise ValueError(f"Unknown model: {args.model}")


def train_model(args, x_train, y_train, x_val, y_val):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(args, x_train.shape[-1]).to(device)

    train_ds = TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
    val_x = torch.from_numpy(x_val).to(device)
    val_y = torch.from_numpy(y_val).to(device)
    loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = nn.MSELoss()
    history = []

    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for xb, yb in loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(xb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            losses.append(loss.item())

        model.eval()
        with torch.no_grad():
            val_pred = model(val_x)
            val_loss = criterion(val_pred, val_y).item()
        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "val_loss": val_loss}
        history.append(row)
        print(f"epoch={epoch} train_loss={row['train_loss']:.6f} val_loss={val_loss:.6f}")

    return model, history, device


def predict(model, x, device, batch_size):
    model.eval()
    preds = []
    loader = DataLoader(torch.from_numpy(x), batch_size=batch_size, shuffle=False)
    with torch.no_grad():
        for xb in loader:
            pred = model(xb.to(device)).detach().cpu().numpy()
            preds.append(pred)
    return np.concatenate(preds)


def mark_eligibility(df, min_trade_amount, min_price, max_price):
    eligible = pd.Series(True, index=df.index)
    if min_trade_amount > 0:
        eligible &= df["amount"] >= min_trade_amount
    if min_price > 0:
        eligible &= df["close"] >= min_price
    if max_price > 0:
        eligible &= df["close"] <= max_price
    result = df.copy()
    result["is_eligible"] = eligible
    return result


def daily_ic(pred_df):
    values = []
    for date, group in pred_df.groupby("trade_date"):
        if len(group) < 3:
            continue
        if group["score"].nunique() < 2 or group["y_true"].nunique() < 2:
            continue
        values.append({"trade_date": date, "ic": group["score"].corr(group["y_true"])})
    ic = pd.DataFrame(values)
    if ic.empty:
        return ic, {"ic_mean": float("nan"), "ic_std": float("nan"), "icir": float("nan")}
    mean = ic["ic"].mean()
    std = ic["ic"].std(ddof=1)
    return ic, {"ic_mean": mean, "ic_std": std, "icir": mean / std if std and not np.isnan(std) else float("nan")}


def evaluate_strategy_grid(pred_df, transaction_cost_bps):
    rows = []
    for top_n, rebalance_k in [
        (10, 1),
        (10, 2),
        (10, 3),
        (20, 2),
        (20, 3),
        (20, 5),
        (30, 2),
        (30, 3),
        (30, 5),
        (50, 3),
        (50, 5),
        (50, 8),
    ]:
        _, metrics = backtest_topk(pred_df, top_n, rebalance_k, transaction_cost_bps)
        if not metrics:
            continue
        rows.append({"top_n": top_n, "rebalance_k": rebalance_k, **metrics})
    return pd.DataFrame(rows)


def backtest_topk(pred_df, top_n, rebalance_k, transaction_cost_bps):
    portfolio = []
    rows = []
    one_way_cost = transaction_cost_bps / 10000.0
    for date, day in pred_df.sort_values("trade_date").groupby("trade_date", sort=True):
        day = day.dropna(subset=["score", "strategy_return"]).sort_values("score", ascending=False)
        available = list(day["ts_code"])
        if not available:
            continue

        old_portfolio = list(portfolio)
        if not portfolio:
            portfolio = available[:top_n]
            traded_names = len(portfolio)
        else:
            score_map = day.set_index("ts_code")["score"].to_dict()
            held_with_scores = [(code, score_map.get(code, -np.inf)) for code in portfolio]
            sell = {
                code
                for code, _ in sorted(held_with_scores, key=lambda item: item[1])[:rebalance_k]
            }
            portfolio = [code for code in portfolio if code not in sell and code in set(available)]
            before_buy = set(portfolio)
            for code in available:
                if len(portfolio) >= top_n:
                    break
                if code not in portfolio:
                    portfolio.append(code)
            bought = set(portfolio) - before_buy
            traded_names = len(sell) + len(bought)

        held = day[day["ts_code"].isin(portfolio)]
        if held.empty:
            continue
        gross_ret = held["strategy_return"].mean()
        turnover = traded_names / max(1, top_n)
        cost = turnover * one_way_cost
        ret = gross_ret - cost
        rows.append(
            {
                "trade_date": date,
                "gross_return": gross_ret,
                "transaction_cost": cost,
                "turnover": turnover,
                "daily_return": ret,
                "nav": np.nan,
                "holdings": ",".join(portfolio),
                "prev_holdings": ",".join(old_portfolio),
            }
        )

    bt = pd.DataFrame(rows)
    if bt.empty:
        return bt, {}
    bt["nav"] = (1.0 + bt["daily_return"]).cumprod()
    total_return = bt["nav"].iloc[-1] - 1.0
    annual_return = bt["nav"].iloc[-1] ** (252.0 / len(bt)) - 1.0
    vol = bt["daily_return"].std(ddof=1)
    sharpe = bt["daily_return"].mean() / vol * math.sqrt(252.0) if vol and not np.isnan(vol) else float("nan")
    drawdown = bt["nav"] / bt["nav"].cummax() - 1.0
    metrics = {
        "total_return": total_return,
        "annual_return": annual_return,
        "sharpe": sharpe,
        "max_drawdown": drawdown.min(),
        "avg_turnover": bt["turnover"].mean(),
        "total_transaction_cost": bt["transaction_cost"].sum(),
        "days": int(len(bt)),
    }
    return bt, metrics


def benchmark_returns(data_dir, index_code, dates, horizon):
    path = data_dir / "market" / f"{index_code}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["trade_date"] = df["trade_date"].astype(int)
    df = df.sort_values("trade_date")
    entry_open = df["open"].shift(-1)
    exit_close = df["close"].shift(-horizon)
    df["daily_return"] = exit_close / entry_open - 1.0
    bench = df[df["trade_date"].isin(set(dates))][["trade_date", "daily_return"]].dropna()
    bench["nav"] = (1.0 + bench["daily_return"]).cumprod()
    return bench


def plot_outputs(history, bt, bench, output_dir):
    hist = pd.DataFrame(history)
    plt.figure(figsize=(7, 4))
    plt.plot(hist["epoch"], hist["train_loss"], label="train")
    plt.plot(hist["epoch"], hist["val_loss"], label="validation")
    plt.xlabel("Epoch")
    plt.ylabel("MSE loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "loss_curve.png", dpi=160)
    plt.close()

    if not bt.empty:
        plt.figure(figsize=(8, 4))
        plt.plot(bt["trade_date"].astype(str), bt["nav"], label="GRU top-k")
        if not bench.empty:
            aligned = bench[bench["trade_date"].isin(set(bt["trade_date"]))]
            plt.plot(aligned["trade_date"].astype(str), aligned["nav"], label="CSI 300")
        step = max(1, len(bt) // 8)
        plt.xticks(range(0, len(bt), step), bt["trade_date"].astype(str).iloc[::step], rotation=30)
        plt.ylabel("Net asset value")
        plt.legend()
        plt.tight_layout()
        plt.savefig(output_dir / "backtest_nav.png", dpi=160)
        plt.close()


def save_latest_recommendations(pred_df, stock_info, output_dir, top_n):
    latest_date = pred_df["trade_date"].max()
    latest = pred_df[pred_df["trade_date"] == latest_date].sort_values("score", ascending=False)
    latest = latest.head(top_n).merge(stock_info, on="ts_code", how="left")
    latest.to_csv(output_dir / "latest_recommendations.csv", index=False)
    return latest_date, latest


def main():
    args = parse_args()
    set_seed(args.seed)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    stock_pool = select_stock_pool(args.data_dir, args.max_stocks)
    print(f"selected_stocks={len(stock_pool)}")
    panel = load_daily_panel(args.data_dir, stock_pool, args.start_date, args.end_date)
    data = add_features_and_label(panel, args.horizon)
    data = add_training_label(data, args.target, args.label_transform)
    x_train, y_train, x_val_loss, y_val_loss, x_eval, _y_eval, meta = build_windows(
        data,
        args.lookback,
        args.train_end,
        args.val_start,
        args.val_end,
        args.max_train_samples,
        args.max_val_samples,
        args.target,
    )
    if len(x_train) == 0 or len(x_val_loss) == 0 or len(x_eval) == 0:
        raise ValueError("Empty train or validation set. Check date ranges and stock pool.")

    print(
        f"train_samples={len(x_train)} "
        f"val_loss_samples={len(x_val_loss)} "
        f"eval_samples={len(x_eval)} "
        f"features={len(FEATURES)}"
    )
    model, history, device = train_model(args, x_train, y_train, x_val_loss, y_val_loss)
    meta["score"] = predict(model, x_eval, device, args.batch_size)
    meta = mark_eligibility(meta, args.min_trade_amount, args.min_price, args.max_price)
    meta.to_csv(args.output_dir / "validation_predictions.csv", index=False)

    eval_meta = meta[meta["is_eligible"]].copy()
    if eval_meta.empty:
        raise ValueError("No eligible validation rows. Relax liquidity or price filters.")

    ic, ic_metrics = daily_ic(eval_meta)
    ic.to_csv(args.output_dir / "daily_ic.csv", index=False)
    direction_acc = (np.sign(eval_meta["score"]) == np.sign(eval_meta["y_true"])).mean()
    bt, bt_metrics = backtest_topk(eval_meta, args.top_n, args.rebalance_k, args.transaction_cost_bps)
    bt.to_csv(args.output_dir / "backtest.csv", index=False)
    grid = evaluate_strategy_grid(eval_meta, args.transaction_cost_bps)
    if not grid.empty:
        grid.to_csv(args.output_dir / "strategy_grid.csv", index=False)
    bench = benchmark_returns(args.data_dir, "000300.SH", bt["trade_date"] if not bt.empty else [], args.horizon)
    if not bench.empty:
        bench.to_csv(args.output_dir / "benchmark_000300.csv", index=False)

    latest_x, latest_meta = build_latest_windows(data, args.lookback)
    latest_meta["score"] = predict(model, latest_x, device, args.batch_size)
    latest_meta = mark_eligibility(latest_meta, args.min_trade_amount, args.min_price, args.max_price)
    latest_date, latest = save_latest_recommendations(
        latest_meta[latest_meta["is_eligible"]].copy(), stock_pool, args.output_dir, args.top_n
    )
    plot_outputs(history, bt, bench, args.output_dir)
    eval_counts = eval_meta.groupby("trade_date")["ts_code"].nunique()

    metrics = {
        "args": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
        "model": args.model,
        "features": FEATURES,
        "target": args.target,
        "label_transform": args.label_transform,
        "backtest_return": "next_open_to_close_return",
        "transaction_cost_bps": args.transaction_cost_bps,
        "min_trade_amount": args.min_trade_amount,
        "min_price": args.min_price,
        "max_price": args.max_price,
        "train_samples": int(len(x_train)),
        "val_samples": int(len(eval_meta)),
        "val_loss_samples": int(len(x_val_loss)),
        "eval_samples": int(len(x_eval)),
        "eligible_eval_samples": int(len(eval_meta)),
        "eval_days": int(eval_meta["trade_date"].nunique()),
        "eval_min_stocks_per_day": int(eval_counts.min()),
        "eval_median_stocks_per_day": float(eval_counts.median()),
        "eval_max_stocks_per_day": int(eval_counts.max()),
        "final_train_loss": history[-1]["train_loss"],
        "final_val_loss": history[-1]["val_loss"],
        "direction_accuracy": float(direction_acc),
        **{k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in ic_metrics.items()},
        "backtest": {k: float(v) if isinstance(v, (np.floating, float)) else v for k, v in bt_metrics.items()},
        "latest_recommendation_date": int(latest_date),
    }
    with open(args.output_dir / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)

    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    print("latest recommendations:")
    print(latest[["trade_date", "ts_code", "name", "score"]].to_string(index=False))


if __name__ == "__main__":
    main()
