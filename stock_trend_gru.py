import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


FEATURES = [
    "ret_1",
    "intraday_ret",
    "high_low_spread",
    "close_vwap_gap",
    "log_vol",
    "log_amount",
    "turnover_proxy",
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
    for col in ["open", "high", "low", "close", "pre_close", "vol", "amount", "vwap"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df["ret_1"] = df["pct_chg"].astype(float) / 100.0
    df["intraday_ret"] = df["close"] / df["open"] - 1.0
    df["high_low_spread"] = df["high"] / df["low"] - 1.0
    df["close_vwap_gap"] = df["close"] / df["vwap"] - 1.0
    df["log_vol"] = np.log1p(df["vol"])
    df["log_amount"] = np.log1p(df["amount"])
    df["turnover_proxy"] = df["amount"] / df["vol"].replace(0, np.nan)
    df["turnover_proxy"] = np.log1p(df["turnover_proxy"])

    future_close = df.groupby("ts_code")["close"].shift(-horizon)
    df["label"] = future_close / df["close"] - 1.0

    needed = FEATURES + ["close"]
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=needed)
    return df


def deterministic_cap(indices, max_samples):
    if max_samples is None or max_samples <= 0 or len(indices) <= max_samples:
        return indices
    positions = np.linspace(0, len(indices) - 1, max_samples).astype(int)
    return indices[positions]


def build_windows(df, lookback, train_end, val_start, val_end, max_train, max_val):
    xs = []
    ys = []
    dates = []
    codes = []
    closes = []

    for code, group in df.groupby("ts_code", sort=False):
        group = group.sort_values("trade_date")
        feat = group[FEATURES].to_numpy(dtype=np.float32)
        label = group["label"].to_numpy(dtype=np.float32)
        date = group["trade_date"].to_numpy(dtype=np.int64)
        close = group["close"].to_numpy(dtype=np.float32)

        for i in range(lookback - 1, len(group)):
            if np.isnan(label[i]):
                continue
            window = feat[i - lookback + 1 : i + 1]
            mean = window.mean(axis=0, keepdims=True)
            std = window.std(axis=0, keepdims=True) + 1e-6
            xs.append((window - mean) / std)
            ys.append(label[i])
            dates.append(date[i])
            codes.append(code)
            closes.append(close[i])

    x = np.stack(xs).astype(np.float32)
    y = np.asarray(ys, dtype=np.float32)
    dates = np.asarray(dates, dtype=np.int64)
    closes = np.asarray(closes, dtype=np.float32)
    codes = np.asarray(codes)

    train_idx = np.where(dates <= train_end)[0]
    val_idx = np.where((dates >= val_start) & (dates <= val_end))[0]
    train_idx = deterministic_cap(train_idx, max_train)
    val_idx = deterministic_cap(val_idx, max_val)

    meta = pd.DataFrame(
        {
            "trade_date": dates[val_idx],
            "ts_code": codes[val_idx],
            "close": closes[val_idx],
            "y_true": y[val_idx],
        }
    )
    return x[train_idx], y[train_idx], x[val_idx], y[val_idx], meta


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
        rows.append({"trade_date": int(last["trade_date"]), "ts_code": code, "close": float(last["close"])})
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


def backtest_topk(pred_df, top_n, rebalance_k):
    portfolio = []
    rows = []
    for date, day in pred_df.sort_values("trade_date").groupby("trade_date", sort=True):
        day = day.dropna(subset=["score", "y_true"]).sort_values("score", ascending=False)
        available = list(day["ts_code"])
        if not available:
            continue

        if not portfolio:
            portfolio = available[:top_n]
        else:
            score_map = day.set_index("ts_code")["score"].to_dict()
            held_with_scores = [(code, score_map.get(code, -np.inf)) for code in portfolio]
            sell = {
                code
                for code, _ in sorted(held_with_scores, key=lambda item: item[1])[:rebalance_k]
            }
            portfolio = [code for code in portfolio if code not in sell and code in set(available)]
            for code in available:
                if len(portfolio) >= top_n:
                    break
                if code not in portfolio:
                    portfolio.append(code)

        held = day[day["ts_code"].isin(portfolio)]
        if held.empty:
            continue
        ret = held["y_true"].mean()
        rows.append(
            {
                "trade_date": date,
                "daily_return": ret,
                "nav": np.nan,
                "holdings": ",".join(portfolio),
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
        "days": int(len(bt)),
    }
    return bt, metrics


def benchmark_returns(data_dir, index_code, dates):
    path = data_dir / "market" / f"{index_code}.csv"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["trade_date"] = df["trade_date"].astype(int)
    df = df.sort_values("trade_date")
    df["daily_return"] = df["close"].shift(-1) / df["close"] - 1.0
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
    x_train, y_train, x_val, y_val, meta = build_windows(
        data,
        args.lookback,
        args.train_end,
        args.val_start,
        args.val_end,
        args.max_train_samples,
        args.max_val_samples,
    )
    if len(x_train) == 0 or len(x_val) == 0:
        raise ValueError("Empty train or validation set. Check date ranges and stock pool.")

    print(f"train_samples={len(x_train)} val_samples={len(x_val)} features={len(FEATURES)}")
    model, history, device = train_model(args, x_train, y_train, x_val, y_val)
    meta["score"] = predict(model, x_val, device, args.batch_size)
    meta.to_csv(args.output_dir / "validation_predictions.csv", index=False)

    ic, ic_metrics = daily_ic(meta)
    ic.to_csv(args.output_dir / "daily_ic.csv", index=False)
    direction_acc = (np.sign(meta["score"]) == np.sign(meta["y_true"])).mean()
    bt, bt_metrics = backtest_topk(meta, args.top_n, args.rebalance_k)
    bt.to_csv(args.output_dir / "backtest.csv", index=False)
    bench = benchmark_returns(args.data_dir, "000300.SH", bt["trade_date"] if not bt.empty else [])
    if not bench.empty:
        bench.to_csv(args.output_dir / "benchmark_000300.csv", index=False)

    latest_x, latest_meta = build_latest_windows(data, args.lookback)
    latest_meta["score"] = predict(model, latest_x, device, args.batch_size)
    latest_date, latest = save_latest_recommendations(latest_meta, stock_pool, args.output_dir, args.top_n)
    plot_outputs(history, bt, bench, args.output_dir)

    metrics = {
        "args": vars(args) | {"data_dir": str(args.data_dir), "output_dir": str(args.output_dir)},
        "model": args.model,
        "features": FEATURES,
        "train_samples": int(len(x_train)),
        "val_samples": int(len(x_val)),
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
