"""Training pipeline: TCN → LightGBM Stacking.

Architecture:
  1. Pretrain TCN on ALL industries mixed (data augmentation)
  2. Finetune per-industry TCN (optional)
  3. Extract TCN outputs → use as feature for LightGBM
  4. LightGBM trains on: [tcn_reg, delta_sentiment, news_count, news_heat, interactions]
  5. IsolationForest on news volume features

Loguru handles console output. WandbHandler pushes metrics to wandb dashboard.
"""

from __future__ import annotations

import pickle
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
import torch
import torch.nn as nn
from loguru import logger
from scipy import stats
from sklearn.metrics import r2_score
from torch.optim.adam import Adam
from torch.utils.data import DataLoader, TensorDataset

from trainer.config import get_config
from trainer.src.config import load_config
from trainer.src.config.signals import SignalsConfig
from trainer.src.datasets.signals import (
    WeeklySignalDataset,
    _get_market_cap_weight,
    build_lgbm_features,
    build_sequences,
    build_sub_category_sequences,
    compute_global_leader_sentiment,
    compute_market_beta,
    export_phase2_dataset,
)
from trainer.src.models.signals import TCN, TCNFanIn, export_tcn_fanin_to_onnx
from trainer.wandb_handler import WandbHandler, WandbRegistry

# ─── Model Training ─────────────────────────────────────────────────────────────


def train_tcn_pretrain(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    cfg: SignalsConfig,
    wb: WandbHandler,
    device: torch.device,
) -> TCN:
    """Step A: Pretrain TCN on ALL industries mixed."""
    tc = cfg.training
    sc = cfg.tcn
    model = TCN(
        input_size=6,
        hidden_size=sc.hidden_size,
        num_layers=sc.num_layers,
        dropout=sc.dropout,
    ).to(device)

    optimizer = Adam(model.parameters(), lr=tc.lr, weight_decay=1e-3)
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.BCELoss()

    dataset = TensorDataset(torch.FloatTensor(X), torch.FloatTensor(y_reg), torch.FloatTensor(y_cls))
    loader = DataLoader(dataset, batch_size=tc.batch_size, shuffle=True)

    for epoch in range(tc.epochs_pretrain):
        model.train()
        total_loss, total_reg, total_cls = 0.0, 0.0, 0.0
        for bx, by_reg, by_cls in loader:
            bx, by_reg, by_cls = bx.to(device), by_reg.to(device), by_cls.to(device)
            optimizer.zero_grad()
            pred_reg, pred_cls = model(bx)
            loss = reg_criterion(pred_reg, by_reg) + 0.01 * cls_criterion(pred_cls, by_cls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_reg += reg_criterion(pred_reg, by_reg).item()
            total_cls += cls_criterion(pred_cls, by_cls).item()

        n = len(loader)
        avg_loss = total_loss / n
        avg_reg = total_reg / n
        avg_cls = total_cls / n
        wb.log_epoch("pretrain", epoch + 1, avg_loss, {"reg_loss": avg_reg, "cls_loss": avg_cls})
        logger.info(
            f"  [Pretrain] epoch {epoch + 1}/{tc.epochs_pretrain} "
            f"loss={avg_loss:.4f} reg={avg_reg:.4f} cls={avg_cls:.4f}"
        )

    return model


def finetune_per_industry(
    sentiment_df: pl.DataFrame,
    industries: list[str],
    base_model: TCN,
    cfg: SignalsConfig,
    wb: WandbHandler,
    device: torch.device,
) -> TCN:
    """Step B: Finetune TCN per industry (freeze all but last temporal block)."""
    tc = cfg.training
    seq_len = cfg.tcn.sequence_length

    # Freeze all but last TCN block + heads
    num_layers = cfg.tcn.num_layers
    for i, block in enumerate(base_model.network):
        freeze = i < num_layers - 1
        for param in block.parameters():
            param.requires_grad = not freeze

    optimizer = Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=tc.lr * 0.5)
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.BCELoss()

    has_ohlcv = all(c in sentiment_df.columns for c in ["volume", "high", "low", "close", "open"])

    for ind in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == ind).sort("date")
        if len(ind_df) < seq_len + 2:
            continue
        vals = ind_df["sentiment_mean"].to_numpy()
        rets = ind_df["return"].to_numpy() if "return" in ind_df.columns else np.zeros_like(vals)

        if has_ohlcv:
            vol_arr = ind_df["volume"].to_numpy()
            high_arr = ind_df["high"].to_numpy()
            low_arr = ind_df["low"].to_numpy()
            close_arr = ind_df["close"].to_numpy()
        else:
            vol_arr = high_arr = low_arr = close_arr = np.zeros_like(vals)

        X_ind, y_reg_ind, y_cls_ind = [], [], []
        for i in range(len(vals) - seq_len - 1):
            # 6-channel window: sentiment_mean, sentiment_std, news_count,
            # avg_confidence, volume_ratio, intraday_vol
            ch0 = vals[i : i + seq_len]
            ch1 = (
                ind_df["sentiment_std"][i : i + seq_len].to_numpy()
                if "sentiment_std" in ind_df.columns
                else np.zeros(seq_len)
            )
            ch2 = ind_df["news_count"][i : i + seq_len].to_numpy()
            ch3 = (
                ind_df["avg_confidence"][i : i + seq_len].to_numpy()
                if "avg_confidence" in ind_df.columns
                else np.zeros(seq_len)
            )
            vol_ma5 = np.array([np.mean(vol_arr[max(0, j - 4) : j + 1]) for j in range(i, i + seq_len)])
            close_safe = np.where(close_arr[i : i + seq_len] != 0, close_arr[i : i + seq_len], 1.0)
            ch4 = np.where(vol_arr[i : i + seq_len] > 0, vol_arr[i : i + seq_len] / (vol_ma5 + 1e-9), 0.0)
            ch5 = (high_arr[i : i + seq_len] - low_arr[i : i + seq_len]) / close_safe

            X_ind.append(np.stack([ch0, ch1, ch2, ch3, ch4, ch5], axis=1))  # (seq_len, 6)
            # Continuous target (same as pretrain)
            target = np.clip(
                (vals[i + seq_len] - vals[i + seq_len - 1]) / (np.abs(vals[i + seq_len - 1]) + 1e-9),
                -1,
                1,
            )
            y_reg_ind.append(target)
            nr = rets[i + seq_len]
            y_cls_ind.append(1 if abs(nr) > tc.anomaly_threshold else 0)

        if len(X_ind) < 2:
            continue

        X_t = torch.FloatTensor(np.array(X_ind, dtype=np.float32)).reshape(-1, seq_len, 6).to(device)
        y_reg_t = torch.FloatTensor(np.array(y_reg_ind, dtype=np.float32)).reshape(-1, 1).to(device)
        y_cls_t = torch.FloatTensor(np.array(y_cls_ind, dtype=np.float32)).reshape(-1, 1).to(device)

        dataset = TensorDataset(X_t, y_reg_t, y_cls_t)
        loader = DataLoader(dataset, batch_size=min(32, len(X_ind)), shuffle=True)

        for epoch in range(tc.epochs_finetune):
            base_model.train()
            total_loss = 0.0
            for bx, by_reg, by_cls in loader:
                optimizer.zero_grad()
                pred_reg, pred_cls = base_model(bx)
                loss = reg_criterion(pred_reg, by_reg) + 0.01 * cls_criterion(pred_cls, by_cls)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            wb.log_epoch("finetune", epoch + 1, avg_loss, {"industry": ind})
        logger.info(f"  [Finetune] {ind} done ({tc.epochs_finetune} ep)")

    return base_model


def train_tcn_fanin(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray | None = None,
    cfg: SignalsConfig | None = None,
    wb: WandbHandler | None = None,
    device: torch.device = torch.device("cpu"),
) -> TCNFanIn:
    """Train fan-in TCN: (batch, seq_len, 47, 6) → (batch, 8)."""
    tc = cfg.training if cfg else None
    sc = cfg.tcn if cfg else None

    model = TCNFanIn(
        n_sub=47,
        n_meta=8,
        input_size=6,
        hidden_size=sc.hidden_size if sc else 64,
        num_layers=sc.num_layers if sc else 4,
        dropout=sc.dropout if sc else 0.2,
        spatial_dropout_p=0.3,
    ).to(device)

    epochs = tc.epochs_pretrain if tc else 50
    lr = tc.lr if tc else 1e-3
    batch_size = tc.batch_size if tc else 32

    optimizer = Adam(model.parameters(), lr=lr, weight_decay=5e-3)
    reg_criterion = nn.MSELoss()

    # Build classification labels if not provided
    if y_cls is None:
        y_cls = (np.abs(y_reg) > 0.05).astype(np.float32)
        if y_cls.ndim == 1:
            y_cls = y_cls.reshape(-1, 1)

    cls_criterion = nn.BCELoss()

    dataset = TensorDataset(
        torch.FloatTensor(X),
        torch.FloatTensor(y_reg),
        torch.FloatTensor(y_cls),
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        model.train()
        total_loss, total_reg = 0.0, 0.0
        for bx, by_reg, by_cls in loader:
            bx, by_reg, by_cls = bx.to(device), by_reg.to(device), by_cls.to(device)
            optimizer.zero_grad()
            pred_reg, pred_cls = model(bx)

            # pred_reg: (batch, 8), by_reg: (batch, 8) or (batch, 1)
            if by_reg.shape != pred_reg.shape:
                by_reg = by_reg.expand_as(pred_reg)

            loss = reg_criterion(pred_reg, by_reg)
            if pred_cls.shape == by_cls.shape:
                loss = loss + 0.01 * cls_criterion(pred_cls, by_cls)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_reg += reg_criterion(pred_reg, by_reg).item()

        n = len(loader)
        avg_loss = total_loss / n
        avg_reg = total_reg / n

        if wb is not None:
            wb.log_epoch("fanin", epoch + 1, avg_loss, {"reg_loss": avg_reg})

        if epoch % 10 == 0 or epoch == epochs - 1:
            logger.info(f"  [FanIn] epoch {epoch + 1}/{epochs} loss={avg_loss:.4f} reg={avg_reg:.4f}")

    return model


def train_lgbm_stacking(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray | None = None,
    cfg: SignalsConfig | None = None,
    wb: WandbHandler | None = None,
) -> Any:
    """Step C: Train LightGBM on stacking features.

    If dates is provided, splits by time (last 20% by date) instead of random.
    """
    import lightgbm as lgb

    if dates is not None:
        # Time-based split: sort by date, use last 20% as validation
        order = np.argsort(dates)
        X, y = X[order], y[order]
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]
    else:
        split = int(len(X) * 0.8)
        X_train, X_val = X[:split], X[split:]
        y_train, y_val = y[:split], y[split:]

    model = lgb.LGBMRegressor(
        num_leaves=cfg.lightgbm.num_leaves if cfg else 7,
        learning_rate=cfg.lightgbm.learning_rate if cfg else 0.02,
        n_estimators=cfg.lightgbm.n_estimators if cfg else 500,
        min_child_samples=10,
        lambda_l1=0.5,
        lambda_l2=0.5,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            lgb.early_stopping(stopping_rounds=10),
            lgb.log_evaluation(period=50),
        ],
    )

    train_pred = model.predict(X_train)
    val_pred = model.predict(X_val)

    train_score = r2_score(y_train, train_pred)
    val_score = r2_score(y_val, val_pred)

    if wb is not None:
        wb.log({"lgbm_train_r2": train_score, "lgbm_val_r2": val_score})
    logger.info(f"  [LightGBM] train_r2={train_score:.4f} val_r2={val_score:.4f}")

    return model


# ─── Evaluation Metrics ───────────────────────────────────────────────────────


def compute_industry_ic(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    industries: np.ndarray,
    stage: str,
    wb: WandbHandler | None = None,
) -> dict[str, float]:
    """Compute Pearson IC per industry. Returns dict of industry → IC value."""
    ic_dict: dict[str, float] = {}
    unique_industries = np.unique(industries)
    for ind in unique_industries:
        mask = industries == ind
        if mask.sum() < 3:
            ic_dict[ind] = float("nan")
            continue
        ic, p = stats.pearsonr(y_true[mask], y_pred[mask])
        ic_dict[ind] = ic

    logger.info(f"  [{stage}] Industry IC:")
    for ind, ic in ic_dict.items():
        logger.info(f"    {ind}: {ic:.4f}")
    if wb is not None:
        ic_summary = {f"ic/{ind}": ic for ind, ic in ic_dict.items()}
        wb.log_summary(ic_summary)
    return ic_dict


def analyze_residuals(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    dates: np.ndarray | None = None,
    stage: str = "",
    wb: WandbHandler | None = None,
) -> dict[str, Any]:
    """Analyze residual distribution: normality (Shapiro-Wilk), skewness, kurtosis, time-based anomalies."""
    residuals = y_true.astype(float) - y_pred.astype(float)

    # Basic stats
    skew = float(stats.skew(residuals))
    kurt = float(stats.kurtosis(residuals))
    shapiro_stat, shapiro_p = stats.shapiro(residuals[: min(len(residuals), 5000)])

    result = {
        f"{stage}/residual_skew": skew,
        f"{stage}/residual_kurt": kurt,
        f"{stage}/shapiro_stat": float(shapiro_stat),
        f"{stage}/shapiro_p": float(shapiro_p),
        f"{stage}/residual_mean": float(np.mean(residuals)),
        f"{stage}/residual_std": float(np.std(residuals)),
    }

    # Time-based anomalies: split residuals into early/late halves
    if dates is not None:
        order = np.argsort(dates)
        mid = len(residuals) // 2
        early = residuals[order[:mid]]
        late = residuals[order[mid:]]
        result[f"{stage}/residual_early_mean"] = float(np.mean(early))
        result[f"{stage}/residual_late_mean"] = float(np.mean(late))
        result[f"{stage}/residual_early_std"] = float(np.std(early))
        result[f"{stage}/residual_late_std"] = float(np.std(late))

    if wb is not None:
        wb.log_summary(result)

    logger.info(
        f"  [{stage}] Residuals — skew={skew:.3f} kurt={kurt:.3f} shapiro_p={shapiro_p:.4f} (p<0.05→non-normal)"
    )
    return result


def _rolling_std_1d(values: np.ndarray, window: int) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = float(np.std(values[start : i + 1])) if i > start else 0.0
    return out


def _prediction_stability_1d(values: np.ndarray, window: int = 5) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        p = values[start : i + 1]
        dir_consistency = abs(np.sign(p).sum()) / float(len(p))
        dispersion = float(np.std(p) / (np.mean(np.abs(p)) + 1e-9))
        out[i] = float(np.clip(dir_consistency - 0.5 * dispersion, 0.0, 1.0))
    return out


def _rolling_percentile_1d(values: np.ndarray, window: int = 252) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        hist = values[start : i + 1]
        out[i] = float(np.mean(hist <= values[i])) if len(hist) > 1 else 0.5
    return out


# ─── Main Pipeline ───────────────────────────────────────────────────────────


def run_training(force: bool = False) -> dict[str, str]:
    """Primary training entrypoint.

    The maintained path is now the fan-in training pipeline defined in
    docs/hopper.md. The legacy per-industry pipeline is intentionally bypassed.
    """
    del force  # legacy flag kept for CLI compatibility

    try:
        signals_cfg = get_config().signals
    except AssertionError:
        signals_cfg = load_config().signals
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wb = (
        WandbRegistry.get("signals")
        if "signals" in WandbRegistry._handlers
        else WandbHandler(
            run_name=f"signals-{datetime.now():%m%d-%H%M}",
            tags=["signals", "TCN", "LightGBM", "IsolationForest"],
        )
    )

    sentiment_path = Path("data/industry_sentiment.parquet")
    if not sentiment_path.exists():
        raise FileNotFoundError("Missing data/industry_sentiment.parquet for fan-in training.")

    mapping_path = Path("data/meta_sector_mapping.json")
    if not mapping_path.exists():
        raise FileNotFoundError("Missing data/meta_sector_mapping.json for fan-in training.")

    import json

    sentiment_df = pl.read_parquet(sentiment_path)
    with open(mapping_path, encoding="utf-8") as f:
        meta_sector_map = json.load(f)

    logger.info(f"[Train] Device: {device}")
    logger.info(f"[Train] Fan-in sentiment source: {sentiment_path}")
    return run_training_fanin(sentiment_df, meta_sector_map, cfg=signals_cfg, wb=wb, device=device)


def run_training_fanin(
    sentiment_df: pl.DataFrame,
    meta_sector_map: dict,
    cfg: SignalsConfig | None = None,
    wb: WandbHandler | None = None,
    device: torch.device | None = None,
) -> dict[str, str]:
    """Fan-in TCN pipeline: TCNFanIn → 8×LightGBM → IForest → SHAP."""
    if cfg is None:
        try:
            cfg = get_config().signals
        except AssertionError:
            cfg = load_config().signals
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if wb is None:
        wb = (
            WandbRegistry.get("signals")
            if "signals" in WandbRegistry._handlers
            else WandbHandler(
                run_name=f"fanin-{datetime.now():%m%d-%H%M}",
                tags=["signals", "TCNFanIn", "LightGBM", "SHAP"],
            )
        )

    logger.info(f"[FanIn Train] Device: {device}")

    # Step A: Build sub-category sequences
    logger.info("[Step A] Build sub-category sequences (47×6 fan-in)...")
    X, y, dates, sub_industries = build_sub_category_sequences(sentiment_df, meta_sector_map, lookback_days=5)
    logger.info(f"  Data: X={X.shape}, y={y.shape}, {len(sub_industries)} sub-industries")

    if len(X) == 0:
        logger.error("  No training data! Check sentiment_df and meta_sector_map.")
        return {}

    # Time-based split (80/20)
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]
    train_dates = np.array(dates[:split])
    test_dates = np.array(dates[split:])

    # Step B: Train TCNFanIn
    logger.info("[Step B] Train TCNFanIn...")
    tcn_model = train_tcn_fanin(X_train, y_train, cfg=cfg, wb=wb, device=device)

    # Step C: Train 8 independent LightGBMs
    logger.info("[Step C] Train 8 independent LightGBMs...")
    import lightgbm as lgb

    meta_sectors = list(meta_sector_map.get("meta_sectors", {}).keys())
    lgbm_models = {}

    checkpoint_root = cfg.training.output_checkpoint or Path("trainer/checkpoints/signals")
    checkpoint_dir = checkpoint_root / f"signals-{datetime.now():%m%d-%H%M}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    lgbm_dir = checkpoint_dir / "lgbm"
    lgbm_dir.mkdir(exist_ok=True)

    sector_col = "sub_category" if "sub_category" in sentiment_df.columns else "industry"
    sent_col = "sentiment_mean" if "sentiment_mean" in sentiment_df.columns else "sentiment_weighted"
    all_dates = sorted(sentiment_df["date"].unique().to_list())
    date_to_idx = {d: idx for idx, d in enumerate(all_dates)}
    global_leader_map = meta_sector_map.get("global_leader_map", {})

    meta_sentiment: dict[str, np.ndarray] = {}
    meta_news_count: dict[str, np.ndarray] = {}
    for ms in meta_sectors:
        arr_sent = np.zeros(len(all_dates), dtype=np.float32)
        arr_news = np.zeros(len(all_dates), dtype=np.float32)
        ms_subs = meta_sector_map.get("meta_sectors", {}).get(ms, {}).get("sub_categories", [])
        ms_df = sentiment_df.filter(pl.col(sector_col).is_in(ms_subs)).sort("date")
        for d in all_dates:
            day_df = ms_df.filter(pl.col("date") == d)
            if not day_df.is_empty():
                weights = np.array(
                    [_get_market_cap_weight(sub, meta_sector_map) for sub in day_df[sector_col].to_list()],
                    dtype=np.float32,
                )
                sentiments = np.array(day_df[sent_col].to_list(), dtype=np.float32)
                arr_sent[date_to_idx[d]] = (
                    float(np.average(sentiments, weights=weights)) if weights.sum() > 0 else float(sentiments.mean())
                )
                arr_news[date_to_idx[d]] = float(day_df["news_count"].sum()) if "news_count" in day_df.columns else 0.0
        meta_sentiment[ms] = arr_sent
        meta_news_count[ms] = arr_news

    meta_global_leader: dict[str, np.ndarray] = {}
    global_leader_df = compute_global_leader_sentiment(sentiment_df, meta_sector_map)
    if not global_leader_df.is_empty():
        for ms in meta_sectors:
            col = f"global_leader_{ms}"
            if col in global_leader_df.columns:
                meta_global_leader[ms] = global_leader_df.sort("date")[col].to_numpy().astype(np.float32)
            else:
                meta_global_leader[ms] = np.zeros(len(all_dates), dtype=np.float32)
    else:
        for ms in meta_sectors:
            leaders = global_leader_map.get(ms, [])
            if leaders:
                leader_series = []
                for leader in leaders:
                    if leader in meta_sectors:
                        leader_series.append(meta_sentiment[leader])
                    else:
                        for other_ms, info in meta_sector_map.get("meta_sectors", {}).items():
                            if leader in info.get("sub_categories", []):
                                leader_series.append(meta_sentiment[other_ms])
                                break
                meta_global_leader[ms] = (
                    np.mean(leader_series, axis=0) if leader_series else np.zeros(len(all_dates), dtype=np.float32)
                )
            else:
                meta_global_leader[ms] = np.zeros(len(all_dates), dtype=np.float32)

    # Extract TCN features for LightGBM input
    with torch.no_grad():
        tcn_reg_train, _ = tcn_model(torch.FloatTensor(X_train).to(device))
        tcn_reg_train = tcn_reg_train.cpu().numpy()
        tcn_reg_test, _ = tcn_model(torch.FloatTensor(X_test).to(device))
        tcn_reg_test = tcn_reg_test.cpu().numpy()

    # Step D: IsolationForest first, so LightGBM consumes the same `news_heat`
    logger.info("[Step D] Train IsolationForest...")
    from sklearn.ensemble import IsolationForest

    iforest = IsolationForest(contamination=0.05, n_estimators=100, random_state=42)
    iforest_train_X = X_train.reshape(len(X_train), -1)
    iforest_test_X = X_test.reshape(len(X_test), -1)
    iforest.fit(iforest_train_X)

    iforest_train_scores = -iforest.score_samples(iforest_train_X).astype(np.float32)
    iforest_test_scores = -iforest.score_samples(iforest_test_X).astype(np.float32)
    news_heat_train_all = _rolling_percentile_1d(iforest_train_scores, 252)
    news_heat_test_all = _rolling_percentile_1d(
        np.concatenate([iforest_train_scores, iforest_test_scores], axis=0),
        252,
    )[-len(iforest_test_scores) :]

    lgbm_test_features: dict[str, np.ndarray] = {}

    for m_idx, ms in enumerate(meta_sectors):
        n_train = len(X_train)
        n_test = len(X_test)
        ms_news = meta_news_count[ms]
        ms_sent = meta_sentiment[ms]
        gls = meta_global_leader[ms]

        train_idx = np.array([date_to_idx[d] for d in train_dates], dtype=int)
        test_idx = np.array([date_to_idx[d] for d in test_dates], dtype=int)

        tcn_reg_delta_train = np.zeros(n_train, dtype=np.float32)
        tcn_reg_delta_train[1:] = tcn_reg_train[1:, m_idx] - tcn_reg_train[:-1, m_idx]
        tcn_reg_delta_test = np.zeros(n_test, dtype=np.float32)
        tcn_reg_delta_test[1:] = tcn_reg_test[1:, m_idx] - tcn_reg_test[:-1, m_idx]

        delta_sentiment_1w = np.zeros(n_train, dtype=np.float32)
        delta_sentiment_1w[1:] = y_train[1:, m_idx] - y_train[:-1, m_idx]
        delta_sentiment_2w = np.zeros(n_train, dtype=np.float32)
        delta_sentiment_2w[2:] = y_train[2:, m_idx] - y_train[:-2, m_idx]

        delta_sentiment_1w_test = np.zeros(n_test, dtype=np.float32)
        delta_sentiment_1w_test[1:] = y_test[1:, m_idx] - y_test[:-1, m_idx]
        delta_sentiment_2w_test = np.zeros(n_test, dtype=np.float32)
        delta_sentiment_2w_test[2:] = y_test[2:, m_idx] - y_test[:-2, m_idx]

        news_count_feature = ms_news[train_idx]
        news_count_test = ms_news[test_idx]
        news_count_std_5d = _rolling_std_1d(ms_news, 5)[train_idx]
        news_count_std_test = _rolling_std_1d(ms_news, 5)[test_idx]
        sentiment_volatility_5d = _rolling_std_1d(ms_sent, 5)[train_idx]
        sent_vol_test = _rolling_std_1d(ms_sent, 5)[test_idx]

        news_heat = news_heat_train_all
        news_heat_test = news_heat_test_all

        tcn_prediction_stability = _prediction_stability_1d(tcn_reg_train[:, m_idx], 5)
        tcn_prediction_stability_test = _prediction_stability_1d(tcn_reg_test[:, m_idx], 5)

        tcn_heat_interaction = tcn_reg_train[:, m_idx] * news_heat
        tcn_heat_test = tcn_reg_test[:, m_idx] * news_heat_test

        rolling_mean_news = np.zeros(len(ms_news), dtype=np.float32)
        for i in range(len(ms_news)):
            start = max(0, i - 4)
            rolling_mean_news[i] = float(np.mean(ms_news[start : i + 1]))
        volume_ratio = np.clip(news_count_feature / (rolling_mean_news[train_idx] + 1e-9), 0.5, 3.0)
        volume_ratio_test = np.clip(news_count_test / (rolling_mean_news[test_idx] + 1e-9), 0.5, 3.0)

        intraday_vol = np.clip(sentiment_volatility_5d * 2, 0.005, 0.1)
        intraday_vol_test = np.clip(sent_vol_test * 2, 0.005, 0.1)
        avg_price = np.clip((ms_sent[train_idx] + 1.0) * 50.0, 50.0, 150.0).astype(np.float32)
        avg_price_test = np.clip((ms_sent[test_idx] + 1.0) * 50.0, 50.0, 150.0).astype(np.float32)

        global_leader_sentiment = gls[train_idx]
        global_leader_test = gls[test_idx]

        if np.std(gls) > 1e-9:
            cov = np.cov(ms_sent, gls)[0, 1]
            var = np.var(gls)
            beta_value = float(cov / (var + 1e-9))
        else:
            beta_value = 1.0
        market_beta = np.full(n_train, np.clip(beta_value, 0.3, 2.0), dtype=np.float32)
        market_beta_test = np.full(n_test, np.clip(beta_value, 0.3, 2.0), dtype=np.float32)

        entropy_full = np.zeros(len(ms_sent), dtype=np.float32)
        for i in range(len(ms_sent)):
            start = max(0, i - 4)
            window = ms_sent[start : i + 1]
            if len(window) > 1 and np.std(window) > 1e-9:
                p = np.abs(window) / (np.sum(np.abs(window)) + 1e-9)
                entropy_full[i] = float(-np.sum(p * np.log(p + 1e-9)))
        sentiment_entropy = np.clip(entropy_full[train_idx] / 2.0, 0.0, 1.0)
        sentiment_entropy_test = np.clip(entropy_full[test_idx] / 2.0, 0.0, 1.0)

        X_sector_train = np.column_stack(
            [
                delta_sentiment_1w,
                delta_sentiment_2w,
                news_count_feature,
                news_heat,
                tcn_reg_train[:, m_idx],
                tcn_reg_delta_train,
                tcn_prediction_stability,
                news_count_std_5d,
                sentiment_volatility_5d,
                tcn_heat_interaction,
                volume_ratio,
                intraday_vol,
                avg_price,
                global_leader_sentiment,
                market_beta,
                sentiment_entropy,
            ]
        ).astype(np.float32)
        X_sector_test = np.column_stack(
            [
                delta_sentiment_1w_test,
                delta_sentiment_2w_test,
                news_count_test,
                news_heat_test,
                tcn_reg_test[:, m_idx],
                tcn_reg_delta_test,
                tcn_prediction_stability_test,
                news_count_std_test,
                sent_vol_test,
                tcn_heat_test,
                volume_ratio_test,
                intraday_vol_test,
                avg_price_test,
                global_leader_test,
                market_beta_test,
                sentiment_entropy_test,
            ]
        ).astype(np.float32)
        y_sector = y_train[:, m_idx]

        model_lgb = lgb.LGBMRegressor(
            num_leaves=cfg.lightgbm.num_leaves,
            learning_rate=cfg.lightgbm.learning_rate,
            n_estimators=cfg.lightgbm.n_estimators,
            min_child_samples=10,
            lambda_l1=0.5,
            lambda_l2=0.5,
            verbose=-1,
        )
        model_lgb.fit(X_sector_train, y_sector, eval_set=[(X_sector_test, y_test[:, m_idx])])
        lgbm_models[ms] = model_lgb
        lgbm_test_features[ms] = X_sector_test

        model_lgb.booster_.save_model(str(lgbm_dir / f"{ms}.txt"))
        logger.info(f"  [LightGBM] {ms} trained")

    iforest_path = checkpoint_dir / "iforest_model.pkl"
    with open(iforest_path, "wb") as f:
        pickle.dump(iforest, f)

    # Step E: Save TCN
    tcn_path = checkpoint_dir / "tcn_fanin.pt"
    torch.save(tcn_model.state_dict(), tcn_path)
    export_tcn_fanin_to_onnx(tcn_model, checkpoint_dir / "tcn.onnx")
    logger.info(f"  [Save] tcn_fanin.pt + tcn.onnx → {checkpoint_dir}")

    export_phase2_dataset(
        sentiment_df=sentiment_df,
        price_df=pl.DataFrame(),
        index_df=pl.DataFrame(),
        meta_sector_map=meta_sector_map,
        tcn_model=tcn_model,
        lgbm_models=lgbm_models,
        iforest_model=iforest,
        device=device,
        output_path=Path("data/agent_features.parquet"),
    )

    # Step F: SHAP Analysis
    logger.info("[Step F] SHAP Analysis...")
    try:
        from trainer.src.utils.signals_xai import SHAPAnalyzer

        for ms in meta_sectors:
            shap_analyzer = SHAPAnalyzer(lgbm_models[ms], lgbm_test_features[ms])
            shap_analyzer.compute_shap_values()

            shap_dir = checkpoint_dir / "shap"
            shap_analyzer.generate_summary_plot(shap_dir / f"shap_summary_{ms}.png")
            shap_analyzer.export_shap_values(test_dates, shap_dir)
            shap_analyzer.check_tcn_dominance()
            logger.info(f"  [SHAP] {ms} analysis complete")
    except ImportError:
        logger.warning("  [SHAP] shap not installed, skipping analysis")
    logger.success("[Done] Fan-in pipeline complete.")

    return {
        "tcn_path": str(tcn_path),
        "lgbm_dir": str(lgbm_dir),
        "iforest_path": str(iforest_path),
    }


def _export_all_onnx(
    tcn_model: TCN,
    lgbm_model: Any,
    iforest: Any,
    iforest_X: np.ndarray,
    checkpoint_dir: Path,
    seq_len: int,
    X_lgbm: np.ndarray,
    device: torch.device,
) -> dict[str, Path | None]:
    """Export TCN, LightGBM, and IsolationForest to ONNX format."""
    import onnxmltools
    from onnxmltools.convert.common.data_types import FloatTensorType

    from trainer.src.models.signals import export_tcn_to_onnx

    results: dict[str, Path | None] = {}

    # ── TCN → ONNX ───────────────────────────────────────────────────────────
    tcn_onnx_path = checkpoint_dir / "tcn.onnx"
    try:
        # Move model to CPU for export to avoid device mismatch
        tcn_model_cpu = tcn_model.cpu()
        export_tcn_to_onnx(
            tcn_model_cpu,
            tcn_onnx_path,
            seq_len=seq_len,
            input_size=6,
        )
        tcn_model.to(device)  # move back to original device
        results["tcn_onnx"] = tcn_onnx_path
        logger.info(f"  [ONNX] TCN → {tcn_onnx_path}")
    except Exception as exc:
        logger.warning(f"  [ONNX] TCN export failed: {exc}")
        results["tcn_onnx"] = None

    # ── LightGBM → ONNX ───────────────────────────────────────────────────────
    lgbm_onnx_path = checkpoint_dir / "lgbm_stacking.onnx"
    try:
        initial_type = [("float_input", FloatTensorType([None, X_lgbm.shape[1]]))]
        lgbm_onnx = onnxmltools.convert_lightgbm(lgbm_model, initial_types=initial_type, target_opset=15)
        with open(lgbm_onnx_path, "wb") as f:
            f.write(lgbm_onnx.SerializeToString())
        results["lgbm_onnx"] = lgbm_onnx_path
        logger.info(f"  [ONNX] LightGBM → {lgbm_onnx_path}")
    except Exception as exc:
        logger.warning(f"  [ONNX] LightGBM export failed: {exc}")
        results["lgbm_onnx"] = None

    # ── IsolationForest → ONNX ────────────────────────────────────────────────
    iforest_onnx_path = checkpoint_dir / "iforest.onnx"
    try:
        example_X = iforest_X[:1].astype(np.float32)
        iforest_onnx = onnxmltools.convert_sklearn(
            iforest,
            initial_types=[("input", FloatTensorType([None, example_X.shape[1]]))],
            target_opset=3,
        )
        with open(iforest_onnx_path, "wb") as f:
            f.write(iforest_onnx.SerializeToString())  # type: ignore
        results["iforest_onnx"] = iforest_onnx_path
        logger.info(f"  [ONNX] IsolationForest → {iforest_onnx_path}")
    except Exception as exc:
        logger.warning(f"  [ONNX] IsolationForest export failed ({type(exc).__name__}): {exc}. Falling back to pickle.")
        results["iforest_onnx"] = None

    return results
