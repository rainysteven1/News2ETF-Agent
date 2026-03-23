"""Training pipeline: LSTM+Attention → LightGBM Stacking.

Architecture:
  1. Pretrain LSTM+Attention on ALL industries mixed (data augmentation)
  2. Finetune per-industry LSTM+Attention (optional)
  3. Extract LSTM hidden states → use as feature for LightGBM
  4. LightGBM trains on: [lstm_hidden, raw_sentiment, news_count] → next-week return

Loguru handles console output. WandbHandler pushes metrics to wandb dashboard.
Both run in parallel — they are complementary.
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
from torch.utils.data import DataLoader, TensorDataset

from trainer.config import TrainerConfig, load_config
from trainer.signals.models import LSTMWithAttention
from trainer.wandb_handler import WandbHandler

# ─── Data Preparation ───────────────────────────────────────────────────────────


def build_sequences(
    sentiment_df: pl.DataFrame,
    industries: list[str],
    seq_len: int,
    anomaly_threshold: float = 0.03,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build LSTM training data from sentiment time series.

    Returns:
        X:     (N, seq_len, 1) — sentiment_mean sequences
        y_reg: (N, 1) — next-week direction: 1 / 0 / -1
        y_cls: (N, 1) — 1 if |return| > threshold else 0
    """
    X_list, y_reg_list, y_cls_list = [], [], []

    for ind in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == ind).sort("date")
        if len(ind_df) < seq_len + 2:
            continue
        vals = ind_df["sentiment_mean"].to_numpy()
        rets = ind_df["return"].to_numpy() if "return" in ind_df.columns else np.zeros_like(vals)

        for i in range(len(vals) - seq_len - 1):
            X_list.append(vals[i : i + seq_len])
            next_ret = rets[i + seq_len]
            y_reg_list.append(1 if next_ret > 0 else (-1 if next_ret < 0 else 0))
            y_cls_list.append(1 if abs(next_ret) > anomaly_threshold else 0)

    X = np.array(X_list, dtype=np.float32).reshape(-1, seq_len, 1)
    y_reg = np.array(y_reg_list, dtype=np.float32).reshape(-1, 1)
    y_cls = np.array(y_cls_list, dtype=np.float32).reshape(-1, 1)
    return X, y_reg, y_cls


def build_lgbm_features(
    sentiment_df: pl.DataFrame,
    industries: list[str],
    seq_len: int,
    lstm_model: LSTMWithAttention,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray]:
    """Build LightGBM feature matrix from LSTM hidden states + raw signals.

    Features:
        [delta_sentiment_1w, delta_sentiment_2w, news_count, news_heat,
         lstm_reg_output, lstm_cls_output]
    """
    feat_rows, label_rows = [], []

    for ind in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == ind).sort("date")
        if len(ind_df) < seq_len + 2:
            continue
        vals = ind_df["sentiment_mean"].to_numpy()
        nc = ind_df["news_count"].to_numpy()
        nh = ind_df["news_heat"].to_numpy() if "news_heat" in ind_df.columns else np.zeros_like(nc)

        for i in range(seq_len, len(vals) - 1):
            delta1 = vals[i] - vals[i - 1]
            delta2 = vals[i] - vals[i - 2]
            news_count = nc[i]
            news_heat = nh[i]
            next_dir = 1 if vals[i + 1] > vals[i] else (-1 if vals[i + 1] < vals[i] else 0)

            x_t = torch.FloatTensor(vals[i - seq_len : i]).reshape(1, seq_len, 1).to(device)
            with torch.no_grad():
                reg_out, cls_out = lstm_model(x_t)
            lstm_reg = reg_out.item()
            lstm_cls = cls_out.item()

            feat_rows.append([delta1, delta2, news_count, news_heat, lstm_reg, lstm_cls])
            label_rows.append(next_dir)

    return np.array(feat_rows, dtype=np.float32), np.array(label_rows, dtype=np.int32)


# ─── Model Training ─────────────────────────────────────────────────────────────


def train_lstm_attention_pretrain(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    cfg: TrainerConfig,
    wb: WandbHandler,
    device: torch.device,
) -> LSTMWithAttention:
    """Step A: Pretrain LSTM+Attention on ALL industries mixed."""
    tc = cfg.training
    sc = cfg.signals
    model = LSTMWithAttention(
        input_size=1,
        hidden_size=sc.hidden_size,
        num_layers=sc.num_layers,
        dropout=sc.dropout,
        num_heads=tc.num_heads,
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=tc.lr)
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
            loss = reg_criterion(pred_reg, by_reg) + 0.3 * cls_criterion(pred_cls, by_cls)
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
    base_model: LSTMWithAttention,
    cfg: TrainerConfig,
    wb: WandbHandler,
    device: torch.device,
) -> LSTMWithAttention:
    """Step B: Finetune LSTM+Attention per industry (freeze all but last LSTM layer)."""
    tc = cfg.training
    # Freeze: only unfreeze last LSTM layer + attention
    for name, param in base_model.named_parameters():
        is_lstm_ih = "lstm" in name and "weight_ih" in name
        is_fc = "reg_head" in name or "cls_head" in name
        param.requires_grad = not is_lstm_ih and not is_fc

    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, base_model.parameters()), lr=tc.lr * 0.5)
    reg_criterion = nn.MSELoss()
    cls_criterion = nn.BCELoss()

    for ind in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == ind).sort("date")
        if len(ind_df) < tc.batch_size + 2:
            continue
        vals = ind_df["sentiment_mean"].to_numpy()
        rets = ind_df["return"].to_numpy() if "return" in ind_df.columns else np.zeros_like(vals)

        X_ind, y_reg_ind, y_cls_ind = [], [], []
        for i in range(len(vals) - tc.batch_size - 1):
            X_ind.append(vals[i : i + tc.batch_size])
            nr = rets[i + tc.batch_size]
            y_reg_ind.append(1 if nr > 0 else (-1 if nr < 0 else 0))
            y_cls_ind.append(1 if abs(nr) > tc.anomaly_threshold else 0)

        if len(X_ind) < 2:
            continue

        X_t = torch.FloatTensor(np.array(X_ind, dtype=np.float32)).reshape(-1, tc.batch_size, 1).to(device)
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
                loss = reg_criterion(pred_reg, by_reg) + 0.3 * cls_criterion(pred_cls, by_cls)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            avg_loss = total_loss / len(loader)
            wb.log_epoch("finetune", epoch + 1, avg_loss, {"industry": ind})
        logger.info(f"  [Finetune] {ind} done ({tc.epochs_finetune} ep)")

    return base_model


def train_lgbm_stacking(
    X: np.ndarray,
    y: np.ndarray,
    cfg: TrainerConfig,
    wb: WandbHandler,
) -> Any:
    """Step C: Train LightGBM on stacking features."""
    import lightgbm as lgb

    split = int(len(X) * 0.8)
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    model = lgb.LGBMRegressor(
        num_leaves=cfg.lightgbm.num_leaves,
        learning_rate=cfg.lightgbm.learning_rate,
        n_estimators=cfg.lightgbm.n_estimators,
        verbose=-1,
    )
    model.fit(
        X_train,
        y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.log_evaluation(period=20)],
    )

    train_score = model.fit(X_train, y_train)
    val_score = model.fit(X_val, y_val)
    wb.log({"lgbm_train_r2": train_score, "lgbm_val_r2": val_score})
    logger.info(f"  [LightGBM] train_r2={train_score:.4f} val_r2={val_score:.4f}")

    return model


# ─── Main Pipeline ───────────────────────────────────────────────────────────


def run_training(
    config_path: str | None = None,
    wandb_project: str | None = None,
    wandb_name: str | None = None,
) -> dict[str, str]:
    """Full pipeline: pretrain → finetune → LightGBM stacking."""
    cfg = load_config(config_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wb = WandbHandler(
        project=wandb_project or cfg.wandb.project,
        entity=cfg.wandb.entity,
        name=wandb_name or cfg.wandb.name or f"etf-train-{datetime.now():%m%d-%H%M}",
        config=cfg,
        tags=["signals"],
        mode=cfg.wandb.mode,
    )

    logger.info(f"[Train] Device: {device}")
    logger.info(f"[Train] Config: seq_len={cfg.signals.sequence_length}, hidden={cfg.signals.hidden_size}")

    # ── Load sentiment data ────────────────────────────────────────────────────
    sentiment_path = cfg.data.output_sentiment
    if not sentiment_path.exists():
        raise FileNotFoundError(f"Industry sentiment not found at {sentiment_path}. Run compute-signals first.")
    sentiment_df = pl.read_parquet(sentiment_path)
    industries = sentiment_df["industry"].unique().to_list()
    logger.info(f"[Train] {len(industries)} industries, {len(sentiment_df)} rows")

    # ── Step A: Pretrain on ALL industries mixed ───────────────────────────────
    logger.info("\n[Step A] Pretrain LSTM+Attention on mixed industries...")
    seq_len = cfg.signals.sequence_length
    tc = cfg.training
    X_all, y_reg_all, y_cls_all = build_sequences(sentiment_df, industries, seq_len, tc.anomaly_threshold)
    logger.info(f"  Mixed data: X={X_all.shape}, y_reg={y_reg_all.shape}, y_cls={y_cls_all.shape}")

    lstm_model = train_lstm_attention_pretrain(X_all, y_reg_all, y_cls_all, cfg, wb, device)

    # ── Step B: Finetune per industry ─────────────────────────────────────────
    logger.info("\n[Step B] Finetune LSTM+Attention per industry...")
    lstm_model = finetune_per_industry(sentiment_df, industries, lstm_model, cfg, wb, device)

    # ── Save LSTM ──────────────────────────────────────────────────────────────
    checkpoint_dir = Path("checkpoints")
    checkpoint_dir.mkdir(exist_ok=True)
    lstm_path = checkpoint_dir / "lstm_attention.pt"
    torch.save(lstm_model.state_dict(), lstm_path)
    logger.info(f"  [Save] lstm_attention.pt → {checkpoint_dir}")

    # ── Step C: LightGBM stacking ───────────────────────────────────────────────
    logger.info("\n[Step C] Build stacking features + train LightGBM...")
    X_lgbm, y_lgbm = build_lgbm_features(sentiment_df, industries, seq_len, lstm_model, device)
    logger.info(f"  LGBM data: X={X_lgbm.shape}, y={y_lgbm.shape}")

    lgbm_model = train_lgbm_stacking(X_lgbm, y_lgbm, cfg, wb)
    lgbm_path = checkpoint_dir / "lgbm_stacking.txt"
    lgbm_model.booster_.save_model(str(lgbm_path))
    logger.info(f"  [Save] lgbm_stacking.txt → {checkpoint_dir}")

    # ── Step D: IsolationForest ───────────────────────────────────────────────
    logger.info("\n[Step D] Train IsolationForest...")
    from sklearn.ensemble import IsolationForest

    iforest_X = []
    for ind in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == ind).sort("date")
        nc = ind_df["news_count"].to_numpy()
        nh = ind_df["news_heat"].to_numpy() if "news_heat" in ind_df.columns else np.zeros_like(nc)
        for i in range(len(nc) - seq_len * 2):
            iforest_X.append(list(nc[i : i + seq_len]) + list(nh[i : i + seq_len]))

    iforest_X = np.array(iforest_X, dtype=np.float32)
    iforest = IsolationForest(
        contamination=cfg.isolation_forest.contamination,
        n_estimators=cfg.isolation_forest.n_estimators,
        random_state=42,
    )
    iforest.fit(iforest_X)
    iforest_path = checkpoint_dir / "iforest_model.pkl"
    with open(iforest_path, "wb") as f:
        pickle.dump(iforest, f)
    logger.info(f"  [Save] iforest_model.pkl → {checkpoint_dir}")

    wb.finish()
    logger.success("[Done] All models trained.")

    return {
        "lstm_path": str(lstm_path),
        "lgbm_path": str(lgbm_path),
        "iforest_path": str(iforest_path),
    }


# ─── CLI ─────────────────────────────────────────────────────────────────────


if __name__ == "__main__":
    import typer

    app = typer.Typer(add_completion=False)

    @app.command()
    def train(
        config: str | None = None,
        project: str = "news2etf",
        name: str | None = None,
    ) -> None:
        """Run full pipeline: pretrain → finetune → LightGBM stacking."""
        run_training(config_path=config, wandb_project=project, wandb_name=name)

    @app.command()
    def train_lstm_only(config: str | None = None) -> None:
        """Pretrain + finetune LSTM+Attention only."""
        cfg = load_config(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wb = WandbHandler(project="news2etf", name="lstm-only", config=cfg, tags=["signals"])

        sentiment_df = pl.read_parquet(cfg.data.output_sentiment)
        industries = sentiment_df["industry"].unique().to_list()
        seq_len = cfg.signals.sequence_length
        tc = cfg.training

        X_all, y_reg_all, y_cls_all = build_sequences(sentiment_df, industries, seq_len, tc.anomaly_threshold)
        lstm_model = train_lstm_attention_pretrain(X_all, y_reg_all, y_cls_all, cfg, wb, device)
        lstm_model = finetune_per_industry(sentiment_df, industries, lstm_model, cfg, wb, device)

        checkpoint_dir = Path("checkpoints")
        lstm_path = checkpoint_dir / "lstm_attention.pt"
        torch.save(lstm_model.state_dict(), lstm_path)
        logger.info(f"Saved → {lstm_path}")
        wb.finish()

    @app.command()
    def train_lgbm_only(config: str | None = None) -> None:
        """Train LightGBM stacking only (requires lstm_attention.pt)."""
        cfg = load_config(config)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        wb = WandbHandler(project="news2etf", name="lgbm-only", config=cfg, tags=["signals"])

        checkpoint_dir = Path("checkpoints")
        lstm_path = checkpoint_dir / "lstm_attention.pt"
        if not lstm_path.exists():
            logger.error("LSTM checkpoint not found. Run train_lstm_only first.")
            raise typer.Exit(1)

        lstm_model = LSTMWithAttention(
            input_size=1,
            hidden_size=cfg.signals.hidden_size,
            num_layers=cfg.signals.num_layers,
            dropout=cfg.signals.dropout,
            num_heads=cfg.training.num_heads,
        ).to(device)
        lstm_model.load_state_dict(torch.load(lstm_path, map_location=device))
        lstm_model.eval()

        sentiment_df = pl.read_parquet(cfg.data.output_sentiment)
        industries = sentiment_df["industry"].unique().to_list()
        X_lgbm, y_lgbm = build_lgbm_features(sentiment_df, industries, cfg.signals.sequence_length, lstm_model, device)
        lgbm_model = train_lgbm_stacking(X_lgbm, y_lgbm, cfg, wb)
        lgbm_path = checkpoint_dir / "lgbm_stacking.txt"
        lgbm_model.booster_.save_model(str(lgbm_path))
        logger.info(f"Saved → {lgbm_path}")
        wb.finish()

    app()
