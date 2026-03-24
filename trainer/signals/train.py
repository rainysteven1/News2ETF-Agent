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

from trainer.config import TrainerConfig, load_config
from trainer.signals.dataset import build_lgbm_features, build_sequences, WeeklySignalDataset
from trainer.signals.models import TCN
from trainer.wandb_handler import WandbHandler

# ─── Model Training ─────────────────────────────────────────────────────────────


def train_tcn_pretrain(
    X: np.ndarray,
    y_reg: np.ndarray,
    y_cls: np.ndarray,
    cfg: TrainerConfig,
    wb: WandbHandler,
    device: torch.device,
) -> TCN:
    """Step A: Pretrain TCN on ALL industries mixed."""
    tc = cfg.training
    sc = cfg.tcn
    model = TCN(
        input_size=1,
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
    cfg: TrainerConfig,
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

    for ind in industries:
        ind_df = sentiment_df.filter(pl.col("industry") == ind).sort("date")
        if len(ind_df) < seq_len + 2:
            continue
        vals = ind_df["sentiment_mean"].to_numpy()
        rets = ind_df["return"].to_numpy() if "return" in ind_df.columns else np.zeros_like(vals)

        X_ind, y_reg_ind, y_cls_ind = [], [], []
        for i in range(len(vals) - seq_len - 1):
            X_ind.append(vals[i : i + seq_len])
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

        X_t = torch.FloatTensor(np.array(X_ind, dtype=np.float32)).reshape(-1, seq_len, 1).to(device)
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


def train_lgbm_stacking(
    X: np.ndarray,
    y: np.ndarray,
    dates: np.ndarray | None = None,
    cfg: TrainerConfig | None = None,
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
        f"  [{stage}] Residuals — skew={skew:.3f} kurt={kurt:.3f} "
        f"shapiro_p={shapiro_p:.4f} (p<0.05→non-normal)"
    )
    return result


# ─── Main Pipeline ───────────────────────────────────────────────────────────


def run_training(force: bool = False) -> dict[str, str]:
    """Full pipeline: pretrain → finetune → LightGBM stacking."""
    cfg = load_config()
    wcfg = cfg.wandb

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    wb = WandbHandler(
        project=wcfg.project,
        entity=cfg.wandb.entity,
        name=f"signals-{datetime.now():%m%d-%H%M}",
        config=cfg,
        tags=["signals", "TCN", "LightGBM", "IsolationForest"],
        mode=cfg.wandb.mode,
    )

    logger.info(f"[Train] Device: {device}")
    logger.info(f"[Train] Config: seq_len={cfg.tcn.sequence_length}, hidden={cfg.tcn.hidden_size}")

    # ── Load / build sentiment data via WeeklySignalDataset ───────────────────
    #   If output_sentiment parquet exists → load directly; otherwise process raw.
    ds = WeeklySignalDataset(cfg.dataset, force=force)
    sentiment_df = ds.sentiment_df
    assert sentiment_df is not None, "sentiment_df is None after dataset init"
    industries = sentiment_df["industry"].unique().to_list()
    logger.info(
        f"[Train] {len(industries)} industries, {len(sentiment_df)} rows "
        f"(cached={ds.output_sentiment is not None and ds.output_sentiment.exists()})"
    )

    # ── Step A: Pretrain on ALL industries mixed ───────────────────────────────
    logger.info("\n[Step A] Pretrain TCN on mixed industries...")
    seq_len = cfg.tcn.sequence_length
    tc = cfg.training
    X_all, y_reg_all, y_cls_all = build_sequences(sentiment_df, industries, seq_len, tc.anomaly_threshold)
    logger.info(f"  Mixed data: X={X_all.shape}, y_reg={y_reg_all.shape}, y_cls={y_cls_all.shape}")

    tcn_model = train_tcn_pretrain(X_all, y_reg_all, y_cls_all, cfg, wb, device)

    # ── Step B: Finetune per industry ─────────────────────────────────────────
    logger.info("\n[Step B] Finetune TCN per industry...")
    tcn_model = finetune_per_industry(sentiment_df, industries, tcn_model, cfg, wb, device)

    # ── Save TCN ──────────────────────────────────────────────────────────────
    checkpoint_dir = cfg.training.output_checkpoint / f"signals-{datetime.now():%m%d-%H%M}"
    checkpoint_dir.mkdir(exist_ok=True, parents=True)
    tcn_path = checkpoint_dir / "tcn.pt"
    torch.save(tcn_model.state_dict(), tcn_path)
    logger.info(f"  [Save] tcn.pt → {checkpoint_dir}")

    # ── Step C: LightGBM stacking ───────────────────────────────────────────────
    logger.info("\n[Step C] Build stacking features + train LightGBM...")
    X_lgbm, y_lgbm, dates, industry_lgbm = build_lgbm_features(
        sentiment_df, industries, seq_len, tcn_model, device
    )
    logger.info(f"  LGBM data: X={X_lgbm.shape}, y={y_lgbm.shape}")

    lgbm_model = train_lgbm_stacking(X_lgbm, y_lgbm, dates, cfg, wb)

    # ── Post-step C: Industry IC + Residual analysis ─────────────────────────────
    # Time-based split (last 20%)
    order = np.argsort(dates)
    split = int(len(X_lgbm) * 0.8)
    val_idx = order[split:]
    y_val_true = y_lgbm[val_idx]
    y_val_pred = lgbm_model.predict(X_lgbm[val_idx])
    industries_val = industry_lgbm[val_idx]

    compute_industry_ic(y_val_true, y_val_pred, industries_val, stage="LGBM_val", wb=wb)
    analyze_residuals(y_val_true, y_val_pred, dates[val_idx], stage="LGBM_val", wb=wb)

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

    # ── Step E: Export all models to ONNX ─────────────────────────────────────
    logger.info("\n[Step E] Exporting all models to ONNX...")
    _export_all_onnx(tcn_model, lgbm_model, iforest, iforest_X, checkpoint_dir, seq_len, X_lgbm, device)

    wb.finish()
    logger.success("[Done] All models trained.")

    return {
        "tcn_path": str(tcn_path),
        "lgbm_path": str(lgbm_path),
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

    from trainer.signals.models import export_tcn_to_onnx

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
            input_size=1,
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
        logger.warning(f"  [ONNX] IsolationForest export failed ({type(exc).__name__}): {exc}. "
                      "Falling back to pickle.")
        results["iforest_onnx"] = None

    return results
