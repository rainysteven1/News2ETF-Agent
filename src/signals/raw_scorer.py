"""ML signal scorer — CPU/GPU adaptive.

CPU mode: keyword-based rules
GPU mode: LSTM + IsolationForest + LightGBM (requires torch)
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import polars as pl

from src.config import AgentRootConfig
from src.utils.device import get_device


class RawScorer:
    """CPU/GPU adaptive signal scorer.

    GPU mode: loads LSTM, IForest, LightGBM from checkpoint_dir
    CPU mode: uses keyword sentiment rules
    """

    def __init__(self, config: AgentRootConfig, checkpoint_dir: Path | None = None):
        self.config = config
        self.checkpoint_dir = checkpoint_dir or Path("checkpoints")
        self.device = get_device()
        self.use_gpu = self.device == "cuda"

        self.lstm = None
        self.iforest = None
        self.lgbm = None

        if self.use_gpu:
            self._load_models()

    def _load_models(self) -> None:
        """Load LSTM, IForest, LightGBM from checkpoint_dir."""
        import torch  # lazy import — only needed when GPU available

        from src.signals.models import LSTMModel

        lstm_path = self.checkpoint_dir / "lstm_model.pt"
        iforest_path = self.checkpoint_dir / "iforest_model.pkl"
        lgbm_path = self.checkpoint_dir / "lgbm_model.txt"

        if lstm_path.exists():
            self.lstm = LSTMModel(
                hidden_size=self.config.model.lstm.hidden_size,
                num_layers=self.config.model.lstm.num_layers,
                dropout=self.config.model.lstm.dropout,
            ).to(self.device)
            self.lstm.load_state_dict(torch.load(lstm_path, weights_only=True))
            self.lstm.eval()
        else:
            self.use_gpu = False
            return

        if iforest_path.exists():
            with open(iforest_path, "rb") as f:
                self.iforest = pickle.load(f)

        if lgbm_path.exists():
            try:
                import lightgbm as lgb

                self.lgbm = lgb.Booster(model_file=str(lgbm_path))
            except Exception:
                self.lgbm = None

    def _get_lookback(
        self,
        sentiment_df: pl.DataFrame,
        industry: str,
        date: str,
        window: int,
    ) -> np.ndarray | None:
        ind_df = (
            sentiment_df.filter(pl.col("industry") == industry).filter(pl.col("date") <= date).sort("date").tail(window)
        )
        if len(ind_df) < window:
            return None
        return ind_df["sentiment_mean"].to_numpy().astype(np.float32)

    def score_industry(
        self,
        sentiment_df: pl.DataFrame,
        industry: str,
        date: str,
    ) -> dict:
        """Compute all signals for one industry on a date."""
        if self.use_gpu and self.lstm is not None:
            return self._score_gpu(sentiment_df, industry, date)
        return self._score_cpu(sentiment_df, industry, date)

    def _score_gpu(
        self,
        sentiment_df: pl.DataFrame,
        industry: str,
        date: str,
    ) -> dict:
        import torch  # lazy import

        if not self.use_gpu:
            return self._score_cpu(sentiment_df, industry, date)

        seq_len = self.config.model.lstm.sequence_length
        seq = self._get_lookback(sentiment_df, industry, date, seq_len)
        if seq is None:
            return self._score_cpu(sentiment_df, industry, date)

        seq_tensor = torch.FloatTensor(seq.reshape(1, seq_len, 1)).to(self.device)
        with torch.no_grad():
            momentum = float(self.lstm(seq_tensor).item())

        heat = 0.5
        if self.iforest is not None:
            ind_df = (
                sentiment_df.filter(pl.col("industry") == industry)
                .filter(pl.col("date") <= date)
                .sort("date")
                .tail(seq_len * 2)
            )
            if len(ind_df) >= seq_len * 2:
                from sklearn.preprocessing import StandardScaler

                scaler = StandardScaler()
                feats = (
                    np.concatenate(
                        [
                            ind_df["news_count"].to_numpy()[-seq_len:],
                            ind_df["news_heat"].to_numpy()[-seq_len:],
                        ]
                    )
                    .reshape(1, -1)
                    .astype(np.float32)
                )
                scaler.fit(feats)
                raw = self.iforest.decision_function(scaler.transform(feats))
                heat = float(np.clip((raw - raw.min()) / (raw.max() - raw.min() + 1e-8), 0, 1)[0])

        composite = momentum
        if self.lgbm is not None:
            ind_df = sentiment_df.filter(pl.col("industry") == industry).filter(pl.col("date") <= date).sort("date")
            n = len(ind_df)
            if n >= seq_len + 1:
                sm = ind_df["sentiment_mean"].to_numpy()
                composite = float(
                    np.clip(
                        self.lgbm.predict(
                            np.array(
                                [
                                    [
                                        sm[-1] - sm[-1 - seq_len],
                                        heat,
                                        ind_df["sentiment_trend"].to_numpy()[-1]
                                        if "sentiment_trend" in ind_df.columns
                                        else 0,
                                        ind_df["news_count"].to_numpy()[-1],
                                    ]
                                ],
                                dtype=np.float32,
                            )
                        )[0],
                        -1,
                        1,
                    )
                )

        return {
            "momentum_score": momentum,
            "heat_anomaly": heat,
            "composite_score": composite,
            "trend_direction": 1 if momentum > 0.1 else (-1 if momentum < -0.1 else 0),
        }

    def _score_cpu(
        self,
        sentiment_df: pl.DataFrame,
        industry: str,
        date: str,
    ) -> dict:
        """CPU fallback: use keyword sentiment on recent news headlines."""
        ind_df = sentiment_df.filter(pl.col("industry") == industry).filter(pl.col("date") <= date).sort("date").tail(5)
        if len(ind_df) == 0:
            return {"momentum_score": 0.0, "heat_anomaly": 0.0, "composite_score": 0.0, "trend_direction": 0}

        sentiment_mean = ind_df["sentiment_mean"].mean()
        momentum = float(np.clip(sentiment_mean, -1, 1))

        news_heat_vals = ind_df["news_heat"].to_numpy() if "news_heat" in ind_df.columns else [1.0]
        heat = float(np.clip(np.mean(news_heat_vals) / 5.0, 0, 1))

        return {
            "momentum_score": momentum,
            "heat_anomaly": heat,
            "composite_score": momentum * heat,
            "trend_direction": 1 if momentum > 0.1 else (-1 if momentum < -0.1 else 0),
        }

    def score_all(
        self,
        sentiment_df: pl.DataFrame,
        industries: list[str],
        date: str,
    ) -> pl.DataFrame:
        rows = []
        for industry in industries:
            s = self.score_industry(sentiment_df, industry, date)
            rows.append({"industry": industry, "date": date, **s})
        return pl.DataFrame(rows)
