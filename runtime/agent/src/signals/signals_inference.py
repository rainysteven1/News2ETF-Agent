"""Signals ONNX inference pipeline for agent/runtime consumption.

This module loads the deployed signals ONNX bundle exported by `signals export-onnx`
and produces daily meta-sector features that the agent and backtest can
consume without depending on in-memory training objects.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from src.signals.signal_helpers import (
    build_sub_category_sequences,
    compute_global_leader_sentiment,
    get_market_cap_weight,
)


def _import_onnxruntime() -> Any:
    try:
        import onnxruntime as ort
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "onnxruntime is required only when agent feature cache is missing and live signals inference is needed."
        ) from exc
    return ort


def _make_ort_session(onnx_path: Path) -> Any:
    ort = _import_onnxruntime()
    opts = ort.SessionOptions()
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    opts.intra_op_num_threads = 1
    opts.inter_op_num_threads = 4
    opts.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
    return ort.InferenceSession(str(onnx_path), opts, providers=["CPUExecutionProvider"])


def _run_first_output(session: Any, array: np.ndarray) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    outputs = session.run(None, {input_name: array.astype(np.float32)})
    arr = np.asarray(outputs[0])
    if arr.ndim == 2 and arr.shape[1] == 1:
        return arr[:, 0]
    return arr


def _run_tcn_reg(session: Any, array: np.ndarray, batch_size: int = 256) -> np.ndarray:
    input_name = session.get_inputs()[0].name
    chunks: list[np.ndarray] = []
    for start in range(0, len(array), batch_size):
        batch = array[start : start + batch_size].astype(np.float32)
        outputs = session.run(None, {input_name: batch})
        chunks.append(np.asarray(outputs[0], dtype=np.float32))
    return np.concatenate(chunks, axis=0) if chunks else np.zeros((0, 0), dtype=np.float32)


def _prediction_stability_2d(values: np.ndarray, window: int = 5) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        hist = values[start : i + 1]
        signs = np.sign(hist)
        dir_consistency = np.abs(np.sum(signs, axis=0)) / float(len(hist))
        dispersion = np.std(hist, axis=0) / (np.mean(np.abs(hist), axis=0) + 1e-9)
        out[i] = np.clip(dir_consistency - 0.5 * dispersion, 0.0, 1.0)
    return out


def _rolling_std_1d(values: np.ndarray, window: int = 5) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        out[i] = float(np.std(values[start : i + 1]))
    return out


def _rolling_percentile_1d(values: np.ndarray, window: int = 252) -> np.ndarray:
    out = np.zeros(len(values), dtype=np.float32)
    for i in range(len(values)):
        start = max(0, i - window + 1)
        hist = values[start : i + 1]
        out[i] = float(np.mean(hist <= values[i])) if len(hist) > 1 else 0.5
    return out


def _normalize_date(value: Any) -> str:
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return str(value)


class SignalsONNXInferencePipeline:
    """Runtime inference over the exported signals ONNX bundle."""

    def __init__(
        self,
        bundle_dir: str | Path,
        meta_sector_mapping_path: str | Path,
    ) -> None:
        self.bundle_dir = Path(bundle_dir)
        self.meta_sector_mapping_path = Path(meta_sector_mapping_path)
        manifest_path = self.bundle_dir / "manifest.json"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Signals ONNX manifest not found: {manifest_path}")

        with open(manifest_path, encoding="utf-8") as f:
            self.manifest = json.load(f)
        with open(self.meta_sector_mapping_path, encoding="utf-8") as f:
            self.meta_sector_map = json.load(f)

        self.seq_len = int(self.manifest["seq_len"])
        self.label_stats_path = self.manifest.get("label_stats_path") or None
        self.tcn_session = _make_ort_session(self.bundle_dir / "tcn.onnx")
        self.lgbm_sessions: dict[str, Any] = {}
        lgbm_dir = self.bundle_dir / "lgbm"
        if lgbm_dir.exists():
            for path in sorted(lgbm_dir.glob("*.onnx")):
                self.lgbm_sessions[path.stem] = _make_ort_session(path)

        self.iforest_session: Any | None = None
        iforest_path = self.bundle_dir / "iforest.onnx"
        if iforest_path.exists():
            self.iforest_session = _make_ort_session(iforest_path)

    def infer_feature_frame(
        self,
        sentiment_df: pl.DataFrame,
        *,
        start_date: str | None = None,
        end_date: str | None = None,
        output_path: str | Path | None = None,
    ) -> pl.DataFrame:
        X_all, _, sample_dates, sub_industries = build_sub_category_sequences(
            sentiment_df,
            self.meta_sector_map,
            lookback_days=self.seq_len,
            price_df=pl.DataFrame(),
            label_stats_path=self.label_stats_path,
            target_mode=self.manifest.get("target_mode", "meta_excess_return"),
        )
        if len(X_all) == 0:
            return pl.DataFrame()

        all_sample_dates = [_normalize_date(d) for d in sample_dates]
        tcn_reg_all = _run_tcn_reg(self.tcn_session, X_all)
        reg_delta_all = np.zeros_like(tcn_reg_all, dtype=np.float32)
        reg_delta_all[1:] = tcn_reg_all[1:] - tcn_reg_all[:-1]
        stability_all = _prediction_stability_2d(tcn_reg_all, 5)

        if self.iforest_session is not None:
            flat = X_all.reshape(len(X_all), -1).astype(np.float32)
            outputs = self.iforest_session.run(None, {self.iforest_session.get_inputs()[0].name: flat})
            raw_iforest = None
            for out in reversed(outputs):
                arr = np.asarray(out)
                if arr.shape[0] != len(flat):
                    continue
                if arr.ndim == 2 and arr.shape[1] >= 2:
                    raw_iforest = arr[:, -1].astype(np.float32)
                    break
                if np.issubdtype(arr.dtype, np.number):
                    raw_iforest = arr.reshape(len(flat)).astype(np.float32)
                    break
            if raw_iforest is None:
                raw_iforest = np.zeros(len(flat), dtype=np.float32)
        else:
            raw_iforest = np.zeros(len(X_all), dtype=np.float32)
        news_heat_all = _rolling_percentile_1d(raw_iforest, 252)

        sector_col = "sub_category" if "sub_category" in sentiment_df.columns else "industry"
        sent_col = "sentiment_mean" if "sentiment_mean" in sentiment_df.columns else "sentiment_weighted"
        full_dates = [_normalize_date(d) for d in sentiment_df["date"].unique().sort().to_list()]
        date_to_idx = {d: idx for idx, d in enumerate(full_dates)}
        sub_to_idx = {sub: idx for idx, sub in enumerate(sub_industries)}
        sample_date_to_seq_idx = {d: idx for idx, d in enumerate(all_sample_dates)}
        meta_sectors = list(self.meta_sector_map.get("meta_sectors", {}).keys())

        meta_sentiment: dict[str, np.ndarray] = {}
        meta_news_count: dict[str, np.ndarray] = {}
        for ms in meta_sectors:
            arr_sent = np.zeros(len(full_dates), dtype=np.float32)
            arr_news = np.zeros(len(full_dates), dtype=np.float32)
            ms_subs = self.meta_sector_map.get("meta_sectors", {}).get(ms, {}).get("sub_categories", [])
            ms_df = sentiment_df.filter(pl.col(sector_col).is_in(ms_subs)).sort("date")
            for raw_date in full_dates:
                day_df = ms_df.filter(pl.col("date").cast(pl.Utf8) == raw_date)
                if day_df.is_empty():
                    continue
                weights = np.array(
                    [get_market_cap_weight(sub, self.meta_sector_map) for sub in day_df[sector_col].to_list()],
                    dtype=np.float32,
                )
                sentiments = np.array(day_df[sent_col].to_list(), dtype=np.float32)
                arr_sent[date_to_idx[raw_date]] = (
                    float(np.average(sentiments, weights=weights)) if weights.sum() > 0 else float(sentiments.mean())
                )
                arr_news[date_to_idx[raw_date]] = float(day_df["news_count"].sum()) if "news_count" in day_df.columns else 0.0
            meta_sentiment[ms] = arr_sent
            meta_news_count[ms] = arr_news

        gl_df = compute_global_leader_sentiment(sentiment_df, self.meta_sector_map)
        global_leader_map: dict[str, dict[str, float]] = {}
        if not gl_df.is_empty():
            for row in gl_df.iter_rows(named=True):
                global_leader_map[_normalize_date(row["date"])] = {
                    k: float(v or 0.0) for k, v in row.items() if k != "date"
                }

        entropy_by_sector: dict[str, np.ndarray] = {}
        std_news_by_sector: dict[str, np.ndarray] = {}
        sent_std_by_sector: dict[str, np.ndarray] = {}
        volume_ratio_by_sector: dict[str, np.ndarray] = {}
        global_leader_by_sector: dict[str, np.ndarray] = {}
        market_beta_by_sector: dict[str, np.ndarray] = {}
        for ms in meta_sectors:
            ms_sent = meta_sentiment[ms]
            ms_news = meta_news_count[ms]

            entropy = np.zeros(len(ms_sent), dtype=np.float32)
            for i in range(len(ms_sent)):
                start = max(0, i - 4)
                window = ms_sent[start : i + 1]
                if len(window) > 1 and np.std(window) > 1e-9:
                    p = np.abs(window) / (np.sum(np.abs(window)) + 1e-9)
                    entropy[i] = float(-np.sum(p * np.log(p + 1e-9)))
            entropy_by_sector[ms] = np.clip(entropy / 2.0, 0.0, 1.0)
            std_news_by_sector[ms] = _rolling_std_1d(ms_news, 5)
            sent_std_by_sector[ms] = _rolling_std_1d(ms_sent, 5)

            rolling_mean_news = np.zeros(len(ms_news), dtype=np.float32)
            for i in range(len(ms_news)):
                start = max(0, i - 4)
                rolling_mean_news[i] = float(np.mean(ms_news[start : i + 1]))
            volume_ratio_by_sector[ms] = np.clip(ms_news / (rolling_mean_news + 1e-9), 0.5, 3.0)

            gl_series = np.zeros(len(full_dates), dtype=np.float32)
            for i, raw_date in enumerate(full_dates):
                gl_series[i] = float(global_leader_map.get(raw_date, {}).get(f"global_leader_{ms}", 0.0))
            global_leader_by_sector[ms] = gl_series
            if np.std(gl_series) > 1e-9:
                cov = np.cov(ms_sent, gl_series)[0, 1]
                var = np.var(gl_series)
                beta_value = float(cov / (var + 1e-9))
            else:
                beta_value = 1.0
            market_beta_by_sector[ms] = np.full(len(full_dates), np.clip(beta_value, 0.3, 2.0), dtype=np.float32)

        rows: list[dict[str, Any]] = []
        for sample_pos, raw_date in enumerate(all_sample_dates):
            if start_date and raw_date < start_date:
                continue
            if end_date and raw_date > end_date:
                continue

            date_idx = date_to_idx[raw_date]
            seq_idx = sample_date_to_seq_idx[raw_date]
            row: dict[str, Any] = {
                "date": raw_date,
                "iforest_score": float(raw_iforest[seq_idx]),
            }

            for m_idx, ms in enumerate(meta_sectors):
                ms_subs = self.meta_sector_map.get("meta_sectors", {}).get(ms, {}).get("sub_categories", [])
                residuals = [float(X_all[seq_idx, -1, sub_to_idx[sub], 5]) for sub in ms_subs if sub in sub_to_idx]

                row[f"tcn_reg_{ms}"] = float(tcn_reg_all[seq_idx, m_idx])
                row[f"tcn_reg_delta_{ms}"] = float(reg_delta_all[seq_idx, m_idx])
                row[f"tcn_prediction_stability_{ms}"] = float(stability_all[seq_idx, m_idx])
                row[f"news_heat_{ms}"] = float(news_heat_all[seq_idx])
                row[f"meta_sentiment_{ms}"] = float(meta_sentiment[ms][date_idx])
                row[f"global_leader_sentiment_{ms}"] = float(global_leader_by_sector[ms][date_idx])
                row[f"sentiment_vs_price_residual_{ms}"] = float(np.mean(residuals)) if residuals else 0.0

                delta1 = float(meta_sentiment[ms][date_idx] - meta_sentiment[ms][max(0, date_idx - 1)])
                delta2 = float(meta_sentiment[ms][date_idx] - meta_sentiment[ms][max(0, date_idx - 2)])
                news_count = float(meta_news_count[ms][date_idx])
                sent_vol = float(sent_std_by_sector[ms][date_idx])
                lgbm_x = np.array(
                    [
                        [
                            delta1,
                            delta2,
                            news_count,
                            float(news_heat_all[seq_idx]),
                            float(tcn_reg_all[seq_idx, m_idx]),
                            float(reg_delta_all[seq_idx, m_idx]),
                            float(stability_all[seq_idx, m_idx]),
                            float(std_news_by_sector[ms][date_idx]),
                            sent_vol,
                            float(tcn_reg_all[seq_idx, m_idx] * news_heat_all[seq_idx]),
                            float(volume_ratio_by_sector[ms][date_idx]),
                            float(np.clip(sent_vol * 2.0, 0.005, 0.1)),
                            float(np.clip((meta_sentiment[ms][date_idx] + 1.0) * 50.0, 50.0, 150.0)),
                            float(global_leader_by_sector[ms][date_idx]),
                            float(market_beta_by_sector[ms][date_idx]),
                            float(entropy_by_sector[ms][date_idx]),
                        ]
                    ],
                    dtype=np.float32,
                )
                safe_ms = ms.replace("/", "_")
                if safe_ms in self.lgbm_sessions:
                    row[f"lgbm_score_{ms}"] = float(_run_first_output(self.lgbm_sessions[safe_ms], lgbm_x)[0])
                else:
                    row[f"lgbm_score_{ms}"] = 0.0

            rows.append(row)

        feature_df = pl.DataFrame(rows)
        if output_path is not None and not feature_df.is_empty():
            out = Path(output_path)
            out.parent.mkdir(parents=True, exist_ok=True)
            feature_df.write_parquet(out)
        return feature_df
