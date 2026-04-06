"""Analyze label consistency for SetFit sub-category data.

For each major category, computes:
  - Intra-class vs inter-class embedding distance (separability ratio)
  - Pairwise sub-category similarity matrix
  - Confused pairs (high cross-category similarity)
  - Per-class confidence distribution

Usage:
    python scripts/analyze_label_consistency.py [--majors 主题策略 科技信息] [--sample-size 500]
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl
from rich.console import Console
from rich.table import Table
from sklearn.metrics.pairwise import cosine_similarity

console = Console()

RAW_PATH = Path("trainer/data/labeled/setfit/raw.parquet")
MODEL_PATH = Path("trainer/data/pretrained_models/mengzi-bert-base-fin")


def encode_texts(texts: list[str], model_path: str, batch_size: int = 256) -> np.ndarray:
    from sentence_transformers import SentenceTransformer

    model = SentenceTransformer(model_path)
    return model.encode(texts, show_progress_bar=True, batch_size=batch_size)


def analyze_major(
    df: pl.DataFrame,
    major: str,
    model_path: str,
    sample_size: int = 500,
) -> dict:
    sub = df.filter(pl.col("major_category") == major)
    labels = sorted(sub["sub_category"].unique().to_list())
    console.print(f"\n[bold cyan]{'=' * 60}[/bold cyan]")
    console.print(f"[bold cyan]Major: {major} ({len(sub)} samples, {len(labels)} sub-categories)[/bold cyan]")

    # Sample per class for embedding
    sampled_parts = []
    for label in labels:
        label_df = sub.filter(pl.col("sub_category") == label)
        n = min(sample_size, len(label_df))
        sampled_parts.append(label_df.sample(n=n, shuffle=True, seed=42))
    df_sampled = pl.concat(sampled_parts)

    texts = (df_sampled["title"] + " " + df_sampled["content"].fill_null("")).to_list()
    text_labels = df_sampled["sub_category"].to_list()

    console.print(f"Encoding {len(texts)} texts...")
    embeddings = encode_texts(texts, model_path)

    # Compute per-class centroids
    centroids: dict[str, np.ndarray] = {}
    class_embeddings: dict[str, np.ndarray] = {}
    for label in labels:
        mask = [i for i, l in enumerate(text_labels) if l == label]
        emb = embeddings[mask]
        class_embeddings[label] = emb
        centroids[label] = emb.mean(axis=0)

    # Pairwise centroid similarity
    centroid_matrix = np.array([centroids[l] for l in labels])
    sim_matrix = cosine_similarity(centroid_matrix)

    # Separability: intra-class distance vs inter-class distance
    intra_distances = {}
    for label in labels:
        emb = class_embeddings[label]
        centroid = centroids[label]
        dists = 1 - cosine_similarity(emb, centroid.reshape(1, -1)).flatten()
        intra_distances[label] = float(dists.mean())

    avg_intra = np.mean(list(intra_distances.values()))
    inter_dists = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            inter_dists.append(1 - sim_matrix[i][j])
    avg_inter = np.mean(inter_dists) if inter_dists else 1.0
    separability = avg_inter / max(avg_intra, 1e-9)

    # Confidence stats per class
    conf_stats = {}
    for label in labels:
        label_df = sub.filter(pl.col("sub_category") == label)
        conf = label_df["confidence"]
        conf_stats[label] = {
            "count": len(label_df),
            "mean": float(conf.mean()),
            "min": float(conf.min()),
            "std": float(conf.std()),
        }

    # Print similarity matrix
    table = Table(title=f"{major} — Pairwise Centroid Cosine Similarity")
    table.add_column("", style="bold")
    for label in labels:
        table.add_column(label[:12], justify="center")
    for i, label in enumerate(labels):
        row = []
        for j in range(len(labels)):
            val = sim_matrix[i][j]
            style = "bold red" if i != j and val > 0.85 else ""
            row.append(f"[{style}]{val:.3f}[/{style}]" if style else f"{val:.3f}")
        table.add_row(label, *row)
    console.print(table)

    # Print separability
    console.print(f"\n[bold]Separability ratio[/bold]: {separability:.3f} (inter/intra, higher = better)")
    console.print(f"  avg intra-class distance: {avg_intra:.4f}")
    console.print(f"  avg inter-class distance: {avg_inter:.4f}")

    # Print confused pairs
    confused = []
    for i in range(len(labels)):
        for j in range(i + 1, len(labels)):
            if sim_matrix[i][j] > 0.85:
                confused.append((labels[i], labels[j], float(sim_matrix[i][j])))
    if confused:
        console.print("\n[bold red]Confused pairs (similarity > 0.85):[/bold red]")
        for a, b, s in sorted(confused, key=lambda x: -x[2]):
            console.print(f"  {a} ↔ {b}: {s:.3f}")

    # Print confidence stats
    conf_table = Table(title=f"{major} — Per-class Confidence & Count")
    conf_table.add_column("Sub-category", style="bold")
    conf_table.add_column("Count", justify="right")
    conf_table.add_column("Conf Mean", justify="right")
    conf_table.add_column("Conf Min", justify="right")
    conf_table.add_column("Conf Std", justify="right")
    for label in labels:
        s = conf_stats[label]
        conf_table.add_row(label, str(s["count"]), f"{s['mean']:.3f}", f"{s['min']:.3f}", f"{s['std']:.3f}")
    console.print(conf_table)

    return {
        "major": major,
        "n_samples": len(sub),
        "n_classes": len(labels),
        "separability": separability,
        "confused_pairs": confused,
        "confidence": conf_stats,
    }


def main():
    parser = argparse.ArgumentParser(description="Analyze label consistency for SetFit sub-category data")
    parser.add_argument("--majors", nargs="*", default=None, help="Major categories to analyze (default: all)")
    parser.add_argument("--sample-size", type=int, default=500, help="Max samples per class for embedding")
    parser.add_argument("--output", type=str, default=None, help="Save JSON summary to this path")
    args = parser.parse_args()

    df = pl.read_parquet(RAW_PATH)
    all_majors = sorted(df["major_category"].unique().to_list())
    target_majors = args.majors if args.majors else all_majors

    results = {}
    for major in target_majors:
        result = analyze_major(df, major, str(MODEL_PATH), args.sample_size)
        results[major] = result

    if args.output:
        with open(args.output, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2, default=str)
        console.print(f"\n[bold green]Summary saved to {args.output}[/bold green]")


if __name__ == "__main__":
    main()
