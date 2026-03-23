"""TF-IDF based similar news retrieval — find historical cases similar to current news."""

from __future__ import annotations

import polars as pl
from sklearn.feature_extraction.text import TfidfVectorizer


class KnowledgeRetrieval:
    """TF-IDF similarity search over historical news."""

    def __init__(self, news_df: pl.DataFrame, text_column: str = "content"):
        self.news_df = news_df
        self.text_column = text_column

        # Build TF-IDF index
        texts = news_df[text_column].to_list()
        self.vectorizer = TfidfVectorizer(max_features=5000)
        self.tfidf_matrix = self.vectorizer.fit_transform(texts)  # type: ignore[attr-defined]

    def retrieve(
        self,
        query_text: str,
        top_k: int = 5,
    ) -> list[dict]:
        """Find top-k most similar historical news items to query_text."""
        if not query_text or self.tfidf_matrix.shape[0] == 0:
            return []

        query_vec = self.vectorizer.transform([query_text])  # type: ignore[attr-defined]

        # Cosine similarity
        from sklearn.metrics.pairwise import cosine_similarity

        sims = cosine_similarity(query_vec, self.tfidf_matrix)[0]

        # Get top-k indices
        top_indices = sims.argsort()[-top_k:][::-1]

        results = []
        for idx in top_indices:
            row = self.news_df.row(idx, named=True)
            results.append(
                {
                    "similarity": float(sims[idx]),
                    "date": row.get("date", ""),
                    "industry": row.get("industry", ""),
                    "content": row.get(self.text_column, "")[:200],
                }
            )
        return results
