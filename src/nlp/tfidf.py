"""
EPM TF-IDF Vectorizer from Scratch
==================================

Pure NumPy implementation of Term Frequency - Inverse Document Frequency (TF-IDF).

MATHEMATICAL FOUNDATION
-----------------------
1. Why TF-IDF?
   Raw word counts (Bag-of-Words) bias heavily toward common words ("the", "is", "pump")
   that appear everywhere but carry little discriminative information.
   TF-IDF weights words proportionally to their frequency in a specific document,
   discounted by their prevalence across the entire corpus.

2. Term Frequency (TF):
   For term t in document d:
     TF(t, d) = count(t, d) / (total words in d)

3. Inverse Document Frequency (IDF):
   Let N = total documents in corpus |D|, and df(t) = number of documents containing t.
   With standard additive smoothing:
     IDF(t) = ln( (1 + N) / (1 + df(t)) ) + 1.0
   
   Intuition:
     - If word t appears in every single document: df(t) = N -> IDF(t) ≈ 1.0 (low weight)
     - If word t appears in only 1 rare failure document: df(t) = 1 -> IDF(t) is high!

4. Document Vector & L2 Normalization:
   Vector v_d = [ TF(t₁, d) * IDF(t₁), ..., TF(t_|V|, d) * IDF(t_|V|) ]
   Normalize to unit length so document length does not bias distance:
     v̂_d = v_d / ||v_d||₂
"""

from __future__ import annotations

from typing import Sequence
import numpy as np

from src.nlp.preprocessing import simple_tokenize, Vocabulary


class TFIDFVectorizerScratch:
    """
    TF-IDF Vectorizer implemented in pure NumPy.

    Parameters
    ----------
    max_features : int, optional
        Maximum number of vocabulary terms to include.
    min_df : int, default=1
        Minimum document frequency required for a term.
    norm : {"l2", None}, default="l2"
        Vector normalization strategy.
    """

    def __init__(
        self,
        max_features: int | None = None,
        min_df: int = 1,
        norm: str | None = "l2",
    ) -> None:
        self.max_features = max_features
        self.min_df = min_df
        self.norm = norm

        self.vocabulary_: dict[str, int] = {}
        self.feature_names_: list[str] = []
        self.idf_diag_: np.ndarray | None = None
        self.is_fitted: bool = False

    def fit(self, corpus: Sequence[str]) -> TFIDFVectorizerScratch:
        """Compute document frequencies and IDF vector from corpus."""
        n_docs = len(corpus)
        tokenized_docs = [simple_tokenize(doc) for doc in corpus]

        # Count document frequency df(t)
        df_counts: dict[str, int] = {}
        for tokens in tokenized_docs:
            unique_in_doc = set(tokens)
            for t in unique_in_doc:
                df_counts[t] = df_counts.get(t, 0) + 1

        # Filter by min_df
        valid_terms = [t for t, count in df_counts.items() if count >= self.min_df]

        # Sort terms alphabetically or by document frequency
        valid_terms.sort(key=lambda t: (-df_counts[t], t))
        if self.max_features is not None:
            valid_terms = valid_terms[: self.max_features]

        # Build vocabulary lookup
        self.feature_names_ = sorted(valid_terms)
        self.vocabulary_ = {term: idx for idx, term in enumerate(self.feature_names_)}

        # Compute smoothed IDF: ln((1 + N) / (1 + df(t))) + 1.0
        n_features = len(self.feature_names_)
        self.idf_diag_ = np.zeros(n_features, dtype=np.float64)
        for i, term in enumerate(self.feature_names_):
            df = df_counts[term]
            self.idf_diag_[i] = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0

        self.is_fitted = True
        return self

    def transform(self, corpus: Sequence[str]) -> np.ndarray:
        """Transform documents into normalized TF-IDF feature matrix (N, V)."""
        if not self.is_fitted or self.idf_diag_ is None:
            raise RuntimeError("Vectorizer is not fitted. Call fit() first.")

        n_docs = len(corpus)
        n_features = len(self.feature_names_)
        X_tfidf = np.zeros((n_docs, n_features), dtype=np.float64)

        for i, doc in enumerate(corpus):
            tokens = simple_tokenize(doc)
            if not tokens:
                continue

            doc_len = len(tokens)
            # Count term frequencies in this document
            for t in tokens:
                if t in self.vocabulary_:
                    feat_idx = self.vocabulary_[t]
                    X_tfidf[i, feat_idx] += 1.0

            # Compute TF = count / doc_len
            X_tfidf[i, :] /= doc_len

            # Multiply by IDF
            X_tfidf[i, :] *= self.idf_diag_

            # L2 Normalization
            if self.norm == "l2":
                row_norm = np.linalg.norm(X_tfidf[i, :])
                if row_norm > 0:
                    X_tfidf[i, :] /= row_norm

        return X_tfidf

    def fit_transform(self, corpus: Sequence[str]) -> np.ndarray:
        """Fit to data, then transform it."""
        return self.fit(corpus).transform(corpus)

    def get_feature_names(self) -> list[str]:
        """Return list of vocabulary feature names."""
        return self.feature_names_
