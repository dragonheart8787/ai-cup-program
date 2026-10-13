"""PLACEHOLDER reason text selection -- not a real explanation model.

For each defect class the single sentence with the highest mean character n-gram
cosine similarity to all training sentences of that class is used for every
prediction. This only keeps the submission format valid (reason must be non-empty).
"""
from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

logger = logging.getLogger(__name__)

PLACEHOLDER_NOTE = "PLACEHOLDER reason: most central training sentence of the class, identical for every prediction"


def most_central_sentence(texts: Sequence[str]) -> str:
    """Sentence (among distinct texts) with the highest mean cosine to ALL sentences (duplicates counted)."""
    clean = [t for t in texts if isinstance(t, str) and t.strip()]
    if not clean:
        raise ValueError("no reason texts to choose from")
    vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 4))
    x = vec.fit_transform(clean)  # rows are L2-normalised -> dot product is cosine
    uniq = sorted(set(clean))
    xu = vec.transform(uniq)
    mean_sim = np.asarray((xu @ x.T).mean(axis=1)).ravel()
    best = int(np.argmax(mean_sim))  # ties -> first in sorted order (deterministic)
    logger.info("placeholder reason chosen among %d distinct / %d sentences (mean cos %.3f)", len(uniq), len(clean), mean_sim[best])
    return uniq[best]
