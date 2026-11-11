"""Metric tests on small examples computed by hand."""

import math

import numpy as np
import pytest
import scipy.sparse as sp

from src.metrics import (
    catalog_coverage, dcg_at_n, intra_list_diversity, ndcg_at_n, precision_at_n, recall_at_n,
)

RECS = [10, 20, 30, 40, 50]
RELEVANT = {20, 50, 99}  # hits at ranks 2 and 5; 99 never recommended


def test_precision() -> None:
    assert precision_at_n(RECS, RELEVANT, 5) == pytest.approx(2 / 5)
    assert precision_at_n(RECS, RELEVANT, 2) == pytest.approx(1 / 2)
    assert precision_at_n(RECS, set(), 5) == 0.0


def test_recall() -> None:
    assert recall_at_n(RECS, RELEVANT, 5) == pytest.approx(2 / 3)
    assert recall_at_n(RECS, RELEVANT, 1) == 0.0
    assert recall_at_n(RECS, set(), 5) == 0.0


def test_dcg_by_hand() -> None:
    # 1/log2(3) + 1/log2(6)
    assert dcg_at_n(RECS, RELEVANT, 5) == pytest.approx(1 / math.log2(3) + 1 / math.log2(6))


def test_ndcg_by_hand() -> None:
    # IDCG uses min(|test|=3, N=5) = 3 hits: 1 + 1/log2(3) + 1/log2(4)
    idcg = 1 + 1 / math.log2(3) + 0.5
    assert ndcg_at_n(RECS, RELEVANT, 5) == pytest.approx((1 / math.log2(3) + 1 / math.log2(6)) / idcg)
    # N=2: IDCG has min(3, 2) = 2 hits; one hit at rank 2
    assert ndcg_at_n(RECS, RELEVANT, 2) == pytest.approx((1 / math.log2(3)) / (1 + 1 / math.log2(3)))


def test_ndcg_extremes() -> None:
    assert ndcg_at_n([1, 2, 3], {1, 2, 3}, 3) == pytest.approx(1.0)
    assert ndcg_at_n([1, 7, 8], {1}, 3) == pytest.approx(1.0)  # IDCG with 1 hit
    assert ndcg_at_n([7, 8, 9], {1}, 3) == 0.0
    assert ndcg_at_n([1, 2], set(), 2) == 0.0


def test_catalog_coverage() -> None:
    assert catalog_coverage([[0, 1, 2], [2, 3]], 10) == pytest.approx(0.4)
    assert catalog_coverage([], 10) == 0.0


def test_intra_list_diversity_by_hand() -> None:
    # rows 0 and 1 identical, row 2 orthogonal; unnormalized on purpose
    X = sp.csr_matrix(np.array([[2.0, 0.0], [1.0, 0.0], [0.0, 3.0]]))
    # pair cosines: (0,1)=1, (0,2)=0, (1,2)=0 -> mean 1/3
    assert intra_list_diversity([0, 1, 2], X) == pytest.approx(1 - 1 / 3)
    assert intra_list_diversity([0, 1], X) == pytest.approx(0.0)
    assert intra_list_diversity([0, 2], X) == pytest.approx(1.0)
    assert intra_list_diversity([0], X) == 0.0
