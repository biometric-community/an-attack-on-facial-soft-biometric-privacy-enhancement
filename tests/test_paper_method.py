"""The attack and metrics, checked against the definitions in the paper (Sec. III-B, IV-C)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from softatk.attack.similarity_attack import _aggregate
from softatk.eval import best_acc_eer
from softatk.privacy.pe_miu import PrivacyEnhancingMIU


def paper_decision(scores: np.ndarray, genders: np.ndarray, n: int, mode: str) -> int:
    """Decision rule of Sec. III-B: argmax over per-attribute evidence c(a)."""
    if mode == "majority":
        top = genders[np.argsort(-scores)[:n]]
        return int(np.bincount(top, minlength=2).argmax())
    evidence = []
    for attribute in (0, 1):
        s = np.sort(scores[genders == attribute])[::-1][:n]
        i = np.arange(1, len(s) + 1)
        if mode == "average":
            w = np.ones(len(s))
        elif mode == "linear":
            w = 1.0 - i / (n + 1)
        else:
            w = -np.log(i / (n + 1))
        evidence.append(float(np.mean(w * s)))
    return int(np.argmax(evidence))


@pytest.mark.parametrize(
    "mode,n",
    [("majority", 5), ("majority", 11), ("average", 10), ("linear", 10), ("log", 10), ("log", 50)],
)
def test_decision_rule_matches_paper(mode, n):
    rng = np.random.RandomState(0)
    mismatches = 0
    for _ in range(300):
        genders = rng.randint(0, 2, size=200)
        scores = rng.normal(size=200) + 0.15 * genders
        mismatches += _aggregate(scores, genders, n, mode) != paper_decision(scores, genders, n, mode)
    assert mismatches == 0


def test_averaging_uses_similarity_scores_not_label_counts():
    """Attribute 1 has the higher mean score although attribute 0 wins the top-3 vote."""
    genders = np.array([0, 0, 0, 1, 1, 1])
    scores = np.array([0.80, 0.79, 0.10, 0.99, 0.78, 0.77])
    assert paper_decision(scores, genders, 3, "average") == 1
    assert _aggregate(scores, genders, 3, "average") == 1


def test_majority_vote_is_only_configured_for_odd_n(default_cfg):
    """Sec. V-B: the majority vote uses odd n, so a vote can never tie."""
    if "majority" in default_cfg["attack"]["modes"]:
        assert [n for n in default_cfg["attack"]["n_list"] if n % 2 == 0] == []


def test_pe_miu_shuffle_only_permutes_blocks():
    v = np.random.RandomState(1).normal(size=128)
    pe = PrivacyEnhancingMIU(block_size=16)
    shuffled = pe.shuffle(v)
    assert sorted(map(tuple, shuffled.reshape(-1, 16))) == sorted(map(tuple, v.reshape(-1, 16)))


def test_pe_miu_reconstruct_recovers_a_shuffled_template():
    v = np.random.RandomState(2).normal(size=128)
    pe = PrivacyEnhancingMIU(block_size=16)
    assert pe.cos_sim(v, pe.reconstruct(v, pe.shuffle(v))) == pytest.approx(1.0)


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def test_eer_is_the_point_where_fmr_equals_fnmr():
    """Sec. IV-C: EER is where FMR and FNMR are equal, not the minimum of their mean."""
    lo, hi = 0.0, 1.0
    for _ in range(80):
        t = 0.5 * (lo + hi)
        fmr, fnmr = 1.0 - _norm_cdf(t), _norm_cdf((t - 1.0) / 0.1)
        lo, hi = (t, hi) if fmr > fnmr else (lo, t)
    expected = 1.0 - _norm_cdf(t)

    rng = np.random.RandomState(3)
    scores = np.concatenate([rng.normal(1.0, 0.1, 200000), rng.normal(0.0, 1.0, 200000)])
    labels = np.concatenate([np.ones(200000, dtype=int), np.zeros(200000, dtype=int)])
    _, eer = best_acc_eer(scores, labels)
    assert eer == pytest.approx(expected, abs=0.01)
