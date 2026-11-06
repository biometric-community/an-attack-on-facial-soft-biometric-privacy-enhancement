"""The paper's findings, reproduced with this repo's code on synthetic embeddings.

Embeddings carry an identity component and a shared gender direction, which is
the "broad homogeneity" the attack exploits (Sec. III-A).
"""

from __future__ import annotations

import random

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from softatk.attack.similarity_attack import attack_success, attack_success_pemiU
from softatk.eval import best_acc_eer
from softatk.privacy.pe_miu import PrivacyEnhancingMIU

DIM = 128


def make_embeddings(n: int, seed: int, gender_dir: np.ndarray):
    rng = np.random.RandomState(seed)
    gender = np.arange(n) % 2
    x = rng.normal(size=(n, DIM)) + 1.2 * np.where(gender[:, None] == 1, 1.0, -1.0) * gender_dir
    return x / np.linalg.norm(x, axis=1, keepdims=True), gender


@pytest.fixture(scope="module")
def data():
    random.seed(0)
    # Blocks of the gender direction sum to zero, so block shuffling hides it from a
    # linear classifier, matching the near-chance classifiers of Table IV.
    blocks = np.random.RandomState(99).normal(size=(DIM // 16, 16))
    gender_dir = (blocks - blocks.mean(axis=0)).reshape(-1)
    pe = PrivacyEnhancingMIU(block_size=16)
    gallery, gallery_gender = make_embeddings(120, 1, gender_dir)
    probe, probe_gender = make_embeddings(60, 2, gender_dir)
    return pe, gallery, gallery_gender, probe, probe_gender


def test_attack_recovers_gender_from_pe_miu_templates(data):
    """Sec. V-B: PE-MIU is highly vulnerable; Table IX reports around 88%."""
    pe, gallery, gallery_gender, probe, probe_gender = data
    result = attack_success_pemiU(
        pe, pe.shuffle(probe), probe_gender, pe.shuffle(gallery), gallery_gender, [5], ["majority"]
    )
    assert result["majority"]["5"] >= 0.75


def test_attack_beats_a_classifier_on_protected_templates(data):
    """Table IV vs V: classifiers fall to near chance while the attack still succeeds."""
    pe, gallery, gallery_gender, probe, probe_gender = data
    protected_gallery, protected_probe = pe.shuffle(gallery), pe.shuffle(probe)
    clf = LogisticRegression(max_iter=2000).fit(protected_gallery, gallery_gender)
    classifier = clf.score(protected_probe, probe_gender)
    attack = attack_success_pemiU(
        pe, protected_probe, probe_gender, protected_gallery, gallery_gender, [5], ["majority"]
    )["majority"]["5"]
    assert classifier <= 0.70
    assert attack > classifier


def test_gender_is_readable_from_unprotected_templates(data):
    """Table IV: on original templates gender is predicted well."""
    _, gallery, gallery_gender, probe, probe_gender = data
    clf = LogisticRegression(max_iter=2000).fit(gallery, gallery_gender)
    assert clf.score(probe, probe_gender) >= 0.85
    result = attack_success(probe, probe_gender, gallery, gallery_gender, [5], ["majority"])
    assert result["majority"]["5"] >= 0.85


def test_pe_miu_preserves_verification():
    """Eq. (3) and Table III: protected and original error rates stay close."""
    random.seed(1)
    rng = np.random.RandomState(4)
    pe = PrivacyEnhancingMIU(block_size=16)
    clean, protected, labels = [], [], []
    for k in range(600):
        a = rng.normal(size=DIM)
        same = k % 2 == 0
        b = a + 0.9 * rng.normal(size=DIM) if same else rng.normal(size=DIM)
        clean.append(pe.cos_sim(a, b))
        sa, sb = pe.shuffle(a), pe.shuffle(b)
        protected.append(pe.cos_sim(sa, pe.reconstruct(sa, sb)))
        labels.append(int(same))
    labels = np.array(labels)
    _, err_clean = best_acc_eer(np.array(clean), labels)
    _, err_protected = best_acc_eer(np.array(protected), labels)
    assert abs(err_protected - err_clean) <= 0.03
