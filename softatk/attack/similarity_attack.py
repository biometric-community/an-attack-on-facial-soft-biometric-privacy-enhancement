"""Black-box soft-biometric attack via gallery similarity (paper Alg.)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _aggregate(scores: np.ndarray, genders: np.ndarray, n: int, mode: str) -> int:
    """Infer gender from top-n gallery matches."""
    if mode == "majority":
        top = genders[np.argsort(-scores)[:n]].astype(np.int64)
        return int(np.bincount(top, minlength=2).argmax())
    if mode not in ("average", "linear", "log"):
        raise ValueError(mode)
    # Sec. III-B: one list of top-n similarity scores per attribute, argmax of its evidence
    evidence = []
    for attribute in (0, 1):
        s = np.sort(scores[genders == attribute])[::-1][:n]
        if len(s) == 0:
            evidence.append(-np.inf)
            continue
        i = np.arange(1, len(s) + 1)
        if mode == "average":
            w = np.ones(len(s))
        elif mode == "linear":
            w = 1.0 - i / (n + 1)
        else:
            w = -np.log(i / (n + 1))
        evidence.append(float(np.mean(w * s)))
    return int(np.argmax(evidence))


def attack_success(
    probe_embs: np.ndarray,
    probe_gender: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_gender: np.ndarray,
    n_list: List[int],
    modes: List[str],
) -> Dict:
    """
    Cosine similarity gallery attack.
    For PE-MIU, embeddings should already be in comparable form
    (e.g. probe reconstructed to each gallery entry — expensive);
    here we use shuffled embeddings with direct cosine as a lite
    approximation when reconstruct_per_gallery=False.
    """
    # normalize
    pe = probe_embs / (np.linalg.norm(probe_embs, axis=1, keepdims=True) + 1e-8)
    ge = gallery_embs / (np.linalg.norm(gallery_embs, axis=1, keepdims=True) + 1e-8)
    sim = pe @ ge.T  # (P, G)
    out = {}
    for mode in modes:
        out[mode] = {}
        for n in n_list:
            n_use = min(n, gallery_embs.shape[0])
            preds = [
                _aggregate(sim[i], gallery_gender, n_use, mode) for i in range(len(probe_gender))
            ]
            acc = float((np.array(preds) == probe_gender).mean())
            out[mode][str(n)] = acc
    return out


def attack_success_pemiU(
    pe_miu,
    probe_embs: np.ndarray,
    probe_gender: np.ndarray,
    gallery_embs: np.ndarray,
    gallery_gender: np.ndarray,
    n_list: List[int],
    modes: List[str],
    max_gallery_for_recon: int = 500,
) -> Dict:
    """
    Faithful PE-MIU matching: for each probe, reconstruct probe to each
    gallery template then cosine (paper-style). Cap gallery for runtime.
    """
    G = min(len(gallery_embs), max_gallery_for_recon)
    ge = gallery_embs[:G]
    gg = gallery_gender[:G]
    sim = np.zeros((len(probe_embs), G), dtype=np.float64)
    for i, p in enumerate(probe_embs):
        for j, g in enumerate(ge):
            adj = pe_miu.reconstruct(g, p)
            sim[i, j] = pe_miu.cos_sim(g, adj)
    out = {}
    for mode in modes:
        out[mode] = {}
        for n in n_list:
            n_use = min(n, G)
            preds = [_aggregate(sim[i], gg, n_use, mode) for i in range(len(probe_gender))]
            acc = float((np.array(preds) == probe_gender).mean())
            out[mode][str(n)] = acc
    return out
