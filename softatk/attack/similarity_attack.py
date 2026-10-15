"""Black-box soft-biometric attack via gallery similarity (paper Alg.)."""

from __future__ import annotations

from typing import Dict, List

import numpy as np


def _aggregate(scores: np.ndarray, genders: np.ndarray, n: int, mode: str) -> int:
    """Infer gender from top-n gallery matches."""
    order = np.argsort(-scores)[:n]
    g = genders[order].astype(np.float64)
    s = scores[order]
    if mode == "majority":
        return int(np.round(g.mean()))
    if mode == "average":
        # soft vote by gender of top-n (unweighted mean of labels)
        return int(np.round(g.mean()))
    if mode == "linear":
        w = np.linspace(n, 1, num=len(g))
        return int(np.round(np.average(g, weights=w)))
    if mode == "log":
        w = 1.0 / np.log2(np.arange(2, len(g) + 2))
        return int(np.round(np.average(g, weights=w)))
    raise ValueError(mode)


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
