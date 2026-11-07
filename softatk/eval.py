"""LFW utility (clean vs PE-MIU) + CelebA gender similarity attack."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from tqdm import tqdm

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from softatk.attack.similarity_attack import attack_success, attack_success_pemiU
from softatk.data.celeba import CelebAGenderDataset, resolve_celeba_root
from softatk.data.pairs import build_lfw
from softatk.models.fr_net import FaceEmbedder
from softatk.privacy.pe_miu import PrivacyEnhancingMIU


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def _device(cfg: dict, override: Optional[str] = None) -> torch.device:
    name = override or cfg.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    if str(name).startswith("cuda") and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(name)


def equal_error_rate(scores: np.ndarray, labels: np.ndarray) -> float:
    """Error rate at the threshold where FMR and FNMR are closest."""
    mated = labels[np.argsort(scores)] == 1
    if mated.all() or not mated.any():
        return 1.0
    fnmr = np.cumsum(mated) / mated.sum()
    fmr = 1.0 - np.cumsum(~mated) / (~mated).sum()
    i = int(np.argmin(np.abs(fmr - fnmr)))
    return float(0.5 * (fmr[i] + fnmr[i]))


def best_acc_eer(scores: np.ndarray, labels: np.ndarray) -> Tuple[float, float]:
    thresholds = np.linspace(scores.min(), scores.max(), 200)
    best_acc = 0.0
    for t in thresholds:
        pred = (scores >= t).astype(np.int32)
        best_acc = max(best_acc, float((pred == labels).mean()))
    return best_acc, equal_error_rate(scores, labels)


@torch.no_grad()
def embed_loader(model, loader, device) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    embs, labels, genders = [], [], []
    for batch in tqdm(loader, desc="embed", leave=False):
        if len(batch) == 3 and not torch.is_tensor(batch[2]):
            # pair dataset: a,b,y
            a, b, y = batch
            ea = model.embed(a.to(device))
            eb = model.embed(b.to(device))
            embs.append((ea * eb).sum(dim=1).cpu().numpy())
            labels.append(y.numpy())
            genders = None
        else:
            images, lab, gen = batch
            e = model.embed(images.to(device)).cpu().numpy()
            embs.append(e)
            labels.append(lab.numpy())
            genders.append(gen.numpy())
    if genders is None:
        return np.concatenate(embs), np.concatenate(labels), None
    return np.concatenate(embs), np.concatenate(labels), np.concatenate(genders)


@torch.no_grad()
def eval_lfw_clean(model, cfg, device) -> Dict:
    image_size = int(cfg["model"].get("image_size", 112))
    max_p = cfg["eval"].get("max_lfw_pairs")
    max_p = int(max_p) if max_p is not None else None
    ds = build_lfw(cfg, image_size, max_p)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["eval"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
    )
    scores, labels = [], []
    model.eval()
    for a, b, y in tqdm(loader, desc="lfw_clean", leave=False):
        ea = model.embed(a.to(device))
        eb = model.embed(b.to(device))
        scores.append((ea * eb).sum(dim=1).cpu().numpy())
        labels.append(y.numpy())
    s = np.concatenate(scores)
    y = np.concatenate(labels)
    acc, eer = best_acc_eer(s, y)
    print(f"[lfw_clean] acc={acc:.4f} eer={eer:.4f} n={len(y)}")
    return {"accuracy": acc, "eer": eer, "n_pairs": int(len(y))}


@torch.no_grad()
def eval_lfw_pemiU(model, pe_miu, cfg, device) -> Dict:
    """PE-MIU utility: shuffle both, reconstruct probe→ref, cosine."""
    image_size = int(cfg["model"].get("image_size", 112))
    max_p = cfg["eval"].get("max_lfw_pairs")
    max_p = int(max_p) if max_p is not None else None
    ds = build_lfw(cfg, image_size, max_p)
    loader = DataLoader(
        ds,
        batch_size=int(cfg["eval"].get("batch_size", 64)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
    )
    scores, labels = [], []
    model.eval()
    for a, b, y in tqdm(loader, desc="lfw_pemiU", leave=False):
        ea = model.embed(a.to(device)).cpu().numpy()
        eb = model.embed(b.to(device)).cpu().numpy()
        ea = pe_miu.shuffle(ea)
        eb = pe_miu.shuffle(eb)
        for i in range(len(y)):
            adj = pe_miu.reconstruct(ea[i], eb[i])
            scores.append(pe_miu.cos_sim(ea[i], adj))
            labels.append(int(y[i]))
    s = np.array(scores, dtype=np.float64)
    y = np.array(labels)
    acc, eer = best_acc_eer(s, y)
    print(f"[lfw_pemiU] acc={acc:.4f} eer={eer:.4f} n={len(y)}")
    return {"accuracy": acc, "eer": eer, "n_pairs": int(len(y))}


@torch.no_grad()
def run_attack(model, pe_miu, cfg, device) -> Dict:
    root = resolve_celeba_root(cfg)
    image_size = int(cfg["model"].get("image_size", 112))
    seed = int(cfg["train"].get("seed", 42))
    max_gal = cfg["attack"].get("max_gallery")
    max_probe = cfg["attack"].get("max_probe")
    max_gal = int(max_gal) if max_gal is not None else None
    max_probe = int(max_probe) if max_probe is not None else None

    # subject-disjoint: train identities = gallery, val = probes
    gal_ds = CelebAGenderDataset(
        root,
        split="train",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_gal,
        seed=seed,
        train=False,
    )
    probe_ds = CelebAGenderDataset(
        root,
        split="val",
        image_size=image_size,
        train_ratio=float(cfg["data"].get("train_ratio", 0.7)),
        max_n=max_probe,
        seed=seed,
        train=False,
    )
    gal_loader = DataLoader(
        gal_ds,
        batch_size=int(cfg["eval"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
    )
    probe_loader = DataLoader(
        probe_ds,
        batch_size=int(cfg["eval"].get("batch_size", 128)),
        shuffle=False,
        num_workers=int(cfg["train"].get("num_workers", 4)),
    )
    g_emb, _, g_gen = embed_loader(model, gal_loader, device)
    p_emb, _, p_gen = embed_loader(model, probe_loader, device)

    # protect with PE-MIU shuffle
    g_prot = pe_miu.shuffle(g_emb)
    p_prot = pe_miu.shuffle(p_emb)

    n_list: List[int] = list(cfg["attack"].get("n_list", [1, 5, 10, 50]))
    modes: List[str] = list(cfg["attack"].get("modes", ["majority", "average", "linear", "log"]))
    use_recon = bool(cfg["attack"].get("reconstruct_match", True))
    max_g_recon = int(cfg["attack"].get("max_gallery_recon", 400))

    if use_recon:
        attack = attack_success_pemiU(
            pe_miu,
            p_prot,
            p_gen,
            g_prot,
            g_gen,
            n_list,
            modes,
            max_gallery_for_recon=max_g_recon,
        )
    else:
        attack = attack_success(p_prot, p_gen, g_prot, g_gen, n_list, modes)

    # baseline: attack on unprotected embeddings
    baseline = attack_success(p_emb, p_gen, g_emb, g_gen, n_list, modes)

    print("[attack] PE-MIU protected:", json.dumps(attack, indent=2))
    print("[attack] unprotected baseline:", json.dumps(baseline, indent=2))
    return {
        "n_gallery": int(len(g_gen)),
        "n_probe": int(len(p_gen)),
        "protected": attack,
        "unprotected": baseline,
        "reconstruct_match": use_recon,
    }


def evaluate(cfg: dict, device_override: Optional[str] = None) -> Dict:
    device = _device(cfg, device_override)
    ckpt = torch.load(
        Path(cfg["paths"]["checkpoint_dir"]) / "last.pt",
        map_location=device,
        weights_only=False,
    )
    model = FaceEmbedder(int(ckpt["num_classes"]), pretrained=False).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    block = int(cfg["privacy"].get("block_size", 16))
    pe_miu = PrivacyEnhancingMIU(block_size=block)

    results = {
        "lfw_clean": eval_lfw_clean(model, cfg, device),
        "lfw_pemiU": eval_lfw_pemiU(model, pe_miu, cfg, device),
        "attack": run_attack(model, pe_miu, cfg, device),
        "block_size": block,
    }
    log_dir = Path(cfg["paths"]["log_dir"])
    log_dir.mkdir(parents=True, exist_ok=True)
    out = log_dir / "eval_results.json"
    with open(out, "w") as f:
        json.dump(results, f, indent=2)
        f.write("\n")
    print(f"Wrote {out}")
    return results


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--device", default=None)
    args = ap.parse_args()
    evaluate(load_config(args.config), device_override=args.device)


if __name__ == "__main__":
    main()
