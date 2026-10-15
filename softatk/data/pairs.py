"""LFW verification pairs (utility eval)."""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


def resolve_lfw(cfg: dict) -> Path:
    root = Path(cfg["data"].get("lfw_root", "../../datasets/lfw"))
    if not root.is_absolute():
        root = (Path(__file__).resolve().parents[2] / root).resolve()
    if not (root / "lfw").is_dir() or not (root / "pairs.txt").is_file():
        raise FileNotFoundError(root)
    return root


def _tf(image_size: int) -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
        ]
    )


def _subsample(pairs, max_pairs: Optional[int]):
    if max_pairs is None or len(pairs) <= max_pairs:
        return pairs
    pos = [p for p in pairs if p[2] == 1]
    neg = [p for p in pairs if p[2] == 0]
    n = max_pairs // 2
    out = pos[:n] + neg[:n]
    return out[:max_pairs]


def _parse_lfw(pairs_path: Path, max_pairs: Optional[int] = None):
    out = []
    for line in pairs_path.read_text().strip().splitlines()[1:]:
        toks = line.split()
        if len(toks) == 3:
            name, n1, n2 = toks
            a = f"{name}/{name}_{int(n1):04d}.jpg"
            b = f"{name}/{name}_{int(n2):04d}.jpg"
            out.append((a, b, 1))
        elif len(toks) == 4:
            n1, i1, n2, i2 = toks
            a = f"{n1}/{n1}_{int(i1):04d}.jpg"
            b = f"{n2}/{n2}_{int(i2):04d}.jpg"
            out.append((a, b, 0))
    return _subsample(out, max_pairs)


class PairDataset(Dataset):
    def __init__(self, img_root: Path, pairs: List[Tuple[str, str, int]], image_size: int = 112):
        self.img_root = Path(img_root)
        self.pairs = pairs
        self.transform = _tf(image_size)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index: int):
        a, b, label = self.pairs[index]
        img_a = self.transform(Image.open(self.img_root / a).convert("RGB"))
        img_b = self.transform(Image.open(self.img_root / b).convert("RGB"))
        return img_a, img_b, label


def build_lfw(cfg, image_size, max_pairs):
    root = resolve_lfw(cfg)
    return PairDataset(root / "lfw", _parse_lfw(root / "pairs.txt", max_pairs), image_size)
