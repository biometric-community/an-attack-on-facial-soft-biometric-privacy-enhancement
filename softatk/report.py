"""Figures + REPORT.md."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from softatk.plot_style import FIGSIZE, LINEWIDTH_MAIN, MULTI_SERIES_COLORS, apply_rcparams, apply_style


def plot_train(history, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(
        [h["epoch"] for h in history],
        [h["loss"] for h in history],
        color=MULTI_SERIES_COLORS[0],
        linewidth=LINEWIDTH_MAIN,
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Victim FR training (CelebA)")
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_utility(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    names = ["lfw_clean", "lfw_pemiU"]
    labels = ["LFW clean", "LFW PE-MIU"]
    acc = [eval_res[n]["accuracy"] * 100 for n in names if n in eval_res]
    labs = [labels[i] for i, n in enumerate(names) if n in eval_res]
    x = np.arange(len(acc))
    ax.bar(x, acc, color=MULTI_SERIES_COLORS[0], edgecolor="#002080")
    ax.set_xticks(x)
    ax.set_xticklabels(labs)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Verification utility")
    apply_style(ax, grid_axis="y")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def plot_attack(eval_res: dict, out: Path) -> None:
    apply_rcparams()
    fig, ax = plt.subplots(figsize=FIGSIZE)
    atk = eval_res.get("attack", {})
    prot = atk.get("protected", {})
    unprot = atk.get("unprotected", {})
    # prefer majority mode
    mode = "majority" if "majority" in prot else (next(iter(prot), None))
    if mode is None:
        return
    ns = sorted(prot[mode].keys(), key=lambda x: int(x))
    ax.plot(
        [int(n) for n in ns],
        [prot[mode][n] * 100 for n in ns],
        color=MULTI_SERIES_COLORS[0],
        linewidth=LINEWIDTH_MAIN,
        marker="o",
        label="PE-MIU protected",
    )
    if mode in unprot:
        ax.plot(
            [int(n) for n in ns],
            [unprot[mode][n] * 100 for n in ns],
            color=MULTI_SERIES_COLORS[1],
            linewidth=LINEWIDTH_MAIN,
            marker="s",
            label="unprotected",
        )
    ax.set_xlabel("Top-n gallery matches")
    ax.set_ylabel("Gender attack accuracy (%)")
    ax.set_title(f"Soft-biometric attack ({mode})")
    ax.legend()
    apply_style(ax)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out, format="svg")
    plt.close(fig)


def write_report(cfg, train_s, eval_res, path: Path) -> None:
    protocol = cfg["train"].get("protocol", "full")
    atk = eval_res.get("attack", {})
    lines = [
        f"# REPORT: Soft-biometric privacy attack — {protocol}",
        "",
        f"**Protocol:** `{protocol}`  ",
        f"**Victim privacy:** PE-MIU (block_size={eval_res.get('block_size')})  ",
        "",
        "## Verification utility",
        "",
        "| Setting | Accuracy | EER | Pairs |",
        "|---------|----------|-----|-------|",
    ]
    for key, name in [("lfw_clean", "LFW clean"), ("lfw_pemiU", "LFW PE-MIU")]:
        if key not in eval_res:
            continue
        r = eval_res[key]
        lines.append(
            f"| {name} | {r['accuracy']*100:.2f}% | {r['eer']*100:.2f}% | {r['n_pairs']} |"
        )
    lines += [
        "",
        "## Gender attack (CelebA subject-disjoint)",
        "",
        f"- Gallery: {atk.get('n_gallery')} · Probe: {atk.get('n_probe')}",
        f"- Reconstruct match: `{atk.get('reconstruct_match')}`",
        "",
        "| Mode / n | " + " | ".join(sorted(atk.get("protected", {}).get("majority", {}).keys(), key=int)) + " |",
        "|----------|" + "|".join(["------"] * len(atk.get("protected", {}).get("majority", {}))) + "|",
    ]
    prot = atk.get("protected", {})
    for mode, by_n in prot.items():
        ns = sorted(by_n.keys(), key=int)
        cells = " | ".join(f"{by_n[n]*100:.1f}%" for n in ns)
        lines.append(f"| protected/{mode} | {cells} |")
    unprot = atk.get("unprotected", {})
    for mode, by_n in unprot.items():
        ns = sorted(by_n.keys(), key=int)
        cells = " | ".join(f"{by_n[n]*100:.1f}%" for n in ns)
        lines.append(f"| unprotected/{mode} | {cells} |")
    lines += [
        "",
        "## Training",
        "",
        f"- Epochs: {train_s.get('epochs')}",
        f"- Train images: {train_s.get('n_train')}",
        f"- Identities: {train_s.get('num_classes')}",
        f"- Elapsed: {train_s.get('elapsed_sec', 0):.2f} s",
        "",
        "## Figures",
        "",
        "![train_loss](figures/train_loss.svg)",
        "",
        "![utility](figures/utility_acc.svg)",
        "",
        "![attack](figures/attack_acc.svg)",
        "",
        "## Reproduce",
        "",
        "```bash",
        "bash scripts/smoke.sh",
        "bash scripts/train_full.sh",
        "```",
        "",
        "## Notes",
        "",
        "- PE-MIU from pterhoer/PrivacyPreservingFaceRecognition; attack per paper Alg.",
        "- PFRNet victim omitted (weights unavailable); CelebA gallery/probe (D1).",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/default.yaml")
    args = ap.parse_args()
    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    log_dir = Path(cfg["paths"]["log_dir"])
    fig_dir = Path(cfg["paths"]["figure_dir"])
    train_s = json.loads((log_dir / "train_summary.json").read_text())
    eval_res = json.loads((log_dir / "eval_results.json").read_text())
    plot_train(train_s.get("history", []), fig_dir / "train_loss.svg")
    plot_utility(eval_res, fig_dir / "utility_acc.svg")
    plot_attack(eval_res, fig_dir / "attack_acc.svg")
    tracked = _ROOT / "figures"
    tracked.mkdir(exist_ok=True)
    for name in ("train_loss.svg", "utility_acc.svg", "attack_acc.svg"):
        src = fig_dir / name
        if src.exists():
            (tracked / name).write_bytes(src.read_bytes())
    write_report(cfg, train_s, eval_res, _ROOT / "REPORT.md")
    print(f"Wrote {_ROOT / 'REPORT.md'}")


if __name__ == "__main__":
    main()
