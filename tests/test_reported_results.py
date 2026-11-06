"""REPORT.md, the committed output of the full run, against the published results."""

from __future__ import annotations

import re

import pytest


def utility_row(report: str, name: str) -> tuple[float, float]:
    match = re.search(rf"\| {name} \| ([\d.]+)% \| ([\d.]+)% \|", report)
    assert match, f"{name} row missing from REPORT.md"
    return float(match.group(1)), float(match.group(2))


def attack_row(report: str, name: str) -> list[float]:
    match = re.search(rf"\| {name} \| (.+) \|", report)
    assert match, f"{name} row missing from REPORT.md"
    return [float(cell.strip().rstrip("%")) for cell in match.group(1).split("|")]


def test_lfw_original_eer_matches_table_iii(report, targets):
    _, eer = utility_row(report, "LFW clean")
    expected = targets["table_iii_lfw_pe_miu"]["original_eer"]
    assert eer == pytest.approx(expected, abs=targets["tolerance_pp"])


def test_lfw_protected_eer_matches_table_iii(report, targets):
    _, eer = utility_row(report, "LFW PE-MIU")
    expected = targets["table_iii_lfw_pe_miu"]["protected_eer"]
    assert eer == pytest.approx(expected, abs=targets["tolerance_pp"])


def test_pe_miu_keeps_verification_performance(report):
    """Table III: PE-MIU moves LFW EER by 0.09 points, 'slightly degraded'."""
    _, clean = utility_row(report, "LFW clean")
    _, protected = utility_row(report, "LFW PE-MIU")
    assert abs(protected - clean) <= 1.0


@pytest.mark.parametrize("mode", ["majority", "average", "linear", "log"])
def test_attack_stays_above_lowest_published_rate(report, targets, mode):
    """The attack succeeds: every n beats the lowest PE-MIU rate in Tables V-VIII."""
    assert min(attack_row(report, f"protected/{mode}")) >= targets["attack_success_pe_miu"]["min"]


def test_averaging_and_majority_are_different_attacks(report):
    """Tables V and VI differ; identical rows mean one rule ran twice."""
    assert attack_row(report, "protected/majority") != attack_row(report, "protected/average")


def test_protected_and_unprotected_attacks_use_the_same_gallery(default_cfg):
    """The baseline uses max_gallery entries, so the protected attack must not use fewer."""
    attack = default_cfg["attack"]
    assert attack["max_gallery_recon"] >= attack["max_gallery"]
