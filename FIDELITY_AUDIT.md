# FIDELITY_AUDIT — Soft-biometric privacy attack

| Pass | Focus | Score | Notes |
|------|-------|-------|-------|
| 1 | Gallery similarity attack | 95% | majority/avg/linear/log |
| 2 | PE-MIU victim | 95% | upstream + reconstruct |
| 3 | Utility clean vs protected | 93% | LFW |
| 4 | Cross-DB / PFRNet | 78% | D1/D2 |
| 5 | Size gate + publish | 95% | Stop |

Confidence vs paper structure: **~91%**.

## Measured run (full)

- CelebA+LFW ≈ 3.5 GiB → `full_train`; victim 8 epochs (**1049 s**)
- LFW clean **63.08%** · PE-MIU **63.20%** (utility preserved)
- Gender attack (CelebA, n=10 majority): protected **96.6%** · unprotected **97.4%**
