# An Attack on Facial Soft-Biometric Privacy Enhancement

[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-ee4c2c.svg)](https://pytorch.org/)
[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org/)
[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by/4.0/)

Lab twin of the **black-box soft-biometric (gender) attack** on PE-MIU-protected face embeddings.

**Repository:** https://github.com/biometric-community/an-attack-on-facial-soft-biometric-privacy-enhancement

**Paper:** [10.1109/TBIOM.2022.3172724](https://doi.org/10.1109/TBIOM.2022.3172724) · [arXiv:2111.12405](https://arxiv.org/abs/2111.12405)

## Setup

```bash
bash scripts/setup_env.sh
```

## Train / Eval / Report

```bash
bash scripts/smoke.sh
bash scripts/train_full.sh
```

See [REPORT.md](./REPORT.md) and [DEVIATIONS.md](./DEVIATIONS.md).

## License

[CC BY 4.0](./LICENSE)
