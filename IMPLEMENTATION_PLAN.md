# Implementation Plan — Soft-biometric privacy attack

1. Train ResNet-18 FR on CelebA (victim embedder).
2. PE-MIU (block shuffle + Hungarian reconstruct) from public code.
3. LFW utility: clean vs PE-MIU; CelebA subject-disjoint gender attack (majority/average/linear/log).
4. Smoke → size gate → full → publish.
