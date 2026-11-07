# DEVIATIONS — Soft-biometric privacy attack

| ID | Item | Paper | Ours | Reason |
|----|------|-------|------|--------|
| D1 | Datasets | LFW, Adience, CelebA, ColorFeret cross-DB | CelebA gallery/probe + LFW utility | Adience/ColorFeret not local |
| D2 | Victims | PFRNet + PE-MIU | PE-MIU only | PFRNet weights unavailable |
| D3 | FR backbone | VGGFace2 / FaceNet | CelebA-trained ResNet-18 | Local data / compute |
| D4 | Gallery size | Up to n=200 reported | Cap gallery/recon for runtime | Lab twin budget |
| D5 | Epochs | N/A (pretrained FR) | 8 CE epochs on CelebA | Train victim locally |
| D6 | Averaging n | n=10, 50 (Tables VI-VIII) | n=11, 51, same as majority (Table V) | One odd n list for all attacks |
