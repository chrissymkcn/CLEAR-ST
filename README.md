# CLEAR-ST (CLEARmodel / undiff)

**CLEAR‑ST** is a probabilistic, physics‑informed framework for spatial transcriptomics
decontamination. It models lateral mRNA diffusion with a graph‑Laplacian forward operator
and learns clean expression with a denoising autoencoder, using Pyro for uncertainty‑aware
inference.

---

## Highlights

- **Probabilistic** diffusion rate inference (Pyro)
- **Physics‑informed** anisotropic diffusion operator (graph Laplacian)
- **Image‑aware** optional niche regularization from histology

---

## Installation

From repo root after cloning:

````bash
pip install -e .