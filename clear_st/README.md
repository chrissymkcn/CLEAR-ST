# clear_st / CLEARmodel

A Pyro + PyTorch implementation of a probabilistic spatial decontamination model
that learns a denoising autoencoder for clean expression and a niche-aware,
anisotropic diffusion (graph Laplacian) forward operator that explains observed
contamination in spatial transcriptomics data.

**Design goals**
- **Probabilistic**: models uncertainty in per-spot diffusion parameters via Pyro.
- **Interpretable**: exposes niche-specific anisotropic kernel parameters and
	spot-wise diffusion rates.
- **Flexible**: supports image-guided contrastive regularization and optional
	X_init pre-decontamination input.

**Quick Tags**: `probabilistic-model`, `spatial-transcriptomics`, `decontamination`,
`diffusion`, `graph-laplacian`, `Pyro`, `PyTorch`, `denoising-autoencoder`, `image-integration`

**Quick Start**

- Install the environment with your preferred package manager. Required libraries
	include: `torch`, `pyro-ppl`, `scanpy`, `squidpy`, `scikit-learn`, and their
	dependencies.
- Example: run the example runner `run_CLEARmodel.py` (edit paths/args as needed):

```bash
python run_CLEARmodel.py --data_input /path/to/slide.h5ad --epochs 1000 --n_genes 2000
```

**Python API (minimal example)**

```python
import torch
from clear_st.clear_model import CLEARmodel

# Prepare tensors (examples):
Y = torch.tensor(..., dtype=torch.float32)                 # (n_spots, n_genes)
spatial_coords = torch.tensor(..., dtype=torch.float32)    # (n_spots, 2)
spot_types = torch.tensor(..., dtype=torch.long)           # (n_spots,)
niche_labels = torch.tensor(..., dtype=torch.long)        # (n_spots,)
diffrate_labels = torch.tensor(..., dtype=torch.long)     # (n_spots,)
spatial_con = ...   # sparse adjacency (torch.sparse_coo_tensor)
spatial_dist = ...  # sparse distance matrix (torch.sparse_coo_tensor)

model = CLEARmodel(
		Y=Y,
		spatial_coords=spatial_coords,
		spot_types=spot_types,
		niche_labels=niche_labels,
		diffrate_labels=diffrate_labels,
		spatial_con=spatial_con,
		spatial_dist=spatial_dist,
		image_features=None,  # optional
)

# Pretrain AE (recommended) and run SVI/guide per examples in `run_CLEARmodel.py`
model.pretrain_autoencoder(n_epochs=200)
# then use `guide()` / SVI loop per runner
```

**Inputs / Outputs**
- **Inputs**: `Y` (observed counts), `spatial_coords`, `spatial_con` (adjacency),
	`spatial_dist`, optional `image_features`, optional `X` (X_init), and spot-level
	categorical arrays (`spot_types`, `niche_labels`, `diffrate_labels`).
- **Main outputs**: deterministic site `true_expression_expanded` (decoded clean
	expression), `contaminated_expression` (predicted observed), learned
	per-spot diffusion rates, and niche kernel parameters. The runner also
	writes `restored_adata.h5ad` when invoked via `run_CLEARmodel.py`.

**Recommended diagnostics & evaluation**
- **Fit metrics**: Spearman/Pearson between predicted and observed spot-wise
	expression; JS divergence on normalized gene distributions.
- **Spatial checks**: Moran's I on cell-type / abundance fields; visualize
	learned diffusion rates across the tissue and compare to background regions.
- **Biological checks**: marker-gene spatial patterns before/after correction.
- **Stability**: inspect posterior variance of diffusion parameters from Pyro.

**Strengths & Limitations**
- Strengths: explicit generative diffusion model, interpretable kernel params,
	support for image-guided regularization, uncertainty quantification via Pyro.
- Limitations: computationally heavy for large slides, requires careful graph
	setup and hyperparameter tuning; AE + VI training may need monitoring.

**Suggested next steps**
- Add an `from_anndata()` helper to convert `scanpy.AnnData` → tensors (I can
	add this helper if you want).
- Include a small diagnostic writer that saves per-spot diffusion rates and
	niche kernel summaries to `model_dir` for downstream `eval.py` compatibility.

---
For details and usage patterns see `run_CLEARmodel.py` and
`clear_model.py`.

