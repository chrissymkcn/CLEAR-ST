
#### Below is version 2, with niche-specific weights only ####
# If
# λ∼Gamma(α,β)
# and
# x∼Poisson(λ)
# then
# x∼NegativeBinomial(r=α,p=β/(β+1))

# This means:

# Gamma-Poisson parameters:
# concentration = α (shape)
# rate = β (rate)

# NB parameters:
# total_count = r = α
# probs = p = β / (β + 1)

from sklearn.preprocessing import scale
import torch
import pyro
import pyro.distributions as dist
from pyro.nn import PyroModule, PyroSample, PyroParam
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pyro.poutine as poutine


def _graph_laplacian_from_sparse(rows, cols, weights, n):
    """
    Compute the graph Laplacian L = D - W as a sparse COO tensor, preserving
    gradient flow through ``weights``.

    Both the off-diagonal entries (-weights) and the diagonal degree entries
    are assembled into a **single** sparse_coo_tensor so that autograd can
    differentiate through the values tensor end-to-end.

    Parameters
    ----------
    rows, cols : LongTensor  – edge source/target indices (self-loops excluded)
    weights    : FloatTensor – differentiable edge weights (requires_grad OK)
    n          : int         – number of nodes

    Returns
    -------
    Sparse COO tensor of shape (n, n) representing the graph Laplacian.
    """
    device = weights.device
    diag_idx = torch.arange(n, dtype=torch.long, device=device)

    # Degree for each row node: D[i] = sum_j w_{ij}
    # Use scatter on a differentiable path (index_add on a zeros tensor).
    degree = torch.zeros(n, dtype=weights.dtype, device=device)
    degree = degree.index_add(0, rows, weights)  # differentiable w.r.t. weights

    # Concatenate all indices and values into a single COO tensor.
    # Off-diagonal: (rows, cols) → -weights
    # Diagonal:     (i,    i   ) →  degree[i]
    all_rows = torch.cat([rows,     diag_idx])
    all_cols = torch.cat([cols,     diag_idx])
    all_vals = torch.cat([-weights, degree])   # both live on the autograd graph

    L = torch.sparse_coo_tensor(
        torch.stack([all_rows, all_cols]),
        all_vals,
        (n, n),
        device=device,
    ).coalesce()
    return L


class DenoisingAutoencoder(nn.Module):
    """Denoising autoencoder with ConGaR-style cross-modal alignment.

    - Gene branch: expression encoder/decoder for denoising reconstruction.
    - Image branch (optional): independent image encoder used only for
      cross-modal contrastive alignment (ConGaR/ConGcR-style).
    - Neighborhood head (always available): predicts neighborhood relations
      from latent embeddings, enabling graph-structure supervision even when
      ``use_image=False``.
    """

    def __init__(
        self,
        n_genes: int,
        n_img: int,
        hidden_dim: int | None = 128,
        img_hidden: int = 128,
        latent_dim: int = 64,
        dropout: float = 0.1,
        use_image: bool = True,
    ) -> None:
        super().__init__()

        # Enable image branch only if requested and n_img > 0
        self.use_image = bool(use_image) and (n_img is not None) and (n_img > 0)
        self.n_img = n_img if self.use_image else 0

        if hidden_dim is None:
            hidden_dim = 128

        self.latent_dim = latent_dim if self.use_image else hidden_dim // 2

        # Gene AE branch (always present)
        self.encoder = nn.Sequential(
            nn.Linear(n_genes, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(self.latent_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_genes),
        )

        # ConGaR-style cross-modal branch (only when image features are enabled)
        if self.use_image:
            self.image_encoder = nn.Sequential(
                nn.Linear(n_img, img_hidden),
                nn.BatchNorm1d(img_hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(img_hidden, self.latent_dim),
                nn.BatchNorm1d(self.latent_dim),
                nn.ReLU(),
            )
            self.gene_proj_head = nn.Linear(self.latent_dim, self.latent_dim)
            self.image_proj_head = nn.Linear(self.latent_dim, self.latent_dim)
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.1), dtype=torch.float32))  # inverse temperature parameterization (logit_scale = log(1/temperature))
        else:
            self.image_encoder = None
            self.gene_proj_head = None
            self.image_proj_head = None
            self.logit_scale = None

        # Neighborhood prediction head (always on)
        self.neighborhood_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def encode(self, x_expr: torch.Tensor, x_img: torch.Tensor | None = None) -> torch.Tensor:
        return self.encoder(x_expr)

    def encode_image(self, x_img: torch.Tensor) -> torch.Tensor:
        if not self.use_image or self.image_encoder is None:
            raise RuntimeError("Image encoder is disabled (use_image=False).")
        return self.image_encoder(x_img)

    def contrastive_loss(self, z_gene, x_img, temperature_floor: float = 1e-4):
        if not self.use_image:
            return z_gene.new_tensor(0.0)

        z_img = self.encode_image(x_img)
        g = F.normalize(self.gene_proj_head(z_gene), dim=1)  # project and normalize gene latent
        v = F.normalize(self.image_proj_head(z_img), dim=1)  # project and normalize image latent

        scale = torch.exp(self.logit_scale).clamp(min=1.0, max=1000.0)  # inverse temperature
        logits_gv = (g @ v.T) * scale
        logits_vg = logits_gv.T

        labels = torch.arange(logits_gv.size(0), device=logits_gv.device)
        return 0.5 * (F.cross_entropy(logits_gv, labels) + F.cross_entropy(logits_vg, labels))

    def decode_genes(self, z: torch.Tensor, x_img: torch.Tensor | None = None) -> torch.Tensor:
        """Decode latent z to log-scale gene expression."""
        return self.decoder(z)

    def neighborhood_logits(self, z: torch.Tensor) -> torch.Tensor:
        """Predict pairwise neighborhood logits from latent embedding."""
        h = F.normalize(self.neighborhood_head(z), dim=1)
        return h @ h.T

    def forward(self, x_expr: torch.Tensor, x_img: torch.Tensor | None = None) -> torch.Tensor:
        z = self.encode(x_expr, x_img)
        return self.decode_genes(z, x_img)


class DenoisingAutoencoderLite(nn.Module):
    """
    Lower-capacity AE to reduce ability to mimic diffusion:
    - LayerNorm instead of BatchNorm
    - smaller latent
    - fewer layers
    Keeps the same interface as your original DenoisingAutoencoder.
    """
    def __init__(
        self,
        n_genes: int,
        n_img: int,
        hidden_dim: int = 96,
        img_hidden: int = 96,
        latent_dim: int = 32,
        dropout: float = 0.05,
        use_image: bool = True,
    ) -> None:
        super().__init__()
        self.use_image = bool(use_image) and (n_img is not None) and (n_img > 0)
        self.n_img = n_img if self.use_image else 0
        self.latent_dim = latent_dim

        def block(in_dim, out_dim, p=dropout):
            return nn.Sequential(
                nn.Linear(in_dim, out_dim),
                nn.LayerNorm(out_dim),
                nn.GELU(),
                nn.Dropout(p),
            )

        # Gene encoder: shallow + lower capacity
        self.encoder = nn.Sequential(
            block(n_genes, hidden_dim),
            nn.Linear(hidden_dim, self.latent_dim),
            nn.LayerNorm(self.latent_dim),
            nn.GELU(),
        )

        # Gene decoder: shallow + lower capacity
        self.decoder = nn.Sequential(
            block(self.latent_dim, hidden_dim),
            nn.Linear(hidden_dim, n_genes),
        )

        # Optional image branch
        if self.use_image:
            self.image_encoder = nn.Sequential(
                block(n_img, img_hidden),
                nn.Linear(img_hidden, self.latent_dim),
                nn.LayerNorm(self.latent_dim),
                nn.GELU(),
            )
            self.gene_proj_head = nn.Linear(self.latent_dim, self.latent_dim)
            self.image_proj_head = nn.Linear(self.latent_dim, self.latent_dim)
            self.logit_scale = nn.Parameter(torch.tensor(np.log(1.0 / 0.1), dtype=torch.float32))
        else:
            self.image_encoder = None
            self.gene_proj_head = None
            self.image_proj_head = None
            self.logit_scale = None

        # Neighborhood head (keep, but also lower capacity)
        self.neighborhood_head = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.GELU(),
            nn.Linear(self.latent_dim, self.latent_dim),
        )

    def encode(self, x_expr: torch.Tensor, x_img: torch.Tensor = None) -> torch.Tensor:
        return self.encoder(x_expr)

    def encode_image(self, x_img: torch.Tensor) -> torch.Tensor:
        if not self.use_image or self.image_encoder is None:
            raise RuntimeError("Image encoder is disabled (use_image=False).")
        return self.image_encoder(x_img)

    def contrastive_loss(self, z_gene: torch.Tensor, x_img: torch.Tensor, temperature_floor: float = 1e-4) -> torch.Tensor:
        if not self.use_image:
            return z_gene.new_tensor(0.0)
        z_img = self.encode_image(x_img)
        g = F.normalize(self.gene_proj_head(z_gene), dim=1)
        v = F.normalize(self.image_proj_head(z_img), dim=1)
        scale = torch.exp(self.logit_scale).clamp(min=10.0, max=100.0)
        logits_gv = (g @ v.T) * scale
        labels = torch.arange(logits_gv.size(0), device=logits_gv.device)
        return 0.5 * (F.cross_entropy(logits_gv, labels) + F.cross_entropy(logits_gv.T, labels))

    def decode_genes(self, z: torch.Tensor, x_img: torch.Tensor = None) -> torch.Tensor:
        # still returns "log-scale" per your pipeline expectations
        return self.decoder(z)

    def neighborhood_logits(self, z: torch.Tensor) -> torch.Tensor:
        h = F.normalize(self.neighborhood_head(z), dim=1)
        return h @ h.T

    def forward(self, x_expr: torch.Tensor, x_img: torch.Tensor = None) -> torch.Tensor:
        z = self.encode(x_expr, x_img)
        return self.decode_genes(z, x_img)


class CLEARmodel(PyroModule):
    """
    Pyro model for spatial transcriptomics decontamination with multiscale diffusion patterns.
    """
    
    def __init__(self,
                Y: torch.Tensor,
                spatial_coords: torch.Tensor,
                spot_types: torch.Tensor,  # 0: background, 1: tissue, 2: no-cell tissue
                niche_labels: torch.Tensor,
                diffrate_labels: torch.Tensor,
                X: torch.Tensor = None,       # optional pre-decontaminated counts (X_init)
                spatial_con: torch.Tensor = None,  # shape [n_spots, n_spots]
                spatial_dist: torch.Tensor = None,
                image_features: torch.Tensor = None,
                cell_counts: torch.Tensor = None,
                gene_likelihood: str = "nb",  # available: "nb", "poisson", "gamma_poisson", "zip", "zinb"
                n_diffusion_steps: int = 1,  # Number of diffusion steps in forward operator
                use_image: bool = True,        # whether to use image features at all
                img_penalty_weight: float = 1.0,  # weight on image embedding penalty in the model
                ):
        
        super().__init__()
        
        print("Initializing CLEARmodel model...")
        # Register data tensors as buffers so they get moved with .to(device)
        self.register_buffer('Y', Y)  # shape [n_spots, n_genes]
        # X: optional pre-decontaminated counts (X_init from OT transport).
        # When provided, the AE encoder trains on X (cleaner signal) instead of Y.
        # When None, the model falls back to Y-only (default behaviour).
        if X is not None:
            assert X.shape == Y.shape, f"X.shape {X.shape} must match Y.shape {Y.shape}"
            self.register_buffer('X', X.to(Y.dtype))
            print("  X_init provided: AE encoder will use X (pre-decontaminated counts).")
        else:
            self.X = None  # plain attribute; register_buffer(None) unsupported in some PyTorch versions
        self.register_buffer('spatial_coords', spatial_coords)  # shape [n_spots, 2]
        self.n_spots, self.n_genes = self.Y.shape
        self.n_diffusion_steps = n_diffusion_steps  # Store for initialization
        
        self.spatial_dist = spatial_dist
        self.spatial_con = spatial_con

        # Store image_features and cell_counts
        self.use_image = bool(use_image)
        print("Storing image features and cell densities...")
        if self.use_image:
            if not isinstance(image_features, torch.Tensor):
                raise ValueError("image_features must be a torch.Tensor of shape (n_spots, n_img)")
            if image_features.shape[0] != self.n_spots:
                raise ValueError(
                    f"image_features must have one row per spot: got {image_features.shape[0]} vs n_spots={self.n_spots}"
                )
            self.register_buffer('image_features', image_features.to(Y.dtype))
            self.n_img = self.image_features.shape[1]
        else:
            # create an empty placeholder so downstream code can safely reference .image_features
            empty_img = Y.new_zeros(self.n_spots, 0)
            self.register_buffer('image_features', empty_img)
            self.n_img = 0

        self.register_buffer('cell_counts', cell_counts if isinstance(cell_counts, torch.Tensor) else None)
        self.register_buffer('total_counts', torch.sum(Y, dim=1))
        self.register_buffer('spot_types', spot_types)
        self.register_buffer('niche_labels', niche_labels)
        self.register_buffer('diffrate_labels', diffrate_labels)
        self.n_niche_types = len(torch.unique(self.niche_labels))
        self.gene_likelihood = gene_likelihood
        # Weight for image embedding penalty term in the generative model
        self.img_penalty_weight = float(img_penalty_weight)
        self.use_laplacian_diffusion = True

        # Extract sparse indices safely
        print(f"Extracting sparse indices...")
        try:
            adj_idx = self.spatial_con.coalesce().indices()  # shape (2, nnz)
            # print(f"[DEBUG] Extracted indices shape: {adj_idx.shape}")
            r = adj_idx[0]      # source spots
            c = adj_idx[1]      # target spots
            keep = (r != c)  # remove indices for self-loops
            r = r[keep]      # self-loops removed
            c = c[keep]      # self-loops removed
            # print(f"[DEBUG] After removing self-loops: {len(r)} edges")
            self.register_buffer('adj_rows', r)        # (nnz,)  numerical spot indices of target spots
            self.register_buffer('adj_cols', c)        # (nnz,)  numerical spot indices of source spots
            self.nnz = self.adj_rows.numel()  # number of non-zero entries = total number of neighborhood connections
            # print(f"[DEBUG] Adjacency indices registered successfully")
        except Exception as e:
            print(f"[ERROR] Failed to extract adjacency indices: {e}")
            raise
        
        # Regularization defaults
        lib_sizes = self.Y.sum(dim=1)
        self.register_buffer('lib_sizes', lib_sizes)
        
        self.register_buffer('zero_penalty_weight', torch.tensor(1.0))  # Reduced from 100000000
        self.register_buffer('zero_mask_epsilon', torch.tensor(0.0))
        self.penalize_only_in_tissue = True
                
        # Identify spot indices
        background_spots = (self.spot_types == 0).nonzero(as_tuple=True)[0]  # background spots (type 0)
        self.register_buffer('background_spots', background_spots)
        tissue_spots = (self.spot_types != 0).nonzero(as_tuple=True)[0]   # include both tissue and no-cell tissue as "tissue" for this purpose
        self.register_buffer('tissue_spots', tissue_spots)
        no_cell_spots = (self.spot_types == 2).nonzero(as_tuple=True)[0]
        self.register_buffer('no_cell_spots', no_cell_spots)
        self.register_buffer('all_non_cell', torch.cat([self.background_spots, self.no_cell_spots]))  # use both background and no-cell tissue as "non-cell" for initialization purposes
        
        
        # clear_model.py (inside CLEARmodel.__init__)
        # self.denoising_ae = DenoisingAutoencoder(
        #     n_genes=self.n_genes,
        #     n_img=self.n_img,        # dynamic
        #     hidden_dim=128,
        #     img_hidden=128,
        #     latent_dim=64,
        #     dropout=0.1,
        #     use_image=self.use_image,
        # )
        self.denoising_ae = DenoisingAutoencoderLite(
            n_genes=self.n_genes,
            n_img=self.n_img,
            hidden_dim=96,
            img_hidden=96,
            latent_dim=32,
            dropout=0.05,
            use_image=self.use_image,
        )
        self.latent_dim = self.denoising_ae.latent_dim
        self.niche_classifier = nn.Sequential(
            nn.Linear(self.latent_dim, self.latent_dim),
            nn.ReLU(),
            nn.Linear(self.latent_dim, self.n_niche_types),
        )

        # Training-time auxiliary losses (used during Pyro fitting)
        self.enable_aux_losses = False
        self.aux_lambda_niche = 0.0

        # Precompute initialization parameters
        print("Computing initialization parameters...")
        self._compute_initialization_parameters()

    def configure_auxiliary_losses(
        self,
        enable: bool = False,
        lambda_niche: float = 0.0,
    ):
        """Configure auxiliary losses for the main Pyro training phase."""
        self.enable_aux_losses = bool(enable)
        self.aux_lambda_niche = float(max(0.0, lambda_niche))
    
    def normalize(self):
        """Normalize expression data for autoencoder input.

        When ``self.X`` is available (X_init mode), the *clean* reference X is used
        as the reconstruction target/encoder input for tissue spots.  Y is still used
        for all-spot normalization (noisy input to the denoising objective).
        """
        # Clean reference for tissue spots: use X if available, else Y
        src = self.X if (self.X is not None) else self.Y
        X_tissue = torch.log1p(src[self.tissue_spots])

        # Noisy input: always from Y (observed, potentially contaminated)
        # X_expr = self.X if (self.X is not None) else self.Y
        X_expr = self.Y
        X_expr = torch.log1p(X_expr[self.tissue_spots])
    
        # Image standardization over all spots (only if images are used)
        if self.use_image and self.n_img > 0:
            X_img = self.image_features[self.tissue_spots] if self.image_features.shape[0] > X_tissue.shape[0] else self.image_features
            img_mean = X_img.mean(0, keepdim=True)
            img_std  = X_img.std(0, keepdim=True) + 1e-8
            X_img_z  = (X_img - img_mean) / img_std
        else:   
            X_img_z = None
        return X_tissue, X_expr, X_img_z

    
    def _compute_initialization_parameters(self):
        """Compute initialization parameters for spot-specific diffusion rates."""
        
        total_out_perc = 1 - self.Y[self.tissue_spots].sum() / (self.Y.sum() + 1e-8)
        self.register_buffer('total_out_perc', total_out_perc)  # shape: (n_genes,)
        print(f"Estimated total contamination fraction: {total_out_perc.item():.4f}")

        print("Computing spatial distance median...")
        self.register_buffer('dist_median', self.spatial_dist.coalesce().values().median())
        
        # 1. Compute background spot niche influence 
        print("Computing background niche influence...")
        self._compute_background_niche_influence()

        # 2. Compute spot diffusion rate initialization (spot-level scalar)
        print("Computing spot diffusion rate initialization...")
        self._compute_diffusion_rate_initialization()
                
        # 3. Compute local kernel width initialization
        print("Computing local kernel width initialization...")
        self._compute_local_kernel_width()  

    def _compute_diffusion_rate_initialization(self):
        """
        Dispatcher to select diffusion rate initialization method.
        Currently uses: _compute_diffusion_rate_neighborhood_based()
        Other options (for testing):
          - _compute_diffusion_rate_niche_based() [original cluster-based]
          - _compute_diffusion_rate_uniform() [constant for all spots]
        """
        # Call the neighborhood-based version (NEW APPROACH)
        # self._compute_diffusion_rate_neighborhood_based()
        
        # Optionally, uncomment below to test other versions:
        self._compute_diffusion_rate_niche_based()
        # self._compute_diffusion_rate_uniform()

    def _compute_diffusion_rate_neighborhood_based(self):
        """
        Neighborhood-based diffusion-rate initialization (3-stage):

        Stage 1 (edge tissue):
            For every tissue spot that has ≥1 background (non-tissue) neighbor,
            rate = avg_background_neighbor_counts / spot_counts.
        Stage 2 (inner tissue):
            For tissue spots with no background neighbors, propagate rates inward
            by taking the row-weighted average of already-assigned tissue neighbors.
            Iterate until all tissue spots are assigned (or use a global fallback).
        Stage 3 (non-tissue):
            For background (non-tissue) spots, rate = avg_neighborhood_counts / spot_counts.

        All rates are then clamped to [0, 0.5] and divided by n_diffusion_steps to
        represent per-step coefficients.
        """
        print("Computing diffusion rates (edge→inward propagation + background).")

        eps = 1e-8
        device = self.Y.device
        n_steps = max(1, int(self.n_diffusion_steps))

        # Masks and per-spot total counts
        tissue_mask = (self.spot_types == 1)                  # tissue = 1
        non_tissue_mask = ~tissue_mask                        # background / no-cell
        counts = self.Y.sum(dim=1)                            # (n_spots,)
        n_spots = int(self.n_spots)

        # Build row-normalized inverse-distance weights from spatial_dist
        # (This is consistent with your previous init logic, just vectorized) [1](https://connecthkuhk-my.sharepoint.com/personal/chrissy1_connect_hku_hk/Documents/Microsoft%20Copilot%20Chat%20Files/clear_model.py)
        coo = self.spatial_dist.coalesce()
        if coo.values().numel() == 0:
            print("WARNING: spatial_dist is empty. Using uniform diffusion rates.")
            self._compute_diffusion_rate_uniform()
            return

        rows = coo.indices()[0]   # source i
        cols = coo.indices()[1]   # neighbor j
        dvals = coo.values()      # distances(i,j)

        w = 1.0 / (dvals + eps)
        row_sum = torch.zeros(n_spots, dtype=w.dtype, device=w.device)
        row_sum.index_add_(0, rows, w)
        w = w / (row_sum[rows] + eps)  # row-normalized weights, sum_j w_ij = 1

        # -----------------------------
        # Stage 1: edge tissue spots
        # -----------------------------
        # Select directed edges i->j where i is tissue and j is non-tissue (background)
        mask_edge = tissue_mask[rows] & non_tissue_mask[cols]
        rows_e = rows[mask_edge]   # tissue sources
        cols_e = cols[mask_edge]   # background neighbors
        w_e    = w[mask_edge]

        # Weighted average of background neighbors' counts, per tissue source
        bg_num   = torch.zeros(n_spots, dtype=counts.dtype, device=device)
        bg_denom = torch.zeros_like(bg_num)
        if rows_e.numel() > 0:
            contrib = w_e * counts[cols_e]
            bg_num.index_add_(0, rows_e, contrib)
            bg_denom.index_add_(0, rows_e, w_e)

        edge_tissue = (bg_denom > 0) & tissue_mask
        rates = torch.zeros(n_spots, dtype=counts.dtype, device=device)
        rates[edge_tissue] = (bg_num[edge_tissue] / (bg_denom[edge_tissue] + eps)) / (counts[edge_tissue] + eps)

        # -----------------------------
        # Stage 2: inward propagation over tissue graph
        # -----------------------------
        assigned = edge_tissue.clone()

        # Build tissue↔tissue edge lists and weights (still row-normalized) for fast aggregation
        mask_tt = tissue_mask[rows] & tissue_mask[cols]
        rows_t  = rows[mask_tt]   # tissue source
        cols_t  = cols[mask_tt]   # tissue neighbor
        w_t     = w[mask_tt]

        # Iterate frontier → assign inner tissue spots by averaging rates of already-assigned tissue neighbors
        # Stop when all tissue assigned or no progress (disconnected islands)
        # (Bounded iterations to be safe; typically converges in a handful of rounds)
        max_iter = min(64, n_spots)  # conservative cap
        for _ in range(max_iter):
            if bool(assigned[tissue_mask].all()):
                break
            # Consider edges where src is unassigned tissue, dst is assigned tissue
            mask_front = (~assigned[rows_t]) & assigned[cols_t]
            if not mask_front.any():
                break

            r = rows_t[mask_front]
            c = cols_t[mask_front]
            wf = w_t[mask_front]

            num = torch.zeros(n_spots, dtype=counts.dtype, device=device)
            den = torch.zeros_like(num)
            num.index_add_(0, r, wf * rates[c])   # accumulate neighbor-rate * weight
            den.index_add_(0, r, wf)              # accumulate weights

            # Spots we can assign this round: tissue, unassigned, den>0
            can = (den > 0) & tissue_mask & (~assigned)
            if not can.any():
                break
            rates[can] = num[can] / (den[can] + eps)
            assigned[can] = True

        # Remaining tissue spots (e.g., isolated islands) → global fallback
        remain = tissue_mask & (~assigned)
        if remain.any():
            fallback = (
                rates[assigned & tissue_mask].mean()
                if (assigned & tissue_mask).any()
                else torch.clamp(self.total_out_perc / n_steps, 1e-6, 0.5)
            )
            rates[remain] = fallback

        # -----------------------------
        # Stage 3: non-tissue spots
        # -----------------------------
        # For each background spot i: rate = (avg neighbor counts)/count_i, using same row-normalized w_ij.
        # Compute weighted neighbor average counts for all i, then restrict to background. [1](https://connecthkuhk-my.sharepoint.com/personal/chrissy1_connect_hku_hk/Documents/Microsoft%20Copilot%20Chat%20Files/clear_model.py)
        num_all = torch.zeros(n_spots, dtype=counts.dtype, device=device)
        den_all = torch.zeros_like(num_all)
        num_all.index_add_(0, rows, w * counts[cols])
        den_all.index_add_(0, rows, w)

        mask_nt = non_tissue_mask & (den_all > 0)
        neighbor_avg = torch.zeros_like(counts)
        neighbor_avg[mask_nt] = num_all[mask_nt] / (den_all[mask_nt] + eps)
        rates[non_tissue_mask] = neighbor_avg[non_tissue_mask] / (counts[non_tissue_mask] + eps)

        # -----------------------------
        # Finalize: clamp and per-step scaling
        # -----------------------------
        rates = torch.clamp(rates, min=0.0, max=0.5) / float(n_steps)
        rates = torch.where(torch.isfinite(rates), rates, torch.full_like(rates, 1e-6))

        # Save buffers expected downstream
        self.register_buffer('spot_diffusion_rate_init', rates)
        self.register_buffer('contamination_score', rates * n_steps)  # pre-step scale (diagnostics)
        self.register_buffer('calibration_alpha', torch.tensor(1.0, dtype=torch.float32, device=device))

        # Diagnostics
        n_edge = int(edge_tissue.sum().item())
        n_tissue = int(tissue_mask.sum().item())
        n_assigned = int((assigned & tissue_mask).sum().item())
        print(f"  Edge tissue initialized: {n_edge}")
        print(f"  Tissue assigned after propagation: {n_assigned}/{n_tissue}")
        print(f"  Diffusion rates: min={rates.min().item():.6f}, max={rates.max().item():.6f}, mean={rates.mean().item():.6f}")

    def _compute_diffusion_rate_niche_based(self):
        """
        Initialize spot-level diffusion rates based on cluster (niche) assignments.
        
        This OLD approach grouped spots by niche label and estimated contamination per niche,
        then mapped uniformly to all spots in that cluster. This can lead to cluster-biased
        diffusion rates that ignore spatial heterogeneity.
        
        The diffusion coefficient D represents how strongly spatial smoothing occurs at each step.
        """
        print("[OLD] Computing diffusion rates via niche-based (cluster) aggregation...")
        total_out_perc = self.total_out_perc
        n_diffusion_steps = self.n_diffusion_steps  # Use the configured number of steps
        
        # First, compute niche-level TOTAL diffusion rates from data
        diffrate_n_niches = len(torch.unique(self.diffrate_labels))
        niche_total_diffusion = torch.zeros(diffrate_n_niches, dtype=self.Y.dtype, device=self.Y.device)
        
        # Average expression in background/no-cell spots
        background = self.Y[self.all_non_cell]  # shape (n_non_cell, n_genes)
        
        # Check if background_niche_weights exists
        if not hasattr(self, 'background_niche_weights') or self.background_niche_weights is None:
            # Use uniform weights if not computed
            background_niche_weights = torch.ones(len(self.all_non_cell), diffrate_n_niches) / diffrate_n_niches
        else:
            background_niche_weights = self.background_niche_weights
        
        # Niche-weighted average expression in background spots shape (n_niche_types, n_genes)
        background_avg = background_niche_weights.T @ background
        
        # For each niche type 
        for niche_idx in range(diffrate_n_niches):
            niche_mask = self.diffrate_labels == niche_idx
            niche_tissue_mask = niche_mask & (self.spot_types == 1).bool()  # Only tissue spots
            
            if niche_tissue_mask.sum() == 0:
                # No tissue spots in this niche, use global average
                niche_total_diffusion[niche_idx] = float(total_out_perc.item()) * 0.1  # small fraction of global average if no tissue spots
                continue
                
            niche_expr = self.Y[niche_tissue_mask]  # dim (n_spots_in_niche, n_genes)
            niche_total = niche_expr.sum()  # Total counts in niche
            
            # Background counts attributed to this niche
            niche_background_mask = niche_mask & ((self.spot_types == 0) | (self.spot_types == 2)).bool()
            if niche_background_mask.sum() > niche_expr.shape[0] * 0.1:
                niche_background_total = self.Y[niche_background_mask].sum()
            else:
                # Use weighted background average
                niche_background_total = background_avg[niche_idx].sum() * len(self.all_non_cell) / diffrate_n_niches
            
            # Total diffusion fraction: background counts / (tissue counts + background counts)
            # This represents the fraction that escaped from tissue to background
            total_diffusion = niche_background_total / (niche_total + niche_background_total + 1e-6)
            
            # Clamp to reasonable range
            total_diffusion = torch.clamp(
                total_diffusion,
                min=0.001,  # At least 0.1% diffusion
                max=total_out_perc * 1.5  # At most 1.5x global average
            )
            
            niche_total_diffusion[niche_idx] = total_diffusion
        
        # Convert total diffusion to per-step diffusion coefficient
        # For the heat equation, D controls smoothing strength at each step
        # We use a conservative estimate: D = total_diffusion / (n_steps * smoothing_factor)
        # where smoothing_factor accounts for accumulation across steps
        niche_diffusion_per_step = niche_total_diffusion / (n_diffusion_steps)
        
        # Map to spots based on niche labels
        spot_diffusion_rate_init = niche_diffusion_per_step[self.diffrate_labels.long()]
        
        # Ensure values are in valid range
        spot_diffusion_rate_init = torch.clamp(spot_diffusion_rate_init, min=1e-5, max=0.5)
        
        self.register_buffer('niche_diffusion_per_step', niche_diffusion_per_step)  # For diagnostics
        self.register_buffer('spot_diffusion_rate_init', spot_diffusion_rate_init)
        self.register_buffer('niche_total_diffusion_estimated', niche_total_diffusion)  # For diagnostics
    
    def _compute_diffusion_rate_uniform(self):
        """
        [FOR TESTING] Initialize all spots with the SAME diffusion rate.
        
        Useful for testing whether spatial heterogeneity in diffusion rates helps model convergence.
        Uses the global contamination fraction as the uniform rate.
        
        The diffusion coefficient D represents how strongly spatial smoothing occurs at each step.
        """
        print("[TEST] Computing UNIFORM diffusion rates for all spots...")
        
        total_out_perc = self.total_out_perc
        n_diffusion_steps = self.n_diffusion_steps
        
        # Convert total diffusion to per-step coefficient
        uniform_diffusion_per_step = total_out_perc / (n_diffusion_steps)
        
        # Clamp to valid range
        delta_min, delta_max = 1e-5, 0.9
        uniform_diffusion_per_step = torch.clamp(uniform_diffusion_per_step, min=delta_min, max=delta_max)
        
        # Create uniform diffusion rate for all spots
        spot_diffusion_rate_init = torch.full(
            (self.n_spots,),
            uniform_diffusion_per_step.item(),
            dtype=self.Y.dtype,
            device=self.Y.device
        )
        
        self.register_buffer('spot_diffusion_rate_init', spot_diffusion_rate_init)
        
        print(f"  Uniform diffusion rate: {uniform_diffusion_per_step.item():.6f}")
        print(f"  Applied to all {self.n_spots} spots")
        
                
    def _compute_background_niche_influence(self):
        """Compute how each background/no-cell spot is influenced by each niche.
        Specifically, for each background/no-cell spot, compute a weight vector over niches indicating
        the relative influence of each niche on that spot based on spatial proximity (closer niches have more influence).     
        
        Returns:
            background_niche_weights: Tensor of shape (n_non_cell_spots, n_niche_types)   
        """
        diffrate_n_niches = len(torch.unique(self.diffrate_labels))
        if len(self.all_non_cell) == 0:
            self.register_buffer('background_niche_weights', torch.zeros(0, diffrate_n_niches, device=self.spatial_coords.device))
            return
        
        # Get device from model tensors (accounts for .to(device) call)
        device = self.spatial_coords.device
        
        # Use all non-cell spots (background + no-cell tissue)
        non_cell_coords = self.spatial_coords[self.all_non_cell]
        # Create niche_influence on the SAME device as model parameters
        niche_influence = torch.zeros(len(self.all_non_cell), diffrate_n_niches, device=device, dtype=self.spatial_coords.dtype)
        distance_median = self.dist_median + 1e-6  # avoid zero
        
        # print("Computing niche influence for non-cell spots...")
        for niche_idx in range(diffrate_n_niches):
            niche_spots = (self.diffrate_labels == niche_idx).nonzero(as_tuple=True)[0]
            
            if len(niche_spots) > 0:
                # print(f"  Niche {niche_idx} with {len(niche_spots)} spots")
                niche_coords = self.spatial_coords[niche_spots]
                
                # Compute niche centroid to avoid memory explosion from cdist on large niche groups
                # print("    Computing niche centroid...")
                niche_centroid = niche_coords.mean(dim=0, keepdim=True)  # shape: (1, 2)
                
                # Compute distance from each non-cell spot to niche centroid
                # print("    Computing distances to niche centroid...")
                # Use L2 distance: sqrt(sum((x - centroid)^2))
                dist_to_niche = torch.sqrt(((non_cell_coords - niche_centroid) ** 2).sum(dim=1) + 1e-8)  # shape: (n_non_cell,)
                
                # Convert to influence weight (closer = more influence)
                niche_influence[:, niche_idx] = torch.exp(-dist_to_niche / distance_median)  # shape: (n_non_cell,)
                
        print("Normalizing niche influence weights...")
        # Normalize to sum to 1 for each non-cell spot
        niche_sum = niche_influence.sum(dim=1, keepdim=True) + 1e-4
        niche_influence = niche_influence / niche_sum  # shape: (n_non_cell, n_niche_types)
        
        self.register_buffer('background_niche_weights', niche_influence)

    def _compute_average_niche_separation(self):
        """Compute average distance between different niches."""
        diffrate_n_niches = len(torch.unique(self.diffrate_labels))
        device = self.spatial_coords.device
        niche_centroids = torch.zeros(diffrate_n_niches, 2, device=device, dtype=self.spatial_coords.dtype)  # shape: (n_niche_types, 2)
        valid_niches = []
        
        # Compute centroid for each niche
        for niche_idx in range(diffrate_n_niches):
            niche_spots = (self.diffrate_labels == niche_idx).nonzero(as_tuple=True)[0]
            # Filter by tissue spots only
            niche_spots_idx = torch.isin(niche_spots, self.tissue_spots)
            niche_spots = niche_spots[niche_spots_idx]
            
            if len(niche_spots) > 0:
                niche_coords = self.spatial_coords[niche_spots]
                niche_centroids[niche_idx] = niche_coords.mean(dim=0)
                valid_niches.append(niche_idx)
        
        if len(valid_niches) <= 1:
            return self.dist_median * 0.5
        
        # Compute pairwise distances between niche centroids
        valid_centroids = niche_centroids[valid_niches]
        # Manually compute distances to avoid torch.cdist memory issues
        # ||a - b||^2 = ||a||^2 - 2*a*b + ||b||^2
        centroid_sq = (valid_centroids ** 2).sum(dim=1, keepdim=True)  # shape: (n_valid, 1)
        centroid_prod = valid_centroids @ valid_centroids.T  # shape: (n_valid, n_valid)
        centroid_distances = torch.sqrt(
            (centroid_sq + centroid_sq.T - 2 * centroid_prod).clamp(min=0)
        )  # shape: (n_valid_niches, n_valid_niches)
        
        # Get upper triangle (exclude diagonal and lower triangle)
        upper_triangle = torch.triu(centroid_distances, diagonal=1)
        non_zero_distances = upper_triangle[upper_triangle > 0]
        
        if len(non_zero_distances) > 0:
            # Use mean distance between niches, scaled down for sigma
            avg_niche_separation = non_zero_distances.mean() * 0.5  # Scale down for sigma
        else:
            avg_niche_separation = self.dist_median * 0.5
        
        return avg_niche_separation
        
    def _compute_local_kernel_width(self):
        """Compute spatial kernel initialization based on nearest neighbor distances."""

        # Initialize sigma based on spatial scale
        # Local contamination: ~1-2 nearest neighbors
        local_sigma_init = self.dist_median
        
        # Store initialization values
        self.register_buffer('local_sigma_init', local_sigma_init)  # shape: (1,)    
    
    # def _compute_neighborhood_niche_concentration(self):
    #     niche_concentration = torch.full((self.n_spots, self.n_niche_types), 0.5)

    #     neighbor_filt = self.spatial_con > 0  # Filter to have neighborhood graph shape [n_spots, n_spots]
    #     # for each spot, compute niche concentration based on neighbors
    #     for spot_idx in range(self.n_spots):
    #         if neighbor_filt[spot_idx].any():
    #             # Get neighbors' niche labels
    #             neighbor_niches = self.niche_labels[neighbor_filt[spot_idx]]
    #             # Count occurrences of each niche in neighbors
    #             niche_counts = torch.bincount(neighbor_niches, minlength=self.n_niche_types)  # shape: (n_niche_types,)
    #             niche_concentration[spot_idx] = niche_counts.float() + 0.1
    #     self.register_buffer('niche_concentration', niche_concentration)
        
    def _compute_anisotropic_kernel(self, sigma_x, sigma_y, theta, distance):
        """Compute anisotropic Gaussian kernel for anistropic diffusion."""
        cos_theta = torch.cos(theta)
        sin_theta = torch.sin(theta)
        
        x_coords = self.spatial_coords[:, 0:1]
        y_coords = self.spatial_coords[:, 1:2]
        
        x_rot = cos_theta * x_coords - sin_theta * y_coords
        y_rot = sin_theta * x_coords + cos_theta * y_coords
        
        dx = x_rot.unsqueeze(1) - x_rot.unsqueeze(0)
        dy = y_rot.unsqueeze(1) - y_rot.unsqueeze(0)
        
        aniso_dist = (dx**2 / sigma_x**2) + (dy**2 / sigma_y**2)
        
        return torch.exp(-0.5 * aniso_dist).squeeze()
        
    
    def _compute_contamination(
        self,
        true_expression: torch.Tensor,         # (n_spots, n_genes)
        diffusion_coeff: torch.Tensor,         # (n_spots,) or (n_spots, n_genes) or scalar
        L: torch.Tensor,                       # sparse (n_spots, n_spots) graph Laplacian – built fresh each call
        n_steps: int = 1,
        teleport: float | torch.Tensor = 0.01, # damping γ ∈ [0, 1]
    ) -> torch.Tensor:
        """
        Forward contamination diffusion using NON-ELLIPTICAL ROTATED LAPLACIAN KERNEL.
        
        Applies discrete heat equation with learned spatial kernel:
            y_{k+1} = y_k - D·L·y_k + γ·(ȳ_k - y_k)
        
        where:
        - L: Graph Laplacian built from non-elliptical rotated kernel weights
        - D: Per-spot diffusion coefficient (learned, controls magnitude of smoothing)
        - γ: Global damping/teleportation parameter (prevents unbounded growth)
        
        The Laplacian L = D_diagonal - W is constructed where W contains non-elliptical
        anisotropic edge weights from the rotated coordinate system:
          - Rotation angle θ aligns asymmetry axis
          - Directional sigmas (σ_x+, σ_x-, σ_y+, σ_y-) define asymmetric spread
          - Sharpness τ modulates Gaussian decay
        
        This kernel enables modeling RNA contamination with directional bias.
        
        ``L`` must be passed in freshly constructed from the current learned edge
        weights each call so that gradients flow back to those parameters.
        """
        # Broadcast D to (n_spots, n_genes)
        if diffusion_coeff.dim() == 0:
            D = diffusion_coeff.view(1, 1).expand(self.n_spots, self.n_genes)
        elif diffusion_coeff.dim() == 1:
            D = diffusion_coeff.unsqueeze(1).expand(-1, self.n_genes)
        else:
            D = diffusion_coeff
        D = torch.clamp(D, min=1e-5, max=1.0)

        # Scalar γ
        if isinstance(teleport, (float, int)):
            gamma = true_expression.new_tensor(float(teleport)).clamp(0.0, 1.0)
        else:
            gamma = teleport.clamp(0.0, 1.0)

        y = true_expression  # keep in graph – no .clone() / .detach()

        for _ in range(max(1, int(n_steps))):
            # Laplacian term: gradient flows through L's values and through y
            Ly = torch.sparse.mm(L, y)                          # (n_spots, n_genes)
            y_avg = y.mean(dim=0, keepdim=True)                 # (1, n_genes)
            y = y - D * Ly + gamma.view(1, 1) * (y_avg - y)
            y = torch.clamp(y, min=0.0)

        return y
    

    def model(self):
        """
        Deterministic forward model with learned diffusion and gene distribution prediction.
        
        Each gene is normalized to a probability distribution (sum-to-1 per spot).
        The model:
        1. Encodes tissue spots to latent representation
        2. Decodes to clean true expression
        3. Applies spatial diffusion to predict observed gene distribution
        4. Compares predicted vs observed distributions via MSE/KL loss
        """
        
        # ========== SPOT-SPECIFIC DIFFUSION RATES (DETERMINISTIC) ==========
        # spot_diffusion_rate_init is precomputed; learn multiplicative factor
        # diffusion_multiplier = pyro.param(
        #     "diffusion_multiplier",
        #     torch.tensor(1.0, device=self.Y.device),
        #     constraint=dist.constraints.positive,
        # )
        # spot_diffusion_rates = self.spot_diffusion_rate_init * diffusion_multiplier
        # spot_diffusion_rates = torch.clamp(spot_diffusion_rates, min=1e-5, max=1.0)
        # pyro.deterministic("spot_diffusion_rates", spot_diffusion_rates)

        prior_strength = self.Y.new_tensor(2.0)
        # prior_strength = pyro.param(
        #     "diffusion_rate_prior_strength",
        #     prior_strength,
        #     constraint=dist.constraints.positive
        # )
        with pyro.plate("spots", self.n_spots):
            # use mean of initialization as prior mean to impose a conservative assumption
            rate_mean = self.spot_diffusion_rate_init.mean() * torch.ones_like(self.spot_diffusion_rate_init)
            alpha_diff = prior_strength * rate_mean + 1e-5
            beta_diff = prior_strength * (1.0 - rate_mean) + 1e-5
            spot_diffusion_rates = pyro.sample("spot_diffusion_rates", dist.Beta(alpha_diff, beta_diff))

        # ========== SPATIAL DIFFUSION KERNEL DISTRIBUTION PARAMETERS ==========
        # 
        # KERNEL CATEGORY: NON-ELLIPTICAL ROTATED LAPLACIAN DIFFUSION KERNEL
        # 
        # This kernel is characterized by:
        #   1. Non-elliptical asymmetry: Directional elongation differs in positive/negative directions
        #      - σ_x_pos controls rightward spread, σ_x_neg controls leftward spread
        #      - σ_y_pos controls upward spread, σ_y_neg controls downward spread
        #   2. Rotated coordinate system: Rotation angle θ aligns the asymmetry axis with 
        #      the preferred spatial diffusion direction
        #   3. Laplacian-based: Edge weights feed into graph Laplacian L = D - W, then applied
        #      via discrete heat equation: y_{k+1} = y_k - D·L·y_k + γ(ȳ_k - y_k)
        #   4. Sharpness modulation: Parameter τ controls the steepness of the Gaussian decay
        #
        # Kernel formula (per edge i→j in rotated frame):
        #   q = (u_pos/σ_x_pos)² + (u_neg/σ_x_neg)² + (v_pos/σ_y_pos)² + (v_neg/σ_y_neg)²
        #   w(i,j) = exp(-0.5·q/τ²)
        # where u,v are rotated coordinates and (·)± = relu(±·)
        #
        # IMPORTANCE: Diffusion coefficient D (spot-specific) controls magnitude;
        # kernel parameters control only the directional distribution of diffused mass.

        eps = 1e-6

        # Learnable non-elliptical kernel parameters (scalar version, use one param to describe )
        theta_by_niche = pyro.param(
            "theta_by_niche",
            self.spatial_coords.new_full((1,), 0.1),
            constraint=dist.constraints.interval(-np.pi, np.pi),
        )
        sigma_x_pos_by_niche = pyro.param(
            "sigma_x_pos_by_niche",
            self.local_sigma_init * self.Y.new_ones((1,)),
            constraint=dist.constraints.positive,
        )
        sigma_x_neg_by_niche = pyro.param(
            "sigma_x_neg_by_niche",
            self.local_sigma_init * self.Y.new_ones((1,)),
            constraint=dist.constraints.positive,
        )
        sigma_y_pos_by_niche = pyro.param(
            "sigma_y_pos_by_niche",
            self.local_sigma_init * self.Y.new_ones((1,)),
            constraint=dist.constraints.positive,
        )
        sigma_y_neg_by_niche = pyro.param(
            "sigma_y_neg_by_niche",
            self.local_sigma_init * self.Y.new_ones((1,)),
            constraint=dist.constraints.positive,
        )
        tau_by_niche = pyro.param(
            "tau_by_niche",
            self.Y.new_ones((1,)),
            constraint=dist.constraints.positive,
        )

        # Ensure adjacency indices are available
        if not hasattr(self, "adj_rows") or not hasattr(self, "adj_cols"):
            adj = self.spatial_con.coalesce()
            self.adj_rows = adj.indices()[0]
            self.adj_cols = adj.indices()[1]

        # Non-elliptical rotated kernel with niche-specific parameters.
        # Use source-spot niche for outgoing edge distribution i -> j.
        x = self.spatial_coords[:, 0]
        y = self.spatial_coords[:, 1]
        src = self.adj_rows
        dst = self.adj_cols

        theta_e = theta_by_niche.expand(src.size(0), -1).squeeze()  # shape: (n_edges,)
        sx_pos_e = sigma_x_pos_by_niche.expand(src.size(0), -1).squeeze()  # shape: (n_edges,)
        sx_neg_e = sigma_x_neg_by_niche.expand(src.size(0), -1).squeeze()  # shape: (n_edges,)
        sy_pos_e = sigma_y_pos_by_niche.expand(src.size(0), -1).squeeze()  # shape: (n_edges,)
        sy_neg_e = sigma_y_neg_by_niche.expand(src.size(0), -1).squeeze()  # shape: (n_edges,)
        tau_e = tau_by_niche.expand(src.size(0), -1).squeeze()  # shape: (n_edges,)

        dx = x[dst] - x[src]
        dy = y[dst] - y[src]

        cos_t = torch.cos(theta_e)
        sin_t = torch.sin(theta_e)
        u = cos_t * dx - sin_t * dy  # Rotated x-axis coordinate
        v = sin_t * dx + cos_t * dy  # Rotated y-axis coordinate

        u_pos = torch.relu(u)  # Positive part of u
        u_neg = torch.relu(-u)  # Negative part of u
        v_pos = torch.relu(v)  # Positive part of v
        v_neg = torch.relu(-v)  # Negative part of v

        q = (
            (u_pos / (sx_pos_e + eps)) ** 2 +
            (u_neg / (sx_neg_e + eps)) ** 2 +
            (v_pos / (sy_pos_e + eps)) ** 2 +
            (v_neg / (sy_neg_e + eps)) ** 2
        )  # shape: (n_edges,), represents squared distance in anisotropic rotated space normalized by niche-specific sigmas
        w_local = torch.exp(-0.5 * q / ((tau_e + eps) ** 2))

        # Row-wise normalization: each source spot distributes outgoing mass across neighbors.
        row_sums_local = w_local.new_zeros(self.n_spots)
        row_sums_local = row_sums_local.index_add(0, src, w_local)
        w_local = w_local / (row_sums_local[src] + eps)

        row_sums = w_local.new_zeros(self.n_spots)
        row_sums.index_add_(0, src, w_local)  # compute row sums for normalization
        H = - (w_local * (w_local.clamp_min(1e-8)).log())  # per-edge contribution
        row_H = w_local.new_zeros(self.n_spots)
        row_H.index_add_(0, src, H)  # sum contributions for each source spot to get row-wise entropy
        entropy_penalty = pyro.param("w_entropy_lambda", self.Y.new_tensor(1e-3), constraint=dist.constraints.positive)
        pyro.factor("kernel_entropy", -entropy_penalty * row_H.mean())

        # Build the graph Laplacian fresh from the current learned weights so
        # that gradients flow back through w_local → sigma_x/y, theta, etc.
        L_live = _graph_laplacian_from_sparse(
            self.adj_rows, self.adj_cols, w_local, self.n_spots
        )

        # ========== TRUE EXPRESSION MODEL VIA DETERMINISTIC LATENT DECODER ==========
        in_tissue_spots = self.tissue_spots
        n_tissue = len(in_tissue_spots)

        # Deterministic latent from encoder input (clean reference X_tissue).
        X_tissue, _, X_img_z = self.normalize()
        if self.use_image and self.n_img > 0:
            z = self.denoising_ae.encode(X_tissue, X_img_z)
        else:
            z = self.denoising_ae.encode(X_tissue, None)

        pyro.deterministic("z_deterministic", z)

        # ========== AUXILIARY REPRESENTATION LOSS DURING MAIN PYRO TRAINING ==========
        # Keep optional niche supervision only.
        if self.enable_aux_losses and self.aux_lambda_niche > 0.0 and n_tissue >= 2:
            niche_targets = self.niche_labels[in_tissue_spots].long().clamp(min=0, max=self.n_niche_types - 1)
            niche_logits = self.niche_classifier(z)
            loss_niche = F.cross_entropy(niche_logits, niche_targets)
            pyro.factor("aux_niche_penalty", -self.aux_lambda_niche * loss_niche)
            self._last_loss_niche = float(loss_niche.detach().cpu().item())
        else:
            self._last_loss_niche = None
                    
        # Decode deterministic latent to log-scale clean expression for tissue spots.
        log_true_tissue = self.denoising_ae.decode_genes(z, X_img_z)
        
        # Clamp log values to avoid numerical overflow/underflow in exp
        log_true_tissue = torch.clamp(log_true_tissue, min=-20.0, max=15.0)
        
        true_tissue = torch.exp(log_true_tissue)
        
        true_expression_expanded = true_tissue.new_zeros((self.n_spots, self.n_genes))
        true_expression_expanded[in_tissue_spots] = true_tissue
        
        # rescale true expression to match observed library sizes
        true_lib_sizes = true_expression_expanded.sum(dim=1)  # (n_spots,)
        scale_factors = self.lib_sizes / (true_lib_sizes + 1e-8)  # (n_spots,)
        true_expression_expanded = true_expression_expanded * scale_factors.unsqueeze(1)

        # Record clean (pre-diffusion) field as a deterministic site
        pyro.deterministic("true_expression_expanded", true_expression_expanded)
                
        # # --- Anti-smoothness hinge penalty on AE-produced true field ---
        # antismooth_lambda = pyro.param(
        #     "antismooth_lambda",
        #     self.Y.new_tensor(1.0),
        #     constraint=dist.constraints.positive,
        # )
        # E_min = pyro.param(
        #     "antismooth_E_min",
        #     self.Y.new_tensor(1e-3),
        #     constraint=dist.constraints.positive,
        # )

        # # Compute Dirichlet energy on tissue spots only
        # T = true_expression_expanded
        # LT = torch.sparse.mm(L_live, T)  # (n_spots, n_genes)

        # # restrict to tissue spots to avoid boundary artifacts
        # tidx = self.tissue_spots
        # E = (T[tidx] * LT[tidx]).sum()

        # # normalize to make scale roughly stable across datasets
        # E_norm = E / (float(len(tidx)) * float(self.n_genes) + 1e-8)

        # hinge = torch.relu(E_min - E_norm)
        # pyro.factor("antismooth_hinge", -antismooth_lambda * hinge)

        # ========== CONTAMINATION PROCESS ==========
        # Prepare per-step diffusion coefficients (spot-specific, per-gene)
        # Teleportation parameter for spatial mixing (tunable during training)
        laplacian_teleport = pyro.param(
            "laplacian_teleport",
            torch.tensor(0.01, device=self.Y.device),
            constraint=dist.constraints.interval(0.0, 0.05)
        )

        # Compute contaminated expression via forward diffusion process
        # Initialized as total_diffusion / n_steps. They will be learned during inference.
        per_step_diffusion = spot_diffusion_rates.unsqueeze(1).expand(-1, self.n_genes)
        contaminated_expression = self._compute_contamination(
            true_expression=true_expression_expanded,
            diffusion_coeff=per_step_diffusion,  # (n_spots, n_genes)
            L=L_live,                            # fresh Laplacian – gradients flow through w_local
            n_steps=self.n_diffusion_steps,
            teleport=laplacian_teleport,
        )

        # Post-processing: mild per-gene rescaling to preserve total counts
        # This prevents artificial norm shrinkage while allowing model flexibility
        total_before = true_expression_expanded.sum(dim=0)
        total_after = contaminated_expression.sum(dim=0)
        scale = torch.clamp(total_before / (total_after + 1e-8), min=0.0, max=1.3)
        # contaminated_expression = contaminated_expression * scale  # DEBUG

        # ========== OBSERVATION MODEL (likelihood) ==========

        scale = 1.0 / (self.n_spots)   # Scale factor for likelihood to keep magnitudes reasonable

        with poutine.scale(scale=scale):
            with pyro.plate("genes_obs", self.n_genes):
                if self.gene_likelihood == "nb":
                    obs_px_r = pyro.param(
                        "obs_px_r",
                        2.0 * torch.ones(self.n_genes, device=self.Y.device),
                        constraint=dist.constraints.positive
                        ).expand(self.n_spots, -1)  # Shape: (n_spots, n_genes)
                elif self.gene_likelihood == "gamma_poisson":                        
                    obs_concentration = pyro.param(
                        "obs_concentration",
                        2.0 * torch.ones(self.n_genes, device=self.Y.device),
                        constraint=dist.constraints.positive
                        ).expand(self.n_spots, -1)  # Shape: (n_spots, n_genes)
                elif self.gene_likelihood == "poisson":
                    pass
                elif self.gene_likelihood == "zip":
                    obs_zero_prob = pyro.param(
                        "obs_zero_prob",
                        0.1 * torch.ones(self.n_genes, device=self.Y.device),
                        constraint=dist.constraints.positive
                        ).expand(self.n_spots, -1)  # Shape: (n_spots, n_genes)                    
                elif self.gene_likelihood == "zinb":
                    obs_px_r = pyro.param(
                        "obs_px_r",
                        2.0 * torch.ones(self.n_genes, device=self.Y.device),
                        constraint=dist.constraints.positive
                        ).expand(self.n_spots, -1)  # Shape: (n_spots, n_genes)
                    obs_zero_prob = pyro.param(
                        "obs_zero_prob",
                        0.1 * torch.ones(self.n_genes, device=self.Y.device),
                        constraint=dist.constraints.positive
                        ).expand(self.n_spots, -1)  # Shape: (n_spots, n_genes)
                    
                with pyro.plate("spots_obs", self.n_spots):                
                    if self.gene_likelihood == "nb":
                        probs = obs_px_r / torch.clamp(obs_px_r + contaminated_expression, min=1e-6)
                        probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)  # Avoid numerical issues
                        pyro.sample(
                            'observed_expression',
                            dist.NegativeBinomial(total_count=obs_px_r, probs=probs),
                            obs=self.Y
                        )
                    elif self.gene_likelihood == "gamma_poisson":                        
                        obs_rate = obs_concentration / torch.clamp(contaminated_expression, min=1e-4)
                        pyro.sample(
                            'observed_expression',
                            dist.GammaPoisson(concentration=obs_concentration, rate=obs_rate),
                            obs=self.Y
                        )
                    elif self.gene_likelihood == "poisson":
                        safe_contamination = torch.clamp(contaminated_expression, min=1e-6)
                        pyro.sample("observed_expression",
                                    dist.Poisson(safe_contamination),
                                    obs=self.Y)
                    elif self.gene_likelihood == "zip":
                        safe_contamination = torch.clamp(contaminated_expression, min=1e-6)
                        pyro.sample(
                            "observed_expression",
                            dist.ZeroInflatedPoisson(
                                gate=obs_zero_prob,
                                rate=safe_contamination
                            ),
                            obs=self.Y
                        )
                    elif self.gene_likelihood == "zinb":
                        probs = obs_px_r / torch.clamp(obs_px_r + contaminated_expression, min=1e-6)
                        probs = torch.clamp(probs, min=1e-6, max=1.0 - 1e-6)  # Avoid numerical issues
                        safe_contamination = torch.clamp(contaminated_expression, min=1e-6)
                        pyro.sample(
                            "observed_expression",
                            dist.ZeroInflatedNegativeBinomial(
                                gate=obs_zero_prob,
                                total_count=obs_px_r,
                                probs=probs
                            ),
                            obs=self.Y
                        )
                    else:
                        raise ValueError(f"Unsupported gene likelihood: {self.gene_likelihood}")


        return true_expression_expanded, contaminated_expression

    def forward(self, *args, **kwargs):
        """
        Torch/pyro expects a `forward` method on modules for rendering / tracing.
        Delegate to the probabilistic `model` method so pyro.render_model can inspect sampling sites.
        """
        return self.model(*args, **kwargs)

    def guide(self):
        """
        Variational guide for diffusion parameters.
        Matches sampling sites in the model (currently: spot_diffusion_rates).
        """        
        pyro.module("denoising_ae", self.denoising_ae)  # Register autoencoder module

        # ========== SPOT-SPECIFIC DIFFUSION PARAMETERS ==========
        # Use precomputed spot diffusion rate initialization directly
        spot_diffusion_rates_loc = pyro.param(
            "spot_diffusion_rates_loc",
            self.spot_diffusion_rate_init,
            constraint=dist.constraints.unit_interval
        )
        prior_strength = pyro.param(
            "diffusion_rate_prior_strength",
            torch.tensor(5.0, device=self.Y.device),
            constraint=dist.constraints.positive
        )
        with pyro.plate("spots", self.n_spots):
            alpha_diff = prior_strength * spot_diffusion_rates_loc + 1e-4
            beta_diff = prior_strength * (1.0 - spot_diffusion_rates_loc) + 1e-4
            pyro.sample("spot_diffusion_rates", dist.Beta(alpha_diff, beta_diff))
        
    def robust_log(self, x):
        """
        Stable log function to clip extreme values (avoid both 0 and very large values).
        """
        # data informed upper bound
        upper = torch.quantile(x, 0.999, dim=0, keepdim=True)
        lower = torch.tensor(1e-4)
        x = torch.max(torch.min(x, upper), lower)
        return torch.log(x)
    
    def zero_mask_regularizer(
        self,
        true_expression_expanded: torch.Tensor,
        weight: float = None,
        epsilon: float = None,
        only_in_tissue: bool = None,
        normalize: bool = True,
        power: float = 1.0,
    ) -> torch.Tensor:
        """
        Penalize non-zero true expression where observation Y is zero.
        - weight: penalty multiplier (defaults to self.zero_penalty_weight)
        - epsilon: zero threshold on Y (defaults to self.zero_mask_epsilon)
        - only_in_tissue: restrict penalty to tissue spots (defaults to self.penalize_only_in_tissue)
        - normalize: divide by number of zero entries for scale invariance
        - power: exponent for violation magnitude (1=L1, 2=L2-like)
        Returns a scalar tensor.
        """
        w = self.zero_penalty_weight if weight is None else true_expression_expanded.new_tensor(weight)
        eps = self.zero_mask_epsilon if epsilon is None else self.Y.new_tensor(epsilon)
        oit = self.penalize_only_in_tissue if only_in_tissue is None else only_in_tissue

        # Check for NaN in input
        if torch.isnan(true_expression_expanded).any():
            print(f"WARNING: NaN detected in true_expression_expanded. Replacing with small values")
            true_expression_expanded = torch.where(
                torch.isnan(true_expression_expanded),
                torch.tensor(1e-8, device=true_expression_expanded.device),
                true_expression_expanded
            )
        
        # Clamp to valid range to prevent numerical issues
        true_expression_expanded = torch.clamp(true_expression_expanded, min=0.0, max=1e6)

        # mask where observation is (near) zero
        zero_mask = (self.Y <= eps)
        if oit:
            tissue_mask = self.Y.new_zeros(self.n_spots, dtype=torch.bool)
            tissue_mask[self.tissue_spots] = True
            zero_mask = zero_mask & tissue_mask.unsqueeze(1)

        # violations: predicted true expression on observed-zero spots
        violations = true_expression_expanded[zero_mask]
        if violations.numel() == 0:
            return true_expression_expanded.new_zeros(())

        loss = (violations.float() ** power).sum()
        # if power != 1.0:
        #     w = 1.0  # avoid double-scaling
        if normalize:
            loss = loss / (zero_mask.sum().float() + 1e-8)

        # Ensure return value is not NaN
        if torch.isnan(loss):
            print(f"WARNING: NaN in zero_mask_regularizer loss. Returning 0")
            return true_expression_expanded.new_zeros(())

        return w * loss
