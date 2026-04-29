import torch
from torch.nn.functional import pdist
import torch.nn.functional as F

from functools import partial
from joblib import Parallel, delayed

from pde import CartesianGrid, ScalarField
from pde import DiffusionPDE
from pde.tools.numba import jit
from numba.extending import register_jitable
import numpy as np
from scipy.sparse import issparse


class forward_diffusion():
    def __init__(self, 
                grid_sizes: tuple, 
                voxel_sizes: tuple, 
                padding_sizes: tuple,
                x: torch.Tensor,  # Shape: (n_coords,)
                y: torch.Tensor,  # Shape: (n_coords,)
                coords: torch.Tensor,  # Shape: (n_coords, 2) 
                out_tissue_mask: np.ndarray,
                diff_mask: np.ndarray,  
                diffusivity=0.2, noise=0.01, 
                ):
        '''
        Simulate 2d spatial diffusion from initial condition x to reference condition y. 
        
        Args:
            grid_sizes: a tuple or list of two integers defining the grid size (n_x, n_y)
            voxel_sizes: a tuple or list of two floats defining the voxel size (dx, dy)
            padding_sizes: a tuple or list of two floats defining the padding size (pad_x, pad_y) at the scale of original coordinate space (not grids)
            x: a 1D tensor of shape (n_coords,) representing the initial condition at coordinates
            y: a 1D tensor of shape (n_coords,) representing the reference condition at coordinates (for restraining the diffusion)
            out_tissue_mask: a 1D tensor of shape (n_coords,) representing the mask for out-tissue regions (0 for out-of-tissue)
            coords: a 2D tensor of shape (n_coords, 2) representing the coordinates of the points
            diff_mask: a 2D tensor or list of shape (n_coords,) representing the total counts at each coordinate
            diffusivity: a float representing the diffusivity coefficient (default: 0.2)
            noise: a float representing the noise amplitude (default: 0.01)
        '''
        
        self.grid_sizes = grid_sizes
        self.voxel_sizes = voxel_sizes
        self.padding_sizes = padding_sizes
        self.x = x
        self.y = y
        self.coords = coords
        self.diffusivity = diffusivity
        self.noise = noise
        self.out_tissue_mask = out_tissue_mask
        self.diff_mask = diff_mask

    @staticmethod
    def post_step_hook(state_data, t, threshold):
        if state_data.max() < threshold:
            raise StopIteration

    def run(self):
        grids = self.define_grids()
        x_grid = grids['x_grid']
        y_grid = grids['y_grid']
        grid_coords = grids['grid_coords']
        out_tissue_mask = self.out_tissue_mask
        diff_mask = self.diff_mask
        
        # diff params 
        diffusivity = self.diffusivity
        noise = self.noise
        n_x, n_y = self.grid_sizes.tolist()
        grid = CartesianGrid([[0, n_x], [0, n_y]], [n_x, n_y])
        state = ScalarField(grid, x_grid)  # generate initial condition
        post_step_hook = partial(forward_diffusion.post_step_hook, threshold=self.y.max())  # Define a post-step hook to stop simulation if max value exceeds threshold

        eq = SpatialDiffusionPDE(diffusivity=diffusivity, noise=noise, 
                                    out_tissue_mask=out_tissue_mask, 
                                    diff_mask=diff_mask, 
                                    post_step_hook=post_step_hook)
        result = eq.solve(state, t_range=100, dt=0.1)
        
        y_end = result.data[grid_coords[:, 0], grid_coords[:, 1]]
        print('simulated y_end sum, original y sum:')
        print(y_end.sum(), self.y.sum())
        y_end = y_end / (y_end.sum() / self.y.sum())  # normalize to match original data
        return y_end.detach()

    def define_grids(self):
        grid_sizes = self.grid_sizes
        voxel_sizes = self.voxel_sizes
        padding_sizes = self.padding_sizes
        x = self.x
        coords = self.coords
        y = self.y
        
        x_grid, grid_coords = coords_to_filled_grid(
            grid_size=grid_sizes,
            dx=voxel_sizes[0],
            dy=voxel_sizes[1],
            padding_sizes=padding_sizes,
            x=x,
            coords=coords
        )

        y_grid = coords_to_filled_grid(
            grid_size=grid_sizes,
            dx=voxel_sizes[0],
            dy=voxel_sizes[1],
            padding_sizes=padding_sizes,
            x=y,
            coords=coords
        )[0]

        return {
            'x_grid': x_grid.detach().numpy() if x_grid.requires_grad else x_grid.numpy(),
            'y_grid': y_grid.detach().numpy() if y_grid.requires_grad else y_grid.numpy(),
            'grid_coords': grid_coords.detach().numpy() if grid_coords.requires_grad else grid_coords.numpy(),
        }

class EarlyStopping:
    def __init__(self, patience=10, min_delta=0.0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best_score = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, current_score):
        if self.best_score is None:
            self.best_score = current_score
            return False

        improvement = (current_score < self.best_score - self.min_delta) if self.mode == 'min' else (current_score > self.best_score + self.min_delta)

        if improvement:
            self.best_score = current_score
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True

        return self.early_stop

class SpatialDiffusionPDE(DiffusionPDE):
    """Diffusion PDE with custom diffusivity and noise implementations."""

    def __init__(self, diffusivity=0.1, noise=0.1, 
                out_tissue_mask=None, diff_mask=None, post_step_hook=None):
        """
        Parameters:
        -----------
        base_diffusivity : float
            Base diffusion coefficient
        noise : float
            Noise amplitude
        out_tissue_mask : ndarray
            Mask for out-of-tissue regions for noise scaling
        diff_mask : ndarray
            Spatial mask to modify diffusivity at different locations
        """
        super().__init__(diffusivity=diffusivity, noise=noise)
        self.out_tissue_mask = out_tissue_mask
        self.diff_mask = diff_mask if diff_mask is not None else 1.0
        self.post_step_hook = post_step_hook

    # def evolution_rate(self, state, t=0):
    #     """Numpy implementation of the evolution equation"""
    #     # Calculate the Laplacian term
    #     state_lap = state.laplace(bc="auto_periodic_neumann")
    #     # Calculate the gradient terms
    #     state_grad = state.gradient(bc="auto_periodic_neumann")
    #     diff_grad = ScalarField(state.grid, self.diff_mask).gradient(bc="auto_periodic_neumann")
        
    #     # Combine terms: D∇²c + ∇D·∇c
    #     result = (self.diff_mask * state_lap + 
    #              sum(g1 * g2 for g1, g2 in zip(diff_grad, state_grad)))
    #     return self.diffusivity * result

    # def _make_pde_rhs_numba(self, state):
    #     """Numba implementation of the PDE"""
    #     base_diff = float(self.diffusivity)
    #     diff_mask = self.diff_mask.copy()  # Make a copy for numba
        
    #     # Get operators
    #     laplace = state.grid.make_operator("laplace", bc="auto_periodic_neumann")
    #     gradient = state.grid.make_operator("gradient", bc="auto_periodic_neumann")
        
    #     @jit
    #     def pde_rhs(state_data, t):
    #         # Calculate Laplacian term: D∇²c
    #         lap_term = diff_mask * laplace(state_data)
            
    #         # Calculate gradient terms: ∇D·∇c
    #         state_grad = gradient(state_data)
    #         diff_grad = gradient(diff_mask)
            
    #         # Sum up the dot product of gradients
    #         grad_term = np.zeros_like(state_data)
    #         for i in range(len(state_grad)):
    #             grad_term += diff_grad[i] * state_grad[i]
            
    #         return base_diff * (lap_term + grad_term)
            
    #     return pde_rhs

    def _make_noise_realization_numba(self, state):
        """Numba implementation of spatially-dependent noise."""
        # Cache variables for numba
        noise_amplitude = float(self.noise)
        out_tiss_mask = self.out_tissue_mask.copy() if self.out_tissue_mask is not None else 1.0

        @jit
        def noise_realization(state_data, t):
            mask = out_tiss_mask + 0.1
            noise_field = np.random.uniform(0, noise_amplitude, size=state_data.shape)
            return noise_field * mask

        return noise_realization
    
    def make_post_step_hook(self, state):
        """Returns a function that is called after each step.

        Args:
            state (:class:`~pde.fields.FieldBase`):
                An example for the state from which the grid and other information can
                be extracted

        Returns:
            tuple: The first entry is the function that implements the hook. The second
                entry gives the initial data that is used as auxiliary data in the hook.
                This can be `None` if no data is used.

        Raises:
            NotImplementedError: When :attr:`post_step_hook` is `None`.
        """
        if self.post_step_hook is None:
            raise NotImplementedError("`post_step_hook` not set")
        else:
            post_step_hook = register_jitable(self.post_step_hook)

            @register_jitable
            def post_step_hook_impl(state_data, t, post_step_data):
                post_step_hook(state_data, t)

            return post_step_hook_impl, 0  # hook function and initial value


def calculate_domain_parameters(coords: torch.Tensor, divideby: float = 1.0):
    """
    Calculate appropriate domain parameters based on input coordinates.
    
    Args:
        coords: Tensor of shape (n, 2) containing x, y coordinates
        divideby: Factor to divide minimum distance by for voxel size
    
    Returns:
        domain_sizes: Physical size of domain in micrometers
        grid_sizes: Number of voxels in each dimension
        voxel_sizes: Size of each voxel in micrometers
        diffusion_const: Diffusion coefficient in μm²/s
        padding_sizes: Size of padding in each dimension
    """
    # Calculate the physical extent of the domain
    x_min, y_min = torch.min(coords, dim=0)[0]
    x_max, y_max = torch.max(coords, dim=0)[0]
    
    padding = 0.0 # 10% padding
    
    # Add padding to domain boundaries
    domain_width = (x_max - x_min) * (1 + 2*padding)
    domain_height = (y_max - y_min) * (1 + 2*padding)
    domain_sizes = torch.tensor([domain_width, domain_height], device=coords.device)
    padding_sizes = torch.tensor([
        (x_max - x_min) * padding,
        (y_max - y_min) * padding
    ], device=coords.device)
    
    # Calculate minimum distance between coords
    # Handle x coordinates
    x_coords = torch.unique(coords[:, 0:1], dim=0)
    x_dists = torch.unique(pdist(x_coords), return_counts=True)
    min_distance_x = x_dists[0][torch.argmax(x_dists[1])]
    
    # Handle y coordinates
    y_coords = torch.unique(coords[:, 1:2], dim=0)
    y_dists = torch.unique(pdist(y_coords), return_counts=True)
    min_distance_y = y_dists[0][torch.argmax(y_dists[1])]
    
    # Set voxel size to be 1/divideby of minimum distance between coords
    voxel_sizes = torch.tensor([
        min_distance_x / divideby,
        min_distance_y / divideby
    ], device=coords.device)
    
    # Calculate grid size (number of voxels)
    grid_sizes = domain_sizes / voxel_sizes + 1
    grid_sizes = torch.ceil(grid_sizes).to(torch.int64)
    
    # Set diffusion coefficient based on literature x for mRNA
    diffusion_const = 1.0  # μm²/s
    
    return domain_sizes, grid_sizes, voxel_sizes, diffusion_const, padding_sizes


def coords_to_grid(
    grid_sizes,
    dx, 
    dy,
    padding_sizes,
    x: torch.Tensor,  # Shape: (n_coords,)
    coords: torch.Tensor = None,  # Shape: (n_coords, 2)
) -> torch.Tensor:
    """
    Convert point observations to grid x with gap filling
    
    Args:
        coords: Point coordinates (n_coords, 2)
        x: Observed x at coords (n_coords,)
        
    Returns:
        Grid with interpolated x
    """
    # Create empty grid
    grid = torch.zeros(grid_sizes[0], grid_sizes[1], dtype=torch.float32)
    
    # Convert coords to nearest grid indices
    x_idx = torch.round((coords[:, 0] - coords[:, 0].min() + padding_sizes[0]) / dx).long()
    y_idx = torch.round((coords[:, 1] - coords[:, 1].min() + padding_sizes[1]) / dy).long()
    
    # # Clamp indices to valid range
    # x_idx = torch.clamp(x_idx, 0, grid_sizes[0] - 1)
    # y_idx = torch.clamp(y_idx, 0, grid_sizes[1] - 1)
    grid_coords = torch.stack((x_idx, y_idx), dim=1)
    
    # Assign x to nearest grid coords
    for i in range(len(coords)):
        grid[x_idx[i], y_idx[i]] = x[i]
    return grid, grid_coords


def fill_grid_gaps(grid: torch.Tensor, min_neighs=3) -> torch.Tensor:
    """
    Fill gaps in the grid by averaging neighboring non-zero x.
    Args:
        grid: Input grid with some zero x (shape: (H, W))
    Returns:
        grid: Grid with gaps filled
        grid_coords: Coordinates of the grid coords
    """
    # Create mask for original non-zero coords
    original_mask = grid != 0
    
    # Iteratively fill gaps
    max_iterations = 10  # Limit iterations to avoid infinite loops
    for _ in range(max_iterations):
        # Store previous grid for convergence check
        previous_grid = grid.clone()
        
        # Add batch and channel dimensions before padding
        grid_expanded = grid.unsqueeze(0).unsqueeze(0)
        padded_expanded = F.pad(grid_expanded, (1,1,1,1), mode='replicate')
        padded = padded_expanded.squeeze(0).squeeze(0)
        
        # Get neighboring x
        neighbors = torch.stack([
            padded[1:-1, :-2],  # left
            padded[1:-1, 2:],   # right
            padded[:-2, 1:-1],  # up
            padded[2:, 1:-1],   # down
        ])
        
        # Count non-zero neighbors
        non_zero_neighbors = (neighbors != 0).sum(dim=0)
        
        # Calculate mean of non-zero neighbors
        neighbor_sum = neighbors.sum(dim=0)
        neighbor_count = (neighbors != 0).sum(dim=0).clamp(min=1)  # Avoid division by zero
        neighbor_mean = neighbor_sum / neighbor_count
        
        # Create mask for coords to be filled
        fill_mask = (grid == 0) & (non_zero_neighbors >= min_neighs)
        
        # Fill gaps
        grid[fill_mask] = neighbor_mean[fill_mask]
        
        # Check for convergence
        if torch.allclose(grid, previous_grid):
            break
            
    # Restore original x
    grid[original_mask] = previous_grid[original_mask]    
    return grid


def coords_to_filled_grid(
    grid_size,
    dx, 
    dy,
    padding_sizes,
    x: torch.Tensor,  # Shape: (n_coords,)
    coords: torch.Tensor = None,  # Shape: (n_coords, 2)
) :
    """
    Convert point observations to grid x with gap filling
    
    Args:
        coords: Point coordinates (n_coords, 2)
        x: Observed x at coords (n_coords,)
        
    Returns:
        Grid with interpolated x
    """
    grid, grid_coords = coords_to_grid(grid_size, dx, dy, padding_sizes, x, coords)
    filled_grid = fill_grid_gaps(grid)
    
    return filled_grid, grid_coords


# class SparseGPRegression(GPModel):
#     def __init__(
#         self, X, y, kernel, 
#         coords, in_tiss_mask, ttl_cnts,
#         noise=None, approx=None, jitter=1e-6
#     ):
#         assert isinstance(
#             X, torch.Tensor
#         ), "X needs to be a torch Tensor instead of a {}".format(type(X))
#         if y is not None:
#             assert isinstance(
#                 y, torch.Tensor
#             ), "y needs to be a torch Tensor instead of a {}".format(type(y))
            
#         self.n_genes, self.n_spots = X.shape
#         n_inducing = int(torch.sqrt(torch.tensor(self.n_genes)))  # number of inducing points
#         mean_function = partial(self.mean_function, coords=coords, ref_count=y,
#             in_tiss_mask=in_tiss_mask, ttl_cnts=ttl_cnts)
#         super().__init__(X, y, kernel=kernel, mean_function=mean_function, jitter=jitter)
#         X = torch.clamp(X, min=1e-6)  # Ensure non-negative counts
#         self.X_init = X
        
#         X_centered = X - X.mean(dim=1, keepdim=True)
#         X_normalized = X_centered / (X_centered.std(dim=1, keepdim=True) + 1e-10)  # Normalize to unit variance, avoid division by zero
#         # Calculate correlation matrix [n_spots, n_spots]
#         correlation_matrix = torch.mm(X_normalized.t(), X_normalized) / self.n_genes
#         # Add small diagonal regularization to ensure positive-definiteness
#         epsilon = 1e-2
#         correlation_matrix += epsilon * torch.eye(correlation_matrix.size(0), device=correlation_matrix.device)
#         # Cholesky decomposition
#         shared_scale_tril = torch.linalg.cholesky(correlation_matrix)

#         self.X = pyro.nn.PyroSample(
#             lambda self: dist.TransformedDistribution(
#                 base_distribution=dist.MultivariateNormal(
#                     loc=X.log(),  # Transform to log space
#                     scale_tril=shared_scale_tril
#                 ),
#                 transforms=[dist.transforms.ExpTransform()]  # Transform back to original space
#             ).expand([self.n_genes]).to_event(1)
#         )
        
#         Xu = stats.resample(X.log(), n_inducing)  # Initialize in log-space
#         self.Xu = PyroParam(Xu, constraint=constraints.real)
#         assert not torch.isnan(self.Xu).any(), "Xu contains NaNs"
#         assert not torch.isinf(self.Xu).any(), "Xu contains Infs"
        
#         noise = self.X.new_tensor(1.0) if noise is None else noise
#         self.noise = PyroParam(noise, constraints.real)

#         if approx is None:
#             self.approx = "VFE"
#         elif approx in ["DTC", "FITC", "VFE"]:
#             self.approx = approx
#         else:
#             raise ValueError(
#                 "The sparse approximation method should be one of "
#                 "'DTC', 'FITC', 'VFE'."
#             )
        
#         self.in_tiss_mask = in_tiss_mask
#         self.ttl_cnts = ttl_cnts if isinstance(ttl_cnts, torch.Tensor) else torch.tensor(ttl_cnts, dtype=torch.float32)
#         self.coords = coords

#     @pyro_method
#     def model(self):
#         self.set_mode("model")

#         # W = (inv(Luu) @ Kuf).T
#         # Qff = Kfu @ inv(Kuu) @ Kuf = W @ W.T
#         # Fomulas for each approximation method are
#         # DTC:  y_cov = Qff + noise,                   trace_term = 0
#         # FITC: y_cov = Qff + diag(Kff - Qff) + noise, trace_term = 0
#         # VFE:  y_cov = Qff + noise,                   trace_term = tr(Kff-Qff) / noise
#         # y_cov = W @ W.T + D
#         # trace_term is added into log_prob

#         N = self.X.size(0)
#         M = self.Xu.size(0)
#         Kuu = self.kernel(self.Xu).contiguous()
#         Kuu.view(-1)[:: M + 1] += self.jitter  # add jitter to the diagonal
#         assert not torch.isnan(Kuu).any(), "Kuu contains NaNs"
#         assert not torch.isinf(Kuu).any(), "Kuu contains Infs"
#         Luu = torch.linalg.cholesky(Kuu)  # the Cholesky decomposition of Kuu = Luu @ Luu.T
#         Kuf = self.kernel(self.Xu, self.X)
#         W = torch.linalg.solve_triangular(Luu, Kuf, upper=False).t()  # W = inv(Luu).T @ Kuf = Kfu @ inv(Luu).T (an approximation of Kfu @ inv(Kuu))

#         D = self.noise.expand(N)
#         if self.approx == "FITC" or self.approx == "VFE":
#             Kffdiag = self.kernel(self.X, diag=True)  # diagonal of Kff
#             Qffdiag = W.pow(2).sum(dim=-1)  # 
#             if self.approx == "FITC":
#                 D = D + Kffdiag - Qffdiag
#             else:  # approx = "VFE"
#                 trace_term = (Kffdiag - Qffdiag).sum() / self.noise
#                 trace_term = trace_term.clamp(min=0)

#         f_loc = self.mean_function(self.X)
#         if self.y is None:
#             f_var = D + W.pow(2).sum(dim=-1)
#             return f_loc, f_var
#         else:
#             if self.approx == "VFE":
#                 pyro.factor(self._pyro_get_fullname("trace_term"), -trace_term / 2.0)
#             print(f"f_loc shape: {f_loc.shape}")  # Should match y.shape
#             print(f"W shape: {W.shape}")
#             print(f"D shape: {D.shape}")
#             print(f"y shape: {self.y.shape if self.y is not None else None}")
            
#             return pyro.sample(
#                 self._pyro_get_fullname("y"),
#                 dist.LowRankMultivariateNormal(f_loc, W, D)
#                 # .expand_by(self.y.shape[:-1])
#                 .to_event(self.y.dim() - 1),
#                 obs=self.y,
#             )

#     @pyro_method
#     def guide(self):
#         self.set_mode("guide")
#         self._load_pyro_samples()
#         self.X_loc = PyroParam(
#             torch.log(self.X_init).detach().clone().to(self.X.device),
#             constraint=constraints.real
#         )
#         self.X_scale_tril = PyroParam(torch.eye(self.n_spots, device=self.X.device), 
#                                     constraint=constraints.lower_cholesky)
#         self.X = pyro.nn.PyroSample(
#             lambda self: dist.TransformedDistribution(
#                 base_distribution=dist.MultivariateNormal(
#                     loc=self.X_loc, 
#                     scale_tril=self.X_scale_tril
#                 ),
#                 transforms=[dist.transforms.ExpTransform()]  # Transform back to original space
#             ).expand([self.n_genes]).to_event(1)
#         )

#     def forward(self, Xnew, full_cov=False, noiseless=True):
#         r"""
#         Computes the mean and covariance matrix (or variance) of Gaussian Process
#         posterior on a test input data :math:`X_{new}`:

#         .. math:: p(f^* \mid X_{new}, X, y, k, X_u, \epsilon) = \mathcal{N}(loc, cov).

#         .. note:: The noise parameter ``noise`` (:math:`\epsilon`), the inducing-point
#             parameter ``Xu``, together with kernel's parameters have been learned from
#             a training procedure (MCMC or SVI).

#         :param torch.Tensor Xnew: A input data for testing. Note that
#             ``Xnew.shape[1:]`` must be the same as ``self.X.shape[1:]``.
#         :param bool full_cov: A flag to decide if we want to predict full covariance
#             matrix or just variance.
#         :param bool noiseless: A flag to decide if we want to include noise in the
#             prediction output or not.
#         :returns: loc and covariance matrix (or variance) of :math:`p(f^*(X_{new}))`
#         :rtype: tuple(torch.Tensor, torch.Tensor)
#         """
#         self._check_Xnew_shape(Xnew)
#         self.set_mode("guide")

#         # W = inv(Luu) @ Kuf
#         # Ws = inv(Luu) @ Kus
#         # D as in self.model()
#         # K = I + W @ inv(D) @ W.T = L @ L.T
#         # S = inv[Kuu + Kuf @ inv(D) @ Kfu]
#         #   = inv(Luu).T @ inv[I + inv(Luu)@ Kuf @ inv(D)@ Kfu @ inv(Luu).T] @ inv(Luu)
#         #   = inv(Luu).T @ inv[I + W @ inv(D) @ W.T] @ inv(Luu)
#         #   = inv(Luu).T @ inv(K) @ inv(Luu)
#         #   = inv(Luu).T @ inv(L).T @ inv(L) @ inv(Luu)
#         # loc = Ksu @ S @ Kuf @ inv(D) @ y = Ws.T @ inv(L).T @ inv(L) @ W @ inv(D) @ y
#         # cov = Kss - Ksu @ inv(Kuu) @ Kus + Ksu @ S @ Kus
#         #     = kss - Ksu @ inv(Kuu) @ Kus + Ws.T @ inv(L).T @ inv(L) @ Ws

#         N = self.X.size(0)
#         M = self.Xu.size(0)
#         # TODO: cache these calculations to get faster inference
#         Kuu = self.kernel(self.Xu).contiguous()
#         Kuu.view(-1)[:: M + 1] += self.jitter  # add jitter to the diagonal
#         Luu = torch.linalg.cholesky(Kuu)
#         Kuf = self.kernel(self.Xu, self.X)
#         W = torch.linalg.solve_triangular(Luu, Kuf, upper=False)
#         D = self.noise.expand(N)
#         if self.approx == "FITC":
#             Kffdiag = self.kernel(self.X, diag=True)
#             Qffdiag = W.pow(2).sum(dim=0)
#             D = D + Kffdiag - Qffdiag

#         W_Dinv = W / D
#         K = W_Dinv.matmul(W.t()).contiguous()
#         K.view(-1)[:: M + 1] += 1  # add identity matrix to K
#         L = torch.linalg.cholesky(K)

#         # get y_residual and convert it into 2D tensor for packing
#         y_residual = self.y - self.mean_function(self.X)
#         y_2D = y_residual.reshape(-1, N).t()
#         W_Dinv_y = W_Dinv.matmul(y_2D)

#         # End caching ----------

#         Kus = self.kernel(self.Xu, Xnew)
#         Ws = torch.linalg.solve_triangular(Luu, Kus, upper=False)
#         pack = torch.cat((W_Dinv_y, Ws), dim=1)
#         Linv_pack = torch.linalg.solve_triangular(L, pack, upper=False)
#         # unpack
#         Linv_W_Dinv_y = Linv_pack[:, : W_Dinv_y.shape[1]]
#         Linv_Ws = Linv_pack[:, W_Dinv_y.shape[1] :]

#         C = Xnew.size(0)
#         loc_shape = self.y.shape[:-1] + (C,)
#         loc = Linv_W_Dinv_y.t().matmul(Linv_Ws).reshape(loc_shape)

#         if full_cov:
#             Kss = self.kernel(Xnew).contiguous()
#             if not noiseless:
#                 Kss.view(-1)[:: C + 1] += self.noise  # add noise to the diagonal
#             Qss = Ws.t().matmul(Ws)
#             cov = Kss - Qss + Linv_Ws.t().matmul(Linv_Ws)
#             cov_shape = self.y.shape[:-1] + (C, C)
#             cov = cov.expand(cov_shape)
#         else:
#             Kssdiag = self.kernel(Xnew, diag=True)
#             if not noiseless:
#                 Kssdiag = Kssdiag + self.noise
#             Qssdiag = Ws.pow(2).sum(dim=0)
#             cov = Kssdiag - Qssdiag + Linv_Ws.pow(2).sum(dim=0)
#             cov_shape = self.y.shape[:-1] + (C,)
#             cov = cov.expand(cov_shape)

#         return loc + self.mean_function(Xnew), cov

#     def train(self, train_steps=1000):
#         from pyro.infer import SVI, TraceMeanField_ELBO
#         from pyro.optim import ClippedAdam
#         def train_model(model, num_epochs=1000, lr=0.01, patience=10, min_delta=1.0):
#             # Setup optimizer and ELBO
#             optimizer = ClippedAdam({"lr": lr})
#             elbo = TraceMeanField_ELBO()
#             svi = SVI(model.model, model.guide, optimizer, loss=elbo)
#             losses = []
#             # Early stopping monitor
#             early_stopper = EarlyStopping(patience=patience, min_delta=min_delta, mode='min')
#             for epoch in range(num_epochs):
#                 loss = svi.step()
#                 losses.append(loss)
#                 print(f"Epoch {epoch}, ELBO loss: {loss:.4f}")
#                 if early_stopper(loss):
#                     print("Early stopping triggered.")
#                     break
#             return model, losses
#         trained_model, losses = train_model(self, num_epochs=train_steps, lr=0.01, patience=10, min_delta=0.5)
#         # return the model 
#         return trained_model, losses

#     def infer(self, Xnew, full_cov=False, noiseless=True):
#         """
#         Run inference on new data using the trained SparseGPRegression model.

#         Parameters
#         ----------
#         Xnew : torch.Tensor
#             New input data for prediction (shape: [n_test, input_dim]).
#         full_cov : bool, optional
#             Whether to return full covariance matrix. Default is False.
#         noiseless : bool, optional
#             Whether to exclude noise in the prediction. Default is True.

#         Returns
#         -------
#         loc : torch.Tensor
#             Posterior mean of the predictions.
#         cov : torch.Tensor
#             Posterior variance or covariance matrix of the predictions.
#         """
#         self.eval()
#         with torch.no_grad():
#             loc, cov = self.forward(Xnew, full_cov=full_cov, noiseless=noiseless)
#         return loc, cov


#     def save(self, filename_prefix):
#         """
#         Save the model parameters and metadata using Pyro's param store.

#         Parameters
#         ----------
#         filename_prefix : str
#             Prefix for the saved files (e.g., 'model' will create 'model.pt' and 'model_meta.pt')
#         """
#         # Save Pyro parameters
#         pyro.get_param_store().save(f"{filename_prefix}.pt")

#         # Save metadata separately
#         metadata = {
#             'coords': self.coords.detach().cpu(),
#             'in_tiss_mask': self.in_tiss_mask.detach().cpu(),
#             'ttl_cnts': self.ttl_cnts.detach().cpu(),
#             'approx': self.approx,
#             'jitter': self.jitter,
#             'Xu': self.Xu.detach().cpu(),
#             'X': self.X.detach().cpu(),
#             'y': self.y.detach().cpu() if self.y is not None else None,
#             'kernel_state': {
#                 'name': self.kernel.__class__.__name__,
#                 'input_dim': self.kernel.input_dim,
#                 'params': {k: v.detach().cpu() for k, v in self.kernel.named_parameters()}
#             }
#         }
#         torch.save(metadata, f"{filename_prefix}_meta.pt")
#         print(f"Model saved to {filename_prefix}.pt and metadata to {filename_prefix}_meta.pt")

#     @staticmethod
#     def mean_function(X, coords, ref_count, in_tiss_mask, ttl_cnts):
#         domain_sizes, grid_sizes, voxel_sizes, diffusion_const, padding_sizes = calculate_domain_parameters(coords, divideby=1)
#         out_tiss_mask = (in_tiss_mask==0).int()
#         #### Custom noise spatially dependent
#         out_tissue_mask = coords_to_filled_grid(
#             grid_size=grid_sizes,
#             dx=voxel_sizes[0],
#             dy=voxel_sizes[1],
#             padding_sizes=padding_sizes,
#             x=out_tiss_mask,
#             coords=coords
#         )[0]
#         out_tissue_mask = out_tissue_mask.detach().numpy() if out_tissue_mask.requires_grad else out_tissue_mask.numpy()
#         ttl_cnts = torch.tensor(ttl_cnts, dtype=torch.float32) if not isinstance(ttl_cnts, torch.Tensor) else ttl_cnts
#         ttl_grid = coords_to_filled_grid(
#             grid_size=grid_sizes,
#             dx=voxel_sizes[0],
#             dy=voxel_sizes[1],
#             padding_sizes=padding_sizes,
#             x=ttl_cnts,
#             coords=coords
#         )[0]
#         diff_mask = ttl_grid / ttl_grid.max()  # normalize to [0, 1]
#         diff_mask = diff_mask.detach().numpy() if diff_mask.requires_grad else diff_mask.numpy()

#         def process(i):
#             x = X[i, :]
#             y = ref_count[:, i]
#             model = forward_diffusion(
#                 grid_sizes=grid_sizes, voxel_sizes=voxel_sizes, padding_sizes=padding_sizes,
#                 x=x, y=y, coords=coords, 
#                 out_tissue_mask=out_tissue_mask, diff_mask=diff_mask,
#                 diffusivity=0.2, noise=0.01,
#             )
#             return torch.tensor(model.run(), dtype=torch.float32)

#         res = Parallel(n_jobs=-1)(delayed(process)(i) for i in range(X.shape[0]))
#         return torch.stack(res, dim=0).T


#     @staticmethod
#     def load(filename_prefix, custom_mean_func=None):
#         """
#         Load the model parameters and metadata using Pyro's param store.

#         Parameters
#         ----------
#         filename_prefix : str
#             Prefix for the saved files (e.g., 'model' will load 'model.pt' and 'model_meta.pt')
#         custom_mean_func : callable, optional
#             Custom mean function to override the default

#         Returns
#         -------
#         model : SparseGPRegression
#             The loaded model
#         """
#         # Load Pyro parameters
#         params = torch.load(f"{filename_prefix}/model.pt", weights_only=False)
#         pyro.get_param_store().set_state(params)
        
#         # Load metadata
#         metadata = torch.load(f"{filename_prefix}/meta.pt")

#         # Reconstruct kernel
#         kernel_name = metadata['kernel_state']['name']
#         input_dim = metadata['kernel_state']['input_dim']
#         params = metadata['kernel_state']['params']

#         if kernel_name == 'RBF':
#             kernel = gp.kernels.RBF(input_dim=input_dim)
#         elif kernel_name == 'Matern32':
#             kernel = gp.kernels.Matern32(input_dim=input_dim)
#         elif kernel_name == 'Matern52':
#             kernel = gp.kernels.Matern52(input_dim=input_dim)
#         elif kernel_name == 'Periodic':
#             kernel = gp.kernels.Periodic(input_dim=input_dim)
#         elif kernel_name == 'Linear':
#             kernel = gp.kernels.Linear(input_dim=input_dim)
#         else:
#             raise ValueError(f"Unsupported kernel type: {kernel_name}")

#         for name, param in params.items():
#             getattr(kernel, name).data = param

#         # Reconstruct model
#         model = SparseGPRegression(
#             X=metadata['X'],
#             y=metadata['y'],
#             kernel=kernel,
#             coords=metadata['coords'],
#             in_tiss_mask=metadata['in_tiss_mask'],
#             ttl_cnts=metadata['ttl_cnts'],
#             noise=pyro.param("noise") if "noise" in pyro.get_param_store().keys() else None,
#             approx=metadata['approx'],
#             jitter=metadata['jitter']
#         )

#         print(f"Model loaded from {filename_prefix}/model.pt and metadata from {filename_prefix}/meta.pt")
#         return model

#     def infer_latent_X_quantiles(self, quantiles=[0.05, 0.5, 0.95], n_samples=1000):
#         """
#         Sample from the posterior of X and compute quantiles.

#         Parameters
#         ----------
#         quantiles : list of float
#             Quantiles to compute (e.g., [0.05, 0.5, 0.95]).
#         n_samples : int
#             Number of posterior samples to draw.

#         Returns
#         -------
#         torch.Tensor
#             Quantile estimates of X with shape [len(quantiles), n_genes, n_spots].
#         """
#         self.set_mode("guide")
#         self._load_pyro_samples()

#         loc = pyro.param("X_loc")
#         scale_tril = pyro.param("X_shared_scale_tril")
#         dist_X = dist.MultivariateNormal(loc, scale_tril=scale_tril)

#         samples = dist_X.sample((n_samples,))
#         return torch.quantile(samples, torch.tensor(quantiles), dim=0)
    
    
    
    
# class PhysicsInformedSpatialInverter(PyroModule):
#     """
#     Simplified physics-informed model for spatial transcriptomics diffusion correction.
    
#     Key simplifications:
#     1. Direct modeling without complex encoding/decoding
#     2. Spot-specific diffusion coefficients based on tissue features
#     3. Graph Laplacian diffusion with adaptive similarity
#     4. Cell-based penalization
#     """
    
#     def __init__(self, 
#                 X: torch.Tensor,  # [n_spots, n_genes]
#                 y: torch.Tensor,  # [n_spots, n_genes]
#                 coords: torch.Tensor,
#                 spatial_adjacency: torch.Tensor,
#                 pca_emb: torch.Tensor = None,
#                 image_features: torch.Tensor = None,
#                 ecm_scores: torch.Tensor = None,
#                 cell_count: torch.Tensor = None,
#                 cluster_labels: torch.Tensor = None,
#                 diffusion_steps: int = 3,
#                 ):  # Mass conservation weight
        
#         super().__init__()
        
#         self.X = X
#         self.Y = y
#         self.n_spots, self.n_genes = X.shape
#         self.coords = coords
#         self.diffusion_steps = diffusion_steps
#         self.pca_emb = pca_emb
#         # Compute and store total counts per spot (observed)
#         self.total_counts = X.sum(dim=1, keepdim=True)  # [n_spots, 1]
        
#         pca_emb_norm = (pca_emb - pca_emb.mean(dim=0, keepdim=True)) / (pca_emb.std(dim=0, keepdim=True) + 1e-6)
#         cov_matrix = torch.mm(pca_emb_norm.t(), pca_emb_norm) / (pca_emb_norm.size(0))
#         cov_matrix += 1e-2 * torch.eye(cov_matrix.size(0), device=pca_emb.device)  # Ensure positive definiteness
#         self.cov_matrix = torch.linalg.cholesky(cov_matrix)
        
#         # Store masks and features
#         self.cell_count = cell_count if cell_count is not None else torch.ones(self.n_spots, dtype=torch.bool)
#         self.cluster_labels = cluster_labels
        
#         # Adaptive Laplacian matrices
#         self.spatial_adjacency = spatial_adjacency
#         self.spatial_laplacian = self._compute_spatial_laplacian(spatial_adjacency)
#         if pca_emb is not None:
#             self.laplacian_weights = self._compute_laplacian_weights()
#             pca_similarity = torch.cdist(pca_emb, pca_emb)
#             self.adaptive_laplacian = self._compute_adaptive_laplacian(
#                 pca_similarity if pca_similarity is not None else torch.eye(self.n_spots),
#                 spatial_adjacency
#             )
#         else:
#             self.adaptive_laplacian = None
        
#         # Spot-specific diffusion coefficient encoder
#         feature_dim = 0
#         if image_features is not None:
#             feature_dim += image_features.shape[1]
#         if ecm_scores is not None:
#             feature_dim += ecm_scores.shape[1]
            
#         if feature_dim > 0:
#             self.diffusion_encoder = SpotDiffusionEncoder(
#                 feature_dim=feature_dim,
#                 hidden_dim=64
#             )
#             # Concatenate features
#             features = []
#             if image_features is not None:
#                 features.append(image_features)
#             if ecm_scores is not None:
#                 features.append(ecm_scores)
#             self.img_features = torch.cat(features, dim=1)
#         else:
#             self.diffusion_encoder = None
#             self.img_features = None
#         self.scale_factor = self._compute_simple_data_scale()

#     def _compute_simple_data_scale(self):
#         """Simple, fast scale computation from data statistics"""
#         # Use simple data statistics that are fast to compute
#         n_total = self.n_spots * self.n_genes
#         sum_Y = self.Y.sum().item()
#         mean_Y = self.Y.mean().item()
#         max_Y = self.Y.max().item()
        
#         # Rough Poisson likelihood magnitude: sum_Y * log(mean_Y)
#         likelihood_magnitude = sum_Y * np.log(mean_Y + 1)
        
#         # Simple penalty magnitude estimate: roughly O(1) for normalized penalties
#         typical_penalty_magnitude = 1.0
        
#         # Scale to make penalties ~5% of likelihood
#         scale_factor = likelihood_magnitude * 0.1 / typical_penalty_magnitude
        
#         # Add adjustment for count ranges
#         if max_Y > 1000:
#             scale_factor *= 2
#         elif max_Y < 10:
#             scale_factor *= 0.5
        
#         final_scale = max(100.0, min(scale_factor, 50000.0))
        
#         print(f"Simple data scale: {final_scale:.0f} (likelihood est: {likelihood_magnitude:.0f})")
#         return final_scale
    
#     def _compute_spatial_laplacian(self, adjacency):
#         """Compute normalized graph Laplacian from adjacency matrix"""
#         adjacency = adjacency.clone()
#         adjacency.fill_diagonal_(0)  # Remove self-loops
#         adjacency = adjacency / (adjacency.sum(dim=1, keepdim=True) + 1e-10)  # Normalize rows
#         # Degree matrix
#         degree = torch.diag(adjacency.sum(dim=1))
#         laplacian = degree - adjacency        
#         return laplacian
    
#     def _compute_adaptive_laplacian(self, similarity, spatial_adjacency):
#         """Compute expression-similarity based Laplacian constrained to spatial neighbors"""
#         # Apply spatial constraint - only keep similarities for spatial neighbors
#         spatial_adjacency.fill_diagonal_(0)  # Remove self-loops
#         similarity.fill_diagonal_(0)  # Remove self-loops
        
#         # Improved similarity metric - use exponential decay instead of linear scaling
#         similarity = torch.exp(-similarity / similarity.mean())  # RBF kernel on gene expression
#         similarity = similarity * spatial_adjacency.float()  # Apply spatial constraint
        
#         # Convert to Laplacian
#         degree = torch.diag(similarity.sum(dim=1))
#         adaptive_laplacian = degree - similarity
        
#         return adaptive_laplacian

#     def _compute_laplacian_weights(self):
#         """
#         Compute biologically meaningful weights for spatial vs adaptive Laplacians
        
#         Returns:
#             lambda_spatial: weight for spatial Laplacian
#             lambda_adaptive: weight for adaptive Laplacian
#         """
#         if self.cell_count is None:
#             return 0.7, 0.3  # Default: favor spatial
        
#         # OPTION A: Cell density-based (local)
#         # High cell density areas: more adaptive (expression-based) diffusion
#         # Low cell density areas: more spatial (distance-based) diffusion
        
#         # Compute local cell density using spatial neighbors
#         spatial_adj = self.spatial_adjacency.clone()
#         spatial_adj.fill_diagonal_(1.0)  # Include self
        
#         # Local density = average cell count in neighborhood
#         local_cell_density = torch.matmul(spatial_adj.float(), self.cell_count.float()) / spatial_adj.sum(dim=1)
        
#         # Normalize to [0, 1]
#         local_density_norm = (local_cell_density - local_cell_density.min()) / (
#             local_cell_density.max() - local_cell_density.min() + 1e-8
#         )  # shape [n_spots]
        
#         # High density -> more adaptive weighting
#         lambda_adaptive = 0.1 + 0.4 * local_density_norm  # Range: [0.1, 0.5]
#         lambda_spatial = 1.0 - lambda_adaptive             # Range: [0.5, 0.9]
        
#         # Use average weights for simplicity (can be made spot-specific)
#         return lambda_spatial.mean().item, lambda_adaptive.mean().item
    
#     def forward_diffusion(self, true_expression, spot_diffusion_rates):
#         """
#         Apply graph Laplacian diffusion with spot-specific rates
        
#         Args:
#             true_expression: [n_spots, n_genes]
#             spot_diffusion_rates: [n_spots] - learned diffusion coefficients
#         """
#         expression = true_expression.clone()

#         if self.adaptive_laplacian is not None:
#             lambda_spatial, lambda_adaptive = self.laplacian_weights
#             # Combine spatial and adaptive Laplacians
#             combined_laplacian = lambda_spatial * self.spatial_laplacian + lambda_adaptive * self.adaptive_laplacian
#         else:
#             combined_laplacian = self.spatial_laplacian
        
#         for _ in range(self.diffusion_steps):
#             # Apply diffusion: x_new = x - D * L * x
#             diffusion_update = torch.matmul(combined_laplacian, expression)
#             expression = expression - spot_diffusion_rates.unsqueeze(1) * diffusion_update
#             expression = torch.clamp(expression, min=0.0)
        
#         return expression
    
#     def robust_log_transform(self, x, clip_quantile=0.99):
#         # Clip extreme values
#         upper = torch.quantile(x, clip_quantile)
#         x_clipped = torch.clamp(x, max=upper, min=1e-06)
#         x_clipped = torch.log(x_clipped)
#         # Log transform
#         return x_clipped
    
#     @pyro_method
#     def model(self):
#         with pyro.plate("spot_diffrate", self.n_spots):
#                 spot_diffusion_rates = pyro.sample(
#                     "spot_diffusion_rates",
#                     dist.Beta(1.1, 8.0)
#                 )
        
#         # Sample true undiffused expression
#         with pyro.plate("spots_expr", self.n_spots, dim=-2):
#             with pyro.plate("genes", self.n_genes, dim=-1):
#                 true_expression = pyro.sample(
#                     "true_expression",
#                     dist.LogNormal(
#                         self.robust_log_transform(self.X),
#                         torch.ones_like(self.Y) * 0.1  # Small scale for stability
#                     )
#                 )
        
#         # # to test: consider multivariate normal logtransformed?
#         # with pyro.plate('spots_expr', self.n_genes):
#         #     true_expression = pyro.sample(
#         #         'true_expression',
#         #         dist.TransformDistribution(
#         #             dist.MultivariateNormal(
#         #                 loc=self.X.T,
#         #                 scale_tril=self.cov_matrix
#         #             ),
#         #             transform=dist.transforms.ExpTransform()
#         #         )
#         #     )
#         # true_expression = true_expression.T  # [n_spots, n_genes]
        
#         # Forward diffusion process
#         diffused_expression = self.forward_diffusion(true_expression, spot_diffusion_rates)
        
#         # # Observation model
#         # with pyro.plate("obs_spots", self.n_spots, dim=-2):
#         #     with pyro.plate("obs_genes", self.n_genes, dim=-1):
#         #         pyro.sample(
#         #             "obs",
#         #             dist.Poisson(diffused_expression),
#         #             obs=self.Y
#         #         )
        
    
#         # Overdispersion parameter for Negative Binomial
#         concentration = pyro.param("concentration", torch.ones_like(self.Y) * 10.0,  # Start with moderate overdispersion
#                             constraint=dist.constraints.positive)
        
#         # Observation model with Negative Binomial
#         with pyro.plate("obs_spots", self.n_spots, dim=-2):
#             with pyro.plate("obs_genes", self.n_genes, dim=-1):
#                 logits = torch.log(concentration + 1e-8) - torch.log(diffused_expression + 1e-8)
#                 pyro.sample(
#                     "obs",
#                     dist.NegativeBinomial(
#                         total_count=concentration,  # Dispersion parameter
#                         logits=logits  # Log-odds of success
#                     ),
#                     obs=self.Y
#                 )
        
#         # Physics-based penalties
#         self._add_physics_penalties(true_expression)
        
#         return true_expression
    
#     def _add_physics_penalties(self, true_expression):
#         """Add physics-based penalty terms with balanced weights"""
#         scale_factor = self.scale_factor
        
#         # # 1. Mass conservation penalty (keep current weight)
#         # true_total = true_expression.sum(dim=0)
#         # obs_total = self.Y.sum(dim=0)
#         # mass_conservation_loss = torch.sum((true_total - obs_total) ** 2) / torch.sum(obs_total ** 2)
#         # mass_conservation_loss = - mass_conservation_loss * scale_factor
#         # # print(f"Mass conservation loss: {mass_conservation_loss.item()}")
#         # pyro.factor("mass_conservation", mass_conservation_loss)  # Increased from 1.0
        
#         # 2. Spatial smoothness penalty (reduce weight)
#         combined_laplacian = self.spatial_laplacian
#         # Compute per-gene statistics
#         gene_means = self.Y.mean(dim=0)  # [n_genes] - mean expression per gene
#         # Gene-specific smoothness weights
#         # Sparse genes: less smoothness constraint
#         smoothness_weights = torch.sigmoid(gene_means / gene_means.median() - 1.0)  # [0, 1]        
#         # Compute per-gene smoothness
#         gene_smoothness = torch.diag(true_expression.T @ combined_laplacian @ true_expression)  # [n_genes]
#         obs_gene_smoothness = torch.diag(self.Y.T @ combined_laplacian @ self.Y)  # [n_genes]
        
#         # Weighted smoothness penalty
#         relative_smoothness = gene_smoothness / (obs_gene_smoothness + 1e-6)
#         weighted_smoothness = torch.sum(smoothness_weights * relative_smoothness)
#         smoothness_penalty = - 0.1 * weighted_smoothness * scale_factor
        
#         pyro.factor("spatial_smoothness", smoothness_penalty)
        
#         # 3. Non-cell penalty (increase base penalty calculation)
#         if self.cell_count is not None:
#             cell_count_dist = self.cell_count.float() / (self.cell_count.sum() + 1e-8)
#             norm_true_expression = true_expression.sum(dim=1)
#             norm_true_expression = norm_true_expression / (norm_true_expression.sum() + 1e-8)
            
#             # Add small epsilon for numerical stability
#             eps = 1e-8
#             target_dist = cell_count_dist.squeeze() + eps
#             pred_dist = norm_true_expression + eps
            
#             # KL divergence: naturally scaled and meaningful
#             kl_div = torch.sum(target_dist * torch.log(target_dist / pred_dist))
#             cell_count_kl = - 0.5 * kl_div * scale_factor
#             # Apply KL divergence (already well-scaled)
#             pyro.factor("cell_count_kl", cell_count_kl)
                        
#         # 4. Cluster separation loss (keep current)
#         if self.cluster_labels is not None:
#             cluster_separation_loss = self._cluster_separation_loss(true_expression, self.cluster_labels)
#             cluster_separation_loss = - 0.5 * cluster_separation_loss * scale_factor
#             pyro.factor("cluster_separation", cluster_separation_loss)
            
#         # 6. Diffusion rate smoothness
#         alpha = pyro.get_param_store()["alpha_param"]
#         beta = pyro.get_param_store()["beta_param"]
#         mean_diffrates = alpha / (alpha + beta)
#         threshold = 0.2
#         above_threshold = torch.relu(mean_diffrates - threshold)
#         threshold_penalty = torch.sum(above_threshold ** 2) / torch.sum(mean_diffrates ** 2)
#         threshold_penalty = - threshold_penalty * scale_factor
#         # print(f"Diffusion rate threshold penalty: {threshold_penalty.item()}")
#         pyro.factor("diffusion_rate_threshold", threshold_penalty)
        
#         if torch.rand(1) < 0.01:  # Print occasionally
#             print(
#                 # f"Penalties: mass={mass_conservation_loss.item():.3f}, ",
#                 f"spatial_smoothness={smoothness_penalty.item():.3f}, ",
#                 f'cell_count={cell_count_kl.item():.3f}, ',
#                 f"cluster={cluster_separation_loss.item()}, ",
#                 f"diffrate_threshold={threshold_penalty.item():.3f}",
#                 # f"entropy={entropy.item():.3f}"
#                 )
            
#     def _cluster_separation_loss(self, true_expr: torch.Tensor, cluster_labels: torch.Tensor) -> torch.Tensor:
#         """Maximize between-cluster distance, minimize within-cluster distance"""
#         unique_clusters = np.unique(cluster_labels)  # Use torch.unique instead of np.unique
#         n_clusters = len(unique_clusters)
        
#         if n_clusters < 2:
#             return torch.tensor(0.0)
        
#         # Compute cluster centroids
#         centroids = []
#         for cluster_id in unique_clusters:
#             mask = cluster_labels == cluster_id  # [n_spots] boolean mask
#             # FIX: Apply mask to correct dimension (spots, not genes)
#             cluster_expr = true_expr[mask, :]  # [n_spots_in_cluster, n_genes]
#             centroid = cluster_expr.mean(dim=0)   # [n_genes] - average across spots in cluster
#             centroids.append(centroid)
#         centroids = torch.stack(centroids)  # [n_clusters, n_genes]
        
#         # Between-cluster distance (maximize)
#         between_dist = torch.pdist(centroids, p=2).mean()
        
#         # Within-cluster distance (minimize)
#         within_dist = 0.0
#         for i, cluster_id in enumerate(unique_clusters):
#             mask = cluster_labels == cluster_id
#             cluster_expr = true_expr[mask, :]  # FIX: Correct indexing
#             if cluster_expr.shape[0] > 1:  # FIX: Check number of spots, not genes
#                 centroid = centroids[i].unsqueeze(0)  # [1, n_genes]
#                 within_dist += torch.norm(cluster_expr - centroid, p=2, dim=1).mean()  # FIX: dim=1 for genes
        
#         # Minimize within/between ratio
#         return within_dist / (between_dist + 1e-6)
    
#     @pyro_method
#     def guide(self):
#         """Variational guide"""
        
#         # Variational parameters for diffusion rates
#         # if self.diffusion_encoder is not None:
#         #     # Encode tissue features to diffusion parameters
#         #     diffusion_params = self.diffusion_encoder(self.img_features)
#         #     alpha_param = F.softplus(diffusion_params[:, 0]) 
#         #     alpha_param = pyro.param("alpha_param", alpha_param, constraint=dist.constraints.positive)
#         #     beta_param = F.softplus(diffusion_params[:, 1]) + 2.0
#         #     beta_param = pyro.param("beta_param", beta_param, constraint=dist.constraints.positive)
#         # else:
#         alpha_param = pyro.param("alpha_param", torch.ones(self.n_spots) * 1.2, 
#                             constraint=dist.constraints.positive)
#         beta_param = pyro.param("beta_param", torch.ones(self.n_spots) * 10.0,
#                             constraint=dist.constraints.positive)
        
#         with pyro.plate("spot_diffrate", self.n_spots):
#             spot_diffusion_rates = pyro.sample(
#                 "spot_diffusion_rates",
#                 dist.Beta(alpha_param, beta_param)
#             )
        
#         # Variational parameters for true expression
#         loc_param = pyro.param("loc_param", self.robust_log_transform(self.X))
#         scale_param = pyro.param("scale_param", torch.ones_like(self.X) * 0.1,
#                             constraint=dist.constraints.positive)
        
#         with pyro.plate("spots_expr", self.n_spots, dim=-2):
#             with pyro.plate("genes", self.n_genes, dim=-1):
#                 true_expression = pyro.sample(
#                     "true_expression",
#                     dist.LogNormal(loc_param,
#                                 scale_param)
#                 )
        
#         # # to test: consider multivariate normal logtransformed?
#         # loc_param = pyro.param("loc_param", self.X.T)
#         # scale_tril_param = pyro.param("scale_tril_param", torch.eye(self.cov_matrix.size(0)), constraint=dist.constraints.lower_cholesky)
#         # with pyro.plate('spots_expr', self.n_genes):
#         #     true_expression = pyro.sample(
#         #         "true_expression",
#         #         dist.TransformDistribution(
#         #             dist.MultivariateNormal(
#         #                 loc=loc_param.T,
#         #                 scale_tril=scale_tril_param
#         #             ),
#         #             transform=dist.transforms.ExpTransform()
#         #         )
#         #     )
#         # true_expression = true_expression.T  # [n_spots, n_genes]
        
#         return true_expression


# class SpotDiffusionEncoder(PyroModule):
#     """
#     Encoder that learns spot-specific diffusion coefficients from tissue features
#     """
#     def __init__(self, feature_dim, hidden_dim=64):
#         super().__init__()
#         self.encoder = nn.Sequential(
#             nn.Linear(feature_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Dropout(0.1),
#             nn.Linear(hidden_dim, hidden_dim // 2),
#             nn.ReLU(),
#             nn.Linear(hidden_dim // 2, 2)  # Output alpha, beta for Beta distribution
#         )
        
#     def forward(self, features):
#         """
#         Args:
#             features: [n_spots, feature_dim] - concatenated image and ECM features
#         Returns:
#             params: [n_spots, 2] - (alpha, beta) parameters for Beta distribution
#         """
#         return self.encoder(features)
