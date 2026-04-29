## Below is version 1 and 2
from scipy.signal import savgol_filter, find_peaks
import numpy as np
import pandas as pd
import torch

from clear_st.clear_model import CLEARmodel
import pyro
import torch.nn.functional as F

from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.sparse import issparse
from scipy.signal import savgol_filter

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt 
import scanpy as sc 
import squidpy as sq
import pickle as pkl
import os
import subprocess


def pick_free_gpu(min_free_mb: int = 2000) -> torch.device:
    """Return a valid torch.device for CUDA/CPU.

    Key behavior:
    - If `CUDA_VISIBLE_DEVICES` is set, we respect it and return `cuda:0`.
      (Visible devices are re-indexed starting at 0.)
    - Otherwise, we pick the GPU with the most free memory via `nvidia-smi`,
      set `CUDA_VISIBLE_DEVICES` to that index, and return `cuda:0`.
    """
    if not torch.cuda.is_available():
        return torch.device("cpu")

    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if cvd is not None:
        cvd = cvd.strip()
        if cvd == "" or cvd.lower() in {"-1", "none"}:
            return torch.device("cpu")
        # Respect the user's visibility restriction. Use the first visible CUDA device.
        return torch.device("cuda:0")

    best_idx: str | None = None
    best_free: int = -1

    try:
        out = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,memory.total,memory.used",
                "--format=csv,noheader,nounits",
            ]
        )
        lines = out.decode().strip().splitlines()
        for line in lines:
            idx, total, used = [x.strip() for x in line.split(",")]
            free = int(total) - int(used)
            if free > best_free:
                best_free = free
                best_idx = idx

        if best_idx is not None and best_free >= min_free_mb:
            os.environ["CUDA_VISIBLE_DEVICES"] = str(best_idx)
            print(f"Selected GPU {best_idx} (free {best_free} MiB)")
            return torch.device("cuda:0")
    except Exception as e:
        print("Could not query GPUs:", e)

    # Fallback: just use the default CUDA device.
    return torch.device("cuda:0")


class undiff():
    def __init__(self, adata, n_neighs=15):
        self.adata = adata.copy()
        self.var_names = adata.var_names
        self.Y_init = None
        libs = list(self.adata.uns['spatial'].keys())
        libs = [k for k in libs if isinstance(self.adata.uns['spatial'][k], dict)]
        if len(libs) == 0:
            raise ValueError("No valid library found in adata.uns['spatial']. Each library entry should be a dictionary containing spatial information.")
        self.lib_id = libs[0]
        self.adata.uns['spatial'] = {self.lib_id: self.adata.uns['spatial'][self.lib_id]}  # keep only one library_id
        sq.gr.spatial_neighbors(self.adata, n_neighs=n_neighs, coord_type='generic')
        self.spatial_con = self.adata.obsp['spatial_connectivities']
        self.spatial_dist = self.adata.obsp['spatial_distances']
        self.img_key = 'hires' 
        # self.coords = self.scale_spatial()
        # self.coords = self.coords - self.coords.min(dim=0)[0]
        self.coords = torch.tensor(self.adata.obsm['spatial'].copy(), dtype=torch.float32, requires_grad=False)
        self.y1_max, self.y2_max = self.adata.uns['spatial'][self.lib_id]['images'][self.img_key].shape[:2]
        self.in_tiss_mask = torch.tensor(self.adata.obs['in_tissue'].values, dtype=torch.float32, requires_grad=False)
        self.gene_selected = []

        self.device = torch.cuda.current_device() if torch.cuda.is_available() else torch.device("cpu")
        print(f"========== Using device: {self.device} ========== ")


    def round_counts_to_integers(self, input_mat, preserve_sum=True, target_sums=None):
        """
        Round corrected expression counts to integers.
        
        Parameters
        ----------
        input_mat : torch.Tensor
            The input count matrix to be rounded
        preserve_sum : bool, default=True
            If True, preserve the column-wise sums during rounding
            
        Returns
        -------
        rounded_counts : torch.Tensor
            Rounded count matrix (returned only if return_copy=True)
        """
        arrtype = type(input_mat)
        rounded_counts = input_mat.clone() if isinstance(input_mat, torch.Tensor) else torch.tensor(input_mat, dtype=torch.float32)
        device = rounded_counts.device if isinstance(rounded_counts, torch.Tensor) else None
        rounded_counts = rounded_counts.to(torch.float32)

        if preserve_sum:
            # Largest-remainder: floor then add ones to the largest fractional parts
            if target_sums is None:
                target_sums = torch.round(rounded_counts.sum(dim=0))
            target_sums = torch.as_tensor(target_sums, dtype=torch.float32, device=rounded_counts.device)
            base = torch.floor(rounded_counts)
            frac = rounded_counts - base
            deficits = (target_sums - base.sum(dim=0)).to(torch.int64)

            for j in range(base.shape[1]):
                k = deficits[j].item()
                if k <= 0:
                    continue
                k = min(k, base.shape[0])
                topk_idx = torch.topk(frac[:, j], k=k, largest=True).indices
                base[topk_idx, j] += 1

            rounded_counts = base
        else:
            rounded_counts = torch.round(rounded_counts)

        rounded_counts = torch.clamp(rounded_counts, min=0)
        if device is not None:
            rounded_counts = rounded_counts.to(device)
        if arrtype != torch.Tensor:
            rounded_counts = rounded_counts.numpy()
        return rounded_counts
        
        
    @staticmethod
    def clustering_model_init(algo, i):
        if algo == KMeans:
            model = algo(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=0)
        elif algo == AgglomerativeClustering:
            model = algo(n_clusters=i)
        elif algo == SpectralClustering:
            model = algo(n_clusters=i, affinity='nearest_neighbors', n_init=10)
        elif algo == Birch:
            model = algo(n_clusters=i)
        elif algo == BayesianGaussianMixture:
            model = algo(n_components=i, weight_concentration_prior=1.0, max_iter=100, random_state=0)
        elif algo == GaussianMixture:
            model = algo(n_components=i, covariance_type='full', max_iter=100, random_state=0)
        else:
            raise ValueError(f"Unsupported algorithm: {algo}")
        return model

    @staticmethod
    def find_optimal_ncls(data, methods, max_clusters=None, starting_clusters=3):
        if max_clusters is None:
            max_clusters = np.ceil(np.sqrt(data.shape[0]))  # Default to sqrt of number of samples
        max_clusters = int(max_clusters) + 5
        # import common sklearn clustering algorithms
        scs = {}
        # wcsses = {}
        max_clusters = np.ceil(np.sqrt(data.shape[0])).astype(int)
        cluster_range = range(starting_clusters, max_clusters + 1)

        for algo in methods:
            scs[f'{algo.__name__}'] = []
            # wcsses[f'{algo.__name__}'] = [] if algo == KMeans else None
            for i in cluster_range:
                model = undiff.clustering_model_init(algo, i)
                predicted_labels = model.fit_predict(data)
                # if hasattr(model, 'inertia_'):
                #     wcsses[f'{algo.__name__}'].append(model.inertia_)
                scs[f'{algo.__name__}'].append(silhouette_score(data, predicted_labels))

        # calculate the optimal number of clusters by averaging the silhouette scores for each n_clusters
        df = pd.DataFrame(scs)
        df.index += starting_clusters
        optimal_clusters = df.mean(axis=1).sort_values(ascending=False).index[:3]  # top 3 cluster numbers based on average silhouette score
        algo = df.loc[optimal_clusters, :].mean(axis=0).idxmax()  # algo with the highest average silhouette score among the top 3 cluster numbers
        return optimal_clusters, algo

    @staticmethod
    def auto_cluster(data, starting_clusters=3, max_clusters=None):
        """
        Calculate silhouette scores for PCA-reduced data and return the optimal number of clusters.
        """
        # Standardize the data
        data = StandardScaler().fit_transform(data)
        from sklearn.cluster import KMeans, AgglomerativeClustering, SpectralClustering, Birch
        from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
        methods = [KMeans, AgglomerativeClustering, SpectralClustering, BayesianGaussianMixture]  # Birch, GaussianMixture, 
            
        # Reduce dimensionality using PCA
        pca_silhouette_scores = {}
        best_algos = []
        max_clusters = data.shape[0] // 2 if max_clusters is None else max_clusters
        for n_pca in range(5, min(30, data.shape[0]), 5):
            pca = PCA(n_components=n_pca)
            data_pca = pca.fit_transform(data)
            top_ncl, algo = undiff.find_optimal_ncls(data_pca, methods, max_clusters=max_clusters, starting_clusters=starting_clusters)
            pca_silhouette_scores[n_pca] = top_ncl
            best_algos.append(algo)
        
        # Convert to DataFrame for better visualization
        pca_silhouette_df = pd.DataFrame(pca_silhouette_scores).T  # dim: (n_pcas, top_choices=3)
        # Find the optimal number of clusters based on the mode of the silhouette scores
        optimal_clusters = pca_silhouette_df.mode() # here we take the mode of top choices (most chosen number of clusters across all PCA num choices)
        optimal_clusters = optimal_clusters.iloc[0].values.max()  # get the first row of the mode values
        optimal_clusters = int(optimal_clusters)  # convert to int
        # then take the maximum of the mode values (could use [0] instead, 
        # essentially the most chosen top 1 number of clusters, across all PCA numbers)
        # # find the optimal number of PCA components based on the mode of the silhouette scores
        # col = pca_silhouette_df.mode().T.idxmax()[0]
        # # print(col)
        # optimal_pca = pca_silhouette_df.loc[pca_silhouette_df[col] == optimal_clusters, col].index[0]
        # pca = PCA(n_components=optimal_pca).fit_transform(data)
        optimal_algo = np.unique(best_algos)[0]
        actual_algo = [algo for algo in methods if algo.__name__ == optimal_algo][0]
        model = undiff.clustering_model_init(actual_algo, optimal_clusters)
        cluster_labels = model.fit_predict(data)
        # return {
        #     'cluster_labels': cluster_labels,
        #     'pca_silhouette_df': pca_silhouette_df,
        #     'optimal_clusters': optimal_clusters,
        #     # 'pca': pca,
        # }
        return cluster_labels

    def prep_genes_params(self, first_n_genes=None, add_genes=[]):
        self.gene_selected = self.gene_selection(first_n_genes=first_n_genes, out_tiss_perc=0.95, min_count=50)
        self.gene_selected = self.gene_selected + [g for g in add_genes if g not in self.gene_selected]
        self.gene_selected = np.array(self.gene_selected)
        self.gene_indices = [self.adata.var_names.get_loc(g) for g in self.gene_selected]  # gene indices indicate gene indices in original data
        self.Y_init = torch.tensor(self.adata.raw.to_adata()[:, self.gene_indices].X.toarray(), dtype=torch.float32)  # shape [n_spots, n_genes]

    @staticmethod
    def pp(ad, min_cnt, n_hvg, spvg=False, save_raw=False):
        sc.pp.calculate_qc_metrics(ad, inplace=True)
        ad = ad[:, ad.var['total_counts'] >= min_cnt]
        if save_raw:
            ad.raw = ad.copy()
        sc.pp.normalize_total(ad, target_sum=1e4)
        sc.pp.log1p(ad)
        sc.pp.highly_variable_genes(ad, flavor="seurat", n_top_genes=n_hvg)
        if spvg:
            sq.gr.spatial_neighbors(ad, coord_type='generic')
            sq.gr.spatial_autocorr(
                ad,
                mode="moran",
                n_perms=100,
                genes=ad.var_names[ad.var['highly_variable']],
            )        
        return ad
    

    def gene_selection(self, first_n_genes=3000, out_tiss_perc=0.95, min_count=50):
        """
        Select genes that are highly variable or highly expressed in out-of-tissue spots.

        Parameters
        ----------
        first_n_genes : int
            Number of top genes to select based on expression in out-of-tissue spots.
        out_tiss_perc : float
            Minimum percentage of total expression that should come from out-of-tissue spots.
        min_count : int
            Minimum total count for a gene to be considered.
        Returns
        -------
        union_genes : list
            List of selected gene names.
        """
        in_tiss = self.adata[self.adata.obs['in_tissue']==1, ]

        self.adata = self.pp(self.adata, min_cnt=min_count, n_hvg=first_n_genes, spvg=False, save_raw=True)
        in_tiss = self.pp(in_tiss, min_cnt=min_count, n_hvg=first_n_genes + 2000, spvg=True,save_raw=False)  # simulate hvg selection in the regular pipeline using filtered data directly
        
        # out = self.adata.raw.to_adata()
        # out = out[out.obs['in_tissue']==0,]
        # out.var['total_counts'] = out.X.toarray().sum(axis=0)
        
        # # select genes with high expression in out-of-tissue spots
        # sorted_out_idx = out.var.sort_values('total_counts', ascending=False).index
        # for i in range(5, len(out.var_names), 10):
        #     union_genes = sorted_out_idx[:i]  # top i genes by expression in out-of-tissue spots
        #     perc_out = out.var.loc[union_genes,'total_counts'].sum()/out.var['total_counts'].sum()
        #     if perc_out > out_tiss_perc:
        #         break
        intiss_hvgs = in_tiss.var.loc[in_tiss.var['highly_variable'], :].index
        sp_var_genes = in_tiss.uns["moranI"][in_tiss.uns["moranI"]['pval_sim_fdr_bh'] < 0.05].index
        intiss_hvgs = intiss_hvgs.intersection(sp_var_genes)
        # raw_hvgs = raw.var.loc[raw.var['highly_variable'], :].index
        # union_genes = union_genes.union(raw_hvgs.intersection(intiss_hvgs))
        union_genes = intiss_hvgs
        
        # Remove mitochondrial and DEPRECATED genes
        to_remove = ['MT-', 'mt-', 'DEPRECATED','Rik']
        union_genes = [x for x in union_genes if not any(sub in x for sub in to_remove)]
        # redo filtering based on min_count
        union_genes = self.adata.var.loc[self.adata.var['total_counts'] >= min_count, :].sort_values('total_counts', ascending=False).index.intersection(union_genes).to_list()
        # Sort genes by expression similarity
        # union_genes = self.sort_genes_by_similarity(union_genes)        
        return union_genes


    def get_cell_cluster_labels(self, n_neighs=15):
        from spateo.tl import smooth
        self.adata.uns['__type'] = 'UMI'
        sc.pp.scale(self.adata, max_value=10)
        sc.pp.pca(self.adata, use_highly_variable=True)

        # Identify neighbors(KNN)
        sc.pp.neighbors(self.adata, n_neighbors=n_neighs, n_pcs=30)
        sc.tl.umap(self.adata)
        sil_scores_expr = []
        for res in np.arange(0.2, 0.8, 0.1):
            res = np.round(res, 1)
            sc.tl.leiden(self.adata, resolution=res, key_added=f'leiden_{res}', neighbors_key='neighbors')
            if self.adata.obs[f'leiden_{res}'].nunique() < 3:  # silhouette score is not meaningful with too few clusters, penalize such cases
                sil_scores_expr.append(-1)
                continue
            sil_scores_expr.append(silhouette_score(self.adata.obsm['X_pca'], self.adata.obs[f'leiden_{res}'].values) * (self.adata.obs[f'leiden_{res}'].nunique() / self.adata.shape[0]))  # penalize small clusters
        sil_score_df = pd.DataFrame({
            'resolution': np.arange(0.2, 0.8, 0.1),
            'silhouette_expression': sil_scores_expr
        })
        best_row = sil_score_df.loc[sil_score_df['silhouette_expression'].idxmax()]
        best_res = np.round(best_row['resolution'], 1)
        # self.adata.obs['leiden_expr'] = smooth(self.adata, radius=n_neighs, key=f'leiden_{best_res}')
        self.adata.obs['leiden_expr'] = self.adata.obs[f'leiden_{best_res}']

        if self.image_features is not None:
            emb_df = pd.DataFrame(self.image_features.cpu().numpy())
            ad_img = sc.AnnData(emb_df)
            ad_img.obs_names = self.adata.obs_names.copy()
            ad_img.obs['in_tissue'] = self.adata.obs['in_tissue'].values
            ad_img.obsm['spatial'] = self.adata.obsm['spatial']
            ad_img.uns['spatial'] = self.adata.uns['spatial']
            ad_img = ad_img[ad_img.obs['in_tissue']==1,:].copy()

            ad_img.uns['__type'] = 'UMI'
            sc.pp.pca(ad_img)
            sc.pp.neighbors(ad_img)
            sc.tl.umap(ad_img)
            sil_scores_img = []
            sil_scores_expr = []
            for res in np.arange(0.1, 0.8, 0.1):
                res = np.round(res, 1)
                sc.tl.leiden(ad_img, resolution=res, key_added=f'image_leiden_{res}')
                if ad_img.obs[f'image_leiden_{res}'].nunique() < 3:  # silhouette score is not meaningful with too few clusters, penalize such cases
                    sil_scores_img.append(-1)
                    sil_scores_expr.append(-1)
                    continue
                sil_scores_img.append(silhouette_score(ad_img.obsm['X_pca'], ad_img.obs[f'image_leiden_{res}'].values) * (ad_img.obs[f'image_leiden_{res}'].nunique() / ad_img.shape[0]))  # penalize small clusters
                sil_scores_expr.append(silhouette_score(self.adata[self.adata.obs['in_tissue']==1,:].obsm['X_pca'], ad_img.obs[f'image_leiden_{res}'].values) * (ad_img.obs[f'image_leiden_{res}'].nunique() / ad_img.shape[0]))  # penalize small clusters
            sil_score_df = pd.DataFrame({
                'resolution': np.arange(0.1, 0.8, 0.1),
                'silhouette_image': sil_scores_img,
                'silhouette_expression': sil_scores_expr
            })
            sil_score_df['silhouette_avg'] = sil_score_df[['silhouette_image', 'silhouette_expression']].mean(axis=1)
            best_row = sil_score_df.loc[sil_score_df['silhouette_avg'].idxmax()]
            best_res = np.round(best_row['resolution'], 1)
            self.adata.obs['leiden_img'] = ad_img.obs[f'image_leiden_{best_res}'].astype(int).values.max() + 1  # assign a new cluster label for out-of-tissue spots
            self.adata.obs.loc[ad_img.obs_names, 'leiden_img'] = ad_img.obs[f'image_leiden_{best_res}'].astype(int).values
            libid = self.lib_id
            sc.pl.umap(ad_img, color=f'image_leiden_{best_res}')
            sq.pl.spatial_scatter(ad_img, color=f'image_leiden_{best_res}', img=True, library_id=libid)        
        else:
            print("Clustering without image features is not implemented yet. Please provide image features for clustering. Will use leiden clusters based on expression only for now, which may not be ideal.")
            self.adata['leiden_img'] = self.adata['leiden_expr'].copy()  # assign the same cluster labels to out-of-tissue spots for now, which is not ideal but allows the rest of the pipeline to run without image features
        
        libid = list(self.adata.uns['spatial'].keys())[0]
        ks = ['leiden_expr']
        sq.pl.spatial_scatter(self.adata, color=ks, size=1.5, alpha=0.8, legend_loc='on data', title='Leiden Clusters', palette='tab20', library_id=libid)
        sc.pl.umap(self.adata, color=ks, title='Leiden Clusters', palette='tab20')

        self.img_labels = self.adata.obs['leiden_img'].astype(int).values if 'leiden_img' in self.adata.obs else self.adata.obs['leiden_expr'].astype(int).values
        self.img_labels = torch.tensor(self.img_labels, dtype=torch.long)
        self.expr_cluster_labels = self.adata.obs['leiden_expr'].astype(int).values
        self.expr_cluster_labels = torch.tensor(self.expr_cluster_labels, dtype=torch.long)

    # =========================================================================
    # X_init: OT-based transport of out-of-tissue mass into tissue
    # Adapted from undiff_Xinit.py.  Enable via use_x_init=True in define_model.
    # =========================================================================

    def thresholding_acc_qts(self, gene_expr_in, gene_expr_out, cutoffs):
        """Apply quantile-based thresholding to in-tissue and out-of-tissue expression."""
        eps = 0
        in_tiss_mask_2d = self.in_tiss_mask.unsqueeze(1).bool()
        if hasattr(self, "in_no_cell"):
            in_tiss_mask_2d = in_tiss_mask_2d & (self.in_no_cell.unsqueeze(1) == 1)
        in_tiss_filt = torch.where(
            (gene_expr_in >= cutoffs) & in_tiss_mask_2d, gene_expr_in, eps
        )
        out_tiss_filt = torch.where(
            (gene_expr_out < cutoffs) | ~in_tiss_mask_2d, gene_expr_out, eps
        )
        return in_tiss_filt, out_tiss_filt

    def prep_acc_qts(self, invalid_cutoff_vals):
        """Prepare in/out tissue tensors with quantile adjustment."""
        X_g = self.Y_init
        ttcnt = self.Y_init.sum(dim=0)
        out_tiss = X_g.clone()
        in_tiss = X_g.clone()
        in_tiss_filt, out_tiss_filt = self.thresholding_acc_qts(in_tiss, out_tiss, invalid_cutoff_vals)
        out_tiss_sum = out_tiss_filt.sum(dim=0)
        in_tiss_sum = in_tiss_filt.sum(dim=0)
        s = out_tiss_sum + in_tiss_sum
        sum_scale = torch.where(ttcnt > 0, s / (ttcnt + 1e-8), torch.ones_like(s))
        out_tiss_filt = out_tiss_filt / (sum_scale + 1e-8)
        in_tiss_filt = in_tiss_filt / (sum_scale + 1e-8)
        out_tiss_sum = out_tiss_filt.sum(dim=0)
        in_tiss_sum = in_tiss_filt.sum(dim=0)
        out_tiss_prob = torch.where(out_tiss_sum > 0, out_tiss_filt / (out_tiss_sum + 1e-8), torch.zeros_like(out_tiss_filt))
        in_tiss_prob = torch.where(in_tiss_sum > 0, in_tiss_filt / (in_tiss_sum + 1e-8), torch.zeros_like(in_tiss_filt))
        return out_tiss_prob, in_tiss_prob, out_tiss_sum, in_tiss_sum

    def gene_initialization(self, gene_expr_out, gene_expr_in, out_sum, in_sum):
        """Distribute out-tissue mass to in-tissue spots via spatial affinity."""
        eps = 1e-8
        out_mask = self.in_tiss_mask == 0
        in_mask = self.in_tiss_mask == 1
        with torch.no_grad():
            if hasattr(self.spatial_dist, 'tocoo'):
                coo = self.spatial_dist.tocoo()
                rows = torch.tensor(coo.row, dtype=torch.long)
                cols = torch.tensor(coo.col, dtype=torch.long)
                values = torch.tensor(coo.data, dtype=torch.float32)
            else:
                nonzero = self.spatial_dist.nonzero(as_tuple=True)
                rows = torch.tensor(nonzero[0], dtype=torch.long)
                cols = torch.tensor(nonzero[1], dtype=torch.long)
                values = self.spatial_dist[nonzero].float()
            keep_mask = out_mask[rows].to(rows.device) & in_mask[cols].to(cols.device)
            rows_f = rows[keep_mask]; cols_f = cols[keep_mask]; vals_f = values[keep_mask]
            n_spots = gene_expr_out.shape[0]
            W = torch.sparse_coo_tensor(
                torch.stack([rows_f, cols_f], dim=0), vals_f, (n_spots, n_spots),
                dtype=torch.float32
            ).coalesce()
            spatial_pull = torch.sparse.mm(W.t(), gene_expr_out)
            allocation_weights = spatial_pull * gene_expr_in
            weight_sums = allocation_weights.sum(dim=0, keepdim=True)
            allocation_probs = allocation_weights / (weight_sums + eps)
        transported_in = (
            allocation_probs * out_sum.unsqueeze(0) +
            gene_expr_in * in_sum.unsqueeze(0)
        )
        transported_in[out_mask, :] = 0.0
        col_sums = transported_in.sum(dim=0)
        expected_sums = out_sum + in_sum
        max_error = (col_sums - expected_sums).abs().max()
        print(f"Gene initialization completed. Max mass error: {max_error.item()}")
        assert max_error < 300, f"Mass conservation violated: max error {max_error.item()}"
        return transported_in

    def compute_X_init(self, params, batch_size: int = 200):
        """Compute X_init by spatially transporting out-of-tissue counts into tissue (batched)."""
        invalid_cutoff_vals = params['invalid_cutoff_vals']
        genes = list(self.gene_selected)
        out_tiss_filt, in_tiss_filt, out_tiss_sum, in_tiss_sum = self.prep_acc_qts(invalid_cutoff_vals)
        n_genes = len(genes)
        transported_cols = []
        for start in range(0, n_genes, batch_size):
            end = min(start + batch_size, n_genes)
            print(f"X_init batch {start}-{end}/{n_genes}")
            transported_batch = self.gene_initialization(
                out_tiss_filt[:, start:end], in_tiss_filt[:, start:end],
                out_tiss_sum[start:end], in_tiss_sum[start:end]
            )
            transported_cols.append(transported_batch)
        X_init = torch.cat(transported_cols, dim=1)
        self.X_init = self.round_counts_to_integers(X_init)
        assert self.X_init.shape[1] == n_genes

    def compute_X_init_fast(self, params):
        """Fast X_init (no spatial affinity) — distributes out-tissue mass proportional to in-tissue weights."""
        invalid_cutoff_vals = params['invalid_cutoff_vals']
        out_tiss_filt, in_tiss_filt, out_tiss_sum, in_tiss_sum = self.prep_acc_qts(invalid_cutoff_vals)
        q_norm = in_tiss_filt / (in_tiss_filt.sum(dim=0, keepdim=True) + 1e-8)
        X_init = out_tiss_sum.unsqueeze(0) * q_norm + in_tiss_sum.unsqueeze(0) * in_tiss_filt
        self.X_init = self.round_counts_to_integers(X_init)
        assert self.X_init.shape[1] == len(self.gene_selected)

    def impute_qt_vectorized(
        self,
        n_steps: int = 15,
        start_qt: float = 0.0,
        max_qt: float = 0.95,
        min_spots: int = 3,
        window_length: int = 6,
        polyorder: int = 2,
        use_row_standardized_W: bool = True,
        return_quantiles: bool = False,
        min_expr: float = 1.0,
        derivative_tol: float = 0.002,
        peak_prominence: float = 0.01,
        peak_height_min: float = -1,
        min_peak_distance: int = 1,
        min_monotonic_span: int = 3,
    ):
        """Compute per-gene optimal quantile cutoffs from Moran's I curve patterns."""
        from scipy.signal import savgol_filter, find_peaks
        from scipy.sparse import issparse as sp_issparse

        gene_indices = self.gene_indices
        if gene_indices is None or len(gene_indices) == 0:
            raise ValueError("gene_indices not set. Call prep_genes_params() first.")

        X = (self.Y_init * self.in_tiss_mask.unsqueeze(1)).to(dtype=torch.float32)
        qts = np.round(np.linspace(start_qt, max_qt, n_steps), 4)

        if sp_issparse(self.spatial_con):
            coo = self.spatial_con.tocoo()
            rows = torch.tensor(coo.row, dtype=torch.long)
            cols = torch.tensor(coo.col, dtype=torch.long)
            w = torch.tensor(coo.data, dtype=torch.float32) if coo.data is not None else torch.ones_like(rows, dtype=torch.float32)
        else:
            W = self.spatial_con
            if isinstance(W, torch.Tensor):
                W = W.detach().cpu().numpy()
            coo_idx = np.asarray(W).nonzero()
            rows = torch.tensor(coo_idx[0], dtype=torch.long)
            cols = torch.tensor(coo_idx[1], dtype=torch.long)
            w = torch.ones_like(rows, dtype=torch.float32)

        keep = rows != cols
        rows, cols, w = rows[keep], cols[keep], w[keep]
        if use_row_standardized_W:
            row_sums = torch.scatter_add(torch.zeros(int(X.shape[0]), dtype=torch.float32), 0, rows, w)
            w = w / (row_sums[rows] + 1e-8)

        X_cpu = X.detach().cpu()
        w_cpu = w.detach().cpu()
        rows_cpu = rows.detach().cpu()
        cols_cpu = cols.detach().cpu()

        n_spots = X.shape[0]
        n_genes = len(gene_indices)
        morans_by_q = {float(q): torch.full((n_genes,), float("nan")) for q in qts}

        for q in qts:
            q_val = float(q)
            cutoffs = torch.full((n_genes,), float('inf'), device='cpu')
            valid_expr_all = torch.zeros(n_spots, n_genes, dtype=torch.bool, device='cpu')
            for gi in range(n_genes):
                col = X_cpu[:, gi]
                valid_expr = col >= min_expr
                valid_expr_all[:, gi] = valid_expr
                if valid_expr.sum() >= min_spots:
                    cutoffs[gi] = torch.quantile(col[valid_expr], q_val)
            masks = valid_expr_all & (X_cpu > cutoffs.unsqueeze(0))
            z = X_cpu * masks
            N = masks.sum(dim=0)
            valid = N >= min_spots
            mean = torch.where(valid, z.sum(dim=0) / N, 0)
            zc = z - mean.unsqueeze(0)
            mask_rows = masks[rows_cpu]; mask_cols = masks[cols_cpu]
            e_mask = mask_rows & mask_cols
            w_expanded = w_cpu.unsqueeze(1).expand(-1, n_genes)
            w_filtered = w_expanded * e_mask
            numer = (w_filtered * zc[rows_cpu] * zc[cols_cpu]).sum(dim=0)
            denom = ((zc * masks) ** 2).sum(dim=0)
            S0 = w_filtered.sum(dim=0)
            morans_by_q[q_val] = torch.where(valid, (N / (S0 + 1e-8)) * (numer / (denom + 1e-8)), float('nan'))

        best_cutoffs = torch.zeros(n_genes, dtype=torch.float32)
        picked_qts = torch.zeros(n_genes, dtype=torch.float32)
        for gi in range(n_genes):
            xs, ys = [], []
            for q in qts:
                v = morans_by_q[float(q)][gi]
                if not torch.isnan(v):
                    xs.append(float(q)); ys.append(float(v.item()))
            if len(ys) < 3:
                picked_q = float(start_qt)
            else:
                x = np.array(xs, dtype=float); y = np.array(ys, dtype=float)
                wl = min(window_length, len(y) if len(y) % 2 == 1 else max(3, len(y) - 1))
                wl = max(wl, 3); po = min(polyorder, wl - 1)
                try:
                    y_smooth = savgol_filter(y, window_length=wl, polyorder=po, mode="interp")
                except Exception:
                    y_smooth = y
                dy = np.gradient(y_smooth, x)
                peaks, prop = find_peaks(y_smooth, prominence=peak_prominence,
                                         height=peak_height_min, distance=min_peak_distance)
                mono_start = None
                for idx in range(len(dy) - min_monotonic_span):
                    if np.all(dy[idx:] <= derivative_tol):
                        mono_start = idx; break
                is_mono = (mono_start is not None) and (
                    len(peaks) == 0 or np.max(prop.get("prominences", [0])) < peak_prominence * 1.5)
                if is_mono:
                    picked_q = float(x[mono_start])
                elif len(peaks) > 0:
                    picked_q = float(x[peaks[0]])
                else:
                    picked_q = float(x[np.argmax(y_smooth)])
                    if np.isclose(picked_q, x[-1], atol=1e-5) or np.isclose(picked_q, float(max_qt), atol=1e-9):
                        picked_q = float(start_qt)
            col = X[:, gi]
            z_valid = col[col >= min_expr]
            if z_valid.numel() == 0:
                z_valid = col
            best_cutoffs[gi] = torch.quantile(z_valid, picked_q)
            picked_qts[gi] = picked_q

        return (best_cutoffs, picked_qts) if return_quantiles else best_cutoffs

    # =========================================================================

    def get_initialized_embeddings(self, n_genes=100,
                                use_x_init=False,
                                use_simple_init=True,
                                morani_adj=True,
                                qts_prior=0.8,
                                cluster_key=None, neighs=15,
                                add_genes=[], seg_kwargs={}):
        prepare_img_features = seg_kwargs['seg_cell']
        seg_dir = seg_kwargs.get('seg_dir')
        if prepare_img_features:
            self._prepare_tissue_features(seg_cell=seg_kwargs['seg_cell'],
                                         seg_dir=seg_dir)
        else:
            self.cell_count = None
            self.in_no_cell = self.in_tiss_mask

        # Load features from the saved file
        self.prep_genes_params(add_genes=add_genes, first_n_genes=n_genes)
        if cluster_key is None:
            self.get_cell_cluster_labels(n_neighs=neighs)
        else:
            self.adata.obs[cluster_key].values
        
        # ---- Optional X_init: OT-based pre-decontamination ----
        if use_x_init:
            if morani_adj:
                print("Computing Moran's I quantile cutoffs for X_init...")
                self.invalid_cutoff_vals = self.impute_qt_vectorized(
                    n_steps=15, start_qt=0.0, max_qt=qts_prior,
                    min_spots=10, window_length=6, polyorder=2,
                    use_row_standardized_W=True, return_quantiles=False, min_expr=1.0,
                )
            else:
                self.invalid_cutoff_vals = torch.quantile(self.Y_init, 0.0, dim=0)
            if use_simple_init:
                print("Computing X_init (fast, no spatial affinity)...")
                self.compute_X_init_fast({'invalid_cutoff_vals': self.invalid_cutoff_vals})
            else:
                print("Computing X_init (with spatial transport)...")
                self.compute_X_init({'invalid_cutoff_vals': self.invalid_cutoff_vals})
        else:
            self.X_init = None  # Y_init-only mode

        
    def _flag_in_no_cell(self, seg_cell_size_thresh=10, max_no_cell_frac=0.4):
        """
        Compute 'in_no_cell' (0=out-of-tissue, 1=in-cell, 2=in-no-cell) from self.adata.obsm['segmentation_features'].

        Strategy:
        - Primary: label == 0  (no nuclei detected inside spot aperture)
        - Secondary (adaptive): label <= q5 AND area_mean_norm <= max(q10, seg_cell_size_thresh_norm)
        - Cap the fraction of in-no-cell spots (avoid overcalling on noisy slides).
        """
        
        seg_df = self.adata.obsm["segmentation_features"].copy()
        in_mask = (self.adata.obs["in_tissue"].values == 1)
        count_mask = self.adata.obs["total_counts"].values < np.quantile(self.adata.obs["total_counts"].values, 0.01)
                
        # arr = np.array([1, 2, 3, 4, 5, 6, 7, 8, 0,0,0,0,0]) 
        # arr < np.percentile(arr, 5)

        labels = seg_df.loc[in_mask, "segmentation_label"].values
        areas_norm = seg_df.loc[in_mask, "segmentation_area_mean_norm"].values

        # Quantiles to adapt thresholds
        q5_lbl = np.quantile(labels, 0.05)
        q10_area = np.quantile(areas_norm, 0.10)

        # Convert absolute pixel threshold into normalized scale
        # If you stored 'segmentation_area_mean_norm' = area / spot_diameter^2
        # then seg_cell_size_thresh_norm ~ seg_cell_size_thresh / spot_diameter^2
        # Use the empirical q10 as lower bound
        area_thresh_norm = max(q10_area, seg_cell_size_thresh)

        # Base candidates
        primary_no_cell = (seg_df["segmentation_label"].values == 0) & in_mask & count_mask
        secondary_no_cell = (
            (seg_df["segmentation_label"].values <= q5_lbl) &
            (seg_df["segmentation_area_mean_norm"].values <= area_thresh_norm) &
            in_mask & count_mask
        )
        no_cell_candidates = primary_no_cell | secondary_no_cell

        # Cap fraction to avoid overcalling if segmentation is weak
        frac = no_cell_candidates.sum() / in_mask.sum()
        if frac > max_no_cell_frac:
            # keep only strict no-nuclei rule
            no_cell_candidates = primary_no_cell

        # Compose final vector: 0,1,2
        in_no_cell = np.zeros(self.adata.n_obs, dtype=int)  # default 0 (out-of-tissue)
        in_no_cell[in_mask] = 1
        in_no_cell[no_cell_candidates] = 2

        self.adata.obs["in_no_cell"] = in_no_cell
        self.in_no_cell = torch.tensor(in_no_cell, dtype=torch.long)
        print("in_no_cell labels assigned: 0=out-of-tissue, 1=in-cell, 2=in-no-cell")
        print('Counts:')
        print(self.adata.obs['in_no_cell'].value_counts())
        
        
    def _perform_cell_segmentation(self, method='watershed', savedir=None):
        """
        Perform cell segmentation and populate per-spot segmentation features.

        Improvements:
        - Robust coordinate scaling using Visium scalefactors (always applied).
        - Watershed pipeline: Gaussian blur + Otsu + closing + small-object removal.
        - Expanded props: label, area, solidity, eccentricity.
        - Normalize area by spot diameter (pixel scale).
        - Consistent zeroing of props for out-of-tissue spots.
        """
        # --- 0) Image & scalefactors ---
        lib_id = self.lib_id
        img_key = self.img_key
        img_np = self.adata.uns['spatial'][lib_id]['images'][img_key]
        img = sq.im.ImageContainer.from_adata(self.adata, library_id=lib_id, img_key=img_key)

        # Map spot coordinates to current image resolution using scalefactors
        sf = self.adata.uns['spatial'][lib_id]['scalefactors']
        
        # --- 1) Segmentation backends ---
        layer_added = None
        if method == 'cellpose':
            from cellpose import models

            def cellpose_he(img_arr, min_size=15, flow_threshold=0.4, channel_cellpose=0):
                model = models.CellposeModel(model_type="nuclei")
                res, _, _ = model.eval(
                    img_arr,
                    channels=[channel_cellpose, 0],
                    diameter=None,
                    min_size=min_size,
                    invert=True,
                    flow_threshold=flow_threshold,
                )
                return res

            sq.im.segment(
                img=img, layer="image", channel=None, method=cellpose_he,
                flow_threshold=0.4, channel_cellpose=0, min_size=10,
                layer_added="segmentation_cellpose",
                library_id=self.lib_id
            )
            layer_added = "segmentation_cellpose"

        elif method == 'stardist':
            from csbdeep.utils import normalize
            from stardist.models import StarDist2D

            def stardist_2D_versatile_he(img_arr, nms_thresh=None, prob_thresh=0.3):
                # Normalize channels jointly (H&E-friendly)
                img_norm = normalize(img_arr, 1, 99.8, axis=(0, 1, 2))
                model = StarDist2D.from_pretrained("2D_versatile_he")
                labels, _ = model.predict_instances(img_norm,
                                                nms_thresh=nms_thresh,
                                                prob_thresh=prob_thresh)
                return labels

            sq.im.segment(
                img=img, layer="image", channel=None, method=stardist_2D_versatile_he,
                layer_added="segmentation_stardist", prob_thresh=0.3, nms_thresh=None,
                library_id=self.lib_id
            )
            layer_added = "segmentation_stardist"

        else:  # watershed default
            from skimage import filters, morphology, measure
            from skimage.color import rgb2gray
            from skimage.segmentation import watershed
            from scipy import ndimage as ndi
            import os
            import matplotlib.pyplot as plt

            # Build a grayscale preprocessed layer in the container
            # 1) Grayscale
            gray = rgb2gray(img_np) if img_np.ndim == 3 else img_np.astype(float)
            # 2) Gaussian blur (sigma tuned for Visium pixel scale)
            gray_blur = filters.gaussian(gray, sigma=1.0, preserve_range=True)
            # 3) Otsu threshold -> binary mask (nuclei ~ dark)
            th = filters.threshold_otsu(gray_blur)
            binary = gray_blur < th
            # 4) Morphological closing to connect fragmented nuclei
            binary = morphology.binary_closing(binary, morphology.disk(2))
            # 5) Remove tiny objects (debris)
            binary = morphology.remove_small_objects(binary, min_size=30)

            # 6) Distance transform + watershed
            distance = ndi.distance_transform_edt(binary)
            # Markers: local maxima of distance; simple peak threshold to avoid over-seeding
            markers = measure.label(distance > np.percentile(distance, 70))
            labels_ws = watershed(-distance, markers, mask=binary)

            # Put the result into the image container
            img.add_img(labels_ws, layer = "segmentation_watershed")
            layer_added = "segmentation_watershed"
            
            if not os.path.exists(savedir):
                os.makedirs(savedir)
            img.save(f'{savedir}/{layer_added}')
            img.show(layer_added, channelwise=True, save=f'{savedir}/{layer_added}_seg.png')
            plt.close()

        # --- 2) Segmentation features over spots ---
        # Add richer props for downstream filtering
        props = ["label", "area", "eccentricity", "solidity"]
        sq.im.calculate_image_features(
            self.adata,
            img,
            layer=layer_added,
            features="segmentation",
            key_added="segmentation_features",
            features_kwargs={
                "segmentation": {
                    "label_layer": layer_added,
                    "props": props,
                }
            },
            mask_circle=True,  # respects Visium spot aperture
            library_id=self.lib_id
        )

        seg_df = self.adata.obsm["segmentation_features"]

        # Normalize area by spot diameter pixels if available
        spot_diam_px = sf.get("spot_diameter_fullres", None)
        if spot_diam_px is not None and spot_diam_px > 0:
            # area_mean per spot divided by spot diameter squared (to approx area units)
            seg_df["segmentation_area_mean_norm"] = (
                seg_df["segmentation_area_mean"] / (spot_diam_px ** 2)
            )
        else:
            seg_df["segmentation_area_mean_norm"] = seg_df["segmentation_area_mean"]

        # Zero props for out-of-tissue spots (consistent masking)
        out_mask = (self.adata.obs["in_tissue"].values == 0)
        for col in seg_df.columns:
            if col.startswith("segmentation_"):
                seg_df.loc[out_mask, col] = 0.0

        # (Optional) Diagnostics: histograms & scatter
        if savedir is not None:
            import matplotlib.pyplot as plt
            os.makedirs(savedir, exist_ok=True)

            # Histogram of nuclei counts
            plt.figure()
            plt.hist(seg_df["segmentation_label"].values, bins=50, color="steelblue")
            q95 = np.quantile(seg_df["segmentation_label"], 0.95)
            plt.axvline(q95, color="orange", linestyle="--", label="95% quantile")
            plt.legend()
            plt.title("Segmentation: nuclei count per spot")
            plt.xlabel("# nuclei")
            plt.ylabel("spots")
            plt.savefig(f"{savedir}/{layer_added}_segmentation_histogram.png", dpi=150)
            plt.close()

            # Count vs area
            plt.figure()
            plt.scatter(seg_df["segmentation_label"], seg_df["segmentation_area_mean"],
                        alpha=0.5, s=8, c="purple")
            plt.xlabel("Segmentation label (# nuclei)")
            plt.ylabel("Segmentation area mean (px)")
            plt.title("Count vs area mean")
            plt.savefig(f"{savedir}/{layer_added}_label_area_scatter.png", dpi=150)
            plt.close()

        # Store cell_count tensor (post-normalization & masking)
        self.cell_count = torch.tensor(seg_df["segmentation_label"].values, dtype=torch.float32)
        self.adata.obs['cell_count'] = seg_df["segmentation_label"].values
        print(f"Identified cell counts for {self.cell_count.shape[0]} spots in .cell_count")
        # # update self.adata's in-tissue labels based on segmentation (sometimes the in_tissue mask is incorrect)
        # self.adata.obs['in_tissue_orig'] = self.adata.obs['in_tissue'].values
        # self.adata.obs['in_tissue'] = ((seg_df['segmentation_label'] > 0) | (self.adata.obs['in_tissue_orig'] > 0)).astype(int)
        # self.in_tiss_mask = torch.tensor(self.adata.obs['in_tissue'].values, dtype=torch.float32)

        # Keep the DataFrame back in obsm (with normalized area)
        self.adata.obsm["segmentation_features"] = seg_df
    
    def _prepare_tissue_features(self, seg_cell=True, seg_method='watershed', seg_dir=None, seg_cell_size_thresh=10):
        """
        Calculate tissue segmentation for each spot
        """
        print("Preparing/loading tissue image features for initialization...")
        feat_path = os.path.join(seg_dir, f'img_features_segcell{seg_cell}.pkl')
        if os.path.exists(feat_path):
            print(f"Loading precomputed image features from {feat_path}...")
            with open(feat_path, 'rb') as f:
                res = pkl.load(f)
            for k, v in res.items():
                print(f"{k}: {v.dtype if isinstance(v, np.ndarray) else v}")
        else:
            res = {}
            
        self.lib_id = list(self.adata.uns['spatial'].keys())[0]

        if seg_cell:            
            if 'cell_count' not in res or 'in_no_cell' not in res:
                print("Performing cell segmentation...")
                self._perform_cell_segmentation(method=seg_method, savedir=seg_dir)  ## NOTE: the main purpose is to 
                print('Select in-tissue no-cell spots...')
                self._flag_in_no_cell(seg_cell_size_thresh=seg_cell_size_thresh)

                # plot in_no_cell spatial map for visualization
                # Auto-crop to tissue extent if crop_coord is None
                if 2 in self.in_no_cell:
                    in_no_cell_mask = (self.in_no_cell.numpy() == 2)
                    coords = self.adata.obsm['spatial']
                    in_no_cell_coords = coords[in_no_cell_mask, :]
                    span = (coords.max(axis=0) - coords.min(axis=0)) * 0.1
                    # visualize one point
                    x_min, y_min = in_no_cell_coords[0,:] - span
                    x_max, y_max = in_no_cell_coords[0,:] + span
                    
                    crop_coords = (x_min, y_min, x_max, y_max)
                    sq.pl.spatial_scatter(
                        self.adata[self.adata.obs['in_tissue']==1,:],
                        color=['in_no_cell', 'in_tissue'],
                        crop_coord=crop_coords,
                        img_alpha=0.5,
                        save=os.path.join(seg_dir, 'in_no_cell_spatial_cropped.png'),
                        library_id=self.lib_id
                        )
                
                sq.pl.spatial_scatter(
                    self.adata[self.adata.obs['in_tissue']==1,:],
                    color=['in_no_cell', 'in_tissue'],
                    img_alpha=0.5,
                    save=os.path.join(seg_dir, 'in_no_cell_spatial.png'),
                    library_id=self.lib_id
                    )
                res['cell_count'] = self.cell_count
                res['in_no_cell'] = self.in_no_cell
            else:
                print("Cell segmentation and in-no-cell spots already present in loaded data.")
                self.cell_count = torch.tensor(res['cell_count'], dtype=torch.float32)
                self.in_no_cell = torch.tensor(res['in_no_cell'], dtype=torch.long)
                self.adata.obs['cell_count'] = self.cell_count.numpy()
                self.adata.obs['in_no_cell'] = self.in_no_cell.numpy()
        else:
            print("Skipping cell segmentation and in-no-cell spot identification.")
            self.cell_count = None
            self.in_no_cell = self.in_tiss_mask
        
        res = {k: v.detach().cpu().numpy() if isinstance(v, torch.Tensor) else v for k, v in res.items()}
        # for k, v in res.items(): print(f"{k}: {v.dtype if isinstance(v, np.ndarray) else v}")
        with open(feat_path, 'wb') as f:
            pkl.dump(res, f)

    def spot_cropping(self, spot_size=None):
        """Crop per-spot image patches from loaded hires image and flatten.

        Uses:
        - image: ``self.adata.uns['spatial'][self.lib_id]['images'][self.img_key]``
        - centers: ``self.adata.obsm['spatial']`` scaled by
          ``tissue_hires_scalef``
        - crop size: ``spot_diameter_fullres * tissue_hires_scalef``

        Parameters
        ----------
        spot_size : int | float | None
            Optional override for patch side length (pixels). If None, uses
            scaled Visium spot diameter from scalefactors.

        Returns
        -------
        torch.Tensor
            Shape ``(n_spots, patch_h * patch_w * n_channels)``.
        """
        img = self.adata.uns['spatial'][self.lib_id]['images'][self.img_key]
        image = np.asarray(img)

        if image.ndim == 2:
            image = image[..., None]  # add channel dim at the end to handle grayscale

        height, width, n_channels = image.shape

        sf = self.adata.uns['spatial'][self.lib_id]['scalefactors']
        hires_scale = float(sf['tissue_hires_scalef'])
        if spot_size is None:
            spot_size = float(sf['spot_diameter_fullres']) * hires_scale

        patch_size = max(1, int(round(float(spot_size))))
        half = patch_size // 2

        coords = np.asarray(self.adata.obsm['spatial'], dtype=np.float32)
        centers = coords * hires_scale  # (x, y) in hires image coordinate system
        n_spots = centers.shape[0]

        patches_flat = np.zeros(
            (n_spots, patch_size * patch_size * n_channels),
            dtype=np.float32,
        )

        for idx, (x_center, y_center) in enumerate(centers):
            x_center_i = int(round(float(x_center)))
            y_center_i = int(round(float(y_center)))

            left = x_center_i - half
            upper = y_center_i - half
            right = left + patch_size
            lower = upper + patch_size

            x0 = max(0, left)
            y0 = max(0, upper)
            x1 = min(width, right)
            y1 = min(height, lower)

            patch = np.zeros((patch_size, patch_size, n_channels), dtype=np.float32)
            if (x1 > x0) and (y1 > y0):
                px0 = x0 - left
                py0 = y0 - upper
                px1 = px0 + (x1 - x0)
                py1 = py0 + (y1 - y0)
                patch[py0:py1, px0:px1, :] = image[y0:y1, x0:x1, :]

            patches_flat[idx] = patch.reshape(-1)

        self.image_features = torch.tensor(patches_flat, dtype=torch.float32)
        # z normalize per feature (pixel) across spots
        self.image_features = (self.image_features - self.image_features.mean(dim=0, keepdim=True)) / (self.image_features.std(dim=0, keepdim=True) + 1e-8)
        
        return self.image_features
    
    def define_model(self,
                    img_emb=None,
                    n_genes=100,
                    cluster_key=None,
                    neighs=15,
                    n_diffusion_steps=1,
                    add_genes=[],
                    # X_init options
                    use_x_init: bool = False,
                    use_simple_init: bool = True,
                    morani_adj: bool = True,
                    qts_prior: float = 0.95,
                    **kwargs):
        """
        Define model.

        Parameters
        ----------
        use_x_init : bool
            When True, compute X_init (OT-based transport of out-of-tissue mass into
            tissue spots) and use it as the clean reference fed to the AE encoder.
            When False (default), the model operates on Y_init only; the AE encoder
            is trained directly on observed counts (Y_init-only pipeline).
        use_simple_init : bool
            Only relevant when use_x_init=True.
            True  → fast proportional redistribution (no spatial affinity graph).
            False → full spatial-OT transport (slower but more accurate).
        morani_adj : bool
            Only relevant when use_x_init=True.
            Compute per-gene quantile cutoffs from Moran's I curve patterns before
            building X_init.  Set False to skip (use zero cutoff, i.e. no adjustment).
        qts_prior : float
            Upper quantile bound for the Moran's I scan (default 0.8).
        """
        seg_kwargs = {k: v for k, v in kwargs.items() if k.startswith('seg_')}
        spot_size = kwargs.get('spot_size', None)

        if img_emb is None:
            # image features will be from spot cropped image patches
            self.image_features = self.spot_cropping(spot_size)
        elif isinstance(img_emb, str):
            if img_emb.endswith('.csv'):
                self.image_features = torch.tensor(pd.read_csv(img_emb).values, dtype=torch.float32)
            elif img_emb.endswith('.npy'):
                self.image_features = torch.tensor(np.load(img_emb), dtype=torch.float32)
            else:
                raise ValueError(f"Unsupported file format for img_emb: {img_emb}")
        elif isinstance(img_emb, np.ndarray):
            self.image_features = torch.tensor(img_emb, dtype=torch.float32)
        elif isinstance(img_emb, pd.DataFrame):
            self.image_features = torch.tensor(img_emb.values, dtype=torch.float32)
        elif isinstance(img_emb, torch.Tensor):
            self.image_features = img_emb
        else:
            raise ValueError(f"Unsupported type for img_emb: {type(img_emb)}")

        # Initialize expression data (and optionally X_init)
        if not hasattr(self, 'Y_init') or self.Y_init is None:
            self.get_initialized_embeddings(
                n_genes=n_genes,
                cluster_key=cluster_key,
                neighs=neighs,
                add_genes=add_genes,
                use_x_init=use_x_init,
                use_simple_init=use_simple_init,
                morani_adj=morani_adj,
                qts_prior=qts_prior,
                seg_kwargs=seg_kwargs,
            )
        elif use_x_init and (not hasattr(self, 'X_init') or self.X_init is None):
            # Y_init already computed but X_init not yet; compute it now
            if morani_adj:
                print("Computing Moran's I quantile cutoffs for X_init...")
                self.invalid_cutoff_vals = self.impute_qt_vectorized(
                    n_steps=15, start_qt=0.0, max_qt=qts_prior,
                    min_spots=10, window_length=6, polyorder=2,
                    use_row_standardized_W=True, return_quantiles=False, min_expr=1.0,
                )
            else:
                self.invalid_cutoff_vals = torch.quantile(self.Y_init, 0.0, dim=0)
            if use_simple_init:
                self.compute_X_init_fast({'invalid_cutoff_vals': self.invalid_cutoff_vals})
            else:
                self.compute_X_init({'invalid_cutoff_vals': self.invalid_cutoff_vals})
        
        # Convert spatial connectivity to sparse tensor if needed
        # print(f"[DEBUG] spatial_con type: {type(self.spatial_con)}, is_sparse: {issparse(self.spatial_con)}")
        if issparse(self.spatial_con):
            # print(f"[DEBUG] Converting scipy sparse to torch sparse...")
            # print(f"[DEBUG] spatial_con format: {self.spatial_con.format}, shape: {self.spatial_con.shape}, nnz: {self.spatial_con.nnz}")
            # Convert to COO if not already in COO format
            spatial_con_coo = self.spatial_con.tocoo() if self.spatial_con.format != 'coo' else self.spatial_con
            # print(f"[DEBUG] COO shape: {spatial_con_coo.shape}, nnz: {spatial_con_coo.nnz}")
            indices = torch.tensor(np.vstack([spatial_con_coo.row, spatial_con_coo.col]), dtype=torch.long)
            values = torch.tensor(spatial_con_coo.data, dtype=torch.float32)
            # print(f"[DEBUG] indices shape: {indices.shape}, values shape: {values.shape}")
            spatial_con = torch.sparse_coo_tensor(indices, values, spatial_con_coo.shape)
            # print(f"[DEBUG] torch sparse tensor created successfully")
        else:
            # print(f"[DEBUG] Using dense tensor, shape: {self.spatial_con.shape}")
            spatial_con = torch.tensor(self.spatial_con, dtype=torch.float32)
            
        # print(f"[DEBUG] spatial_dist type: {type(self.spatial_dist)}, is_sparse: {issparse(self.spatial_dist)}")
        if issparse(self.spatial_dist):
            # print(f"[DEBUG] Converting scipy sparse distance matrix to torch sparse...")
            # print(f"[DEBUG] spatial_dist format: {self.spatial_dist.format}, shape: {self.spatial_dist.shape}, nnz: {self.spatial_dist.nnz}")
            # Convert to COO if not already in COO format
            spatial_dist_coo = self.spatial_dist.tocoo() if self.spatial_dist.format != 'coo' else self.spatial_dist
            # print(f"[DEBUG] COO shape: {spatial_dist_coo.shape}, nnz: {spatial_dist_coo.nnz}")
            indices = torch.tensor(np.vstack([spatial_dist_coo.row, spatial_dist_coo.col]), dtype=torch.long)
            values = torch.tensor(spatial_dist_coo.data, dtype=torch.float32)
            # print(f"[DEBUG] indices shape: {indices.shape}, values shape: {values.shape}")
            spatial_dist = torch.sparse_coo_tensor(indices, values, spatial_dist_coo.shape)
            # print(f"[DEBUG] torch sparse distance tensor created successfully")
        else:
            spatial_dist = torch.tensor(self.spatial_dist, dtype=torch.float32)        

        default_params = {
            'gene_likelihood': 'zero_inflated',
        }
        # Update default parameters with any additional kwargs
        kwargs_model = {k: v for k, v in kwargs.items() if not k.startswith('seg_')}
        default_params.update(kwargs_model)

        # Determine whether to use image features in the model
        use_image = self.image_features is not None
        default_params['use_image'] = use_image

        self.kwargs = {**default_params, 'use_x_init': use_x_init, 'use_simple_init': use_simple_init,
                       'morani_adj': morani_adj, 'qts_prior': qts_prior}  # store for save/load
        
        coords_normed = (self.coords - self.coords.mean(dim=0, keepdim=True)) / (self.coords.std(dim=0, keepdim=True) + 1e-8)
        
        # Create model on CPU (all tensors should be on CPU at this point)
        print("Defining CLEARmodel model...")
        self.model = CLEARmodel(
            Y=self.Y_init,  # shape [n_spots, n_genes] – noisy observations
            X=self.X_init if (use_x_init and self.X_init is not None) else None,  # clean reference (optional)
            spatial_coords=coords_normed,
            spot_types=self.in_no_cell,
            n_diffusion_steps=n_diffusion_steps,
            niche_labels=self.img_labels,  # shape [n_spots] (niche labels)
            diffrate_labels=self.expr_cluster_labels,  # shape [n_spots] (expression cluster labels, related to diffusion rate estimation)
            spatial_dist=spatial_dist,  # shape [n_spots, n_spots]
            spatial_con=spatial_con,  # shape [n_spots, n_spots]
            image_features=self.image_features,  # shape [n_spots, n_features] or None
            cell_counts=self.cell_count,  # shape [n_spots] (cell counts)
            **default_params
        )
        print('Model defined. Moving to device...')
        # Move entire model to device (this handles all parameters and buffers)
        self.model = self.model.to(self.device)   # move module associated tensors to device (buffers mainly)
        ps = pyro.get_param_store()
        for name, value in list(ps.items()):
            ps[name] = value.to("cuda")  # move global parameter store to device
        
        self.model_name = 'CLEARmodel'
        return self.model

    @staticmethod
    def _set_module_requires_grad(module: torch.nn.Module, flag: bool):
        """Enable/disable gradients for all parameters in a module."""
        for p in module.parameters():
            p.requires_grad_(flag)

    def _run_svi_phase(
        self,
        learning_rate: float,
        n_epochs: int,
        patience: int,
        use_aux_losses: bool,
        lambda_niche_train: float,
        freeze_ae: bool,
        phase_name: str,
    ):
        """Run one SVI phase with optional AE freezing."""

        self._set_module_requires_grad(self.model.denoising_ae, not freeze_ae)

        # Auxiliary losses depend on AE gradients; disable them when AE is frozen.
        aux_enable = bool(use_aux_losses) and (not freeze_ae)
        self.model.configure_auxiliary_losses(
            enable=aux_enable,
            lambda_niche=lambda_niche_train,
        )

        def _coerce_adam_beta(x):
            if isinstance(x, torch.Tensor):
                if x.numel() != 1:
                    x = x.reshape(-1)[0]
                return float(x.item())
            return float(x)

        betas = (_coerce_adam_beta(0.9), _coerce_adam_beta(0.999))

        scheduler = pyro.optim.ReduceLROnPlateau({
            "optimizer": torch.optim.Adam,
            "optim_args": {
                "lr": learning_rate,
                "betas": betas,
                "weight_decay": 1e-4,
            },
            "patience": patience,
            "factor": 0.5,
            "threshold": 0.1,
        }, clip_args={"clip_norm": 5.0})

        elbo = pyro.infer.Trace_ELBO(num_particles=1)
        svi = pyro.infer.SVI(
            model=self.model.model,
            guide=self.model.guide,
            optim=scheduler,
            loss=elbo,
        )

        losses = []
        best_loss = float('inf')
        patience_counter = 0

        for step in range(n_epochs):
            loss = svi.step()
            losses.append(loss)

            if loss < best_loss:
                best_loss = loss
                patience_counter = 0
            else:
                patience_counter += 1

            if patience_counter >= patience:
                print(f"[{phase_name}] Early stopping at step {step}")
                break

            if step % 10 == 0:
                print(f"[{phase_name}] Step {step}: Loss = {loss:.4f}")
                # get factor loss values for logging
                aux_niche_penalty = self.model._last_loss_niche * lambda_niche_train if hasattr(self.model, '_last_loss_niche') and (self.model._last_loss_niche is not None) else None
                aux_zero_penalty = self.model._last_zero_penalty * self.model.zero_penalty_weight if hasattr(self.model, '_last_zero_penalty') and (self.model._last_zero_penalty is not None) else None
                if aux_niche_penalty is not None:
                    print(f"  - Niche penalty (cluster auxiliary loss): {aux_niche_penalty:.4f}")
                if aux_zero_penalty is not None:
                    print(f"  - Zero penalty (for zero-inflated likelihood): {aux_zero_penalty:.4f}")

        # Restore trainability for subsequent phases.
        self._set_module_requires_grad(self.model.denoising_ae, True)
        return losses

    def train(
        self,
        learning_rate=1e-4,
        n_epochs=1000,
        patience=50,
        use_aux_losses: bool = True,
        lambda_niche_train: float = 1.0,
        train_mode: str = "joint",
        # alternating (EM-style) controls
        n_alternations: int = 3,
        diffusion_epochs_per_round: int = 50,
        force_y_only: bool = True,
    ):
        """Train the model with either joint SVI or EM-style alternating optimization.

        Modes
        -----
        - ``train_mode='joint'``: original joint SVI optimization.
        - ``train_mode='alternating'``: coordinate descent / EM-style schedule:
            1) Fix diffusion params, train AE for a short phase.
            2) Fix AE params, fit diffusion params with SVI.
            3) Repeat.

        This is designed to reduce AE-vs-diffusion competition on the same structure.
        """
        train_mode = str(train_mode).lower().strip()
        if train_mode not in {"joint", "alternating"}:
            raise ValueError("train_mode must be one of {'joint', 'alternating'}")

        if train_mode == "joint":
            return self._run_svi_phase(
                learning_rate=learning_rate,
                n_epochs=n_epochs,
                patience=patience,
                use_aux_losses=use_aux_losses,
                lambda_niche_train=lambda_niche_train,
                freeze_ae=False,
                phase_name="joint",
            )

        # -----------------------------
        # Alternating / EM-style mode
        # -----------------------------
        if force_y_only and getattr(self.model, "X", None) is not None:
            print("[alternating] Forcing Y-only mode: ignoring X_init during training.")
            self.model.X = None

        all_losses = []

        # Start from AE as trainable.
        self._set_module_requires_grad(self.model.denoising_ae, True)

        for round_idx in range(int(n_alternations)):
            print("=" * 80)
            print(f"[alternating] Round {round_idx + 1}/{n_alternations}")
            print("=" * 80)

            print("[alternating] Phase B: fitting diffusion params (AE frozen)")
            phase_losses = self._run_svi_phase(
                learning_rate=learning_rate,
                n_epochs=diffusion_epochs_per_round,
                patience=patience,
                use_aux_losses=False,
                lambda_niche_train=lambda_niche_train,
                freeze_ae=True,
                phase_name=f"alt-round-{round_idx + 1}-diffusion",
            )
            all_losses.extend(phase_losses)

        return all_losses
    
    def save(self, path: str):
        """Save the trained model parameters and AnnData."""
        os.makedirs(path, exist_ok=True)
        # Save Pyro parameter store
        pyro.get_param_store().save(f'{path}/model.pt')
        print(f"Model parameters saved to {path}/model.pt")

        # Save all relevant model input and state to AnnData
        self.restore_adata(copy=False, use_mean=True)
        self.adata.uns['undiff'] = {
            'gene_selected': self.gene_selected,
            'Y_init': self.Y_init.detach().cpu().numpy(),
            'cell_count': self.cell_count.detach().cpu().numpy() if hasattr(self, 'cell_count') and self.cell_count is not None else None,
            'in_no_cell': self.in_no_cell.detach().cpu().numpy() if isinstance(self.in_no_cell, torch.Tensor) else self.in_no_cell,
            'image_features': self.image_features.detach().cpu().numpy() if hasattr(self, 'image_features') and self.image_features is not None else None,
            'model_name': self.model_name,
            'img_labels': self.img_labels.detach().cpu().numpy() if hasattr(self, 'img_labels') else None,
            'expr_cluster_labels': self.expr_cluster_labels.detach().cpu().numpy() if hasattr(self, 'expr_cluster_labels') else None,
            'kwargs': self.kwargs,
        }
        self.adata.write_h5ad(f'{path}/restored_adata.h5ad')
        print(f"AnnData saved to {path}/restored_adata.h5ad")
        
    @staticmethod
    def load(path: str, **kwargs):
        """Load a trained model from saved parameters and AnnData."""
        # Load Pyro parameter store
        params = torch.load(f"{path}/model.pt", weights_only=False)
        pyro.get_param_store().set_state(params)
        print(f"Loaded Pyro parameters from {path}/model.pt")

        # Load AnnData and model state
        adata = sc.read_h5ad(f'{path}/restored_adata.h5ad')
        restorer = undiff(adata, **kwargs)
        print(f"Loaded AnnData from {path}/restored_adata.h5ad and initialized model state.")
        return restorer

    def get_restored_counts(self, num_samples: int = 100, use_mean: bool = True) -> np.ndarray:
        """Return an estimate of the clean expression field.

        This uses the learned deterministic encoder/decoder directly, without
        running full posterior predictive sampling (which can be memory-intensive).

        Concretely, we:
        - Normalize tissue counts exactly as in the guide.
                - Pass them through the encoder to get deterministic latent ``z``.
        - Decode to log-scale expression and exponentiate.
        - Scatter into the full ``(n_spots, n_genes)`` tensor and
          rescale to match observed per-spot library sizes.

                ``num_samples`` and ``use_mean`` are kept for API compatibility but ignored.
        """

        model = self.model
        model.denoising_ae.eval()
        device = next(model.parameters()).device

        n_spots, n_genes = model.n_spots, model.n_genes
        in_tissue_spots = model.tissue_spots
        n_tissue = len(in_tissue_spots)

        # Normalize observations exactly as in guide
        # When model.X is available (X_init mode), use it as the encoder input
        # (cleaner signal).  Otherwise fall back to Y (observed counts).
        with torch.no_grad():
            src = model.X if (model.X is not None) else model.Y
            X_tissue = src[in_tissue_spots].to(device)
            X_normed = torch.log1p(X_tissue)
                
            if getattr(model, "use_image", False) and getattr(model, "n_img", 0) > 0:
                X_img = model.image_features[in_tissue_spots]     # (n_tissue, n_img)
                img_mean = X_img.mean(0, keepdim=True)
                img_std  = X_img.std(0, keepdim=True) + 1e-8
                X_img_z  = (X_img - img_mean) / img_std
                h = model.denoising_ae.encode(X_normed, X_img_z)
            else:
                h = model.denoising_ae.encode(X_normed, None)
            if hasattr(model, 'latent_heads'):
                z = model.latent_heads['z_mean'](h)
            else:
                z = h  # if no separate latent head, use the shared representation directly
                
            # Decode deterministic latent to log-scale clean expression
            if getattr(model, "use_image", False) and getattr(model, "n_img", 0) > 0:
                log_true_tissue = model.denoising_ae.decode_genes(z, X_img_z)
            else:
                log_true_tissue = model.denoising_ae.decode_genes(z, None)
            true_tissue = torch.exp(log_true_tissue)

            # Scatter to full field and match library sizes
            true_expression_expanded = true_tissue.new_zeros((n_spots, n_genes), device=device)
            true_expression_expanded[in_tissue_spots] = true_tissue

            true_lib_sizes = true_expression_expanded.sum(dim=1)
            scale_factors = model.lib_sizes.to(device) / (true_lib_sizes + 1e-8)
            true_expression_expanded = true_expression_expanded * scale_factors.unsqueeze(1)
        lat_z_full = z.new_zeros((n_spots, z.shape[1]), device=device)
        lat_z_full[in_tissue_spots] = z
        self.adata.obsm['Z_spdiff'] = lat_z_full.detach().cpu().numpy()
        return true_expression_expanded.detach().cpu().numpy()
    
    def restore_adata(self, copy: bool = False, use_mean: bool = True, num_samples: int = 100) -> sc.AnnData:
        """
        Create AnnData with restored counts at .layers['restored'].
        - If param_key is provided, fetch that param (expects [n_tissue_spots, n_genes]) and place on tissue spots.
        - Else use get_restored_counts(use_mean/sampling).
        """
        restored_counts = self.get_restored_counts(num_samples=num_samples, use_mean=use_mean)
        # Ensure gene dimension matches AnnData
        # store references
        ad = self.adata.copy() if copy else self.adata  
        ad = ad[:, self.gene_selected].copy()
        ad.layers['restored'] = restored_counts
        ad.layers['Y_init'] = self.Y_init.detach().cpu().numpy()
        # Store X_init layer when available (pre-decontaminated reference)
        if self.X_init is not None:
            ad.layers['X_init'] = self.X_init.detach().cpu().numpy()
        # Use X_init as the per-gene total reference when available (it is already
        # mass-conserving after OT transport); otherwise fall back to Y_init.
        ref_layer = 'X_init' if (self.X_init is not None and 'X_init' in ad.layers) else 'Y_init'
        gene_totals_orig = ad.layers[ref_layer].sum(axis=0)
        gene_totals_rest = ad.layers['restored'].sum(axis=0)
        scale = np.divide(gene_totals_orig, gene_totals_rest, out=np.ones_like(gene_totals_orig), where=gene_totals_rest > 0)
        # Enforce exact per-gene totals using target_sums to avoid drift after rounding
        ad.layers['restored'] = self.round_counts_to_integers(
            ad.layers['restored'] * scale,
            preserve_sum=True,
            target_sums=gene_totals_orig
        )
        if copy:
            return ad
    
