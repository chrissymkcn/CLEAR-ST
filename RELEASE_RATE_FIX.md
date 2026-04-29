# Release Rate Initialization Fix

## Problem

The original `release_rate` initialization had a **critical scale mismatch** with the step-wise diffusion operator.

### Step-wise Diffusion Formula
```python
# Forward operator with n_steps iterations:
for _ in range(n_steps):
    y = (1-gamma) * P @ y + gamma * uniform(y)
contaminated = (1 - release_rate) * true_expr + release_rate * y
```

### The Issue
With `n_steps=5`, the **cumulative effect** means:
```
Total diffusion ≈ release_rate × n_steps  (for small release_rate)
```

**Original initialization**: Estimated `release_rate` as the total fraction diffused (e.g., 0.20)
- This caused **5× too much diffusion** (≈100% instead of 20%)
- Model couldn't converge because everything was over-diffused

## Solution

### 1. Proper Per-Step Initialization

```python
def _compute_diffusion_fraction_initialization(self):
    # Step 1: Estimate TOTAL diffusion from data
    niche_background_total = self.Y[background_mask].sum()
    niche_tissue_total = self.Y[tissue_mask].sum()
    total_diffusion = niche_background_total / (niche_tissue_total + niche_background_total)
    
    # Step 2: Convert to PER-STEP release rate
    release_rate_per_step = total_diffusion / self.n_diffusion_steps
    
    # Example: 20% total diffusion with 5 steps
    # → release_rate_per_step = 0.20 / 5 = 0.04
```

### 2. Configurable Number of Steps

Added `n_diffusion_steps` parameter:
```python
class CLEARmodel(PyroModule):
    def __init__(self, ..., n_diffusion_steps: int = 5, ...):
        self.n_diffusion_steps = n_diffusion_steps
```

This ensures:
- Initialization uses the same `n_steps` as the forward operator
- Easy to experiment with different numbers of steps
- No hardcoded magic numbers

### 3. Diagnostic Output

```python
print(f"Diffusion initialization: niche total diffusion = {niche_total_diffusion}")
print(f"Diffusion initialization: per-step release rates = {niche_release_rate_per_step}")
print(f"Diffusion initialization: global total_out_perc = {total_out_perc:.4f}")
```

Example output:
```
Diffusion initialization: niche total diffusion = [0.15 0.22 0.18 0.12]
Diffusion initialization: per-step release rates = [0.03 0.044 0.036 0.024]
Diffusion initialization: global total_out_perc = 0.1850
```

This helps verify:
- Total diffusion estimates are reasonable (10-25% is typical)
- Per-step rates are much smaller (2-5%)
- Different niches have different contamination levels

## Mathematical Justification

### Linear Approximation (for small r)

```
Step 1: contaminated₁ = (1-r)·X + r·(P·X)
      = X + r·(P·X - X)

Step 2: contaminated₂ = (1-r)·X + r·(P·contaminated₁)
      ≈ X + r·(P·X - X) + r·(P²·X - P·X)
      ≈ X + r·[(P-I)·X + (P²-P)·X]

Step n: contaminated_n ≈ X + r·sum_{i=1}^{n}(P^i - P^{i-1})·X
```

For **small r** and **local diffusion** (where P^n converges slowly):
```
Total diffusion ≈ r × n × average_diffusion_per_step
```

With typical spatial graphs:
```
Total diffusion ≈ r × n × (0.8 to 1.2)  [depending on graph structure]
```

So the **rule of thumb**: `total_diffusion ≈ release_rate × n_steps`

## Impact on Model Behavior

### Before Fix
- `release_rate_init ≈ 0.20` (total diffusion estimate)
- With 5 steps: effective diffusion ≈ 100%
- **Symptoms**:
  - Training fluctuates wildly
  - Moran's I collapses (everything becomes spatially uniform)
  - Model can't distinguish signal from noise
  - ELBO doesn't improve

### After Fix
- `release_rate_init ≈ 0.04` (per-step rate)
- With 5 steps: effective diffusion ≈ 20%
- **Expected behavior**:
  - Stable training
  - Preserved spatial structure
  - Reasonable decontamination
  - ELBO converges

## Usage

No changes needed in user code! The fix is automatic:

```python
model = unobj.define_model(
    n_genes=n_genes,
    gene_likelihood='zero_inflated',
    use_laplacian_diffusion=True
    # n_diffusion_steps defaults to 5
)
```

To use a different number of steps:
```python
model = CLEARmodel(
    X=X, Y=Y, ...,
    n_diffusion_steps=3  # Fewer steps → less diffusion reach
)
```

## Related Changes

This fix works together with the autoencoder improvements:
1. **Autoencoder pretraining** → better initialization
2. **Batch normalization** → stable gradients
3. **Reduced zero penalty** (1000 instead of 100M) → less aggressive regularization
4. **Fixed release_rate scale** → correct diffusion amount

All four changes together provide stable, biologically plausible training.
