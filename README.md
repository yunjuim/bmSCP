## Publication
Sooin Yun and Yunju Im (2026). Bayesian Multivariate Spatiotemporal Changepoint Detection with Spatial Dependence: An Application to SEER Cancer Incidence. Manuscript.

## Description
1) BHMnet.jl contains the main functions for generating MCMC samples for the proposed bmSCP model described in the manuscript.
2) simdata.jld2 contains a simulated dataset used in the numerical experiments. 

## Examples
Below is an example illustrating how to run the bmSCP model.

```julia
using StatsBase, LinearAlgebra, Distributions, Random, CSV, DataFrames, JLD2, RCall

# Load model functions and simulated data
include("/mnt/nrdstor/yunjuim/yunjuim/cancer-trend-cp/2026-02-02-ftn-code-clean-up.jl")

@load "simdat.jld2" simdat M0 CA_names

# Extract outcomes and true parameters
y = Matrix(simdat[:, r"y"])
true_phi    = simdat[:, "true_phi"]
true_d      = simdat[:, "true_d"]
true_lambda = simdat[:, "true_lambda"]

# Indices of true changepoints (nonzero effects)
true_idx = findall(true_d .!= 0)

# Construct hyperparameters for the spatiotemporal model
H = construct_hp(y, 2)

# MCMC settings
n_total = 2000
n_burn  = 1000

# Run posterior sampler
s1 = run_sampler(y, M0, H, n_total = n_total)

# Posterior inclusion probabilities for changepoints
PPI = vec(mean(s1.r[:, (n_burn + 1):end], dims = 2))

# Selected changepoints based on PPI threshold
selected = findall(x -> x > 0.5, PPI)

# True positive rate for changepoint detection
TPR = length(intersect(selected, true_idx)) / length(true_idx)

# Posterior mean and RMSE for latent temporal effects (phi)
mphi = vec(mean(s1.phi[:, (n_burn + 1):end], dims = 2))
rmse_phi = sqrt(mean((mphi - true_phi).^2))
```

For more information, please contact Yunju Im at yim@unmc.edu. 
