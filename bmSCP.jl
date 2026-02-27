using Pkg, Plots, Plots.PlotMeasures, ColorSchemes, StatsPlots
using DataFrames, CSV, LinearAlgebra, StatsBase, Distributions, Random, SpecialFunctions, LaTeXStrings, JLD2, BenchmarkTools #, PolyaGammaSamplers
using RCall

R"""
library(BayesLogit)
"""

##################################################
##################################################
function generate_phi_mcar(W::Matrix, rho::Float64, Lambda::Matrix, scale_factor::Float64)
    """
    Generate phi from a Proper MCAR model with a rho parameter.

    Arguments:
    - W      : Adjacency matrix (n × n)
    - rho    : Spatial dependence parameter (scalar)
    - Lambda : q × q covariance matrix for multivariate dependence

    Returns:
    - phi    : (n × q) matrix of sampled phi values
    """

    n, q = size(W, 1), size(Lambda, 1)
    D = Diagonal(sum(W, dims=2)[:])
    Q = (D - rho * W)  # Spatial precision
    Sigma_inv = kron(inv(Lambda), Q); Sigma_inv = 0.5 * (Sigma_inv + Sigma_inv')
    S = inv(Sigma_inv); S = 0.5 * (S + S')
    S_scaled = scale_factor * S  # Reduce magnitude
    
    phi_vec = rand(MvNormal(zeros(n * q), S_scaled)) 
    phi = reshape(phi_vec, n, q)

    return phi
end

##################################################
##################################################
ilogit(z) = 1 / (1 + exp(-z))
ilogit_(z) = 1 / (1 + exp(z))

function f_add_diagnal(x, c)
    xx = copy(x)
    @inbounds for s = 1 : dim(xx)[1]; xx[s, s] += c; end
    xx
end

function compute_uil(i, l, tau1, tau2, T, lambda)
    uu = [tau1 / max(0, (j - lambda)) for j in T if j > lambda]
    isempty(uu) && return Inf  
    minimum([minimum(uu), tau2])
end

function update_mu_d!(mu_d, vmu_d, vd, sigma2d, sigma2mud_, SL_, Mstar_, n, L, nL)
    
    Vl0_ = Mstar_ / sigma2d
    for l in 1 : L
        vidx = (n * (l - 1) + 1): (n * l)
        mask = trues(nL); mask[vidx] .= false
    
        Vl_ = f_add_diagnal(Vl0_, SL_[l, l] * sigma2mud_); Vl_ = 0.5 * (Vl_ + Vl_')
        chol_Vl_ = cholesky(Vl_)

        sum_ll = zeros(n)
        for ll in 1 : L
            if ll == l
                continue
            end
            sum_ll += (SL_[l, ll] * mu_d[:, ll]) * sigma2mud_
        end

        m_tilde_l = Vl_ \ ((Mstar_ * vd[vidx]) / sigma2d - sum_ll)
        mu_d[:, l] = m_tilde_l + chol_Vl_.U \ randn(n)
        vmu_d[vidx] = copy(mu_d[:, l])
    end
end

function update_lambda!(lambda, tt, dt, ss_lambda, vd, tau1, tau2, vphi, a1, bT, psi_y, y, n, L, T)
    
    for l in 1 : L
        for i in 1 : n; 
        
            vidx = (l - 1) * n + i
            mask_i = trues(n); mask_i[i] = false    

            cur = lambda[i, l]
            prop = cur + randn() * ss_lambda 

            if all(0.05 .< prop .< 0.95) 

                t_lambda_cur = max.(T .- cur, 0)
                t_lambda_prop = max.(T .- prop, 0)

                dil = vd[vidx]
                dt_cur = dil .* t_lambda_cur
                dt_prop = dil .* t_lambda_prop

                ind_cur = (sum(abs.(dt_cur)) > tau1) | (abs(dil) > tau2)
                ind_prop = (sum(abs.(dt_prop)) > tau1) | (abs(dil) > tau2)

                star_dt_cur = a1[l] .+ vphi[vidx] .+ bT[vidx, :] .+ dt_cur .* ind_cur
                star_dt_prop = a1[l] .+ vphi[vidx] .+ bT[vidx, :] .+ dt_prop .* ind_prop

                logp_cur = sum(star_dt_cur .* y[vidx, :] .- psi_y[vidx, :] .* log.(1.0 .+ exp.(star_dt_cur)))
                logp_prop = sum(star_dt_prop .* y[vidx, :] .- psi_y[vidx, :] .* log.(1.0 .+ exp.(star_dt_prop)))

                logr = logp_prop - logp_cur
                if log(rand()) < logr; 
                    lambda[i, l] = prop
                    tt[vidx, :] = t_lambda_prop
                    dt[vidx, :] = dt_prop
                end

            end
        end
    end
end

function run_sampler(y, M0, H; n_total = 5000) 

    n, nT, L, T = H.n, H.nT, H.L, H.T
    s2b_, s2a_, Sa_ =  H.s2b_, H.s2a_, H.S_Sa_
    sigma2phi, sigma2phi_ = H.sigma2phi, H.sigma2phi_
    
    a0phi, b0phi = H.a0phi, H.b0phi
    a0psi, b0psi = H.a0psi, H.b0psi
    a0d, b0d = H.a0d, H.b0d
    a0mud, b0mud = H.a0mud, H.b0mud

    tau1, tau2 = H.tau1, H.tau2
    nu0, S_SL = H.nu0, H.S_SL
    ss_lambda = H.ss_lambda
    
    N, nL, NTL = n * nT, n * L, n * nT * L
    @rput y n nT N L nL NTL
    
    r = zeros(Bool, n, L)
    r .= true
    T2 = T .^ 2
    Tt = T'
    
    #####
    n_nb_i = sum(M0, dims = 2)[:]; M_nb = Diagonal(n_nb_i)  
    QQ = (M_nb - M0)
    A_chol_list = Vector{Cholesky{Float64, Matrix{Float64}}}(undef, n)
    B_list = Vector{Vector{Float64}}(undef, n)
    Mstar = copy(M0); 
    Mstar = 0.6 * Mstar; for i in 1 : n; Mstar[i, i] = 1.0; end;  
    eigvals1, eigvecs1 = eigen(Mstar)
    Mstar = Mstar + Diagonal(ones(n) .* maximum([abs(minimum(eigvals1)) + 1e-3, 1e-3]));
    
    for i in 1:n
        mask_i = setdiff(1:n, i)
        A = Mstar[mask_i, mask_i]
        B = Mstar[mask_i, i]
        A_chol_list[i] = cholesky(A)
        B_list[i] = B
    end
    chol_Mstar = cholesky(Mstar)
    Mstar_ = chol_Mstar \ I;
    
    ##### Initial values
    SL = [0.5^(abs(i - j)) for i in 1:L, j in 1:L];
    chol_SL = cholesky(SL)
    SL_ = chol_SL \ I
    
    a1 = rand(MvNormal(0.0 * ones(L), SL / s2a_))
    b1 = rand(MvNormal(0.0 * ones(L), SL / s2b_))
    aa = repeat(a1, inner = n)
    bT = repeat(b1, inner = n) * Tt
    
    d = ones(n, L); vd = vec(d)
    mu_d = zeros(n, L); vmu_d = vec(mu_d)
    lambda = 0.5 * ones(n, L); vlambda = vec(lambda)
    tt = max.(Tt .- vlambda, 0)
    
    phi = generate_phi_mcar(M0, 0.99999, SL, 1.0)
    phi = phi .- mean(phi, dims = 1); vphi = vec(phi)
    
    sigma2d = rand(InverseGamma(a0d, b0d))
    sigma2mud = rand(InverseGamma(a0mud, b0mud)); sigma2mud_ = 1 / sigma2mud
    
    psi = 3.0 * ones(L); psi_mat = repeat(psi, inner = n)
    zz = 0.5 * (y .- psi_mat)

    dt = vd .* tt
    delta_ind = (abs.(sum(dt, dims = 2)[:]) .> tau1).| (abs.(vd) .> tau2)
    delta = dt .* delta_ind;
    psi_y = (psi_mat .+ y)
    
    ###### Record-keeping variables
    d_r = zeros(nL, n_total)
    mu_d_r = zeros(nL, n_total)
    r_r = zeros(Bool, nL, n_total)
    lambda0_r = zeros(nL, n_total)
    lambda_r = zeros(nL, n_total)
    a_r, b_r = zeros(L, n_total), zeros(L, n_total)
    phi_r = zeros(nL, n_total)
    psi_r = zeros(L, n_total)
    SL_r = zeros(L, L, n_total);
    tau_r = zeros(2, n_total) 
    sigma2d_r = zeros(n_total)
    sigma2mud_r = zeros(n_total)
    sigma2phi_r = zeros(n_total)
    
    elapsed_time = (
    @elapsed for iter = 1 : n_total
        
        # update mu_d
        update_mu_d!(mu_d, vmu_d, vd, sigma2d, sigma2mud_, SL_, Mstar_, n, L, nL )
        
        # update lambda
        update_lambda!(lambda, tt, dt, ss_lambda, vd, tau1, tau2, vphi, a1, bT, psi_y, y, n, L, T)
        
        delta_ind = (abs.(sum(dt, dims = 2)[:]) .> tau1).| (abs.(vd) .> tau2)
        delta = dt .* delta_ind

        Eta = aa + vphi .+ bT + delta
        @rput Eta psi_y
        R"""
        vw1 = array(rpg(NTL, psi_y, Eta), dim = c(nL, nT))
        """
        @rget vw1
        
        # update d
        vstar = zz ./ vw1 .- (aa + vphi .+ bT)
        vheart = sum(vw1 .* tt .* vstar, dims = 2); heart = reshape(vheart, n, L) # tt = max.(Tt .- vlambda, 0)
        vtriangle = sum(vw1 .* (tt .^ 2), dims = 2); triangle = reshape(vtriangle, n, L)
    
        r .= true
        sum_tt = sum(tt, dims = 2)[:] 
        for l in 1 : L
            for i in 1 : n 

                uil = minimum([tau1 / sum_tt[(l - 1) * n + i], tau2])

                mask_i = trues(n); mask_i[i] = false
                S_i = Mstar[i, i]

                A_chol = A_chol_list[i]
                B = B_list[i]

                d_diff = d[mask_i, l] - mu_d[mask_i, l]
                cond_di = mu_d[i, l] + B' * (A_chol \ d_diff)
                cond_s2i = sigma2d * (S_i - B' * (A_chol \ B))
                cond_si = sqrt(cond_s2i)

                s2di = 1 / (triangle[i, l] + 1 / cond_s2i)
                sdi = sqrt(s2di)
                m_di = (heart[i, l] + cond_di / cond_s2i) * s2di

                cc = 0.5 * (-log(cond_s2i) - cond_di^2 / cond_s2i +log(s2di) + m_di^2 / s2di)
                B1 = cdf(Normal(cond_di, cond_si), uil) - cdf(Normal(cond_di, cond_si), -uil)
                B2 = exp(cc) *  (1 - cdf(Normal(m_di, sdi), uil))
                B3 = exp(cc + logcdf(Normal(m_di, sdi), -uil))

                BB = B1 + B2 + B3
                BB1, BB2, BB3 = B1 / BB, B2 / BB, B3 / BB

                idx_d = wsample(1:3, [BB1, BB2, BB3])
                if idx_d == 1
                    d[i, l] = rand(truncated(Normal(cond_di, cond_si), -uil, uil))
                    r[i, l] = false 
                elseif idx_d == 2
                    d[i, l] = rand(truncated(Normal(m_di, sdi), uil, Inf))
                else
                    d[i, l] = rand(truncated(Normal(m_di, sdi), -Inf, -uil))
                end
            end
        end
        vd = vcat(d...)    
        dt = vd .* tt
        delta_ind = (abs.(sum(dt, dims = 2)[:]) .> tau1) .| (abs.(vd) .> tau2)
        delta = dt .* delta_ind
        
        # update s2d
        s2a1d = 0.5 * nL + a0d
        s2b1d = 0.5 * sum((chol_Mstar.L \ (d - mu_d)).^2) + b0d
        sigma2d = rand(InverseGamma(s2a1d, s2b1d))
    
        # update sigma2mud
        s2a1mud = 0.5 * nL + a0mud
        s2b1mud = 0.5 * sum((chol_SL.L \ mu_d') .^ 2) + b0mud
        sigma2mud = rand(InverseGamma(s2a1mud, s2b1mud)); sigma2mud_ = 1 / sigma2mud
            
        # update a
        w_sum = sum(reshape(sum(vw1, dims = 2), n, L), dims = 1)
        Ea_ = Diagonal(w_sum[:]) + s2a_ * Sa_
        Ea_ = Symmetric(Ea_)
        chol_Ea_ = cholesky(Ea_)

        star = zz ./ vw1 .- (vphi .+ (delta + bT))
        ma = sum(reshape(sum(vw1 .* star, dims=2), n, L), dims = 1)[:]
        ma = chol_Ea_ \ ma

        rn = randn(L)
        a1 = ma + chol_Ea_.U \ rn
        aa = repeat(a1, inner = n)
            
        # update b
        XtWX = sum(reshape(vw1 * (T2), n, L), dims = 1)[:]
        Eb_ = Diagonal(XtWX) + s2b_ * SL_
        Eb_ = Symmetric(Eb_)
        chol_Eb_ = cholesky(Eb_)

        star = zz ./ vw1 .- (aa + vphi .+ delta);
        XtWstar = sum(reshape((vw1 .* star) * T, n, L), dims = 1)[:]
        mb = chol_Eb_ \ XtWstar
        b1 = mb + chol_Eb_.U \ randn(L)
        bT = repeat(b1, inner = n) * Tt
        
        # update phi
        star = zz ./ vw1 .- (aa .+ bT + delta)
        w_sum = sum(vw1, dims = 2);
        w1star_sum = sum(vw1 .* star, dims = 2); 

        for l in 1 : L
            idx = ((l-1) * n + 1) : l * n
            mask = trues(nL); mask[idx] .= false
            maskL = trues(L); maskL[l] = false  

            SL_lm = SL[maskL, l]
            SL_mm = SL[maskL, maskL]
            SL_mm_ = SL[maskL, maskL] \ I 

            cc_ = 1 / (sigma2phi * (SL[l, l] - SL_lm' * (SL_mm \ SL_lm)))

            S_l = cc_ * QQ
            m_l = (SL_lm' * SL_mm_) .* vphi[mask]

            Ephi_ = Diagonal(w_sum[idx]) + S_l; Ephi_ = Symmetric(Ephi_)
            chol_Ephi_ = cholesky(Ephi_)

            mphi = w1star_sum[idx] + S_l * m_l
            mphi = chol_Ephi_ \ mphi

            rn = randn(n)
            tmp = mphi + chol_Ephi_.U \ rn
            vphi[idx] = tmp .- mean(tmp)
        end
        phi = reshape(vphi, n, L)
            
        # update psi
        Eta = aa + vphi .+ (delta + bT)
        expEta = exp.(Eta)
        theta = expEta ./ (1 .+ expEta)
        theta = clamp.(theta, 0.01, 0.99)
        log_1_minus_theta = log.(1 .- theta);
        for l in 1 : L
            pp = [1; psi[l] ./ (psi[l] .+ (2:maximum(y)) .- 1)]  # Precompute possible pp
            LL = [sum(rand.(Bernoulli.(pp[1 : y[i + (l - 1) * n, t]]))) for i in 1 : n for t = 1 : nT] # Will sum L, so doesn't have to be a matrix
            psi[l] = rand(Gamma(sum(LL) + a0psi, 1 / (b0psi - sum(log_1_minus_theta[((l-1) * n + 1) : l * n, :]))))
        end
        psi_mat = repeat(psi, inner = n)
        zz = 0.5 * (y .- psi_mat);
        psi_y = (psi_mat .+ y)
        
        # update SL
        DD = mu_d' * mu_d
        PP = phi' * QQ * phi

        CC = sigma2mud_ * DD + sigma2phi_ * PP + s2b_ * b1 * b1' + S_SL
        CC = 0.5 * (CC + CC')
        SL = rand(InverseWishart(2 * n + nu0, CC))
        chol_SL = cholesky(SL)
        SL_ = chol_SL \ I
            
        #####
        d_r[:, iter] = vd
        mu_d_r[:, iter] = vcat(mu_d...)
        r_r[:, iter] = vcat(r...)
        lambda_r[:, iter] = vcat(lambda...)

        a_r[:, iter] = a1
        b_r[:, iter] = b1
        phi_r[:, iter] = vphi
        psi_r[:, iter] = psi

        tau_r[1, iter] = tau1
        tau_r[2, iter] = tau2
        
        SL_r[:, :, iter] = SL
        sigma2d_r[iter] = sigma2d
        sigma2mud_r[iter] = sigma2mud
        sigma2phi_r[iter] = sigma2phi    
    end

    )
    println("complete.")
    println("Elapsed time = $elapsed_time seconds")
    
    Result(d_r, mu_d_r, r_r, lambda_r, a_r, b_r, phi_r, psi_r, tau_r, SL_r, sigma2d_r, sigma2mud_r, sigma2phi_r, elapsed_time)
end

struct Result
    d::Array{Float64, 2}
    mu_d::Array{Float64, 2}
    r::Array{Bool, 2}
    lambda::Array{Float64, 2}
    a::Array{Float64, 2}
    b::Array{Float64, 2}
    phi::Array{Float64, 2}
    psi::Array{Float64, 2}
    tau::Array{Float64, 2}
    SL::Array{Float64, 3}
    sigma2d::Array{Float64, 1}
    sigma2mud::Array{Float64, 1}
    sigma2phi::Array{Float64, 1}
    elapsed_time::Float64
end  

mutable struct hp 
    n::Int32
    L::Int32
    nT::Int32
    T::Array{Float64, 1}
    
    tau1::Float64
    tau2::Float64
    
    s2b_::Float64
    s2a_::Float64
    S_Sa_::Array{Float64, 2}
    
    a0phi::Float64
    b0phi::Float64
    
    a0psi::Float64
    b0psi::Float64
    
    a0d::Float64
    b0d::Float64
    
    a0mud::Float64
    b0mud::Float64

    nu0::Float64
    S_SL::Array{Float64, 2}
    
    sigma2phi::Float64
    sigma2phi_::Float64
    
    # MH stepsize
    ss_lambda::Float64
end


function construct_hp(y, L)
    n = Int(size(y)[1] / L)
    nT = size(y)[2]
    T = collect(range(0, stop = 1, length = nT + 1))[2:end]
    
    tau1, tau2 = 2.0, 2.0
    
    s2b_, s2a_, S_Sa_ = 0.1, 0.1, I(L)
 
    nu0, S_SL = L + 1, I(L)
    
    sigma2phi = 1.0
    sigma2phi_ = 1 / sigma2phi
    
    a0phi, b0phi = 1.0, 1.0
    a0psi, b0psi = 1.0, 1.0
    a0d, b0d = 1.0, 1.0
    a0mud, b0mud = 1.0, 1.0
    
    ss_lambda = 0.5  
    
    return hp(n, L, nT, T, tau1, tau2, s2b_, s2a_, S_Sa_, a0phi, b0phi, a0psi, b0psi, a0d, b0d, a0mud, b0mud, nu0, S_SL, sigma2phi, sigma2phi_, ss_lambda)
end
