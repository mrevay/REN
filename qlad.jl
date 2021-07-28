using Plots
using Distributions
using Revise
using LinearAlgebra
using Optim

includet("utils.jl")


# # Original Convex clustering Objective
function convex_clustering_objective(x::AbstractMatrix, mu::AbstractMatrix, kernel::weight_kernel, λi)
    q = size(x, 2)

    # Scaled input
    x_bar = mean(x, dims=2)
    σ = std(norm.(eachcol(x .- x_bar)))
    λ = λi * σ

    W = calculate_weights(x, kernel)
    gram = [W[i, j] * norm(mu[:, i] - mu[:, j]) for (i, j) in product(1:q, 1:q)]
    return sum((x .- mu).^2) / 2 + λ * sum(gram)
end

# Same function as above but takes cluster set as input
function convex_clustering_objective(x, Ω::cluster_set, kernel::weight_kernel, λi)
    n = size(x, 2)
    
    # Scaled input
    x_bar = mean(x, dims=2)
    σ = std(norm.(eachcol(x .- x_bar)))
    λ = λi * σ

    mu = label_data(Ω, x)
    W = calculate_weights(x, kernel)
    gram = [W[i, j] * norm(mu[:, i] - mu[:, j]) for (i, j) in product(1:n, 1:n)]
    return sum((x .- mu).^2) / 2 + λ * sum(gram)
end


# QLAD objective
function Jk(muk, k, xbar, α, mu, n, λ)
    q = length(n)

     # in case of unmerged clusters at c
    indices = filter(idx -> mu[idx] != mu[k], 1:q)
    # indices = filter(idx -> idx != k, 1:q)
    Δmu = norm.(mu[indices] .- [muk])

    S = sum(Δmu .* α[indices, k])
    return n[k] / 2 * sum((xbar - muk).^2) + (2 * λ * S)
end


# # function ∇Jk(muc, c, xbar, mu, n, λ)
# #     dJ = zeros(size(muc))
# #     ∇J!(dJ, muc, c, xbar, mu, n, λ)
# #     return dJ
# # end


# # function ∇Jk!(dJ, muc, c, xbar, mu, n, λ)
# #     q = size(mu, 2)
    
# #     Δmu = mu[:, 1:q .!= c] .- muc
# #     Δmu = Δmu ./ sqrt.(sum(abs2, Δmu, dims=1))  
# #     dJ .= -n[c] * (xbar - muc) - n[c] * λ * (Δmu * n[1:q .!= c])[:, :]
# #     return nothing
# # end

# function test_J()

#     xbar = randn(2, 1)
#     c = 2
#     muc = SizedMatrix{2,1}(randn(2, 1))
#     mu = SizedMatrix{2,100}(randn(2, 100))
#     n = 1 .+ mod.(rand(Int64, 1, 100), 10)
#     λ = 0.1

#     # compute calues
#     J1 = J(muc, c, xbar, mu, n, λ)
    
#     dJ1 = ∇J(muc, c, xbar, mu, n, λ)
#     dJ2 = jacobian(central_fdm(5, 1), x -> J(x, c, xbar, mu, n, λ), muc)
#     print("Error in ∇J: ", norm(dJ2[1]' - dJ1))
# end


function qlad(k, xbar, mu, α, λ, n; merge_clusters=true)

    q = length(n)

    # Check if any of the notches are optima
    if merge_clusters
        for c in 1:q

            # Maybe remove this condition?
            if c == k
                continue
            end

            ℓ = filter(idx -> idx != c && idx != k, 1:q)

            Δmu = mu[ℓ] .- mu[c:c]
            Δmuhat = Δmu ./ norm.(Δmu)
    
            # In case there is only one cluster
            Sk = length(ℓ) > 0 ? sum(Δmuhat .* α[k, ℓ]) : zeros(size(xbar))

            # if λ * sum(α[c, notches]) >= norm(n[c] * (xbar - mu[k]) + λ * Sk)
            if 2 * λ * α[c, k] >= norm(n[k] * (xbar - mu[c]) + 2 * λ * Sk)
                println("Merging clusters ", k, " -> ", c)
                return mu[c]
            end
        end
    end
    # If optima are not notches, optimize via LBFGS
    obj(muc) = Jk(muc, k, xbar, α, mu, n, λ)
    result = Optim.optimize(obj, copy(mu[k]), Optim.LBFGS()) 

    # grad(dJ, muc) = ∇Jk!(dJ, muc, c, xbar, mu, n, λ)
    # result = Optim.optimize(obj, mu[:, c:c], Optim.LBFGS()) 
    
    return Optim.minimizer(result)
end




# TODO: Add Warmstart
# TODO: Fix up return type for cluster
# TODO: Add solution for clusterpath with varying \gamma
# TODO: Add weighted terms to minimization. In particular a guassian weighted kernel is suggested  by LANGE 15
# TODO: Nearest neighbour weighting scheme. Could speed up convergence as gradient calculations are simplified.

