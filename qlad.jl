using Plots
using Distributions
using Revise

using Optim

includet("utils.jl")


# Original Convex clustering Objective
function J(x, mu, W, λ)
    q = size(x, 2)
    gram = [W[i, j] * norm(mu[:, i] - mu[:, j]) for (i, j) in product(1:q, 1:q)]
    return sum((x .- mu).^2) + λ * sum(gram)
end


# Optimzied method for evaluating objective function
#  should probably use static arrays
function Jk(muc, c, xbar, α, mu, n, λ)
    q = length(n)

     # in case of unmerged clusters at c
    indices = filter(idx -> mu[idx] != mu[c], 1:q)
    Δmu = norm.(mu[indices] .- [muc])

    S = sum(Δmu .* α[indices, c])
    return n[c] * norm(xbar - muc)^2 / 2 + (λ * S)
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


function qlad(c, xbar, mu, α, λ, n)
    q = length(n)

    # Check if any of the notches are optima
    for a in 1:q
        if a == c
            continue
        end

        indices = filter(idx -> mu[idx] != mu[a] && idx != c, 1:q)
        notches = filter(idx -> mu[idx] == mu[a] && idx != c, 1:q)
        
        Δmu = mu[indices] .- mu[a:a]
        Δmuhat = Δmu ./ norm.(Δmu)
        
        Sk = length(indices) > 0 ? sum(Δmuhat .* α[indices, c]) : zeros(size(xbar))

        if λ * sum(α[c, notches]) >= norm(-n[c] * (xbar - mu[c]) - λ * Sk)
            println("Merging clusters ", c, " -> ", a)
            return mu[a]
        end
    end

    # If optima are not notches, optimize via LBFGS
    obj(muc) = Jk(muc, c, xbar, α, mu, n, λ)
    result = Optim.optimize(obj, copy(mu[c]), Optim.LBFGS()) 

    # grad(dJ, muc) = ∇Jk!(dJ, muc, c, xbar, mu, n, λ)
    # result = Optim.optimize(obj, mu[:, c:c], Optim.LBFGS()) 
    
    return Optim.minimizer(result)
end




# TODO: Add Warmstart
# TODO: Fix up return type for cluster
# TODO: Add solution for clusterpath with varying \gamma
# TODO: Add weighted terms to minimization. In particular a guassian weighted kernel is suggested  by LANGE 15
# TODO: Nearest neighbour weighting scheme. Could speed up convergence as gradient calculations are simplified.

