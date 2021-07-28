using IterTools
using LinearAlgebra
using Revise

includet("utils.jl")

abstract type weight_kernel
end

struct identity_kernel <: weight_kernel
end

struct nearest_neighbour <: weight_kernel
    k
end

struct gaussian_kernel <: weight_kernel
    Σ
    gaussian_kernel(Σ) = isposdef(Σ) ? new(Σ) : throw(DomainError("Σ must be positive definite"))
end


# Should add check for users
        function isotropic_gaussian(scale)
    if scale <= 0
        throw(DomainError("Scale must be greater than zero"))
    end
    return gaussian_kernel(I / scale^2)
end


function calculate_weights(x::AbstractMatrix, kernel::gaussian_kernel)
    kernel_func(x1, x2) = exp(-(x1 - x2)' * kernel.Σ * (x1 - x2))
    W = [kernel_func(xi, xj) for (xi, xj) in product(eachcol(x), eachcol(x))]
    return W
end

function calculate_weights(x::AbstractMatrix, kernel::identity_kernel)
    n = size(x, 2)
    W = ones(n, n)
return W
end

# Returns the weight matrix between clusters
function collect_weights(W, indices)

    q = length(indices)
    M = zeros(q, q)
    for (l, k) in product(1:q, 1:q)
        M[l, k] = sum(W[indices[l], indices[k]])
    end
    return M
end

function tests()
    x = sort(randn(2, 100), dims=2)
    
    W = calculate_weights(x, isotropic_gaussian(2))
    W = calculate_weights(x, isotropic_gaussian(-2))  # error

    W = calculate_weights(x, gaussian_kernel(I))
    W = calculate_weights(x, gaussian_kernel(-I))

    W = calculate_weights(x, identity_kernel())
end