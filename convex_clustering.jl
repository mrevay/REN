using Pkg
Pkg.activate(".")

using Distributions
using Revise
using Optim
using Plots

includet("utils.jl")
includet("kernels.jl")
includet("qlad.jl")
includet("cluster_struct.jl")


# Returns the weight matrix between clusters
function collect_weights(W, indices)

    q = length(indices)
    M = zeros(q, q)
    for (l, k) in product(1:q, 1:q)
        M[l, k] = sum(W[indices[l], indices[k]])
    end
    return M
end

function cluster(x, kernel::weight_kernel, mu0, λi; continuous_merge=true, maxIters=500, ϵ=1E-4)

    # Calculate cluster weight matrix
    W = calculate_weights(x, kernel)

    x_bar = mean(x, dims=2)
    σ = std(norm.(eachcol(x .- x_bar)))
    λ = λi * σ

    # initially a cluster per points
    Ω = cluster_set(copy(mu0))

    # Main optimization loop
    for iter = 1:maxIters

        n = cluster_sizes(Ω)
        q = length(Ω)

        plot()
        plt = plot_2D_clusters(x, Ω);
        display(plt)
        sleep(0.1)

        # One cyclic descent iteration
        k = 1
        while k <= length(Ω)
            n = cluster_sizes(Ω)
            α = collect_weights(W, Ω.indices)
            centroids = [mean(x[:, idx], dims=2) for idx in Ω.indices]

            muk = qlad(k, centroids[k], Ω.centers, α, λ, n)
            update_clusters!(Ω, k, muk)


            plot()
            plt = plot_2D_clusters(x, Ω);
            display(plt)
            sleep(0.1)
    
            k = k + 1
        end

        # if all(abs.(previous_mu - mu) .< ϵ)
        #     println("Finished clustering. Δμ: ", maximum(abs.(previous_mu - mu)), "\tIterations: ", iter)
        #     break
        # end
    end
    return Ω
end

function cluster_path(x::AbstractMatrix, λ::Vector; weight_kernel::weight_kernel, warmstart=true, merge=true)

    # List to store the results over the centroid path
    cluster_centers = []
    centroids = []
    index_sets = []

    # Calculate regularization path
    mu = x
    for λi in λ * σ
        mu0 = warmstart ? mu : x

        mu, xbar, idx = cluster(x, weight_kernel, mu0, λi)
        append!(cluster_centers, mu)
        append!(centroids, xbar)
        append!(index_sets, idx)
    end

    return cluster_centers, centroids, index_sets
end


function test_cluster()

    N = 2
    # sample data
    
    x = [(randn(N, 5) .- 5.0) (randn(N, 5) .+ 6.0)  (randn(N, 5) .+ [15.0; -5.0]) (randn(N, 5) .+ [20.0; -20.0])]

    # Easy way of calculating k is from the svd of the weight matrix?


    plot(x[1, :], x[2,:]; st=:scatter)
    
    λi = 0.01
    # K = isotropic_gaussian(10.0)
    K = identity_kernel()
    mu0 = collect(eachcol(copy(x)))

    Ω = cluster(x, K, mu0, λi; maxIters=100, ϵ=1E-4);

    plt = plot_2D_clusters(x, Ω);
    display(plt)

 
end