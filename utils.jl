
import LinearAlgebra.cholesky



# A less julianic I but sometimes useful
eye(n) = Matrix{Float64}(I, n, n)
eye(T, n) = Matrix{T}(I, n, n)

const na = [CartesianIndex()]


# converts data in μ to a dictionary
#   μ should be an n × samples matrix
#   key is cluster center 
#   value is list of indices in μ

# E.g.
# μ = [0.5 0.5 0.1 0.2
    #  0.5 0.5 0.6 0.6]
# returns {[0.5; 0.5] => [1, 2],
#          [0.1; 0.6] => [3],
#          [0.2, 0.6 => [4]}

function calc_cluster_indices(μ)
        
    # Construct clusters via hashing
    clusters = Dict()
    for (i, μi) in enumerate(eachcol(μ))
        if haskey(clusters, μi)
            append!(clusters[μi], i)
        else
            clusters[μi] = [i]
        end
    end

    # q = length(clusters)
    # n = size(μ, 1)

    # # covnert to more convenient datastructure
    # centers = SizedMatrix{n,q}(reduce(hcat, k for (k, v) in clusters))

    centers = reduce(hcat, k for (k, v) in clusters)
    indices = reduce(hcat, [v] for (k, v) in clusters)

    return centers, indices
end


function merge_clusters(centers, indices)
    # Construct clusters via hashing
    clusters = Dict()
    for (i, μi) in zip(indices, eachcol(centers))
        if haskey(clusters, μi)
            append!(clusters[μi], i)
        else
            clusters[μi] = i
        end
    end

    # q = length(clusters)
    # n = size(μ, 1)
    # centers = SizedMatrix{n,q}(reduce(hcat, k for (k, v) in clusters))
    
    # covnert to more convenient datastructure
    centers = reduce(hcat, k for (k, v) in clusters)
    indices = reduce(hcat, [v] for (k, v) in clusters)

    return centers, indices
end



function plot_1D_clusters(x, mu, indices, λ)

    plot!(x, 0.0 * ones(size(x)); st=:scatter, c=:black, marker=:o, legend=nothing, ms=4.0)
    
    max_J = 0
    n = length.(indices)
    for i in 1:size(mu, 2)
        nc = n[i]
        xbar = mean(x[:, indices[i]])
        mui = mu[:, i]
        function J(mu_c)
            α = mu_c - xbar
            return nc * α^2 / 2 + λ * sum(abs.(mu_c .- mu) .* n)
        end
        xrange = collect(minimum(x):0.1:maximum(x))
        y = J.(xrange)
        plot!(xrange, y; legend=nothing, c=:black, ls=:dashdot, lw=0.25, alpha=0.5)
        ymin = J.(mui)
        plot!(mui, ymin; st=:scatter, legend=nothing, ms=(1.5 + 10 * log(n[i]) / log(sum(n) / 2)))

        plot!([x[:, indices[i]]; mu[i] * ones(1, n[i])], 
               [zeros(1, n[i]); ymin[1] * ones(1, n[i])]; 
               ls=:dot, legend=nothing)

        max_J = ymin[1] > max_J ? ymin[1] : max_J
    end

    ylims!(-0.0, 1.5 * max_J)
end

function plot_2D_clusters(x, mu, indices)

    # Plotting
    n = length.(indices)
    plt = plot!(x[1, :], x[2, :]; st=:scatter, c=:black, marker=:o, legend=nothing, ms=4.0)
    for c in 1:size(mu, 2)
        one = ones(size(indices[c]))
        plot!([mu[1:1, c:c] * one'; x[1:1, indices[c]]], 
                [mu[2:2, c:c] * one'; x[2:2, indices[c]]]; 
                lw=0.7, alpha=0.7, ls=:dashdot)
        
        ms =  5.0 + 10 * log(n[c]) / log(sum(n) / 2)
        plot!(mu[1:1, c:c], mu[2:2, c:c]; st=:scatter, alpha=0.4, ms=ms)
    end
    return plt
end

function plot_2D_clusters(x, Ω::cluster_set)
    n = cluster_sizes(Ω)
    q = length(Ω)
    plt = plot!(x[1, :], x[2, :]; st=:scatter, c=:black, marker=:o, legend=nothing, ms=4.0)

    for c in 1:q
        one = ones(n[c])
        muc = Ω.centers[c]
        index = Ω.indices[c]
        plot!([muc[1:1,:] * one'; x[1:1, index]], 
                [muc[2:2,:] * one'; x[2:2, index]]; 
                lw=0.7, alpha=0.7, ls=:dashdot)
        
        ms =  5.0 + 10 * log(n[c]) / log(sum(n) / 2)
        plot!(muc[1:1], muc[2:2]; st=:scatter, alpha=0.4, ms=ms)
    end
    return plt
end