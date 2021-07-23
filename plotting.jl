
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