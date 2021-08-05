
using Pkg
Pkg.activate(".")

using Profile
using Distributions
using Revise
using Optim
using Plots

using Convex
using SCS
using LaTeXStrings

includet("cluster_struct.jl")
includet("utils.jl")
includet("kernels.jl")
includet("qlad.jl")



function calculate_descent_indices(Ω, centroids, n, α, λ)
    
    grad_norm = zeros(length(Ω))
    mu = Ω.centers

    for k in 1:length(Ω)
        muk = mu[k]
        xbar = centroids[k][:, 1]
        g = ∇Jk(muk, k, xbar, α, Ω.centers, n, λ)
        grad_norm[k] = norm(g)
    end

    return [argmax(grad_norm)]

end

function calculate_clusters!(Ω::cluster_set, x::Matrix, λi::Real, kernel::weight_kernel;
                            plot_result=true, maxIters=100, ϵ=1E-8, verbose=true,
                            merge_clusters=true, innerIter=100)

    # Calculate cluster weight matrix
    W = calculate_weights(x, kernel)

    x_bar = mean(x, dims=2)
    σ = std(norm.(eachcol(x .- x_bar)))
    λ = λi * σ / norm(W)

    objective = [convex_clustering_objective(x, Ω, kernel, λi)]

    # Plotting and printing
    if verbose
        println("\t Iteration: ", 0, "\tclusters: ", length(Ω), "\t Objective ", objective[end])
    end

    # Main optimization loop
    for iter = 1:maxIters
      
        # One cyclic descent iteration
        for j in 1:innerIter

            n = cluster_sizes(Ω)
            α = collect_weights(W, Ω.indices)
            centroids = [mean(x[:, idx], dims=2) for idx in Ω.indices]

            k = calculate_descent_indices(Ω, centroids, n, α, λ)[1]
            println(k)
            muk = qlad(k, centroids[k], Ω.centers, α, λ, n; merge_clusters=merge_clusters)

            if merge_clusters
                update_clusters!(Ω, k, muk)
            else
                Ω.centers[k] .= muk
            end
            
        end
        append!(objective, convex_clustering_objective(x, Ω, kernel, λi))
        
            
        # Plotting and printing
        if verbose
            println("\t Iteration: ", iter, "\tclusters: ", length(Ω), "\t Objective ", objective[end])
        end

        if plot_result
            plot()
            plt = plot_2D_clusters(x, Ω);
            display(plt)
            sleep(0.0001)
        end

        if abs(objective[end - 1] - objective[end]) / objective[end] .< ϵ
            println("Finished clustering. ΔJ: ", abs(objective[end - 1] - objective[end]), "\tIterations: ", iter)
            break
        end
    end
    return Ω, objective
end


function cluster(x::Matrix, λi::Real, kernel::weight_kernel;
                 maxIters=500, ϵ=1E-8, verbose=true, plot_result=false, merge_clusters=true,
                 innerIter=100)
    Ω = cluster_set(x)
    if verbose
        println("Clustering with λ = ", λi, " with kernel: ", typeof(kernel))
    end
    return calculate_clusters!(Ω, x, λi, kernel; maxIters=maxIters, ϵ=ϵ, 
                                verbose=verbose, plot_result=plot_result, 
                                merge_clusters=merge_clusters, 
                                innerIter=innerIter)
end


function cluster(x::Matrix, λ::AbstractVector, kernel::weight_kernel;
                 maxIters=500, ϵ=1E-8, verbose=true, merge_clusters=true, innerIter=100)

    if verbose
        println("Calculating Regularization path = ", λi, " with kernel: ", typeof(kernel))
    end

    result = []
    objective = []
    Ω = cluster_set(x)
    for λi in λ

        if verbose
            println("Clustering with λ = ", λi, " with kernel: ", typeof(kernel))
        end

        _, obj = calculate_clusters!(Ω, x, λi, kernel; maxIters=maxIters, ϵ=ϵ, 
                                     verbose=verbose, plot_result=false, merge_clusters=merge_clusters,
                                     innerIter=innerIter)
        push!(result, deepcopy(Ω))
        push!(objective, deepcopy(obj))
    end
    return result, objective
end



function convex_cluster(x, λi, K)
    
    q = size(x, 2)
    μ = Variable(size(x))
    W = calculate_weights(x, K)
    
    # Scaled input
    x_bar = mean(x, dims=2)
    σ = std(norm.(eachcol(x .- x_bar)))
    λ = λi * σ / norm(W)

    # Construct Objective
    obj = (sumsquares(x-μ)/2 + λ * sum(W[i, j]*norm(μ[:, i] - μ[:, j]) for (i, j) in product(1:q, 1:q))) / q

    # Solve problem
    problem = Convex.minimize(obj)
    solve!(problem, SCS.Optimizer)

    return μ.value, problem.optval
end


function half_moons()

    T = 20
    
    # Sample half moons
    θ1 = rand(Uniform(-pi, 0), 1, T)
    θ2 = rand(Uniform(0, pi), 1, T)
    r1 = 1 .+ 0.1 * randn(1, T)
    r2 = 1 .+ 0.1 * randn(1, T)

    p1 = [r1 .* cos.(θ1); r1 .* sin.(θ1)] .+ [1.0; 0.0]
    p2 = [r2 .* cos.(θ2); r2 .* sin.(θ2)]

    x = [p1 p2]

    # Calculate clusters
    K = isotropic_gaussian(0.5)
    λ = 10 .^ collect(-4:(2/80):2)
    Ω, obj = cluster(x, λ, K; maxIters=100, ϵ=1E-5);

    
    K = isotropic_gaussian(0.5)
    # λ = 1.5 * 10 .^ collect(-3:(2/100):0)
    λ = 0.01
    Ω_no_merge, obj_no_merg = cluster(x, λ, K; maxIters=100, ϵ=1E-8, merge_clusters=false);
    


    plot(x[1,:], x[2,:]; st=:scatter)
    
    plot()
    plot_regularization_path(x, Ω)
    
end


function time_test()

    N = 10
    x = [(0.2*randn(2, N) .- 5.0) (0.2*randn(2, N) .+ 5.0)  (0.2*randn(2, N) .+ [5.0; -5.0]) (0.2*randn(2, N) .+ [-5.0; 5.0])]

    λi = 1.0
    K = isotropic_gaussian(10.0)
    # K = identity_kernel()
    
    Ω, objective = cluster(x, λi, K; maxIters=100, ϵ=0.0, plot_result=false);
    
    mu, obj = convex_cluster(x, λi, K)
    Ωcvx = cluster_set(mu)

    plot()
    plt = plot_2D_clusters(x, Ωcvx)
    plt = plot_2D_clusters(x, Ω)


    println("cyclic descent: ", objective[end])
    println("Convex: ", obj)

    # Recalculate objective to double check I'm not crazy ... I am?
    # Difference in return value of cluster objective and evaluating afterwards?
    convex_clustering_objective(x, label_data(Ω, x), K, λi)
    convex_clustering_objective(x, label_data(Ω2, x), K, λi)
    convex_clustering_objective(x, label_data(Ωcvx, x), K, λi)

    # Issue is scaling of λi!!!!!!

    plt = plot_2D_clusters(x, Ω2)
end



function test_scaling()
    # Ideally, we should rescale everything to be invariant to the number of points.
    
    nPoints = [10, 20, 30, 40, 60, 80, 100, 120]
    lambdas = [1.0, 5.0, 7.0, 10.0, 20.0]
    K = isotropic_gaussian(10.0)

    times =  zeros(length(nPoints), length(lambdas))
    
    for (i, N) in enumerate(nPoints)    
        x = [(0.5*randn(2, N) .- [5.0; 2.5]) (0.5*randn(2, N) .+ [5.0; 2.5]) (0.5*randn(2, N) .+ [5.0; -2.5]) (0.5*randn(2, N) .- [5.0; -2.5])]
        for (j, λi) in enumerate(lambdas)

            println("Testing runtime for λ:", λi, "\t Points: ", 4*N)
            T = @elapsed Ω, obj_i = cluster(x, λi, K; maxIters=100, ϵ=1E-12, plot_result=false);
            times[i, j] = T
        end
    end

    # Plotting
    plt = plot(; xlabel="Points", ylabel="Clustering Time (S)")
    for i in 1:size(times, 2)
        plot!(4*nPoints, times[:, i]; label=string("λ = ", lambdas[i]))
    end
    plot!()

    savefig(plt, string("./figures/clustering_runtime_vs_lambda.pdf"))
    
    for n in [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400]
        plot!([n, n], [0.1, 800]; label=nothing, c=:black, ls=:dashdot, alpha=0.2)
    end
    plot!(;yscale=:log10, xscale=:log10)
    savefig(plt, string("./figures/clustering_runtime_vs_lambda_logged.pdf"))

    # Profiling
    λi = 5.0
    N = 25
    p = 2
    Δ = []
    nSamples = [2, 4, 8,10, 15, 30, 50, 80, 100, 200]
    for N in nSamples
        x = [(0.5*randn(2, N) .- [5.0; 2.5]) (0.5*randn(2, N) .+ [5.0; 2.5]) (0.5*randn(2, N) .+ [5.0; -2.5]) (0.5*randn(2, N) .- [5.0; -2.5])]

        plot(x[1,:], x[2,:]; st=:scatter)

        Ω, obj_i = cluster(x, λi, K; maxIters=200, ϵ=0.0, plot_result=true, merge_clusters=true, innerIter=50);

        # mu, obj_cvx = convex_cluster(x, λi, K)
        # Ωcvx = cluster_set(mu)
        push!(Δ, obj_i)
    end

    plot()
    for i in 1:length(nSamples)
        err = (Δ[i] .- Δ[i][end]) ./ (Δ[i][1] - Δ[i][end])
        iters = 10*(1:length(Δ[i]))

        plot!(iters[err .> 1E-8], err[err .> 1E-8] ; label=4*nSamples[i])
    end
    plot!()
    xlabel!("Iterations")
    ylabel!(L"\frac{J - J_\star}{J_0 - J_\star}")
    savefig(plt, string("./figures/obj_vs_N_lambda_", λi, ".pdf"))

    plot!(;xscale=:log10, yscale=:log10)
    savefig(plt, string("./figures/obj_vs_N_lambda_", λi, "_log.pdf"))


    plot()
    plt = plot_2D_clusters(x, Ωcvx)
    plt = plot_2D_clusters(x, Ω)

    # Plot objectives
    plt = plot(obj_i .- obj_cvx)
    ylabel!("Objective")
    xlabel!("Iterations")
    savefig(plt, "./figures/objective.pdf")
    
    plt = plot(abs.(obj_i .- obj_cvx))
    ylabel!("Objective")
    xlabel!("Iterations")
    
    plot!(;yscale=:log10)
    savefig(plt, "./figures/objective_log_abs.pdf")

    obj_i[end] .- obj_cvx
    

    Ωcvx.centers

end


