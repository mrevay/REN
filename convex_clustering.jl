using Plots: label_to_string
using Pkg
Pkg.activate(".")

using Distributions
using Revise
using Optim
using Plots

using Convex
using SCS

includet("cluster_struct.jl")
includet("utils.jl")
includet("kernels.jl")
includet("qlad.jl")



function calculate_clusters!(Ω::cluster_set, x::Matrix, λi::Real, kernel::weight_kernel;
                            plot_result=true, maxIters=100, ϵ=1E-8, verbose=true,
                            merge_clusters=true)

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
        k = 1
        while k <= length(Ω)
            n = cluster_sizes(Ω)
            α = collect_weights(W, Ω.indices)
            centroids = [mean(x[:, idx], dims=2) for idx in Ω.indices]

            muk = qlad(k, centroids[k], Ω.centers, α, λ, n; merge_clusters=merge_clusters)
            if merge_clusters
                update_clusters!(Ω, k, muk)
            else
                Ω.centers[k] .= muk
            end
  
            k = k + 1

            append!(objective, convex_clustering_objective(x, Ω, kernel, λi))

        end
            
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
                 maxIters=500, ϵ=1E-8, verbose=true, plot_result=false, merge_clusters=true)
    Ω = cluster_set(x)
    if verbose
        println("Clustering with λ = ", λi, " with kernel: ", typeof(kernel))
    end
    return calculate_clusters!(Ω, x, λi, kernel; maxIters=maxIters, ϵ=ϵ, 
                                verbose=verbose, plot_result=plot_result, 
                                merge_clusters=merge_clusters)
end


function cluster(x::Matrix, λ::AbstractVector, kernel::weight_kernel;
                 maxIters=500, ϵ=1E-8, verbose=true, merge_clusters=true)

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
                                     verbose=verbose, plot_result=false, merge_clusters=merge_clusters)
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
    obj = sumsquares(x-μ)/2 + λ * sum(W[i, j]*norm(μ[:, i] - μ[:, j]) for (i, j) in product(1:q, 1:q))

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
    # Ω2, objective2 = cluster(x, λi, K; maxIters=100, ϵ=0.0, plot_result=true, merge_clusters=false);
    
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
            T = @elapsed Ω, obj_i = cluster(x, λi, K; maxIters=100, ϵ=1E-6, plot_result=false);
            times[i, j] = T
        end
    end

    # Plotting
    plot(; xlabel="Points", ylabel="Clustering Time")
    for i in 1:size(times, 1)
        plot!(4*nPoints, times[:, i]; label=lambdas[i])
    end
    plot!()



    objectives = []
    for λi in [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0, 10.0, 50.0]
        Ω, obj_i = cluster(x, λi, K; maxIters=100, ϵ=1E-9, plot_result=false);
        push!(objectives, obj_i)
        plot()
        plt = plot_2D_clusters(x, Ω)
    end

    plot()
    for i in 1:length(objectives)
        plot!(objectives[i] / objectives[i][1])
    end
    plot!(;yscale=:log10, xscale=:log10)
end