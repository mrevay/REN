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
    λ = λi * σ

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

            muk = qlad(k, centroids[k], Ω.centers, α, λ, n)
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

        if abs(objective[end - 1] - objective[end]) .< ϵ
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


# TODO Scale invariance test.
#       x and 2*x should lead to the same clusters
#       If we use N samples,and  T×N samples
function test_cluster()

    N = 2
    # sample data
    
    x = [(randn(N, 20) .- 5.0) (randn(N, 20) .+ 6.0)  (randn(N, 20) .+ [15.0; -5.0]) (randn(N, 20) .+ [5.0; -15.0])]

    # Easy way of calculating k is from the svd of the weight matrix?
    plot(x[1, :], x[2,:]; st=:scatter)
    
    λi = 0.5
    K = isotropic_gaussian(20.0)
    
    Ω, objective = cluster(x, λi, K; maxIters=100, ϵ=1E-7, plot_result=false);

    plot()
    plt = plot_2D_clusters(x, Ω)
    display(plt)

    # try run again with warmstart
    λ = collect(0.0001:0.05:10.0)
    Ω, obj = cluster(x, λ, K; maxIters=100, ϵ=1E-5);

    Ω2, obj2 = cluster(x, λ, K; maxIters=100, ϵ=1E-6, merge_clusters=false);

    plot()
    plot_regularization_path(x, Ω)

    plot()
    plt = plot_2D_clusters(x, Ω);
    display(plt)

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


function convex_cluster(x, λi, K)
    
    q = size(x, 2)
    μ = Variable(size(x))
    W = calculate_weights(x, K)
    
    # Scaled input
    x_bar = mean(x, dims=2)
    σ = std(norm.(eachcol(x .- x_bar)))
    λ = λi * σ

    # Construct Objective
    obj = sumsquares(x-μ)/2 + λ * sum(W[i, j]*norm(μ[:, i] - μ[:, j]) for (i, j) in product(1:q, 1:q))
    # obj = sum(x-μ)/2 + λ * sum(W[i, j]*norm(μ[:, i] - μ[:, j]) for (i, j) in product(1:q, 1:q))

    # Solve problem
    problem = Convex.minimize(obj)
    solve!(problem, SCS.Optimizer)

    return μ.value, problem.optval
end


function time_test()

    N = 10
    x = [(0.2*randn(2, N) .- 5.0) (0.2*randn(2, N) .+ 5.0)  (0.2*randn(2, N) .+ [5.0; -5.0]) (0.2*randn(2, N) .+ [-5.0; 5.0])]
    
    λi = 1.0
    K = isotropic_gaussian(10.0)
    # K = identity_kernel()
    
    Ω, objective = cluster(x, λi, K; maxIters=100, ϵ=0.0, plot_result=false);
    Ω2, objective2 = cluster(x, λi, K; maxIters=100, ϵ=0.0, plot_result=false, merge_clusters=false);
    
    mu, obj = convex_cluster(x, λi, K)
    Ωcvx = cluster_set(mu)

    plot()
    plt = plot_2D_clusters(x, Ωcvx)
    plt = plot_2D_clusters(x, Ω)
    # plt = plot_2D_clusters(x, Ω2)
    # plot!(mu[1,:], mu[2,:]; st=:scatter, ms=4)


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
