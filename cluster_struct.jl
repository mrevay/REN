
import Base.length
using Test

mutable struct cluster_set
    centers
    indices
end

cluster_set(mu::Matrix) = cluster_set(deepcopy(collect(eachcol(mu))), [[i] for i in 1:size(mu, 2)])
cluster_set(mu::Vector) = cluster_set(mu, [[i] for i in 1:length(mu)])

function update_clusters!(Ω::cluster_set, k::Int, mu_new)
    # Looks through the centers in Ω for μ_new.
    # If it find it, merge the indices, Else update mu[k]
    
    # update center
    Ω.centers[k] .= mu_new
    indices = findall(mui -> mui == mu_new, Ω.centers)

    # merge all centers with common indices
    for idx in indices[end:-1:2]
        append!(Ω.indices[indices[1]], Ω.indices[idx])
        deleteat!(Ω.indices, idx)
        deleteat!(Ω.centers, idx)
    end 
    
    return nothing
end


function label_data(Ω, x)
    labels = zeros(size(x))
    q = length(Ω)

    for i in 1:q
        labels[:, Ω.indices[i]] .= Ω.centers[i]
    end
    return labels
end


length(Ω::cluster_set) = length(Ω.centers)
cluster_sizes(Ω::cluster_set) = length.(Ω.indices)


function test_clusters()
    n = 2
    m = 5
    x = randn(n, m)
    mu = collect(eachcol(x))
    indices = [[i] for i in 1:m]

    Ω = cluster_set(mu, indices)

    Ω.centers
    Ω.indices

    update_clusters!(Ω, 2, Ω.centers[2])

    Ω.centers
    Ω.indices
   
end

