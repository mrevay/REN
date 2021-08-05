using Plots
using LaTeXStrings

a = 1
b = -0.5

xb = (a + b) / 2

λ = 1
xrange = -2:0.1:2

plot()
@gif for λ in 0.01:0.05:10
    f(mu1, mu2) = norm(mu1 - a)^2 + norm(mu2 - b)^2 + λ * abs(mu1 - mu2)
    z = f.(xrange, xrange')
    plt = contour(xrange, xrange, z, levels=40)
    plot!(xrange, xrange)
    plot!([xb], [xb]; st=:scatter, ms=10, marker=:cross)

    # Plot regions where merge will occur
    merge_mu1(mu1, mu2) =  mu2 - a + 2λ > 0
    merge_mu2(mu1, mu2) =  mu1 - b + 2λ > 0

    plot!([-2, 2], [a - 2 * λ, a - 2 * λ], ribbon=(0, 2))
    plot!([b - 2 * λ, b - 2 * λ], [-2, 2], ribbon=(0, 2))

    display(plt)
    sleep(0.001)
end


# Plot for when 
λ = 0.5
xrange = -2:0.01:2
f(mu1, mu2) = norm(mu1 .- a)^2 + norm(mu2 .- b)^2 .+ λ * abs.(mu1 - mu2)
z = f.(xrange, xrange')
plt = contour(xrange, xrange, z, levels=120)
plot!(xrange, xrange; ls=:dashdot, c=:black)
xlabel!(L"\mu_1")
ylabel!(L"\mu_2")

T = 10
mu = zeros(2, T)
mu1 = [1]
mu2 = [1.1]

plot!(mu2, mu1; st=:scatter, c=:blue, label="start")

for iter in 1:T
    mu[1, iter] = mu1[1]
    mu[2, iter] = mu2[1]
    
    mu1 = Optim.minimizer(Optim.optimize(mu1 -> f(mu1, mu2)[1], zeros(size(mu1)), Optim.BFGS()))    
    mu2 = Optim.minimizer(Optim.optimize(mu2 -> f(mu1, mu2)[1], zeros(size(mu2)), Optim.BFGS()))
    
end

plot!(mu2, mu1; st=:scatter, c=:blue, label="Finish",marker=:cross)

savefig(plt, "./figures/landscape.pdf")


plot(xrange, )