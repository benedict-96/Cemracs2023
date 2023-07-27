using Pkg
cd("/Users/zeyuan/Documents/GitHub/GeometricMachineLearning.jl")
Pkg.activate(".")

using Lux
using JLD2
using MLUtils: DataLoader,splitobs
using Random
using ProgressMeter
using GeometricIntegrators
using Plots


params = (m1 = 1, m2 = 2, k1 = 0.1, k2 = 0.1, k = 0.2)

function q̇(v, t, q, p, params)
    v[1] = p[1]/params.m1
    v[2] = p[2]/params.m2
end


function ṗ(f, t, q, p, params)
    f[1] = -params.k1 * q[1] - params.k * Lux.σ(q[1]) * (q[1] - q[2]) - params.k /2 * (q[2] - q[1])^2 * Lux.σ(q[1]) ^2 * exp(-q[1])
    f[2] = -params.k2 * q[2] + params.k * Lux.σ(q[1]) * (q[1] - q[2]) 
end


p1 = plot(xlims=[0,1000], xlab="t", ylab="x(t)", legend=:none)
p2 = plot( xlab="x(t)", ylab="y(t)", legend=:none)

# x = Vector{Vector}()
q1list = []
q2list = []
p1list = []
p2list = []

for x0 in LinRange(0.1, 2, 3) 
    for x1 in LinRange(0.1, 2, 3) 
        params = (m1 = 2, m2 = 2, k1 = 1.5, k2 = 0.1, k = 0.2)
        pode = PODEProblem(q̇, ṗ, (0.0, 1000.0), 1., [x0,0.4], [x1, 0.]; parameters = params)
        sol = integrate(pode,ImplicitMidpoint())
        # Add for input and output
        push!(q1list, sol.q[:,1].parent)
        push!(q2list, sol.q[:,2].parent)

        push!(p1list, sol.p[:,1].parent)
        push!(p2list, sol.p[:,2].parent)
        # plot!(p1,sol.t, sol.q[:,1])
        # plot!(p2, sol.q[:,1], sol.q[:,2])
    end
end


data = Dict("q1list" => q1list,
            "q2list" => q2list,
            "p1list" => p1list,
            "p2list" => p2list,)
filename="Ocilator_9Samples_1000steps_2707.jld2"
save(filename,data)
