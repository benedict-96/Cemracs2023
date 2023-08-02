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


params = (m1 = 3, m2 = 5, k1 = 0.1, k2 = 0.1, k = 0.2)
x0 = 0.5
x1 = 1
pode = PODEProblem(q̇, ṗ, (0.0, 100.0), 1., [x0,0.4], [x1, 0.]; parameters = params)
sol = integrate(pode,ImplicitMidpoint())
plot(sol.q[:,1].parent,sol.q[:,2].parent)


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

# for m1 in [1,2,3,4,5]
#     for m2 in [1,2,3,4,5]
@showprogress for x0 in 0.:0.05:2
    for x1 in 0.:0.05:2
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
#     end
# end

data = Dict("q1list" => q1list,
            "q2list" => q2list,
            "p1list" => p1list,
            "p2list" => p2list,)
filename="Ocilator_400Samples_1000steps_3107.jld2"
save(filename,data)


######### Second model 
params = (m1 = 1, m2 = 2, k1 = 0.1, k2 = 0.1, k = 0.2)
t_integration = 1000

function q̇(v, t, q, p, params)
    v[1] = p[1]/params.m1
    v[2] = p[2]/params.m2
end

function ṗ(f, t, q, p, params)
    f[1] = -params.k1 * q[1] - params.k * (cos(10*q[1]) + 1) * (q[1] - q[2]) + params.k /2 * (q[1] - q[2])^2 * 10 * sin(10 * q[1])
    f[2] = -params.k2 * q[2] + params.k * (cos(10*q[1]) + 1) * (q[1] - q[2]) 
end


p1 = plot(xlims=[0,t_integration], xlab="t", ylab="x(t)", legend=:none)
p2 = plot( xlab="x(t)", ylab="y(t)", legend=:none)

x = Vector{Vector}()
x = []

for x0 in LinRange(0.1, 2, 5) 
    for x1 in LinRange(0.1, 2, 5) 
        params = (m1 = 2, m2 = 2, k1 = 1.5, k2 = 0.1, k = 0.2)
        pode = PODEProblem( q̇, ṗ, (0.0, t_integration), 0.1, [x0,0.4], [x1, 0.4]; parameters = params)
        sol = integrate(pode,ImplicitMidpoint())
        # Add for input and output
        push!(q1list, sol.q[:,1].parent)
        push!(q2list, sol.q[:,2].parent)

        push!(p1list, sol.p[:,1].parent)
        push!(p2list, sol.p[:,2].parent)
    end
end

data = Dict("q1list" => q1list,
            "q2list" => q2list,
            "p1list" => p1list,
            "p2list" => p2list,)
filename="/Users/zeyuan/Documents/GitHub/Cemracs2023/LSTM_scripts/Ocilator_cos_25Samples_1000steps_0208.jld2"
save(filename,data)
