using Pkg
cd("/Users/zeyuan/Documents/GitHub/GeometricMachineLearning.jl")
# cd("./Cemracs2023/LSTM_scripts/")
Pkg.activate(".")

using Lux
using JLD2
using MLUtils: DataLoader,splitobs
using Random
using ProgressMeter
using GeometricIntegrators
using Plots
# using Flux:loadmodel!,loadparams!


function q̇(v, t, q, p, params)
    v[1] = p[1]/params.m1
    v[2] = p[2]/params.m2
end

function σ(x::T) where T
    T(1)/(T(1)+exp(-x))
end

function ṗ1(f, t, q, p, params)
    f[1] = -params.k1 * q[1] - params.k * (q[1] - q[2]) * σ(q[1]) - params.k /2 * (q[1] - q[2])^2 * σ(q[1])^2 * exp(-q[1])
    f[2] = -params.k2 * q[2] + params.k * (q[1] - q[2]) * σ(q[1])
end

function ṗ2(f, t, q, p, params)
    f[1] = -params.k1 * q[1] - params.k * (q[1] - q[2]) * cos(q[1]) + params.k /2 * (q[1] - q[2])^2 * sin(q[1])
    f[2] = -params.k2 * q[2] + params.k * (q[1] - q[2]) * cos(q[1])
end


xt = [1, 0., 0.9, 0.]
ini_con= (q = [1.0, 0.0], p = [0.66, 0.0])
params = (m1=2, m2=0.25, k1=1.5, k2=0.1, k=2)
pode = PODEProblem(q̇, ṗ1, (0.0, 2000), .1, ini_con; parameters = (m1=2, m2=0.25, k1=1.5, k2=0.1, k=2))
sol = integrate(pode,ImplicitMidpoint())


q1list = []
q2list = []
p1list = []
p2list = []

push!(q1list, sol.q[:,1].parent)
push!(q2list, sol.q[:,2].parent)
push!(p1list, sol.p[:,1].parent)
push!(p2list, sol.p[:,2].parent)    

p1list = hcat(p1list...)'
p2list = hcat(p2list...)'
q1list = hcat(q1list...)'
q2list = hcat(q2list...)'

truth = cat(p1list,p2list,q1list,q2list,dims=3)
perm = [3,2,1]
truth = permutedims(truth,perm)
input = truth[:,1:20,:]


model = Recurrence(LSTMCell(4 => 4),return_sequence = false)
rng = Random.default_rng()
Random.seed!(rng, 0)
# ps,st=Lux.setup(Random.default_rng(),model)

@load "/Users/zeyuan/Documents/GitHub/Cemracs2023/LSTM_scripts/lstm_63000samples_seqlen20_shift1_0708.jld2"  ps st
model(input,ps,st)[1]


for _ in range(1,200)
    # @show size(x)
    # @show size(y)
    y_pred, st = model(input, ps, st)
    # @show size(y_pred[end][:,1])
    # @show size(y_pred[end-test_len:end])
    # for i in 1:test_len
    input = cat(input, y_pred, dims=2)
    # end
    # @show size(x)
    # x = x[:, 2:end, :]
    # println(y_pred)
    # @show size(input)
end

plot(input[3,:],label="Prediction")
plot!(truth[3,1:200,:],label="Truth")
# savefig("/Users/zeyuan/Documents/GitHub/Cemracs2023/LSTM_scripts/lstm_xt = [1, 0., 0.9, 0.].png")
