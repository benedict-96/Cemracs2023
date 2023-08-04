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


#####

params_collection = (  (m1=2, m2=2, k1=1.5, k2=0.1, k=0.2),
            (m1=2, m2=1, k1=1.5, k2=0.1, k=0.2),
            (m1=2, m2=.5, k1=1.5, k2=0.1, k=0.2),
            (m1=2, m2=.25, k1=1.5, k2=0.1, k=0.2)
)

initial_conditions_collection = ( (q=[1.,0.], p=[2.,0.]),
                    (q=[1.,0.], p=[1.,0.]),
                    (q=[1.,0.], p=[0.5,0.])
)

t_integration = 1000
                  
function q̇(v, t, q, p, params)
    v[1] = p[1]/params.m1
    v[2] = p[2]/params.m2
end

function ṗ(f, t, q, p, params)
    f[1] = -params.k1 * q[1] - params.k * (q[1] - q[2]) #* (cos(10*q[1]) + 1) + params.k /2 * (q[1] - q[2])^2 * 10 * sin(10 * q[1])
    f[2] = -params.k2 * q[2] + params.k * (q[1] - q[2]) #* (cos(10*q[1]) + 1) 
end

q1list = []
q2list = []
p1list = []
p2list = []
for params in params_collection
    for initial_conditions in initial_conditions_collection
        pode = PODEProblem(q̇, ṗ, (0.0, t_integration), .1, initial_conditions; parameters = params)
        sol = integrate(pode,ImplicitMidpoint())

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
filename="/Users/zeyuan/Documents/GitHub/Cemracs2023/LSTM_scripts/Ocilator_quad_25Samples_1000steps_0208.jld2"
save(filename,data)



params_collection = (  (m1=2, m2=0.25, k1=1.5, k2=0.1, k=2),)
initial_conditions_collection = ( (q=[1.,0.], p=[0.66,0.]),)


t_integration = 5000

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


function extend_tuple(tuple, p0)
    (tuple...,  (q = [1.0, 0.0], p = [p0, 0.0]))
end

for p0 in LinRange(0.1, 2, 10) 
    global initial_conditions_collection = extend_tuple(initial_conditions_collection, p0)
end

initial_conditions_collection
for initial_conditions in initial_conditions_collection
    @show initial_conditions
end

q1list = []
q2list = []
p1list = []
p2list = []

for params in params_collection
    for initial_conditions in initial_conditions_collection
        pode = PODEProblem(q̇, ṗ1, (0.0, t_integration), .1, initial_conditions; parameters = params)
        sol = integrate(pode,ImplicitMidpoint())
        # push!(sols, sol)
        push!(q1list, sol.q[:,1].parent)
        push!(q2list, sol.q[:,2].parent)

        push!(p1list, sol.p[:,1].parent)
        push!(p2list, sol.p[:,2].parent)    
    end
end


p1list = hcat(p1list...)'
p2list = hcat(p2list...)'
q1list = hcat(q1list...)'
q2list = hcat(q2list...)'

data = cat(p1list,p2list,q1list,q2list,dims=3)
perm = [3,2,1]
data = permutedims(data,perm)


sequence_len = 20
shift = 1
train_input = data[:,1:20,:]
train_target = data[:,sequence_len+shift,:]
for _ in 1:3000
    start_point = rand(1:49980)
    train_input = cat(train_input,data[:,start_point:start_point+sequence_len-1,:],dims=3)
    train_target = cat(train_target,data[:,start_point+sequence_len,:],dims=2)
end
size(train_input)

data = Dict("q1list" => q1list,
            "q2list" => q2list,
            "p1list" => p1list,
            "p2list" => p2list,
            "train_input" => train_input,
            "train_target" =>train_target)


filename="/Users/zeyuan/Documents/GitHub/Cemracs2023/LSTM_scripts/Ocilator_sigmoid_63021Samples_1000steps_0208.jld2"
save(filename,data)