using Pkg
Pkg.activate("./GeometricMachineLearning.jl")

using GeometricMachineLearning, KernelAbstractions, LinearAlgebra, ProgressMeter, Zygote
using ChainRulesCore
using CUDA
using Random
using Plots

include("generate_data.jl")

backend = CUDABackend()
T = Float32

data_raw = generate_data()
dim, n_params, n_time_steps = size(data_raw)

data = KernelAbstractions.allocate(backend, T, size(data_raw))
copyto!(data, data_raw)

model = Chain(  MultiHeadAttention(dim,2,Stiefel=false),
                ResNet(dim,tanh),
                MultiHeadAttention(dim,2,Stiefel=false),
                ResNet(dim,tanh),
                MultiHeadAttention(dim,2,Stiefel=false),
                ResNet(dim))
ps = initialparameters(backend, T, model)

const seq_length = 20
const batch_size = 200
const n_epochs = 200

o = Optimizer(AdamOptimizer(), ps)

batch = KernelAbstractions.allocate(backend, T, dim, seq_length, batch_size)
output = KernelAbstractions.allocate(backend, T, dim, batch_size)
output_estimate = KernelAbstractions.allocate(backend, T, dim, batch_size)

# this kernel draws a batch based on arrays of parameters and time_steps
@kernel function assign_batch_kernel!(batch::AbstractArray{T, 3}, data::AbstractArray{T, 3}, params, time_steps) where T
    i,j,k = @index(Global, NTuple)
    time_step = time_steps[k]
    param = params[k]
    batch[i,j,k] = data[i,param,time_step-1+j]
end
assign_batch! = assign_batch_kernel!(backend)

# this kernel assigns the output based on the batch
@kernel function assign_output_kernel!(output::AbstractMatrix{T}, data::AbstractArray{T,3}, params, time_steps) where T 
    i,j = @index(Global, NTuple)
    time_step = time_steps[j]
    param = params[j]
    output[i,j] = data[i,param,time_step+seq_length]
end
assign_output! = assign_output_kernel!(backend)

# this kernel assigns the output estimate
@kernel function assign_output_estimate_kernel!(output_estimate::AbstractMatrix{T}, batch::AbstractArray{T,3}) where T
    i,j= @index(Global, NTuple)
    output_estimate[i,j] = batch[i,seq_length,j]
end
assign_output_estimate! = assign_output_estimate_kernel!(backend)
function assign_output_estimate(batch::AbstractArray{T, 3}) where T
    output_estimate = KernelAbstractions.allocate(backend, T, dim, batch_size)
    assign_output_estimate!(output_estimate, batch, ndrange=size(output_estimate))
    output_estimate
end

# draw batch (for one training step)
function draw_batch!(batch::AbstractArray{T, 3}, output::AbstractMatrix{T}) where T
	params = KernelAbstractions.allocate(backend, T, batch_size)
	time_steps = KernelAbstractions.allocate(backend, T, batch_size)
	rand!(Random.default_rng(), params)
	rand!(Random.default_rng(), time_steps)
	params = Int.(ceil.(n_params*params))
    time_steps = Int.(ceil.((n_time_steps-seq_length)*time_steps)) 
    assign_batch!(batch, data, params, time_steps, ndrange=size(batch))
    assign_output!(output, data, params, time_steps, ndrange=size(output))
end

function loss(ps, batch::AbstractArray{T, 3}, output::AbstractMatrix{T}) where T 
    batch_output = model(batch, ps)
    output_estimate = assign_output_estimate(batch_output)
    norm(output - output_estimate)/sqrt(batch_size)
end
loss(ps) = loss(ps, batch, output)

# these three are AD routines
@kernel function augment_zeros_kernel!(zero_tensor::AbstractArray{T, 3}, output_diff::AbstractMatrix{T}) where T
    i,j = @index(Global, NTuple)
    zero_tensor[i,seq_length,j] = output_diff[i,j]
end
augment_zeros! = augment_zeros_kernel!(backend)
function augment_zeros(output_diff::AbstractMatrix{T}) where T
    zero_tensor = KernelAbstractions.zeros(backend, T, dim, seq_length, batch_size)
    augment_zeros!(zero_tensor, output_diff, ndrange=size(output_diff))
    zero_tensor
end
function ChainRulesCore.rrule(::typeof(assign_output_estimate), batch::AbstractArray{T, 3}) where T
    output_estimate = assign_output_estimate(batch)
    function assign_output_estimate_pullback(output_diff::AbstractMatrix)
        f̄ = NoTangent()
        batch_diff = @thunk augment_zeros(output_diff)
        return f̄, batch_diff
    end
    return output_estimate, assign_output_estimate_pullback
end     

n_training_steps_per_epoch = Int(ceil(n_time_steps/batch_size))
n_training_steps = n_epochs*n_training_steps_per_epoch
ProgressMeter.ijulia_behavior(:clear)
p = Progress(n_epochs; enabled=true)
for epochs in 1:n_epochs
    for t in 1:n_training_steps_per_epoch
        draw_batch!(batch, output)
        dx = Zygote.gradient(loss, ps)[1]
        optimization_step!(o, model, ps, dx)
        # t % n_training_steps_per_epoch == 0 ? println(loss(ps)) : nothing
    end
    ProgressMeter.next!(p; showvalues = [(:Loss,loss(ps))])
end



# Constructing 8 time steps to be unrolled for the transformer
n_int = 1000
xt = [1, 0., 0.75, 0.]
n_rolled_steps = 1
params = (m1=2, m2=0.25, k1=1.5, k2=0.1, k=2)
pode = PODEProblem(q̇, ṗ1, (0.0, 0.1 * seq_length), 0.1, xt[1:2],  xt[3:4]; parameters = params)
sol = integrate(pode,ImplicitMidpoint())
x_transformer = []
push!(x_transformer, [sol.q[:,1], sol.q[:,2], sol.p[:,1], sol.p[:,2]])


@kernel function to_matrix!(matrix, x)
    i,j,k = @index(Global, NTuple)
    matrix[i,j,k] = x[i][j][k]
end
kernel! = to_matrix!(CPU())
x_transformer_matrix = zeros(1,4,seq_length)
kernel!(x_transformer_matrix, x_transformer, ndrange = size(x_transformer_matrix))
x_transformer_matrix = Float32.(x_transformer_matrix)

input_matrix = x_transformer_matrix[1,:,1:seq_length]

# Actual integration for the NN model and the numeric model
xt = input_matrix[:,end]
X = (q1 = [], q2 = [], p1 = [], p2 = [])
for t in 1:n_int
    output_matrix = model(input_matrix |> cu, ps)
    push!(X.q1, output_matrix[1,end])
    push!(X.q2, output_matrix[2,end])
    push!(X.p1, output_matrix[3,end])
    push!(X.p2, output_matrix[4,end])
    input_matrix = hcat(input_matrix[:,2 : end] ,output_matrix[:,end])
    # input_matrix = output_matrix
end
pode = PODEProblem(q̇, ṗ1, (0.0, n_int*0.1), 0.1, [Float64(xt[1]), Float64(xt[2])],[Float64(xt[3]), Float64(xt[4])]; parameters = params)
sol = integrate(pode,ImplicitMidpoint())
p1 = plot(xlims=[0,n_int], xlab="t", ylab="x(t)", legend=:bottomright)
plot!(p1,X.q1, label="NN model")
plot!(p1,sol.q[:,1], label="numeric")