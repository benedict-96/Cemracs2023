using Metal
using GeometricMachineLearning
using GeometricMachineLearning: map_to_cpu, DummyTransformer, DummyNNIntegrator
using Plots
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.RigidBody: odeensemble, default_parameters
import Random 

# hyperparameters for the problem 
const tstep = .2
const tspan = (0., 20.)
const ics₁ = [[sin(val), 0., cos(val)] for val in 0.1:.01:(2*π)]
const ics₂ = [[0., sin(val), cos(val)] for val in 0.1:.01:(2*π)]
const ics = [ics₁..., ics₂...]

ensemble_problem = odeensemble(ics; tspan = tspan, tstep = tstep, parameters = default_parameters)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl₁ = DataLoader(ensemble_solution)

# hyperparameters concerning architecture 
const sys_dim = size(dl₁.input, 1)
const n_heads = 1
const L = 1 # transformer blocks 
const activation = tanh
const n_linear = 1
const n_blocks = 1
const skew_sym = false
const transformer_dim = 5

# backend 
const backend = MetalBackend()

# data type 
const T = Float32

# data loader 
const dl = backend == CPU() ? DataLoader(dl₁.input |> Array{T}) : DataLoader(dl₁.input |> MtlArray{T})

# hyperparameters concerning training 
const n_epochs = 3000
const batch_size = 4096
const seq_length = 5
const opt_method = AdamOptimizer(T)
const resnet_activation = tanh
const upscaling_activation = identity

# parameters for evaluation 
ics_val = (q = [sin(1.1), 0., cos(1.1)], )
const t_validation = 10

function setup_and_train(model::Union{GeometricMachineLearning.Architecture, GeometricMachineLearning.Chain}, batch::Batch)
    Random.seed!(123)

    nn₀ = NeuralNetwork(model, backend, T)
    o₀ = Optimizer(opt_method, nn₀)

    loss_array = o₀(nn₀, dl, batch, n_epochs)

    GeometricMachineLearning.map_to_cpu(nn₀), loss_array
end

feedforward_batch = Batch(batch_size)
transformer_batch = Batch(batch_size, seq_length, seq_length)

# attention only
model₁ = Chain(Dense(sys_dim, transformer_dim, tanh), VolumePreservingAttention(transformer_dim, seq_length; skew_sym = skew_sym), Linear(transformer_dim, sys_dim))

model₂ = Chain(Dense(sys_dim, transformer_dim, tanh), Chain(VolumePreservingFeedForward(transformer_dim, n_blocks * L, n_linear, resnet_activation)).layers..., Linear(transformer_dim, sys_dim))

model₃ = Chain(Dense(sys_dim, transformer_dim, tanh), Chain(VolumePreservingTransformer(transformer_dim, seq_length, n_blocks, n_linear, L, resnet_activation; skew_sym = skew_sym)).layers..., Linear(transformer_dim, sys_dim))

model₄ = RegularTransformerIntegrator(sys_dim, transformer_dim, n_heads, L, upscaling_activation, resnet_activation; add_connection = false)

nn₁, loss_array₁ = setup_and_train(model₁, transformer_batch) 
nn₂, loss_array₂ = setup_and_train(model₂, feedforward_batch)
nn₃, loss_array₃ = setup_and_train(model₃, transformer_batch) 
nn₄, loss_array₄ = setup_and_train(model₄, transformer_batch) 

function numerical_solution(sys_dim::Int, t_integration::Int, tstep::Real, ics_val::NamedTuple)
    validation_problem = odeproblem(ics_val; tspan = (0.0, t_integration), tstep = tstep, parameters = default_parameters)
    sol = integrate(validation_problem, ImplicitMidpoint())

    numerical_solution = zeros(sys_dim, length(sol.t))
    for i in axes(sol.t, 1) numerical_solution[:, i+1] = sol.q[i] end 

    t_array = zeros(length(sol.t))
    for i in axes(sol.t, 1) t_array[i+1] = sol.t[i] end

    T.(numerical_solution), T.(t_array) 
end


function plot_validation(t_validation; nn₁ = nn₁, nn₂ = nn₂, nn₃ = nn₃, nn₄ = nn₄, plot_regular_transformer = false, plot_vp_transformer = false)

    numerical, t_array = numerical_solution(sys_dim, t_validation, tstep, ics_val)

    nn₁ = NeuralNetwork(GeometricMachineLearning.DummyTransformer(seq_length), nn₁.model, nn₁.params)
    nn₂ = NeuralNetwork(GeometricMachineLearning.DummyNNIntegrator(), nn₂.model, nn₂.params)
    nn₃ = NeuralNetwork(GeometricMachineLearning.DummyTransformer(seq_length), nn₃.model, nn₃.params)
    nn₄ = NeuralNetwork(GeometricMachineLearning.DummyTransformer(seq_length), nn₄.model, nn₄.params)

    nn₁_solution = iterate(nn₁, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / tstep)) + 1)
    nn₂_solution = iterate(nn₂, numerical[:, 1]; n_points = Int(floor(t_validation / tstep)) + 1)
    nn₃_solution = iterate(nn₃, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / tstep)) + 1)
    nn₄_solution = iterate(nn₄, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / tstep)) + 1)

    ########################### plot validation

    p_validation = plot(t_array, numerical[1, :], label = "numerical solution", color = 1, linewidth = 2)

    plot!(p_validation, t_array, nn₁_solution[1, :], label = "attention only", color = 2, linewidth = 2)

    plot!(p_validation, t_array, nn₂_solution[1, :], label = "feedforward", color = 3, linewidth = 2)

    if plot_vp_transformer
        plot!(p_validation, t_array, nn₃_solution[1, :], label = "transformer", color = 4, linewidth = 2)
    end

    if plot_regular_transformer
        plot!(p_validation, t_array, nn₄_solution[1, :], label = "standard transformer", color = 5, linewidth = 2)
    end

    p_validation
end

p_validation = plot_validation(t_validation; plot_regular_transformer = true, plot_vp_transformer = true)
p_validation100 = plot_validation(100)

########################### plot training loss

p_training_loss = plot(loss_array₁, label = "attention only", color = 2, linewidth = 2, yaxis = :log)

plot!(p_training_loss, loss_array₂, label = "feedforward", color = 3, linewidth = 2)

plot!(p_training_loss, loss_array₃, label = "transformer", color = 4, linewidth = 2)

plot!(p_training_loss, loss_array₄, label = "standard transformer", color = 5, linewidth = 2)

########################## plot 3d validation 
ic = (q = [sin(1.1), 0., cos(1.1)], )

function make_validation_plot3d(t_validation::Int, nn::NeuralNetwork)
    numerical, _ = numerical_solution(sys_dim, t_validation, tstep, ic)

    nn₁_solution = iterate(nn, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / tstep)) + 1)

    ########################### plot validation

    p_validation = plot(numerical[1, :], numerical[2, :], numerical[3, :], label = "numerical solution", color = 1, linewidth = 2)

    plot!(p_validation, nn₁_solution[1, :], nn₁_solution[2,:], nn₁_solution[3, :], label = "volume-preserving feedforward", color = 2, linewidth = 2)

    p_validation
end

p_validation3d = make_validation_plot3d(20, NeuralNetwork(GeometricMachineLearning.DummyTransformer(seq_length), nn₃.model, nn₃.params))

png(p_validation, joinpath(@__DIR__, "simulations/vpt/validation"))
png(p_training_loss, joinpath(@__DIR__, "simulations/vpt/training_loss"))
png(p_validation3d, joinpath(@__DIR__, "simulations/vpt/validation3d"))