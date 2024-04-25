using CUDA
using Plots; pyplot()
using GeometricMachineLearning
using GeometricMachineLearning: map_to_cpu
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.RigidBody: odeproblem, odeensemble, default_parameters
using LaTeXStrings
using LinearAlgebra: norm
import Random 

# hyperparameters for the problem 
const tstep = .2
const tspan = (0., 20.)
const ics₁ = [[sin(val), 0., cos(val)] for val in .1:.01:(2*π)]
const ics₂ = [[0., sin(val), cos(val)] for val in .1:.01:(2*π)]
const ics = [ics₁..., ics₂...]

ensemble_problem = odeensemble(ics; tspan = tspan, tstep = tstep, parameters = default_parameters)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

dl₁ = DataLoader(ensemble_solution)

# hyperparameters concerning architecture 
const sys_dim = size(dl₁.input, 1)
const n_heads = 1
const L = 3 # transformer blocks 
const activation = tanh
const n_linear = 1
const n_blocks = 2
const skew_sym = true

# backend 
const backend = CUDABackend()

# data type 
const T = Float32

# data loader 
const dl = backend == CPU() ? DataLoader(dl₁.input) : DataLoader(dl₁.input |> CuArray{T})

# hyperparameters concerning training 
const n_epochs = 500000
const batch_size = 16384
const seq_length = 3
const opt_method = AdamOptimizerWithDecay(n_epochs, T; η₁ = 1e-2, η₂ = 1e-6)
const resnet_activation = tanh

# parameters for evaluation 
ics_val = [sin(1.1), 0., cos(1.1)]
ics_val₂ = [0., sin(1.1), cos(1.1)]
const t_validation = 14
const t_validation_long = 100

function train_the_network(nn₀::GeometricMachineLearning.NeuralNetwork, batch::Batch)
    Random.seed!(1234)

    o₀ = Optimizer(opt_method, nn₀)

    loss_array = o₀(nn₀, dl, batch, n_epochs)

    GeometricMachineLearning.map_to_cpu(nn₀), loss_array
end

function setup_and_train(model::GeometricMachineLearning.Architecture, batch::Batch)
    Random.seed!(1234)

    nn₀ = NeuralNetwork(model, backend, T)

    train_the_network(nn₀, batch)
end

function setup_and_train(model::GeometricMachineLearning.Chain, batch::Batch)
    Random.seed!(1234)

    nn₀ = NeuralNetwork(model, backend, T)
    nn₀ = NeuralNetwork(GeometricMachineLearning.DummyTransformer(seq_length), nn₀.model, nn₀.params)

    train_the_network(nn₀, batch)
end

feedforward_batch = Batch(batch_size)
transformer_batch = Batch(batch_size, seq_length, seq_length)

# attention only
# model₁ = Chain(VolumePreservingAttention(sys_dim, seq_length; skew_sym = skew_sym))

model₂ = VolumePreservingFeedForward(sys_dim, n_blocks * L, n_linear, resnet_activation)

model₃ = VolumePreservingTransformer(sys_dim, seq_length; n_blocks = n_blocks, n_linear = n_linear, L = L, activation = resnet_activation, skew_sym = skew_sym)

model₄ = RegularTransformerIntegrator(sys_dim, sys_dim, n_heads; n_blocks = n_blocks, L = L, resnet_activation = resnet_activation, add_connection = false)

# nn₁, loss_array₁ = setup_and_train(model₁, transformer_batch)
nn₂, loss_array₂ = setup_and_train(model₂, feedforward_batch)
nn₃, loss_array₃ = setup_and_train(model₃, transformer_batch)
nn₄, loss_array₄ = setup_and_train(model₄, transformer_batch)

function numerical_solution(sys_dim::Int, t_integration::Int, tstep::Real, ics_val::Vector)
    validation_problem = odeproblem(ics_val; tspan = (0.0, t_integration), tstep = tstep, parameters = default_parameters)
    sol = integrate(validation_problem, ImplicitMidpoint())

    numerical_solution = zeros(sys_dim, length(sol.t))
    for i in axes(sol.t, 1) numerical_solution[:, i+1] = sol.q[i] end 

    t_array = zeros(length(sol.t))
    for i in axes(sol.t, 1) t_array[i+1] = sol.t[i] end

    T.(numerical_solution), T.(t_array) 
end

function compute_neural_network_prediction(nn::NeuralNetwork{<:GeometricMachineLearning.TransformerIntegrator}, numerical::AbstractMatrix, t_validation::Integer)
    iterate(nn, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / tstep)) + 1, prediction_window = seq_length)
end

function compute_neural_network_prediction(nn::NeuralNetwork{<:GeometricMachineLearning.NeuralNetworkIntegrator}, numerical::AbstractMatrix, t_validation::Integer)
    iterate(nn, numerical[:, 1]; n_points = Int(floor(t_validation / tstep)) + 1)
end

function plot_validation(t_validation; nn₂=nn₂, nn₃=nn₃, nn₄=nn₄, plot_regular_transformer = false, plot_vp_transformer = false)

    numerical, t_array = numerical_solution(sys_dim, t_validation, tstep, ics_val)

    # nn₁_solution = iterate(nn₁, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / tstep)) + 1)
    nn₂_solution = compute_neural_network_prediction(nn₂, numerical, t_validation)
    nn₃_solution = compute_neural_network_prediction(nn₃, numerical, t_validation)
    nn₄_solution = compute_neural_network_prediction(nn₄, numerical, t_validation)

    ########################### plot validation

    p_validation = plot(t_array, numerical[1, :], label = "implicit midpoint", color = 1, linewidth = 2, dpi = 400, ylabel = "z", xlabel = "time")

    # plot!(p_validation, t_array, nn₁_solution[1, :], label = "attention only", color = 2, linewidth = 2)

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
p_validation_long = plot_validation(t_validation_long)

########################### plot training loss

p_training_loss = plot(loss_array₃, label = "transformer", color = 4, linewidth = 2, yaxis = :log, dpi = 400, ylabel = "training loss", xlabel = "epoch")

# plot!(loss_array₁, label = "attention only", color = 2, linewidth = 2)

plot!(p_training_loss, loss_array₂, label = "feedforward", color = 3, linewidth = 2)

plot!(p_training_loss, loss_array₄, label = "standard transformer", color = 5, linewidth = 2)

########################## plot 3d validation 

function sphere(r, C)   # r: radius; C: center [cx,cy,cz]
    n = 100
    u = range(-π, π; length = n)
    v = range(0, π; length = n)
    x = C[1] .+ r*cos.(u) * sin.(v)'
    y = C[2] .+ r*sin.(u) * sin.(v)'
    z = C[3] .+ r*ones(n) * cos.(v)'
    return x, y, z
end

function make_validation_plot3d(t_validation::Int, nn::NeuralNetwork, network_description = "volume-preserving transformer", color_type = 4)
    numerical, _ = numerical_solution(sys_dim, t_validation, tstep, ics_val)
    numerical₂, _ = numerical_solution(sys_dim, t_validation, tstep, ics_val₂)

    prediction_window = typeof(nn) <: NeuralNetwork{<:GeometricMachineLearning.TransformerIntegrator} ? seq_length : 1

    nn₁_solution = compute_neural_network_prediction(nn, numerical, t_validation)
    nn₁_solution₂ = compute_neural_network_prediction(nn, numerical₂, t_validation)

    ########################### plot validation

    p_validation = surface(sphere(1., [0., 0., 0.]), alpha = .2, colorbar = false, dpi = 400, xlabel = L"z_1", ylabel = L"z_2", zlabel = L"z_3", xlims = (-1, 1), ylims = (-1, 1), zlims = (-1, 1), aspect_ratio = :equal)
    
    plot!(p_validation, numerical[1, :], numerical[2, :], numerical[3, :], label = "implicit midpoint", color = 1, linewidth = 2, dpi = 400, legendfontsize = 15)
    plot!(p_validation, numerical₂[1, :], numerical₂[2, :], numerical₂[3, :], label = nothing, color = 1, linewidth = 2, dpi = 400)

    plot!(p_validation, nn₁_solution[1, :], nn₁_solution[2,:], nn₁_solution[3, :], label = network_description, color = color_type, linewidth = 2, legendfontsize = 15)
    plot!(p_validation, nn₁_solution₂[1, :], nn₁_solution₂[2,:], nn₁_solution₂[3, :], label = nothing, color = color_type, linewidth = 2)

    p_validation
end

p_validation3d = make_validation_plot3d(t_validation_long, nn₃)

p_validation3d_standard_transformer = make_validation_plot3d(t_validation_long, nn₄, "standard transformer", 5)

p_validation3d_feedforward = make_validation_plot3d(t_validation_long, nn₂, "volume-preserving feedforward", 3)

############################### plot error evolution 

function plot_error_evolution(t_validation; nn₂=nn₂, nn₃=nn₃, nn₄=nn₄, plot_regular_transformer = false, plot_vp_transformer = true)

    numerical, t_array = numerical_solution(sys_dim, t_validation, tstep, ics_val)

    # nn₁_solution = iterate(nn₁, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / tstep)) + 1)
    nn₂_solution = compute_neural_network_prediction(nn₂, numerical, t_validation)
    nn₃_solution = compute_neural_network_prediction(nn₃, numerical, t_validation)
    nn₄_solution = compute_neural_network_prediction(nn₄, numerical, t_validation)

    ########################### plot validation
    _norm(A::Matrix) = [norm(A[:, i]) for i in axes(A, 2)]

    p_validation = plot(t_array, _norm(numerical - nn₂_solution) ./ _norm(numerical), label = "feedforward", color = 3, linewidth = 2, dpi = 400, ylabel = " relative error", xlabel = "time")

    if plot_vp_transformer
        plot!(p_validation, t_array, _norm(numerical - nn₃_solution) ./ _norm(numerical), label = "volume-preserving transformer", color = 4, linewidth = 2)
    end

    if plot_regular_transformer
        plot!(p_validation, t_array, _norm(numerical - nn₄_solution) ./ _norm(numerical), label = "standard transformer", color = 5, linewidth = 2)
    end

    p_validation
end

p_error_evolution = plot_error_evolution(t_validation)
p_error_evolution_long = plot_error_evolution(t_validation_long)

##############################

png(p_validation, joinpath(@__DIR__, "simulations/vpt_"*string(T)*"/validation_"*string(seq_length)))
png(p_training_loss, joinpath(@__DIR__, "simulations/vpt_"*string(T)*"/training_loss_"*string(seq_length)))
png(p_validation3d, joinpath(@__DIR__, "simulations/vpt_"*string(T)*"/validation3d_"*string(seq_length)))
# png(p_validation3d_standard_transformer, joinpath(@__DIR__, "simulations/vpt_"*string(T)*"/standard_transformer_validation3d_"*string(seq_length)))
png(p_validation3d_feedforward, joinpath(@__DIR__, "simulations/vpt_"*string(T)*"/feedforward_validation3d_"*string(seq_length)))
png(p_error_evolution, joinpath(@__DIR__, "simulations/vpt_"*string(T)*"/error_evolution_"*string(seq_length)))
png(p_error_evolution_long, joinpath(@__DIR__, "simulations/vpt_"*string(T)*"/error_evolution_long_"*string(seq_length)))