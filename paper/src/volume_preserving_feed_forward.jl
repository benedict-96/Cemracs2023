using CUDA
using GeometricMachineLearning
using GeometricMachineLearning: map_to_cpu
using Plots
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.RigidBody: odeproblem, default_parameters 
using GeometricEquations: EnsembleProblem
using LinearAlgebra: norm 
import Random 

const ics₁ = [(q = [sin(val), 0., cos(val)], ) for val in 0.1:.01:(2*π)]
const ics₂ = [(q = [0., sin(val), cos(val)], ) for val in 0.1:.01:(2*π)]
const ics = [ics₁..., ics₂...]

const tstep = .2
const tspan = (0., 20.)

const sys_dim = length(ics[1].q)

const n_blocks = 2
const n_linear = 1
const activation = tanh
const model = VolumePreservingFeedForward(sys_dim, n_blocks, n_linear, activation)

const backend = CUDABackend()
# const T = backend == CPU() ? Float64 : Float32
const T = Float32

const batch_size = 16384
const opt_method = AdamOptimizer(T)
const n_epochs = 1000

const t_validation = 14

ensemble_problem = EnsembleProblem(odeproblem().equation, tspan, tstep, ics, default_parameters)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())
const dl₁ = DataLoader(ensemble_solution)

function train_network(T::Type{<:Number}, backend::Backend)
    dl = backend == CPU() ? dl₁ : DataLoader(dl₁.input |> CuArray{T})

    nn = NeuralNetwork(model, backend, T)

    o = Optimizer(opt_method, nn)

    batch = Batch(batch_size, 1)

    loss_array₁ =  o(nn, dl, batch, n_epochs)
end

loss_array₁ = train_network(T, backend)

ic = (q = [sin(1.1), 0., cos(1.1)], )

function numerical_solution(sys_dim::Int, t_integration::Int, tstep::Real, ic::NamedTuple)
    validation_problem = odeproblem(ic; tspan = (0.0, t_integration), tstep = tstep, parameters = default_parameters)
    sol = integrate(validation_problem, ImplicitMidpoint())

    numerical_solution = zeros(sys_dim, length(sol.t))
    for i in axes(sol.t, 1) numerical_solution[:, i+1] = sol.q[i] end 

    t_array = zeros(length(sol.t))
    for i in axes(sol.t, 1) t_array[i+1] = sol.t[i] end

    T.(numerical_solution), T.(t_array) 
end

function make_validation_plot(t_validation::Int, nn::NeuralNetwork)

    numerical, t_array = numerical_solution(sys_dim, t_validation, tstep, ic)

    nn₁_solution = iterate(nn, numerical[:, 1]; n_points = Int(floor(t_validation / tstep)) + 1)

    ########################### plot validation

    p_validation = plot(t_array, numerical[1, :], label = "numerical solution", color = 1, linewidth = 2)

    plot!(p_validation, t_array, nn₁_solution[1, :], label = "volume-preserving feedforward", color = 2, linewidth = 2)

    p_validation
end

function make_validation_plot3d(t_validation::Int, nn::NeuralNetwork)
    numerical, _ = numerical_solution(sys_dim, t_validation, tstep, ic)

    nn₁_solution = iterate(nn, numerical[:, 1]; n_points = Int(floor(t_validation / tstep)) + 1)

    ########################### plot validation

    p_validation = plot(numerical[1, :], numerical[2, :], numerical[3, :], label = "numerical solution", color = 1, linewidth = 2)

    plot!(p_validation, nn₁_solution[1, :], nn₁_solution[2,:], nn₁_solution[3, :], label = "volume-preserving feedforward", color = 2, linewidth = 2)

    p_validation
end

nn₁ = NeuralNetwork(GeometricMachineLearning.DummyNNIntegrator(), nn.model, map_to_cpu(nn.params))

p_validation = make_validation_plot(t_validation, nn₁)

p_validation₂ = make_validation_plot(100, nn₁)
########################### plot training loss

p_training_loss = plot(loss_array₁, label = "volume-preserving feedforward", color = 2, linewidth = 2, yaxis = :log)


########################### plot trajectories on the sphere

p_validation3d = make_validation_plot3d(100, nn₁)

########################### save figures

png(p_validation, joinpath(@__DIR__, "simulations/vpff/validation"))
png(p_validation₂, joinpath(@__DIR__, "simulations/vpff/validation2"))
png(p_validation3d, joinpath(@__DIR__, "simulations/vpff/validation3d"))
png(p_training_loss, joinpath(@__DIR__, "simulations/vpff/training_loss"))