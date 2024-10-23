using GeometricMachineLearning
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricProblems.RigidBody: odeproblem, odeensemble, default_parameters
using JLD2
using CairoMakie

morange = RGBf(255 / 256, 127 / 256, 14 / 256) # hide
mred = RGBf(214 / 256, 39 / 256, 40 / 256) # hide
mpurple = RGBf(148 / 256, 103 / 256, 189 / 256) # hide
mblue = RGBf(31 / 256, 119 / 256, 180 / 256) # hide
mgreen = RGBf(44 / 256, 160 / 256, 44 / 256)

# hyperparameters concerning architecture 
const sys_dim = 3
const seq_length = 3
const n_heads = 1
const L = 3 # transformer blocks 
const activation = tanh
const n_linear = 1
const n_blocks = 2
const skew_sym = false
const resnet_activation = tanh

const _tstep = .2

# backend 
const backend = CPU()

const t_validation = 50
const t_validation_long = 100
const t_validation_very_long = 50000

# data type 
const T = Float32

model_vpff = VolumePreservingFeedForward(sys_dim, n_blocks * L, n_linear, resnet_activation)

model_vpt = VolumePreservingTransformer(sys_dim, seq_length; n_blocks = n_blocks, n_linear = n_linear, L = L, activation = resnet_activation, skew_sym = skew_sym)

model_st = StandardTransformerIntegrator(sys_dim; n_heads = n_heads, n_blocks = n_blocks, L = L, resnet_activation = resnet_activation, add_connection = false)

# get correct parameters from jld2 file # hide
f = load("transformer_rigid_body.jld2")  # hide
f2 = load("transformer_rigid_body_short_training.jld2")
nn_vpff = NeuralNetwork(model_vpff, Chain(model_vpff), f["nn2_params"], backend) # hide
nn_vpt = NeuralNetwork(model_vpt, Chain(model_vpt), f["nn3_params"], backend) # hide
nn_st = NeuralNetwork(model_st, Chain(model_st), f2["nn4_params"], backend) # hide

ics_val = [sin(1.1), 0., cos(1.1)]

function numerical_solution(sys_dim::Int, t_integration::Int, tstep::Real, ics_val::Vector)
    validation_problem = odeproblem(ics_val; tspan = (0.0, t_integration), tstep = tstep, parameters = default_parameters)
    sol = integrate(validation_problem, ImplicitMidpoint())

    numerical_solution = zeros(sys_dim, length(sol.t))
    for i in axes(sol.t, 1) numerical_solution[:, i+1] = sol.q[i] end 

    t_array = zeros(length(sol.t))
    for i in axes(sol.t, 1) t_array[i+1] = sol.t[i] end

    T.(numerical_solution), T.(t_array) 
end

function compute_neural_network_prediction(nn::NeuralNetwork{<:GeometricMachineLearning.TransformerIntegrator}, numerical::AbstractMatrix, t_validation::Integer, tstep::Real)
    iterate(nn, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / tstep)) + 1, prediction_window = seq_length)
end

function compute_neural_network_prediction(nn::NeuralNetwork{<:GeometricMachineLearning.NeuralNetworkIntegrator}, numerical::AbstractMatrix, t_validation::Integer, tstep::Real)
    iterate(nn, numerical[:, 1]; n_points = Int(floor(t_validation / tstep)) + 1)
end

norm2(mat) = sqrt.(sum(mat.^2; dims = 1))
function plot_validation(t_validation; nn₂=nn_vpff, nn₃=nn_vpt, nn₄=nn_st, plot_regular_transformer = false, plot_vp_transformer = false, tstep = _tstep)

    fig = Figure()
    ax = Axis(fig[1, 1], xlabel = L"t", ylabel = L"||z||_2")

    @time "numerical solution" numerical, t_array = numerical_solution(sys_dim, t_validation, _tstep, ics_val)

    # nn₁_solution = iterate(nn₁, numerical[:, 1:seq_length]; n_points = Int(floor(t_validation / tstep)) + 1)
    @time "vpff" nn₂_solution = compute_neural_network_prediction(nn₂, numerical, t_validation, tstep)
    @time "vpt" nn₃_solution = compute_neural_network_prediction(nn₃, numerical, t_validation, tstep)
    @time "st" nn₄_solution = compute_neural_network_prediction(nn₄, numerical, t_validation, tstep)

    ########################### plot validation

    # plot!(p_validation, t_array, nn₁_solution[1, :], label = "attention only", color = 2, linewidth = 2)

    lines!(ax, t_array, norm2(numerical)[1, :]; label = "implicit midpoint", color = mblue, linewidth = 2)

    lines!(ax, t_array, norm2(nn₂_solution)[1, :]; label = "volume-preserving feedforward", color = mgreen, linewidth = 2)

    if plot_vp_transformer
        lines!(ax, t_array, norm2(nn₃_solution)[1, :]; label = "volume-preserving transformer", color = mpurple, linewidth = 2)
    end

    if plot_regular_transformer
        lines!(ax, t_array, norm2(nn₄_solution)[1, :]; label = "standard transformer", color = morange, linewidth = 2)
    end

    axislegend(; position = (.82, .75)) # hide
    fig, ax
end

fig_validation, ax = plot_validation(t_validation; plot_regular_transformer = true, plot_vp_transformer = true)
fig_validation_long, ax = plot_validation(t_validation_long; plot_regular_transformer = true, plot_vp_transformer = true)
fig_validation_very_long, ax = plot_validation(t_validation_very_long; plot_regular_transformer = true, plot_vp_transformer = true)
fig_validation, ax = plot_validation(t_validation; plot_regular_transformer = true, plot_vp_transformer = true)
fig_validation_long, ax = plot_validation(t_validation_long; plot_regular_transformer = true, plot_vp_transformer = true)
fig_validation_very_long, ax = plot_validation(t_validation_very_long; plot_regular_transformer = true, plot_vp_transformer = true)

save("violate_invariant.png", fig_validation)
save("violate_invariant_long.png", fig_validation_long)
save("violate_invariant_very_long.png", fig_validation_very_long)
