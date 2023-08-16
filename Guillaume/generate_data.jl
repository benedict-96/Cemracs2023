using GeometricIntegrators, KernelAbstractions

integration_time_step = 0.4

# here the second point mass is altered
params_collection = (  (m1=2, m2=1., k1=1.5, k2=0.3, k=2),
)

for k0 in LinRange(1, 4, 5) 
    global params_collection = (params_collection...,  (m1=2, m2=1., k1=1.5, k2=0.3, k=k0))
end


initial_conditions_collection = ((q=[1.,0.], p=[2.,0.]),
)
function extend_tuple(tuple, p0)
    (tuple...,  (q = [1.0, 0.0], p = [p0, 0.0]))
end

# for p0 in LinRange(0.1, 2, 8) 
#     global initial_conditions_collection = extend_tuple(initial_conditions_collection, p0)
# end




t_integration = 1000

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

sols = []
for params in params_collection
    for initial_conditions in initial_conditions_collection
        pode = PODEProblem(q̇, ṗ1, (0.0, t_integration), .1, initial_conditions; parameters = params)
        sol = integrate(pode,ImplicitMidpoint())
        push!(sols, sol)
    end
end

time_steps = length(sols[1].q)
data_tensor = zeros(4, length(sols), time_steps)
@kernel function create_tensor_kernel_q!(data_tensor, sols)
    i,j,k = @index(Global, NTuple)
    data_tensor[i,j,k] = sols[j].q[k-1][i]
end
@kernel function create_tensor_kernel_p!(data_tensor, sols)
    i,j,k = @index(Global, NTuple)
    data_tensor[i+2,j,k] = sols[j].p[k-1][i]
end
function assign_tensor(data_tensor, sols)
    assign_q! = create_tensor_kernel_q!(CPU())
    assign_p! = create_tensor_kernel_p!(CPU())
    dims = (2, size(data_tensor,2), size(data_tensor,3))
    assign_q!(data_tensor, sols, ndrange=dims)
    assign_p!(data_tensor, sols, ndrange=dims)
end

function generate_data()
    sols = []
    for params in params_collection
        for initial_conditions in initial_conditions_collection
            pode = PODEProblem(q̇, ṗ1, (0.0, t_integration), integration_time_step , initial_conditions; parameters = params)
            sol = integrate(pode,ImplicitMidpoint())
            push!(sols, sol)
        end
    end

    time_steps = length(sols[1].q)
    data_tensor = zeros(4, length(sols), time_steps)
    assign_tensor(data_tensor, sols)
    data_tensor 
end