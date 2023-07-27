using Pkg
# cd("../Cemracs2023")
cd("/Users/zeyuan/Documents/GitHub/GeometricMachineLearning.jl")
Pkg.activate(".")


using GeometricIntegrators
using Plots
using Zygote
using JLD2

# define Hamiltonian
H(x) = x[2]^2 / 2 + (1-cos(x[1]))
H(q, p) = H([q[1], p[1]])
H(t, q, p, params) = H(q, p)

# compute vector field
∇H(x) = Zygote.gradient(χ -> H(χ), x)[1]
dH(x) = [0 1;-1 0] * ∇H(x)

# vector field methods
function v(v, t, q, p, params)
    v[1] = p[1]
end
function f(f, t, q, p, params)
    f[1] = -sin(q[1])
end


# get data set (includes data & target)
function get_data_set(num=10, xymin=-1.2, xymax=+1.2)
	#range in which the data should be in
	rang = range(xymin, stop=xymax, length=num)

	# all combinations of (x,y) points
	data = [[x,y] for x in rang, y in rang]

	#compute the value of the vector field 
	target = dH.(data)

	return (data, target)
end

function pendulum_data(; tspan = (0., 20.), tstep = 0.1, q₀ = randn(1), p₀ = randn(1))
    # simulate data with geometric Integrators
    ode = HODEProblem(v, f, H, tspan, tstep, q₀, p₀)

    # sol = integrate(ode, SymplecticEulerA())
    sol = integrate(ode, ImplicitMidpoint())

    q = sol.q[:,1]
    p = sol.p[:,1]

    return (q, p)
end

p,q = pendulum_data();
plot(p[1:80],q[1:80])



data_amount = 3400
plist = []
qlist = []

i = 0
@showprogress for i in 1:data_amount
    q, p = pendulum_data()
    if minimum(q.parent) < -15 || maximum(q.parent) >15
        continue
    else
        push!(qlist,q.parent)
        push!(plist,p.parent)
    end
end

plist

data = Dict("qlist" => qlist[1:3000],"plist" => plist[1:3000])
filename="pendulum_3000PureSamples_200steps_qlist_plist_2707.jld2"
save(filename,data)

q,p = pendulum_data()
q.parent
minimum