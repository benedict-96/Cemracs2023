# The rigid body

The differential equation for the rigid body (see [hairer2006geometric](@cite)) describes a solid object fixed at a point.

![](tikz/ellipsoid.png)

The motion of a point ``(x, y, z)^T`` in the rigid body ``\mathcal{B}`` can be described through: 

```math
v = \omega\times{}\begin{pmatrix} x \\ y \\ z \end{pmatrix} = \begin{pmatrix} \omega_2z - \omega_3y \\ \omega_3x - \omega_1z \\ \omega_1y - \omega_2x \end{pmatrix} = \begin{pmatrix} 0 & - \omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix}\begin{pmatrix} x \\ y \\ z \end{pmatrix}.
```

In order to get the entire kinetic energy of ``\mathcal{B}`` we have to integrate over the entire volume of the body:

```math
T   & = \frac{1}{2}\int_\mathcal{B}||\omega\times{}x||^2dm \\
    & = \frac{1}{2}\int_\mathcal{B}\left( (\omega_2x_3 - \omega_3x_2)^2 + (\omega_3x_1 - \omega_1x_3)^2 + (\omega_1x_2 - \omega_2x_1)^2 \right)dm = \frac{1}{2}\omega^T\Theta\omega,        
```

where 

```math
\begin{aligned}
    \Theta_{ij} = \begin{cases} \int_\mathcal{B}(x_k^2 + x_\ell^2)dm \text{ if }i = j, \\  
                             -  \int_\mathcal{B}x_ix_jdm, \end{cases}
\end{aligned}
```

with ``i\neq{}k\neq\ell\neq{}i`` for the first case.

For the mathematical description of a rigid body (modulo rotations) we do not need to know the entire shape of the body! It is enough to know the eigenvalues of the matrix ``\Theta`` which are called *moments of inertia* and denoted by ``I_k`` for ``k = 1, 2, 3``.

We can now use Newtorn's first law, that the angular momentum of the body stays constant, to reformulate these equations. The angular momentum is: 

```math 
L = \int_\mathcal{B}(x\times{}v)dm = \int_\mathcal{B}\left(x \times (\omega \times x)\right)dm.
```

This leads to ``L = \Theta\omega`` or in the distinct basis mentioned above: ``L_k = I_k\omega_k`` (for ``k=1,2,3``). 

We then obtain: 

```math
\begin{pmatrix} \dot{L}_1 \\ \dot{L}_2 \\ \dot{L}_3 \end{pmatrix} = \begin{pmatrix} (I_3^{-1} - I_2^{-1})L_3L_2 
\\ (I_1^{-1} - I_3^{-1})L_1L_3 \\ (I_2^{-1} - I_1^{-1})L_2L_1 \end{pmatrix}.
```

These are the equations we will treat. For simplicity we write: ``x := L_1``, ``y := L_2``, ``z := L_3``, ``A := I_3^{-1} - I_2^{-1}``, ``B := I_1^{-1} - I_3^{-1}`` and ``C := I_2^{-1} - I_1^{-1}``. The differential equation for ``L`` thus becomes: 

```math
\frac{d}{dt}\begin{pmatrix} x \\  y \\ z  \end{pmatrix}  = \begin{pmatrix} Ayz \\ Bxz \\ Cxy \end{pmatrix}. 
```

We further set ``I_1 = 1``, ``I_2 = 2`` and ``I_3 = 2/3``, yielding ``A = 1``, ``B = -1/2`` and ``C = -1/2``. In the following we plot some of the trajectories:

```@eval 
using GeometricProblems.RigidBody: odeproblem, tspan, tstep, default_parameters
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricEquations: EnsembleProblem
using GeometricSolutions: GeometricSolution
using Plots; pyplot()

ics = [
        (q = [sin(1.1), 0., cos(1.1)], ),
        (q = [sin(2.1), 0., cos(2.1)], ),
        (q = [sin(2.2), 0., cos(2.2)], ),
        (q = [0., sin(1.1), cos(1.1)], ),
        (q = [0., sin(1.5), cos(1.5)], ), 
        (q = [0., sin(1.6), cos(1.6)], )
]

ensemble_problem = EnsembleProblem(odeproblem().equation, tspan, tstep, ics, default_parameters)
ensemble_solution = integrate(ensemble_problem, ImplicitMidpoint())

function plot_geometric_solution!(p::Plots.Plot, solution::GeometricSolution; kwargs...)
    plot!(p, solution.q[:, 1], solution.q[:, 2], solution.q[:, 3]; kwargs...)
end

function sphere(r, C)   # r: radius; C: center [cx,cy,cz]
           n = 100
           u = range(-π, π; length = n)
           v = range(0, π; length = n)
           x = C[1] .+ r*cos.(u) * sin.(v)'
           y = C[2] .+ r*sin.(u) * sin.(v)'
           z = C[3] .+ r*ones(n) * cos.(v)'
           return x, y, z
       end

p = surface(sphere(1., [0., 0., 0.]), alpha = .2, colorbar = false)

for (i, solution) in zip(1:length(ensemble_solution), ensemble_solution)
    plot_geometric_solution!(p, solution; color = i, label = "trajectory "*string(i))
end

savefig(p, "rigid_body.png")

nothing
```

![](rigid_body.png)