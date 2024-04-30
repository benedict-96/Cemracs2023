# The rigid body

In the following, we will consider the rigid body as an example to study the performance of our new volume-preserving transformer.
The differential equation for the rigid body [hairer2006geometric, arnold1978mathematical](@cite) describes the dynamics of a solid object fixed at a point. M[fig:RigidBody]m(@latex) shows an example for a rigid body. 
Its dynamics can always be described in terms of an ellipsoid. To see this, let us consider the derivation of the rigid body equations. The motion of a point ``(x_1, x_2, x_3)^T`` in the rigid body ``\mathcal{B}`` can be described as follows: 
```math
v := \frac{d}{dt} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
= \omega \times \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
= \begin{pmatrix} \omega_2x_3 - \omega_3x_2 \\ \omega_3x_1 - \omega_1x_3 \\ \omega_1x_2 - \omega_2x_1 \end{pmatrix}
= \begin{pmatrix} 0 & - \omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix}\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix},
```
where ``\omega`` is the angular velocity. 
The total kinetic energy of ``\mathcal{B}`` is obtained by integrating over the entire volume of the body:
```math
T  = \frac{1}{2} \int_\mathcal{B} ||v||^2 dm = \frac{1}{2} \int_\mathcal{B} || \omega \times x ||^2 dm = \frac{1}{2} \omega^T \Theta \omega,        
```
where 
```math
\Theta_{ij} = \begin{cases}
    \int_\mathcal{B}(x_k^2 + x_\ell^2)dm & \text{ if } i = j, \\  
    -  \int_\mathcal{B}x_ix_jdm & \text{else} ,
\end{cases}
```
``dm`` indicates an integral over the mass density of the rigid body, and ``i\neq{}k\neq\ell\neq{}i`` for the first case.

The mathematical description of a rigid body (modulo rotations) does not require knowledge of the precise shape of the body. As ``\Theta`` is a symmetric matrix, we can write the kinetic energy as:
```math
T = \frac{1}{2}\omega^T\Theta\omega = \frac{1}{2}\omega^TU^TIU\omega,
```
with $I = \mathrm{diag} (I_1, I_2, I_3)$.
This shows that it is sufficient to know the eigenvalues of the matrix ``\Theta`` which are called *moments of inertia* and denoted by ``I_k`` for ``k = 1, 2, 3``. From this point of view every rigid body is equivalent to an ellipsoid. 

```@raw latex
\begin{figure}
\includegraphics[width=.5\textwidth]{tikz/ellipsoid.png}
\caption{Any rigid body fixed at a point (left) can be described through an ellipsoid (right) through $I_1$, $I_2$ and $I_3$.}
\label{fig:RigidBody}
\end{figure}
```

## Formulation of the equations of motion in the Euler-Poincaré framework

The dynamics of the rigid body can be described through a rotational matrix ``Q(t)``, i.e. each point of the rigid body ``x(0)\in\mathcal{B}`` can be described through ``x(t) = Q(t)x(0)`` where ``Q(t)^TQ(t) = \mathbb{I}``. We can therefore describe the evolution of the rigid body through a differential equation on the Lie group ``G := \{Q\in\mathbb{R}^{d\times{}d}:Q^TQ = \mathbb{I}\}``. The associated tangent vector ``\dot{Q}\in{}T_QG`` can be mapped to the Lie algebra[^1] ``\mathfrak{g}=T_\mathbb{I}G`` by:
```math
W := \dot{Q}Q^T = \begin{pmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix} \in \mathfrak{g}. 
\label{eq:LieAlgebraRepresentation}
```
[^1]: This is the Lie algebra of skew-symmetric matrices: ``\mathfrak{g} = \{W\in\mathbb{R}^{d\times{}d}:W^T = -W\}``.

As was indicated in equation M[eq:LieAlgebraRepresentation]m(@latex), the components of the skew-symmetric matrix ``W`` are equivalent to those of the angular velocity ``\omega`` as can easily be checked: 
```math
\dot{x} (t)
= \frac{d}{dt}Q(t)x(0) = \dot{Q}(t)x(0) = \dot{Q}(t)Q^T(t)x(t) = W(t)x(t)
% = Wx
% = \begin{pmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}
% = \\ \begin{pmatrix}  -\omega_3x_2 + \omega_2x_3 \\ \omega_3x_1 - \omega_1x_3 \\ -\omega_2x_1 + \omega_1x_2 \end{pmatrix}
= \omega (t) \times x (t) .
```

With this description, the kinetic energy can be written as: 
```math
T = \frac{1}{2}\mathrm{tr}(WDW^T),
\label{eq:KineticEnergyForLieGroup}
```
where ``D`` is a diagonal matrix[^2] that satisfies ``I_1 = d_2 + d_3``, ``I_2 = d_3 + d_1`` and ``I_3 = d_1 + d_2``. We now write ``z := I^{-1}\omega`` and introduce the following notation: 
```math
\hat{z} = \begin{bmatrix} z_1 \\ z_2 \\ z_3 \end{bmatrix}^{\widehat{\,}} = \begin{bmatrix} \frac{\omega_1}{I_1} \\ \frac{\omega_2}{I_2} \\ \frac{\omega_3}{I_3} \end{bmatrix}^{\widehat{\,}} = \begin{pmatrix} 0 & -\frac{\omega_3}{I_3} & \frac{\omega_2}{I_2} \\ \frac{\omega_3}{I_3} & 0 & -\frac{\omega_1}{I_1} \\ -\frac{\omega_2}{I_2} & \frac{\omega_1}{I_1} & 0 \end{pmatrix} .
```

[^2]: This matrix is equivalent to the diagonal entries of the *coefficient of inertia matrix* in [holm2009geometric](@cite).

and obtain via the Euler-Poincaré equations[^3] for M[eq:KineticEnergyForLieGroup]m(@latex): 
```math
\widehat{I\dot{\omega}} = [\widehat{I\omega}, W],
```

[^3]: For the Euler-Poincaré equations we have to compute variations of M[eq:KineticEnergyForLieGroup]m(@latex) with respect to ``\delta{}W = \delta(\dot{Q}Q^{-1} = \dot{\Sigma} - [W,\Sigma]`` where ``\Sigma := \delta{}QQ^{-1}``. For more details on this see [holm2009geometric](@cite).

or equivalently:
```math
\frac{d}{dt}\begin{bmatrix} z_1 \\  z_2 \\ z_3  \end{bmatrix} 
= \begin{bmatrix} Az_2z_3 \\ Bz_1z_3 \\ Cz_1z_2 \end{bmatrix}.
\label{eq:FinalRigidBodyEquations}
```

In the above equation, we defined ``A := I_3^{-1} - I_2^{-1}``, ``B := I_1^{-1} - I_3^{-1}`` and ``C := I_2^{-1} - I_1^{-1}``. We further set ``I_1 = 1``, ``I_2 = 2`` and ``I_3 = 2/3`` throughout the remainder of this paper, thus yielding ``A = 1``, ``B = -1/2`` and ``C = -1/2``. In M[fig:RigidBodyCurves]m(@latex), we show some trajectories. We immediately see that the vector field M[eq:FinalRigidBodyEquations]m(@latex) is trivially divergence-free.


```@eval 
using GeometricProblems.RigidBody: odeensemble, tspan, tstep, default_parameters
using GeometricIntegrators: integrate, ImplicitMidpoint
using GeometricEquations: EnsembleProblem
using GeometricSolutions: GeometricSolution
using LaTeXStrings
using Plots; pyplot()

ics = [
        [sin(1.1), 0., cos(1.1)],
        [sin(2.1), 0., cos(2.1)],
        [sin(2.2), 0., cos(2.2)],
        [0., sin(1.1), cos(1.1)],
        [0., sin(1.5), cos(1.5)], 
        [0., sin(1.6), cos(1.6)]
]

ensemble_problem = odeensemble(ics; tspan = tspan, tstep = tstep, parameters = default_parameters)
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

p = surface(sphere(1., [0., 0., 0.]), alpha = .2, colorbar = false, dpi = 400, xlabel = L"z_1", ylabel = L"z_2", zlabel = L"z_3", xlims = (-1, 1), ylims = (-1, 1), zlims = (-1, 1), aspect_ratio = :equal)

for (i, solution) in zip(1:length(ensemble_solution), ensemble_solution)
    plot_geometric_solution!(p, solution; color = i, label = "trajectory "*string(i))
end

savefig(p, "rigid_body.png")

nothing
```

```@raw latex
\begin{figure}
\includegraphics[width=.5\textwidth]{rigid_body.png}
\caption{Trajectories for $I_1 = 1$, $I_2 = 2$ and $I_3 = 2/3$ and various initial conditions.}
\label{fig:RigidBodyCurves}
\end{figure}
```