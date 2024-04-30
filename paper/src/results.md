# Experimental results

In the following, we will consider the rigid body as an example to study the performance of our new volume-preserving transformer.
We will solve the following equations (see [The rigid body](@ref) for the derivation):
```math
\frac{d}{dt}\begin{bmatrix} z_1 \\  z_2 \\ z_3  \end{bmatrix} 
= \begin{bmatrix} Az_2z_3 \\ Bz_1z_3 \\ Cz_1z_2 \end{bmatrix} ,
\label{eq:RigidBodyEquations}
```
with ``A = 1``, ``B = -1/2`` and ``C = -1/2``.
We immediately see that the vector field M[eq:RigidBodyEquations]m(@latex) is trivially divergence-free.
In M[fig:RigidBodyCurves]m(@latex), we show some trajectories.


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
\includegraphics[width=.75\textwidth]{rigid_body.png}
\caption{Rigid body trajectories for $A = 1$, $B = -1/2$ and $C = -1/2$ and different initial conditions.}
\label{fig:RigidBodyCurves}
\end{figure}
```


We will compare three different neural network architectures that are trained on simulation data of the rigid body. 
These architectures are:

| Architecture    | Description                     | `n_linear` | `n_blocks` | L | Total number of parameters |
| :-------------  |:------------------------------  |:---------: |:---------: |:-:|:-------------------------: |
| VPFF            | Volume-preserving feed-forward  | 1          | 6          | - | 135                        |
| VPT             | Volume-preserving transformer   | 1          | 2          | 3 | 162                        |
| ST              | Standard transformer            | -          | 2          | 3 | 213                        |


For the standard transformer, we further remove the optional add connection (i.e. the green line in M[fig:TransformerArchitecture]m(@latex)) to have a better comparison with the volume-preserving transformer which does not have an add connection. For the standard transformer, `n_blocks` refers to the number of ResNet layers we use (the last ResNet layer always has a linear activation).

## Training data

As training data we take solutions of M[eq:RigidBodyEquations]m(@latex) for various initial conditions: 
```math
\mathtt{ics} = \left\{ \begin{pmatrix} \sin(v) \\ 0 \\ \cos(v) \end{pmatrix}, \begin{pmatrix} 0 \\ \sin(v) \\ \cos(v) \end{pmatrix}: v\in0.1:0.01:2\pi \right\},
\label{eq:Ics}
```

where ``v\in0.1:0.01:2\pi`` means that we incrementally increase ``v`` from 0.1 to ``2\pi`` by steps of size 0.01. We then integrate M[eq:RigidBodyEquations]m(@latex) for the various initial conditions in M[eq:Ics]m(@latex) with implicit midpoint for the interval ``[0,12]`` and a step size of ``0.2``. The integration is done with `GeometricIntegrators.jl` [Kraus:2020:GeometricIntegrators](@cite). In M[fig:RigidBodyCurves]m(@latex), we show some of the curves for the following initial conditions: 

```math
\left\{
\begin{pmatrix} \sin(1.1) \\  0.       \\  \cos(1.1)\end{pmatrix},
\begin{pmatrix} \sin(2.1) \\  0.       \\  \cos(2.1)\end{pmatrix},
\begin{pmatrix} \sin(2.2) \\  0.       \\  \cos(2.2)\end{pmatrix},
\begin{pmatrix}  0.       \\ \sin(1.1) \\  \cos(1.1)\end{pmatrix},
\begin{pmatrix}  0.       \\ \sin(1.5) \\  \cos(1.5)\end{pmatrix}, 
\begin{pmatrix}  0.       \\ \sin(1.6) \\  \cos(1.6)\end{pmatrix}
\right\}.
```

All solutions lie on a sphere of radius one.

## Loss functions 

For training the feedforward neural network and the transformer we pick similar loss functions. In both cases they take the form: 
```math 
L_{\mathcal{NN}}(input, output) = \frac{ ||output - \mathcal{NN}(input)||_2 }{ ||output||_2 },
```
where ``||\cdot||_2`` is the ``L_2``-norm. The only difference between the two losses for the feedforward neural network and the transformer is that ``input`` and ``output`` are vectors ``\in\mathbb{R}^d`` in the first case and matrices ``\in\mathbb{R}^{d\times{}T}`` in the second. 

## Details on training and choice of hyperparameters

The code is implemented in Julia [bezanson2017julia](@cite) as part of the library `GeometricMachineLearning.jl` [brantner2020GML](@cite). All the computations performed here are done in single precision on an NVIDIA Geforce RTX 4090 GPU [rtx4090](@cite). We use `CUDA.jl` [besard2018juliagpu](@cite) to perform computations on the GPU.

We train the three networks for ``5\cdot10^5`` epochs and use an Adam optimizer [kingma2014adam](@cite) with adaptive learning rate ``\eta``: 
```math
\eta = \exp\left(log\left(\frac{\eta_2}{\eta_1}\right) / \mathtt{n\_epochs}\right)^t\eta_1,
```
where ``\eta_1`` is the initial learning rate and ``\eta_2`` is the final learning rate. We used the following values for the hyperparameters (mostly taken from [goodfellow2016deep](@cite)):

| Name  |``\eta_1`` |``\eta_2`` |``\rho_1`` |``\rho_2`` |``\delta`` |`n_epochs`     |
| ----- |:--------- |:--------- |:--------- |:--------- |:--------- |:------------- |
| Value |``10^{-2}``|``10^{-6}``|``0.9``    |``0.99``   |``10^{-8}``| ``5\cdot10^5``|

With these settings we get the following training times (given as HOURS:MINUTES:SECONDS) for the different networks: 

| Architecture  |   VPFF  |   VPT   |   ST      |
| ------------- | :------ | :------ | :------   |
| Training time | 4:02:09 | 5:58:57 | 3:58:06   |

The time evolution of the different training losses is shown in M[fig:TrainingLoss]m(@latex). The training losses for the volume-preserving transformer and the volume-preserving feedforward neural network reach very low levels (about ``5 \times 10^{-4}``), whereas the standard transformer is stuck at a rather high level (``5 \times 10^{-2}``). The consequences are shown in M[fig:Validation3d]m(@latex), where we plot the predicted trajectories for two initial conditions, ``( \sin(1.1), \, 0, \, \cos(1.1) )^T`` and ``( 0, \, \sin(1.1) , \, \cos(1.1) )^T``, for the time interval ``[0, 100]``. These initial conditions are also shown in M[fig:RigidBodyCurves]m(@latex) as "trajectory 1" and "trajectory 4".

```@raw latex
\begin{figure}
\includegraphics[width = .6\textwidth]{simulations/vpt_Float32/training_loss_3.png}
\caption{Training loss for the different networks.}
\label{fig:TrainingLoss}
\end{figure}
```

```@raw latex
\begin{figure}
\includegraphics[width = .33\textwidth]{simulations/vpt_Float32/feedforward_validation3d_3.png}%
\includegraphics[width = .33\textwidth]{simulations/vpt_Float32/validation3d_3.png}%
\includegraphics[width = .33\textwidth]{simulations/vpt_Float32/standard_transformer_validation3d_3.png}
\caption{Sample trajectories of the rigid body obtained with the three neural networks: volume-preserving feedforward, volume-preserving transformer and the standard transformer, together with the numerical solution for "trajectory 1" and "trajectory 4" in \Cref{fig:RigidBodyCurves}. The volume-preserving feedforward neural network is provided with the initial condition (i.e. $z^{(0)}$) and then starts the prediction and the two transformers are provided with the first three time steps ($z^{(1)}$ and $z^{(2)}$ are obtained via implicit midpoint) and then start the prediction. The prediction is made for the time interval $[0, 100]$, i.e. 500 time steps in total.}
\label{fig:Validation3d}
\end{figure}
```

The standard transformer clearly fails on this task while the volume-preserving feedforward network slowly drifts of. The volume-preserving transformer shows much smaller errors and manages to stay close to the numerical solution. 
M[fig:VPFFvsVPT]m(@latex) shows the time evolution of the relative error (compared to the solution with implicit midpoint) of the two volume-preserving networks.

```@raw latex
\begin{figure}
\includegraphics[width = .7\textwidth]{simulations/vpt_Float32/error_evolution_3.png}
\caption{Comparison between the time evolution of the relative error of the volume-preserving feedforward neural network and the volume-preserving transformer.}
\label{fig:VPFFvsVPT}
\end{figure}
```


## Why does regular attention fail? 

M[fig:Validation3d]m(@latex) shows, that the standard transformer fails to predict the time evolution of the system correctly. The reason behind this could be that it is not sufficiently restrictive, i.e., the three columns making the output of the transformer (see M[eq:StandardTransformerOutput]m(@latex)) are not necessarily linearly independent; a property that the volume-preserving transformer has by construction. We observe that the "trajectory 1" and "trajectory 4" seem to merge at some point, as if there were some kind of attractor in the system. This is not a property of the physical system and seems to be mitigated if we use volume-preserving architectures. 


## A note on parameter-dependent equations

In the example presented here, training data was generated by varying the initial condition of the system, specifically
```math
\{\varphi^t(z^0_\alpha): {t\in(t_0, t_f], z^0_\alpha\in\mathtt{ics}} \},
```
where ``\varphi^t`` is the flow of the differential equation ``\dot{z} = f(z)``, in particular the rigid body from M[eq:RigidBodyEquations]m(@latex), ``t_0`` is the initial time, ``t_f`` the final time, and `ics` denotes the set of initial conditions. 

In applications such as *reduced order modeling* [lee2020model, lassila2014model, fresca2021comprehensive](@cite), one is often concerned with *parametric differential equations* of the form: 
```math
\dot{z} = f(z; \mu) \text{ for $\mu\in\mathbb{P}$},
```
where ``\mathbb{P}`` is a set of parameters on which the differential equation depends. In the example of the rigid body, these parameters could be the moments of inertia ``(I_1, I_2, I_3)`` and thus the parameters ``(A,B,C)`` in M[eq:RigidBodyEquations]m(@latex). A normal feedforward neural network is unable to learn such a parameter-dependent system as it *only sees one point at a time*: 
```math
\mathcal{NN}_\mathrm{ff}: \mathbb{R}^d\to\mathbb{R}^d.
```

Thus a feedforward neural network can only approximate the flow of a differential equation with fixed parameters as the prediction becomes ambiguous in the case of data coming from solutions for different parameters. A transformer neural network[^2] on the other hand, is able to describe solutions with different parameters of the system, as it is able to *consider the history of the trajectory up to that point*. 

[^2]: It should be noted that recurrent neural networks such as LSTMs [hochreiter1997long](@cite) are also able to do this. 
