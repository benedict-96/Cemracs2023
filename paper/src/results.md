# Experimental results

In the following, we will consider the rigid body as an example to study the performance of our new volume-preserving transformer.
We will solve the following equations (see [The Rigid Body](@ref) for the derivation):
```math
\frac{d}{dt}\begin{bmatrix} z_1 \\  z_2 \\ z_3  \end{bmatrix} 
= \begin{bmatrix} \mathfrak{a}z_2z_3 \\ \mathfrak{b}z_1z_3 \\ \mathfrak{c}z_1z_2 \end{bmatrix} ,
\label{eq:RigidBodyEquations}
```
with ``\mathfrak{a} = 1``, ``\mathfrak{b} = -1/2`` and ``\mathfrak{c} = -1/2``.
We immediately see that the vector field in M[eq:RigidBodyEquations]m(@latex) is trivially divergence-free.
In M[fig:RigidBodyCurves]m(@latex) we show some trajectories.


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
\begin{figure}[h]
\includegraphics[width=.75\textwidth]{rigid_body.png}
\caption{Rigid body trajectories for $\mathfrak{a} = 1$, $\mathfrak{b} = -1/2$ and $\mathfrak{c} = -1/2$ and different initial conditions.}
\label{fig:RigidBodyCurves}
\end{figure}
```

We will compare three different neural network architectures that are trained on simulation data of the rigid body. 
These architectures are:

| Architecture                    | `n_linear` | `n_blocks` | L | Total number of parameters |
|:------------------------------  |:---------: |:---------: |:-:|:-------------------------: |
| Volume-preserving feedforward  | 1          | 6          | - | 135                        |
| Volume-preserving transformer   | 1          | 2          | 3 | 162                        |
| Standard transformer            | -          | 2          | 3 | 213                        |


REMARK::
Using transformer neural networks instead of standard feedforward neural networks for *integrating* ordinary differential equations can be motivated similarly to using multi-step methods as opposed to single-step methods in traditional numerical integration (another motivation comes from the possibility to consider parameter-dependent equations as discussed below). In [cellier2006continuous](@cite) it is stated that multi-step methods constitute "[an entire class] of integration algorithms of arbitrary order of approximation accuracy that require only a single function evaluation in every new step". We conjecture that this also holds true for transformer-based integrators: we can hope to build higher-order methods without increased cost.
::


For the standard transformer, we further remove the optional add connection (i.e. the green line in M[fig:TransformerArchitecture]m(@latex)) to have a better comparison with the volume-preserving transformer which does not have an add connection. For the standard transformer, `n_blocks` refers to the number of ResNet layers we use (the last ResNet layer always has a linear activation). The activation functions in the *feedforward layer* (see M[fig:TransformerArchitecture]m(@latex)) and volume-preserving feedfoward layers (the *non-linear layers* in M[fig:VolumePreservingFeedForward]m(@latex) and the *volume-preserving feedforward layers* in M[fig:VolumePreservingTransformerArchitecture]m(@latex)) are all tanh. For the standard transformer and the volume-preserving transformer we further pick ``T = 3``, i.e. we always feed three time steps into the network during training and validation. We also note that strictly speaking ``T`` is not a hyperparameter of the network as its choice does not change the architecture: the dimensions of the matrix ``A`` in the volume-preserving activation in M[eq:VolumePreservingActivation]m(@latex) (or the equivalent for the standard attention mechanism) are independent of the number of time steps ``T`` that we feed into the network.

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

All solutions lie on a sphere of radius one. That they do is a special property of the rigid body (see [The Rigid Body](@ref)) equation and is proofed in [hairer2006geometric; Theorem IV.1.6](@cite) for example.

## Loss functions 

For training the feedforward neural network and the transformer we pick similar loss functions. In both cases they take the form: 
```math 
L_{\mathcal{NN}}(input, output) = \frac{ ||output - \mathcal{NN}(input)||_2 }{ ||output||_2 },
```
where ``||\cdot||_2`` is the ``L_2``-norm. The only difference between the two losses for the feedforward neural network and the transformer is that ``input`` and ``output`` are vectors ``\in\mathbb{R}^d`` in the first case and matrices ``\in\mathbb{R}^{d\times{}T}`` in the second. 

## Details on Training and Choice of Hyperparameters

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

The time evolution of the different training losses is shown in M[fig:TrainingLoss]m(@latex). The training losses for the volume-preserving transformer and the volume-preserving feedforward neural network reach very low levels (about ``5 \times 10^{-4}``), whereas the standard transformer is stuck at a rather high level (``5 \times 10^{-2}``). In addition to the Adam optimizer we also tried stochastic gradient descent (with and without momentum) and the BFGS optimizer [nocedal1999numerical](@cite), but obtained the best results with the Adam optimizer. We also see that training the VPT takes longer than the ST even though it has fewer parameters. This is probably because the softmax activation function requires fewer floating-point operations than the inverse in the Cayley transform. We hence observe that even though `GeometricMachineLearning.jl` has an efficient explicit matrix inverse implemented, this is still slower than computing the exponential in the softmax.

## Using the Networks for Prediction

The corresponding predicted trajectories are shown in M[fig:Validation3d]m(@latex), where we plot the predicted trajectories for two initial conditions, ``( \sin(1.1), \, 0, \, \cos(1.1) )^T`` and ``( 0, \, \sin(1.1) , \, \cos(1.1) )^T``, for the time interval ``[0, 100]``. These initial conditions are also shown in M[fig:RigidBodyCurves]m(@latex) as "trajectory 1" and "trajectory 4". 

REMARK::
Note that for the two transformers we need to supply three vectors as input as opposed to one for the feedforward neural network. We therefore compute the first two steps with implicit midpoint and then proceed by using the transformer. We could however also use the volume-preserving feedforward neural network to predict those first two steps instead of using implicit midpoint. This is necessary if the differential equation is not known.::

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
\caption{Sample trajectories of the rigid body obtained with the three neural networks: volume-preserving feedforward, volume-preserving transformer and the standard transformer, together with the numerical solution (implicit midpoint) for "trajectory 1" and "trajectory 4" in \Cref{fig:RigidBodyCurves}. The volume-preserving feedforward neural network is provided with the initial condition (i.e. $z^{(0)}$) and then starts the prediction and the two transformers are provided with the first three time steps ($z^{(1)}$ and $z^{(2)}$ are obtained via implicit midpoint) and then start the prediction. The prediction is made for the time interval $[0, 100]$, i.e. 500 time steps in total.}
\label{fig:Validation3d}
\end{figure}
```

The standard transformer clearly fails on this task while the volume-preserving feedforward network slowly drifts off. The volume-preserving transformer shows much smaller errors and manages to stay close to the numerical solution.  M[fig:VPFFvsVPT]m(@latex) shows the time evolution of the invariant ``I(z) = ||z||_2`` for implicit midpoint and the three neural network integrators.

```@raw latex
\begin{figure}
\centering
\includegraphics[width = .7\textwidth]{violate_invariant_long.png}
\caption{Time evolution of invariant $I(z) = \sqrt{z_1^2 + z_2^2 + z_3^2} = ||z||_2$ for ``trajectory 1" up for the time interval $[0, 100]$. We see that for implicit midpoint this invariant is conserved and for the volume-preserving transformer it oscillates around the correct value.}
\label{fig:VPFFvsVPT}
\end{figure}
```

In order to get an estimate for the different computational times we perform integration up to time 50000 for all four methods. On CPU we get:

| Method        | IM            |    VPFF      |   VPT          |   ST                  |
| ------------- | :-------      | :-------     | :------        | :------               |
| Evaluation time | 2.51 seconds  | 6.44 seconds | 0.71 seconds   | 0.20 seconds          |

We see that the standard transformer is the fastest, followed by the volume-preserving transformer. The slowest is the volume-preserving feedforward neural network. We attempt to explain these findings:
- we assume that the standard transformer is faster than the volume-preserving transformer because the softmax can be quicker evaluated than our new activation function M[eq:VolumePreservingActivation]m(@latex),
- we assume that implicit midpoint is slower than the two transformers because it involves a Newton solver and the neural networks all perform explicit operations,
- the very poor performance of the volume-preserving feedforward neural network is harder to explain. We suspect that our implementation performs all computations in serial and is therefore slower than the volume-preserving transformer by a factor of three, because we have ``\mathrm{L} = 3`` transformer units. It can furthermore be assumed to be slower by another factor of three because the feedforward neural network only predicts one time step at a time as opposed to three time steps at a time for the two transformer neural networks.

Another advantage of all neural network-based integrators over implicit midpoint is that it is easily suitable for parallel computation on GPU because all operations are explicit[^2]. The biggest motivation for using neural networks to learn dynamics comes however from non-intrusive reduced order modeling as discussed in the introduction; a traditional integrator like implicit midpoint is simply not suitable for this task as we need to recover dynamics from data.

[^2]: The implementation of these architectures in `GeometricMachineLearning.jl` supports parallel computation on GPU.

## Why Does Regular Attention Fail? 

M[fig:Validation3d]m(@latex) shows, that the standard transformer fails to predict the time evolution of the system correctly. The reason behind this could be that it is not sufficiently restrictive, i.e., the matrix which is made up of the three columns in the output of the transformer (see M[eq:StandardTransformerOutput]m(@latex)) does not have full rank (i.e. is not invertible); a property that the volume-preserving transformer has by construction. We observe that the "trajectory 1" and "trajectory 4" seem to merge at some point, as if there were some kind of attractor in the system. This is not a property of the physical system and seems to be mitigated if we use volume-preserving architectures. 


## A Note on Parameter-Dependent Equations

In the example presented here, training data was generated by varying the initial condition of the system, specifically
```math
\{\varphi^t(z^0_\alpha): {t\in(t_0, t_f], z^0_\alpha\in\mathtt{ics}} \},
```
where ``\varphi^t`` is the flow of the differential equation ``\dot{z} = f(z)``, in particular the rigid body from M[eq:RigidBodyEquations]m(@latex), ``t_0`` is the initial time, ``t_f`` the final time, and `ics` denotes the set of initial conditions. 

In applications such as *reduced order modeling* [lee2020model, lassila2014model, fresca2021comprehensive](@cite), one is often concerned with *parametric differential equations* of the form: 
```math
\dot{z} = f(z; \mu) \text{ for $\mu\in\mathbb{P}$},
```
where ``\mathbb{P}`` is a set of parameters on which the differential equation depends. In the example of the rigid body, these parameters could be the moments of inertia ``(I_1, I_2, I_3)`` and thus equivalent to the parameters ``(\mathfrak{a},\mathfrak{b},\mathfrak{c})`` in M[eq:RigidBodyEquations]m(@latex). A normal feedforward neural network is unable to learn such a parameter-dependent system as it *only sees one point at a time*: 
```math
\mathcal{NN}_\mathrm{ff}: \mathbb{R}^d\to\mathbb{R}^d.
```

Thus a feedforward neural network can only approximate the flow of a differential equation with fixed parameters as the prediction becomes ambiguous in the case of data coming from solutions for different parameters. A transformer neural network[^3] on the other hand, is able to describe solutions with different parameters of the system, as it is able to *consider the history of the trajectory up to that point*. 

[^3]: It should be noted that recurrent neural networks such as LSTMs [hochreiter1997long](@cite) are also able to do this. 

We also note that one could build parameter-dependent feedforward neural networks as

```math
\mathcal{NN}_{ff}:\mathbb{R}^d\times\mathbb{P} \to \mathbb{R}^d.
```

A simple single-layer feedforward neural network

```math
    \overline{\mathcal{NN}}_{ff}: x \mapsto \sigma(Ax + b),
```

could be made parameter-dependent by modifying it to

```math
    \mathcal{NN}_{ff}: (x, \mu) \mapsto \sigma\Big(A\begin{bmatrix}x \\ \mu \end{bmatrix} + b\Big),
```

for example. This makes it however harder to imbue the neural network with structure-preserving properties and we do not pursue this approach further in this work.