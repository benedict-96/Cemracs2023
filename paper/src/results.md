# Experimental results

We now compare three different neural network architectures that we all train on the data coming from a rigid body (see [The rigid body](@ref)). Those architectures are ...
1. a volume-preserving feedforward neural network,
2. a volume-preserving transformer,
3. a regular transformer. 

## Loss functions 

For training the feedforward neural network and the transformer we pick similar loss functions. In both cases they take the form: 

```math 
L_{\mathcal{NN}}(input, output) = ||output - \mathcal{NN}(input)||_2/||output||_2,
```

where ``||\cdot||_2`` is the ``L_2``-norm. The only difference between the two losses (for the feedforward neural network and the transformer) is that ``input`` and ``output`` are vectors ``\in\mathbb{R}^d`` in the first case and matrices ``\in\mathbb{R}^{d\times{}T}`` in the second. 

```@raw latex
\begin{figure}
\includegraphics[width = .5\textwidth]{simulations/vpt_Float32/validation_3.png}
\caption{The first component $x$ plotted for the time interval [0, 14].}
\label{fig:Validation}
\end{figure}
```

```@raw latex
\begin{figure}
\includegraphics[width = .5\textwidth]{simulations/vpt_Float32/training_loss_3.png}
\caption{Training loss for the different networks.}
\label{fig:TrainingLoss}
\end{figure}
```

```@raw latex
\begin{figure}
\includegraphics[width = .5\textwidth]{simulations/vpt_Float32/validation3d_3.png}
\caption{Validation plot in 3d.}
\label{fig:Validation3d}
\end{figure}
```

As is shown in m[fig:Validation]m(@latex) the volume-preserving feedforward network manages to predict the time evolution of the rigid body up to a certain point, but then drifts off. The volume-preserving feedforward transformer manages to stay close to the numerical solution much better. It also outperforms the regular transformer while using fewer parameters. 

In m[fig:Validation3d]m(@latex) we show the prediction for two initial conditions, ``\begin{pmatrix} \sin(1.1) & 0 & \cos(1.1) \end{pmatrix}^T`` and ``\begin{pmatrix} 0 & \sin(1.1) & \cos(1.1) \end{pmatrix}^T``, for the time interval ``[0, 100]``. These initial conditions are also shown in m[fig:RigidBodyCurves]m(@latex) as "trajectory 1" and "trajectory 4".

## Details on training and choice of hyperparameters

The code is implemented in Julia [bezanson2017julia](@cite) as part of the library `GeometricMachineLearning.jl` [brantner2020GML](@cite). All the computations performed here are done in single precision and on an NVIDIA Geforce RTX 4090 GPU [rtx4090](@cite) and we use `CUDA.jl` [besard2018juliagpu](@cite) to perform computations on the GPU.

We train the three networks for ``5\cdot10^6`` epochs and use an Adam optimizer [kingma2014adam](@cite) with adaptive learning rate ``\eta``: 

```math
\eta = \exp\left(log\left(\frac{\eta_2}{\eta_1}\right) / \mathtt{n\_epochs}\right)^t\eta_1,
```

where ``\eta_1`` is the initial learning rate and ``\eta_2`` is the final learning rate. We end up with the following choice of hyperparameters (mostly taken from [goodfellow2016deep](@cite)):

| name  |``\eta_1`` |``\eta_2`` |``\rho_1`` |``\rho_2`` |``\delta`` |`n_epochs`     |
| ----- |:--------- |:--------- |:--------- |:--------- |:--------- |:------------- |
| value |``10^{-2}``|``10^{-6}``|``0.9``    |``0.99``   |``10^{-8}``| ``5\cdot10^6``|


With these settings we get the following training times for the different networks[^1]: 

| network type  |   VPFF  |   VPT   |   T     |
| ------------- | :------ | :------ | :------ |
| training time | 4:02:09 | 5:58:57 | 3:58:06 |

[^1]: Times given as HOURS:MINUTES:SECONDS.

## Why does regular attention fail? 

We can see in m[fig:Validation]m(@latex) that the regular transformer looks like a step function, i.e. it predicts a step and then stays there for another two steps before going to a different value again. 

To see how this can happen we look at the input of the attention layer: 

```math
\left[\begin{matrix}
(z^{(1)})^TAz^{(1)} & (z^{(1)})^TAz^{(2)} & (z^{(1)})^TAz^{(3)} \\ 
(z^{(2)})^TAz^{(1)} & (z^{(2)})^TAz^{(2)} & (z^{(2)})^TAz^{(3)} \\ 
(z^{(3)})^TAz^{(1)} & (z^{(3)})^TAz^{(2)} & (z^{(3)})^TAz^{(3)}
\end{matrix}\right] =: \left[\begin{matrix} p^{(1)} & p^{(2)} & p^{(3)} \end{matrix}\right].
\label{eq:ScalarProductResult}
```

The output of the attention layers then is: 

```math
\left[\begin{matrix} \mathrm{softmax}(p^{(1)}) & \mathrm{softmax}(p^{(2)}) & \mathrm{softmax}(p^{(3)}) \end{matrix}\right].
```

The results in figure m[fig:Validation]m(@latex) seem as if ``\mathrm{softmax}(p^{(i)})`` is the same regardless of the integer ``i = 1, 2, 3``. For the volume-preserving attention mechanism introduced in this work this can never happen as the three columns are not treated independently: the result is always a matrix with three independent columns that has determinant 1 or -1[^2]. 

[^2]: By investigating the weight matrix of the attention layer further, we see that the output of m[eq:ScalarProductResult]m(@latex) is a matrix whose entries are all roughly the same. 

__Put a few images to proof this!__

The minimum is achieved when all scalar products map to the same value. Maybe this is because there are not enough degrees of freedom to make more complicated mappings. 


## A note on parameter-dependent equations

The training data for the example presented here was an ODE for which we generated training data by varying the initial condition of the system, i.e. the data were:

```math
\{\varphi^t(z^0_\alpha):\}_{t\in(t_0, t_f], z^0_\alpha\in\mathtt{ics}},
```
where ``\varphi^t`` is the flow of the differential equation ``\dot{z} = f(z)`` (the rigid body from m[eq:RigidBody]m(@latex) in our example), ``t_0`` is the initial time, ``t_f`` the final time and `ics` denotes a set of initial conditions. 

For applications such as *reduced order modeling* (see [lee2020model, lassila2014model, fresca2021comprehensive](@cite)) we usually deal with *parametric differential equations* that are of the form: 

```math
\dot{z} = f(z; \mu) \text{ for $\mu\in\mathbb{P}$},
```

where ``\mathbb{P}`` is a set of parameters on which the differential equation can depend. In the example of the rigid body these parameters could be the moments of inertia ``I_1``, ``I_2`` and ``I_3``. A normal feedforward neural network is unable to learn such a parameter-dependent system as it *only sees one point at a time*: 

```math
\mathcal{NN}_\mathrm{ff}: \mathbb{R}^d\to\mathbb{R}^d.
```

But a feedforward neural network can only approximate the flow of a differential equation with fixed parameters as the prediction becomes ambiguous in the case when we have data coming from solutions for different parameters. In this case a transformer neural network[^3] is needed as it is able to *consider the history of the trajectory up to that point*. 

[^3]: It should be noted that recurrent neural networks such as LSTMs [hochreiter1997long](@cite) are also able to do this. 