# Volume-preserving feedforward neural networks

As a first step to construct a *structure-preserving transformer* we replace the ResNet in the transformer with a feedforward neural network that is volume-preserving. The *volume-preseving feedforward layers* here are inspired by the linear and activation modules from [jin2020sympnets](@cite). Here our key ingredients are upper-triangular matrices ``U`` and lower-triangular matrices ``L`` for whose components we have that ``u_{ij} = 0`` if ``i \geq j`` and ``l_{ij} = 0`` if ``i \leq j``.

In matrix form the ``L`` matrices look like this: 

```math 
L = \begin{pmatrix}
     0 & 0 & \cdots & 0      \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & 0 
\end{pmatrix}.
\label{eq:LinearLower}
```

The Jacobian of a layer 

```math
x \mapsto x + \sigma(Lx + b)
\label{eq:VPFF}
``` 

then is of the form

```math 
J = \begin{pmatrix}
     1 & 0 & \cdots & 0      \\
     b_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     b_{n1} & \cdots & b_{n(n-1)}      & 1 
\end{pmatrix},
```
and the determinant of ``J`` is 1, i.e. the map is volume-preserving. 

```@raw latex
\begin{figure}
\includegraphics[width = .3\textwidth]{tikz/vp_feedforward.png}
\caption{Architecture of the volume-preserving feedforward neural network. "LinearLowerLayer" refers to $x \mapsto x + Lx$ (and similarly for "LinearUpperLayer"). "NonLinearLowerLayer" is shown in \Cref{eq:VPFF}. "Bias" is the addition with a bias vector.}
\label{fig:VolumePreservingFeedForward}
\end{figure}
```

In practice we combine many of those layers where the activation function is either (i) a fixed nonlinearity (tanh in our case) or (ii) identity. This is shown in M[fig:VolumePreservingFeedForward]m(@latex).