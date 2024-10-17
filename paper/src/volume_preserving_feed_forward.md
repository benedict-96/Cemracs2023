# Volume-preserving feedforward neural networks

As a first step to construct a *structure-preserving transformer* we replace the ResNet in the transformer with a feedforward neural network that is volume-preserving. The *volume-preseving feedforward layers* here are inspired by the linear and activation modules from [jin2020sympnets](@cite). The key ingredients are upper-triangular matrices ``U`` and lower-triangular matrices ``L``, whose components are such that ``u_{ij} = 0`` if ``i \geq j`` and ``l_{ij} = 0`` if ``i \leq j``, respectively. Similar volume-preserving feedforward neural networks were introduced before [bajars2023locally](@cite). The difference between the volume-preserving feedforward neural networks in [bajars2023locally](@cite) and the ones presented here is that the ones in [basar2023locally](@cite) are based on ``G``-SympNets, whereas ours are based on ``LA``-SympNets [jin2020sympnets](@cite).

Let us consider a *lower-triangular layer*[^1]:

[^1]: The activation function ``\sigma`` could in principle be chosen arbitrarily, but will be tanh in our experiments.

```math
x \mapsto x + \sigma(Lx + b) ,
\label{eq:VPFF}
``` 
with the matrix ``L`` given by: 
```math 
L = \begin{pmatrix}
     0 & 0 & \cdots & 0      \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & 0 
\end{pmatrix}.
\label{eq:LinearLower}
```
The Jacobian of such a layer is of the form
```math 
J = \begin{pmatrix}
     1 & 0 & \cdots & 0      \\
     b_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     b_{n1} & \cdots & b_{n(n-1)}      & 1 
\end{pmatrix},
```
and the determinant of ``J`` is ``1``, i.e., the map is volume-preserving. 
The same reasoning applies to an "upper-triangular layer" of the form
```math
x \mapsto x + \sigma(Ux + b) .
\label{eq:VPFFU}
``` 

```@raw latex
\begin{figure}
\includegraphics[width = .31\textwidth]{tikz/vp_feedforward.png}
\caption{Architecture of the volume-preserving feedforward neural network. "LinearLowerLayer" refers to $x \mapsto x + Lx$ (and similarly for "LinearUpperLayer"). "NonLinearLowerLayer" is shown in \Cref{eq:VPFF}. "Bias" is the addition with a bias vector.}
\label{fig:VolumePreservingFeedForward}
\end{figure}
```

In practice we combine many of those layers where the activation function is either (i) a fixed nonlinearity (tanh in our case) or (ii) identity. This is shown in M[fig:VolumePreservingFeedForward]m(@latex).
