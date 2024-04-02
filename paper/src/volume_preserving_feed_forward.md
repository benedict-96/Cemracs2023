# Volume-Preserving Feedforward Neural Networks

As a first step to construct a *structure-preserving transformer* we replace the ResNet in the transformer with a feedforward neural network that is volume-preserving. The *volume-preseving feedforward layers* here are inspired by the linear and activation modules from [jin2020sympnets](@cite). Here our key ingredient are upper-triangular matrices ``U`` and lower-triangular matrices ``L`` for whose components we have that ``u_ij = 0`` if ``i \geq j`` and ``l_ij = 0`` if ``i \leq j``.

In matrix form they look like this: 

```math 
L = \begin{pmatrix}
     0 & 0 & \cdots & 0      \\
     a_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     a_{n1} & \cdots & a_{n(n-1)}      & 0 
\end{pmatrix}.
```

The Jacobian of a layer of the above form then is of the form

```math 
J = \begin{pmatrix}
     1 & 0 & \cdots & 0      \\
     b_{21} & \ddots &        & \vdots \\
     \vdots & \ddots & \ddots & \vdots \\
     b_{n1} & \cdots & b_{n(n-1)}      & 1 
\end{pmatrix},
```
and the determinant of ``J`` is 1, i.e. the map is volume-preserving. 

In practice we combine many of those layers where the activation function is either (i) a fixed nonlinearity (tanh in our case) or (ii) identity. 

```@raw latex
\begin{figure}
\includegraphics[width = .5\textwidth]{tikz/vp_feedforward.png}
\caption{Architecture for the volume-preserving feedforward neural network.}
\label{fig:vpf_arch}
\end{figure}
```