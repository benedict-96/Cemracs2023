# Volume-preserving transformer

In this section we introduce a new attention mechanism that we call *volume-preserving attention*. This is strongly inspired by traditional *multi-step methods* (see [feng1998step](@cite)). To this end, we first have to define what volume preservation means for the product space ``\mathbb{R}^{d}\times\cdots\times\mathbb{R}^{d}\equiv\times_\text{$T$ times}\mathbb{R}^{d}``.

Consider an isomorphism ``\hat{}: \times_\text{($T$ times)}\mathbb{R}^{d}\stackrel{\approx}{\longrightarrow}\mathbb{R}^{dT}``. Specifically, this isomorphism takes the form:
```math
Z =  \left[\begin{array}{cccc}
            z_1^{(1)} &  z_1^{(2)} & \quad\cdots\quad & z_1^{(T)} \\
            z_2^{(1)} &  z_2^{(2)} & \cdots & z_2^{(T)} \\
            \cdots &  \cdots & \cdots & \cdots \\
            z_d^{(1)} & z_d^{(2)} & \cdots & z_d^{(T)}
            \end{array}\right] \mapsto 
            \left[\begin{array}{c}  z_1^{(1)} \\ z_1^{(2)} \\ \cdots \\ z_1^{(T)} \\ z_2^{(1)} \\ \cdots \\ z_d^{(T)} \end{array}\right] =: Z_\mathrm{vec}.
\label{eq:isomorphism}
```

The inverse of ``Z \mapsto \hat{Z} `` we refer to as ``Y \mapsto \tilde{Y}``. In the following we also write ``\hat{\varphi}`` for the mapping ``\,\hat{}\circ\varphi\circ\tilde{}\,``.

DEFINITION::
We say that a mapping ``\varphi: \times_\text{$T$ times}\mathbb{R}^{d} \to \times_\text{$T$ times}\mathbb{R}^{d}`` is **volume-preserving** if the associated ``\hat{\varphi}`` is volume-preserving.::

The main difficulty in adapting a transformer-like architecture to be volume-preserving is to adapt the activation function. Indeed, the softmax acts vector-wise and cannot preserve volume. We thus replace the softmax by a different activation function. This new activation function is based on the Cayley transform:

```math
\sigma(Y) = \mathrm{Cayley}(Y) = \frac{1}{2}(\mathbb{I}_{T} - Y)(\mathbb{I}_{T} + Y)^{-1}.
```

The Cayley transform maps skew-symmetric matrices to orthogonal matrices[^1]. This results in a new activation function for our attention mechanism which we denote by ``\Lambda(Z) = \sigma (Z^T A Z)``. Further note that the input into the Cayley transform has to be a skew-symmetric matrix. For this reason we need to constrain ``A`` to be also skew-symmetric. With this ``\Lambda(Z)`` is an orthogonal matrix and the entire mapping is equivalent to a multiplication by a orthogonal matrix in the *big vector representation* shown in m[eq:isomorphism]m(@latex). To see this note that the attention layer performs the following:

[^1]: The orthogonal matrices ``\{B\in\mathbb{R}^{d\times{}d}:B^TB=\mathbb{I}_d\}`` form a Lie group under regular matrix multiplication. The associated Lie algebra is the vector space of skew-symmetric matrices ``\mathfrak{g}=\{C:C+C^T = \mathbb{O}\}`` and the Lie algebra is mapped to the Lie group via the Cayley transform. More details on this can be found in e.g. [hairer2006geometric](@cite).

```math
Z \mapsto Z\Lambda(Z).
\label{eq:LambdaRight}
```

In the transformed coordinate system (in terms of the vector ``Z_\mathrm{vec}`` defined in m[eq:isomorphism]m(@latex)), this is equivalent to multiplication by a sparse matrix ``\tilde\Lambda(Z)`` from the left:

```math
    \tilde{\Lambda}(Z) Z_\mathrm{vec} :=
    \begin{pmatrix}
    \Lambda(Z) & \mathbb{O} & \cdots  & \mathbb{O} \\
    \mathbb{O} & \Lambda(Z) & \cdots & \mathbb{O} \\
    \cdots & \cdots & \ddots & \cdots \\ 
    \mathbb{O} & \mathbb{O} & \cdots & \Lambda(Z) \\
    \end{pmatrix}
    \left[\begin{array}{c}  z_1^{(1)} \\ z_1^{(2)} \\ \ldots \\ z_1^{(T)} \\ z_2^{(1)} \\ \ldots \\ z_d^{(T)} \end{array}\right] .
    \label{eq:LambdaApplication}
```

``\tilde{\Lambda}(Z)`` in m[eq:LambdaApplication]m(@latex) is easily shown to be an orthogonal matrix. 

```@raw latex
\begin{figure}
\includegraphics[width = .3\textwidth]{tikz/vp_transformer.png}
\caption{Architecture for the volume-preserving transformer.}
\label{fig:VolumePreservingTransformerArchitecture}
\end{figure}
```

For the remaining parts of the transformer, the feedforward neural network is replaced by a volume-preserving feedforward network and the first add connection is removed[^2]. The architecture is shown in m[fig:VolumePreservingTransformerArchitecture]m(@latex).

[^2]: Removal of the add connection is necessary as the addition with the input is not a volume-preserving operation. 