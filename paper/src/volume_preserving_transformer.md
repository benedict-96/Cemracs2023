# Volume-preserving transformer

In this section we introduce a new attention mechanism that we call *volume-preserving attention*, which is strongly inspired by traditional *multi-step methods* [feng1998step](@cite). To this end, we first have to define what volume preservation means for the product space
```math
\times_T \mathbb{R}^{d} \equiv \underbrace{ \mathbb{R}^{d} \times \cdots \times \mathbb{R}^{d} }_{\text{$T$ times}} .
```

Consider an isomorphism ``\hat{}: \times_T \mathbb{R}^{d} \stackrel{\approx}{\longrightarrow} \mathbb{R}^{dT}`` of the form:
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

We refer to the inverse of ``Z \mapsto \hat{Z} `` as ``Y \mapsto \tilde{Y}``. In the following we also write ``\hat{\varphi}`` for the mapping ``\,\hat{}\circ\varphi\circ\tilde{}\,``.

DEFINITION::
A mapping ``\varphi: \times_T \mathbb{R}^{d} \to \times_T \mathbb{R}^{d}`` is said to be *volume-preserving* if the associated ``\hat{\varphi}`` is volume-preserving.
::

The main difficulty in adapting a transformer-like architecture to be volume-preserving is to adapt the activation function. Indeed, the softmax acts vector-wise and cannot preserve volume. We thus replace the softmax by a different activation function, which is based on the Cayley transform:

```math
\sigma(Y) = \mathrm{Cayley}(Y) = \frac{1}{2}(\mathbb{I}_{T} - Y)(\mathbb{I}_{T} + Y)^{-1}.
```

The Cayley transform maps skew-symmetric matrices to orthogonal matrices[^1]. This results in a new activation function for the attention mechanism which we denote by ``\Lambda(Z) = \sigma (Z^T A Z)``. Further note that the input into the Cayley transform has to be a skew-symmetric matrix. For this reason we need to constrain ``A`` to be also skew-symmetric. With this, ``\Lambda(Z)`` is an orthogonal matrix and the entire mapping is equivalent to a multiplication by an orthogonal matrix in the *vector representation* shown in M[eq:isomorphism]m(@latex). To see this, note that the attention layer performs the following operation:
```math
Z \mapsto Z\Lambda(Z).
\label{eq:LambdaRight}
```

[^1]: The orthogonal matrices ``\{B\in\mathbb{R}^{d\times{}d}:B^TB=\mathbb{I}_d\}`` form a Lie group under regular matrix multiplication. The associated Lie algebra is the vector space of skew-symmetric matrices ``\mathfrak{g}=\{C:C+C^T = \mathbb{O}\}`` and the Lie algebra is mapped to the Lie group via the Cayley transform. More details on this can be found in e.g. [hairer2006geometric](@cite).

In the transformed coordinate system, that is in terms of the vector ``Z_\mathrm{vec}`` defined in M[eq:isomorphism]m(@latex), this is equivalent to multiplication by a block-diagonal matrix ``\tilde\Lambda(Z)`` from the left:
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

It is easy to see that ``\tilde{\Lambda}(Z)`` in M[eq:LambdaApplication]m(@latex) is an orthogonal matrix. 

```@raw latex
\begin{figure}
\includegraphics[width = .3\textwidth]{tikz/vp_transformer.png}
\caption{Architecture of the volume-preserving transformer. In comparison with the standard transformer in~\Cref{fig:TransformerArchitecture}, the Add layer has been removed, the attention layer has been replaced with a volume-preserving attention layer, and the feed forward layer has been replaced with the volume-preserving feedforward neural network from~\Cref{fig:VolumePreservingFeedForward}.}
\label{fig:VolumePreservingTransformerArchitecture}
\end{figure}
```

While the replacement of the standard transformer attention with this new volume-preserving attention is the biggest change to the transformer architecture, in addition, the feedforward neural network is replaced by a volume-preserving feedforward network, and the first add connection is removed[^2]. The resulting architecture is shown in M[fig:VolumePreservingTransformerArchitecture]m(@latex).

[^2]: Removal of the add connection is necessary as the addition with the input is not a volume-preserving operation. 
