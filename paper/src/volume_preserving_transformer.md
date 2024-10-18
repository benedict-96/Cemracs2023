Next we introduce the volume-preserving attention layer. We then show how this new layer preserves volume.

## The Volume-Preserving Attention Layer

The main difficulty in adapting a transformer-like architecture to be volume-preserving is to adapt the activation function. Indeed, the softmax acts vector-wise and cannot preserve volume. We thus replace the softmax by a different activation function, which is based on the Cayley transform:

```math
\sigma(Y) = \mathrm{Cayley}(Y) = (\mathbb{I}_{T} - Y)(\mathbb{I}_{T} + Y)^{-1}.
```

The Cayley transform maps skew-symmetric matrices to orthogonal matrices[^1], so if ``Y`` has the property that ``Y^T = -Y,`` then we have ``\sigma(Y)^T\sigma(Y) = \mathbb{I}.`` This can be easily shown:

[^1]: The orthogonal matrices ``\{B\in\mathbb{R}^{d\times{}d}:B^TB=\mathbb{I}_d\}`` form a Lie group under regular matrix multiplication. The associated Lie algebra is the vector space of skew-symmetric matrices ``\mathfrak{g}=\{C:C+C^T = \mathbb{O}\}`` and the Lie algebra is mapped to the Lie group via the Cayley transform. More details on this can be found in e.g. [hairer2006geometric](@cite).

```math
\begin{aligned}
\frac{d}{dt}\Big|_{t=0}\sigma(tY)^T\sigma(tY) & = \frac{d}{dt}(\mathbb{I}_{T} - tY)^{-1}(\mathbb{I}_{T} + tY)(\mathbb{I}_{T} - tY)(\mathbb{I}_{T} + tY)^{-1} \\
                                              & = Y + Y - Y - Y = \mathbb{O},
\end{aligned}
```

where we used that ``(\Lambda^{-1})^T = (\Lambda^T)^{-1}`` for an arbitrary invertible matrix ``\Lambda``. We now define a new activation function for the attention mechanism which we denote by 

```math
\Lambda(Z) = \sigma (Z^T A Z),
``` 

where ``A`` is a *learnable skew-symmetric matrix*. Note that the input into the Cayley transform has to be a skew-symmetric matrix in order for the output to be orthogonal; hence we have the restriction on ``A`` to be skew-symmetric. With this, ``\Lambda(Z)`` is an orthogonal matrix and the entire mapping is equivalent to a multiplication by an orthogonal matrix in the *vector representation* shown in M[eq:isomorphism]m(@latex). To see this, note that the attention layer performs the following operation (see M[eq:RightMultiplication]m(@latex) and the comment below that equation):
```math
Z \mapsto Z\Lambda(Z).
\label{eq:LambdaRight}
```

We conclude by formalizing what the volume-preserving attention mechanism:

DEFINITION::
The **volume-preserving attention mechanism** is a map based on the Cayley transform that reweights a collection of input vectors ``Z`` via ``Z \mapsto Z\Lambda(Z),`` where ``\Lambda(Z) = \mathrm{Cayley}(Z^TAZ)`` and ``A`` is a learnable skew-symmetric matrix. 
::

## How is Structure Preserved? 

Here we discuss how *volume-preserving attention* preserves structure. To this end, we first have to define what volume preservation means for the product space
```math
\times_T \mathbb{R}^{d} \equiv \underbrace{ \mathbb{R}^{d} \times \cdots \times \mathbb{R}^{d} }_{\text{$T$ times}} .
```

Consider an isomorphism ``\hat{}: \times_T \mathbb{R}^{d} \stackrel{\approx}{\longrightarrow} \mathbb{R}^{dT}`` of the form:
```math
Z = \begin{pmatrix}
            z_1^{(1)} &  z_1^{(2)} & \quad\cdots\quad & z_1^{(T)} \\
            z_2^{(1)} &  z_2^{(2)} & \cdots & z_2^{(T)} \\
            \cdots &  \cdots & \cdots & \cdots \\
            z_d^{(1)} & z_d^{(2)} & \cdots & z_d^{(T)}
    \end{pmatrix}
\mapsto \hat{Z} = 
\begin{bmatrix}
    z_1^{(1)} \\
    z_1^{(2)} \\
    \cdots \\
    z_1^{(T)} \\
    z_2^{(1)} \\
    \cdots \\
    z_d^{(T)}
\end{bmatrix} 
=: Z_\mathrm{vec}.
\label{eq:isomorphism}
```

We refer to the inverse of ``Z \mapsto \hat{Z} `` as ``Y \mapsto \check{Y}``. In the following we also write ``\hat{\varphi}`` for the mapping ``\,\hat{}\circ\varphi\circ\check{}\,``.

DEFINITION::
A mapping ``\varphi: \times_T \mathbb{R}^{d} \to \times_T \mathbb{R}^{d}`` is said to be **volume-preserving** if the associated ``\hat{\varphi}`` is volume-preserving.
::


In the transformed coordinate system, that is in terms of the vector ``Z_\mathrm{vec}`` defined in M[eq:isomorphism]m(@latex), this is equivalent to multiplication by a block-diagonal matrix ``\widehat{\Lambda(Z)}`` from the left:
```math
    \widehat{\Lambda(Z)} Z_\mathrm{vec} :=
    \begin{pmatrix}
    \Lambda(Z)^T & \mathbb{O} & \cdots  & \mathbb{O} \\
    \mathbb{O} & \Lambda(Z)^T & \cdots & \mathbb{O} \\
    \cdots & \cdots & \ddots & \cdots \\ 
    \mathbb{O} & \mathbb{O} & \cdots & \Lambda(Z)^T \\
    \end{pmatrix}
    \left[\begin{array}{c}  z_1^{(1)} \\ z_1^{(2)} \\ \ldots \\ z_1^{(T)} \\ z_2^{(1)} \\ \ldots \\ z_d^{(T)} \end{array}\right] .
    \label{eq:LambdaApplication}
```

We show that this expression is true:

```math
    \widehat{Z\Lambda(Z)} = \widehat{\sum_{k=1}^Tz_i^{(k)}\lambda_{kj}} = \begin{bmatrix} \sum_{k=1}^T z_1^{(k)}\lambda_{k1} \\ \sum_{k=1}^T z_1^{(k)}\lambda_{k2} \\ \ldots \\ \sum_{k=1}^T z_1^{(k)}\lambda_{kT} \\ \sum_{k=1}^T z_2^{(k)}\lambda_{k1} \\ \ldots \\ \sum_{k=1}^T z_d^{(k)}\lambda_{kT} \end{bmatrix} = \begin{bmatrix} \sum_{k=1}^T \lambda_{k1}z_1^{(k)} \\ \sum_{k=1}^T \lambda_{k2}z_1^{(k)} \\ \ldots \\ \sum_{k=1}^T \lambda_{kT}z_1^{(k)} \\ \sum_{k=1}^T \lambda_{k1}z_2^{(k)} \\ \ldots \\ \sum_{k=1}^T \lambda_{kT}z_d^{(k)} \end{bmatrix} = \begin{bmatrix} [\Lambda(Z)^T z_1^{(\bullet)}]_1 \\ [\Lambda(Z)^T z_1^{(\bullet)}]_2 \\ \ldots \\ [\Lambda(Z)^T z_1^{(\bullet)}]_T \\ [\Lambda(Z)^T z_2^{(\bullet)}]_1 \\ \ldots \\ [\Lambda(Z)^T z_d^{(\bullet)}]_T \end{bmatrix} = \begin{bmatrix} \Lambda(Z)^Tz_1^{(\bullet)} \\ \Lambda(Z)^Tz_2^{(\bullet)} \\ \ldots \\ \Lambda(Z)^Tz_d^{(\bullet)} \end{bmatrix},
```

where we defined:

```math
    z_i^{(\bullet)} := \begin{bmatrix} z_i^{(1)} \\ z_i^{(2)} \\ \ldots \\ z_i^{(T)} \end{bmatrix}.
```

It is easy to see that ``\widehat{\Lambda(Z)}`` in M[eq:LambdaApplication]m(@latex) is an orthogonal matrix. 

```@raw latex
\begin{figure}
\includegraphics[width = .29\textwidth]{tikz/vp_transformer.png}
\caption{Architecture of the volume-preserving transformer. In comparison with the standard transformer in~\Cref{fig:TransformerArchitecture}, the Add layer has been removed, the attention layer has been replaced with a volume-preserving attention layer, and the feed forward layer has been replaced with the volume-preserving feedforward neural network from~\Cref{fig:VolumePreservingFeedForward}.}
\label{fig:VolumePreservingTransformerArchitecture}
\end{figure}
```

The main result of this section was to show that this attention mechanism is *volume-preserving* in a product space that is spanned by the input vectors. To conclude we make the transformer volume-preserving  by (i) replacing the feedforward neural network (ResNet) by a volume-preserving feedforward network (volume-preserving ResNet), replacing standard attention by volume-preserving attention and removing the first add connection[^2]. The resulting architecture is shown in M[fig:VolumePreservingTransformerArchitecture]m(@latex).

[^2]: Removal of the add connection is necessary as the addition with the input is not a volume-preserving operation. 