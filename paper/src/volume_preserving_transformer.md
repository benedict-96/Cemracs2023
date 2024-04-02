# Volume-Preserving Transformer

In this section we introduce a new attention mechanism that we call *structure-preserving attention*. This is strongly inspired by traiditon *multi-step methods* (see [feng1998step](@cite)). To this end, we first have to define what volume preservation means for the product space ``\mathbb{R}^{d}\times\cdots\times\mathbb{R}^{d}\equiv\times_\text{$T$ times}\mathbb{R}^{d}``.

Now consider an isomorphism ``\hat{}: \times_\text{($T$ times)}\mathbb{R}^{d}\stackrel{\sim}{\longrightarrow}\mathbb{R}^{dT}``. Specifically, this isomorphism takes the form:
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

The inverse of ``\hat{}`` we refer to as ``Y \mapsto \tilde{Y}``. In the following we refer to ``\hat{\varphi}`` as the mapping ``\,\hat{}\circ\varphi\circ\tilde{}\,``.

__Definition of volume preservation__: We say that a mapping ``\varphi: \times_\text{$T$ times}\mathbb{R}^{d} \to \times_\text{$T$ times}\mathbb{R}^{d}`` is volume preserving if the associated ``\hat{\varphi}`` is volume-preserving. 

The main difficulty in adapting a transformer-like architecture to be symplectic or structure-preserving is to adapt the activation function. Indeed, the softmax acts vector-wise and cannot preserve symplecticity. We thus replace the softmax by a different activation function. This new activation function is a composition of a skew-symmetrization operation ``\Phi`` and a Cayley transform: 
```math
    \sigma(C) = \mathrm{Cayley}(\Phi(C)),
```

where we have:
```math
\Phi_{ij} = \begin{cases} c_{ij} & \text{if $i<j$}  \\ -c_{ji} & \text{if $i>j$} \\ 0 & \text{else.}\end{cases} \qquad\text{ and }\qquad \mathrm{Cayley}(Y) = \frac{1}{2}(\mathbb{I}_{T} - Y)(\mathbb{I}_{T} + Y)^{-1}.
```

The Cayley transform maps skew-symmetric matrices to orthonormal matrices[^3], and ``\Phi`` maps arbitrary matrices to skew-symmetric ones. This results in a new activation function for our attention mechanism which we denote by ``\Lambda(Z) = \sigma (Z^T A Z)``. Because ``\Lambda(Z)`` is orthonormal the entire mapping is equivalent to a multiplication by a symplectic matrix. To see this note that the attention layer has the following action:

[^3]: The orthonormal matrices ``\{B\in\mathbb{R}^{d\times{}d}:B^TB=\mathbb{I}_d\}`` form a Lie group under regular matrix multiplication. The associated Lie algebra is the vector space of skew-symmetric matrices ``\mathfrak{g}=\{C:C+C^T\}`` and the Lie algebra is mapped to the Lie group via the Cayley transform. More details on this can be found in e.g. [hairer2006geometric](@cite).

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
    \label{eq:lambda_application}
```

```@raw latex
\begin{figure}
\includegraphics[width = .5\textwidth]{tikz/vp_transformer.png}
\caption{Architecture for the volume-preserving transformer.}
\label{fig:VolumePreservingTransformerArchitecture}
\end{figure}
```

M[eq:LambdaRight]m(@latex) is equivalent to a multiplication by a symplectic matrix if we can show that ``\tilde{\Lambda}(Z)^T\mathbb{J}_{2nT}\tilde{\Lambda}(Z) = \mathbb{J}_{2nT}``, where we have defined a big symplectic matrix[^4]:
```math
\mathbb{J}_{2dT} = \begin{pmatrix} \mathbb{O} & \mathbb{I}_{dT} \\ -\mathbb{I}_{dT} & \mathbb{O} \end{pmatrix}.
```
Because the matrix ``\Lambda(Z)`` is orthonormal, i.e. ``\Lambda(Z)^T\Lambda(Z) = \mathbb{I}_T``, the sparse matrix ``\tilde{\Lambda}`` is symplectic:
```math
    (\mathbb{I}_{2d}\otimes\Lambda(Z)^T)(\mathbb{J}_{2d}\otimes\mathbb{I}_{T})(\mathbb{I}_{2d}\otimes\Lambda(Z)) = \mathbb{J}_{2d}\otimes(\Lambda(Z)^T\Lambda(Z)) = \mathbb{J}_{2dT} .
```
Here, we have used tensor product notation, i.e. 
```math
A \otimes B = \begin{pmatrix}   a_{11}B & a_{12}B & \cdots & a_{1n}B \\ 
                                a_{21}B & a_{22}B & \cdots & a_{2n}B \\
                                \cdots  & \cdots  & \cdots & \cdots  \\
                                a_{d1}B & \cdots  & \cdots & a_{dn}  
\end{pmatrix},
```
for a ``d\times{}n`` matrix ``A``.

[^4]: It can be easily checked that ``\tilde{\mathbb{J}}(Z^1, Z^2) = (Z^1_\mathrm{vec})^T\mathbb{J}_{2dT}Z^2_\mathrm{vec}``, so the two notions of symplecticity are equivalent.

For the remaining parts of the transformer, the feedforward neural network is replaced by a SympNet and the first add connection is removed, as shown in figure m[fig:VolumePreservingTransformerArchitecture]m(@latex). Removal of the add connection is necessary as the addition with the input is not a symplectic operation. 
It is important to note that the map in cref{eq:Lambda_right} is not actually symplectic either but only volume-preserving.
Even though ``\Lambda(Z)`` is a symplectic matrix it explicitly depends on ``Z``, leading to additional terms in the Jacobian of this transformation. As a result the modified transformer preserves phasespace volume, but is not symplectic, which is a stronger property. 
