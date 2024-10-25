Next we introduce the volume-preserving attention layer. We then show how this new layer preserves volume.

## The Volume-Preserving Attention Layer

The main difficulty in adapting a transformer-like architecture to be volume-preserving is to adapt the activation function. Indeed, the softmax acts vector-wise and cannot preserve volume. We thus replace the softmax by a different activation function, which is based on the Cayley transform:

```math
\alpha(Y) = \mathrm{Cayley}(Y) = (\mathbb{I}_{T} - Y)(\mathbb{I}_{T} + Y)^{-1}.
```

The Cayley transform maps skew-symmetric matrices to orthogonal matrices[^1], so if ``Y`` has the property that ``Y^T = -Y,`` then we have ``\alpha(Y)^T\alpha(Y) = \mathbb{I}.`` This can be easily shown:

[^1]: The orthogonal matrices ``\{B\in\mathbb{R}^{d\times{}d}:B^TB=\mathbb{I}_d\}`` form a Lie group under regular matrix multiplication. The associated Lie algebra is the vector space of skew-symmetric matrices ``\mathfrak{g}=\{C:C+C^T = \mathbb{O}\}`` and the Lie algebra is mapped to the Lie group via the Cayley transform. More details on this can be found in e.g. [hairer2006geometric](@cite).

```math
\begin{aligned}
\frac{d}{dt}\Big|_{t=0}\alpha(tY)^T\alpha(tY) & = \frac{d}{dt}(\mathbb{I}_{T} - tY)^{-1}(\mathbb{I}_{T} + tY)(\mathbb{I}_{T} - tY)(\mathbb{I}_{T} + tY)^{-1} \\
                                              & = Y + Y - Y - Y = \mathbb{O},
\end{aligned}
```

where we used that ``(\Lambda^{-1})^T = (\Lambda^T)^{-1}`` for an arbitrary invertible matrix ``\Lambda``. We now define a new activation function for the attention mechanism which we denote by 

```math
\Lambda(Z) = \alpha (Z^T A Z),
\label{eq:VolumePreservingActivation}
``` 

where ``A`` is a *learnable skew-symmetric matrix*. Note that the input into the Cayley transform has to be a skew-symmetric matrix in order for the output to be orthogonal; hence we have the restriction on ``A`` to be skew-symmetric. With this, ``\Lambda(Z)`` is an orthogonal matrix and the entire mapping is equivalent to a multiplication by an orthogonal matrix in the *vector representation* shown in M[eq:isomorphism]m(@latex). Note that the attention layer can again be seen as a reweighting of the input sequence and thus as a multiplication by a matrix from the right (see M[eq:RightMultiplication]m(@latex) and the comment below that equation):
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

Note that the input to the transformer ``Z`` is an element of this product space. Now consider the isomorphism ``\hat{}: \times_T \mathbb{R}^{d} \stackrel{\approx}{\longrightarrow} \mathbb{R}^{dT}`` of the form:
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

and it is easy to see that ``\widehat{\Lambda(Z)}`` in M[eq:LambdaApplication]m(@latex) is an orthogonal matrix. We show that M[eq:LambdaApplication]m(@latex) is true:

```math
    \widehat{Z\Lambda(Z)} \equiv \widehat{\sum_{k=1}^Tz_i^{(k)}\lambda_{kj}} \equiv \begin{bmatrix} \sum_{k=1}^T z_1^{(k)}\lambda_{k1} \\ \sum_{k=1}^T z_1^{(k)}\lambda_{k2} \\ \ldots \\ \sum_{k=1}^T z_1^{(k)}\lambda_{kT} \\ \sum_{k=1}^T z_2^{(k)}\lambda_{k1} \\ \ldots \\ \sum_{k=1}^T z_d^{(k)}\lambda_{kT} \end{bmatrix} = \begin{bmatrix} \sum_{k=1}^T \lambda_{k1}z_1^{(k)} \\ \sum_{k=1}^T \lambda_{k2}z_1^{(k)} \\ \ldots \\ \sum_{k=1}^T \lambda_{kT}z_1^{(k)} \\ \sum_{k=1}^T \lambda_{k1}z_2^{(k)} \\ \ldots \\ \sum_{k=1}^T \lambda_{kT}z_d^{(k)} \end{bmatrix} = \begin{bmatrix} [\Lambda(Z)^T z_1^{(\bullet)}]_1 \\ [\Lambda(Z)^T z_1^{(\bullet)}]_2 \\ \ldots \\ [\Lambda(Z)^T z_1^{(\bullet)}]_T \\ [\Lambda(Z)^T z_2^{(\bullet)}]_1 \\ \ldots \\ [\Lambda(Z)^T z_d^{(\bullet)}]_T \end{bmatrix} = \begin{bmatrix} \Lambda(Z)^Tz_1^{(\bullet)} \\ \Lambda(Z)^Tz_2^{(\bullet)} \\ \ldots \\ \Lambda(Z)^Tz_d^{(\bullet)} \end{bmatrix},
    \label{eq:ProductStructure}
```

where we defined:

```math
    z_i^{(\bullet)} := \begin{bmatrix} z_i^{(1)} \\ z_i^{(2)} \\ \ldots \\ z_i^{(T)} \end{bmatrix}.
```

```@raw latex
\begin{figure}
\centering
\includegraphics[width = .29\textwidth]{tikz/vp_transformer.png}
\caption{Architecture of the volume-preserving transformer. In comparison with the standard transformer in~\Cref{fig:TransformerArchitecture}, (i) the feedforward layer has been replaced with the volume-preserving feedforward neural network from~\Cref{fig:VolumePreservingFeedForward}, (ii) the attention layer has been replaced with a volume-preserving attention layer and (iii) the Add layer has been removed. Similar to~\Cref{fig:TransformerArchitecture} the integer ``L'' indicates how often a \textit{transformer unit} is repeated.}   
\label{fig:VolumePreservingTransformerArchitecture}
\end{figure}
```

Also note that in M[eq:ProductStructure]m(@latex) the expression after the first "``\equiv``" sign the ``(i,j)``-th element of the matrix, not the entire matrix (unlike the other terms in M[eq:ProductStructure]m(@latex)).


REMARK::
To our knowledge there is no literature on volume-preserving multi-step methods. There is however significant work on *symplectic multi-step methods* [feng1998step, hairer2006geometric, hairer2008conjugate](@cite). Of the two definitions of symplecticity for multi-step methods given in [hairer2006geometric](@cite), that of *``G``-symplecticity* is similar to the definition of volume preservation given here as it is also defined on a product space. The product structure through which we defined volume preservation also bears strong similarities to ``discrete multi-symplectic structures" defined in [bridges2001multi, yildiz2024structure](@cite).
::

The main result of this section was to show that this attention mechanism is *volume-preserving* in a product space that is spanned by the input vectors. We made the transformer volume-preserving  by (i) replacing the feedforward neural network (ResNet) by a volume-preserving feedforward network (volume-preserving ResNet), (ii) replacing standard attention by volume-preserving attention and (iii) removing the first add connection[^2]. The resulting architecture is shown in M[fig:VolumePreservingTransformerArchitecture]m(@latex).

[^2]: Removal of the add connection is necessary as the addition with the input is not a volume-preserving operation. 