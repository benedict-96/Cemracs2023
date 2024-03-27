# Volume-Preserving Transformer

In this section we introduce a new attention mechanism that we call *structure-preserving attention*. It treats symplecticity the same way it is defined for a multistep method [feng1998step](@cite). To this end, we first define a notion of symplecticity for the product space[^1] ``\mathbb{R}^{2d}\times\cdots\times\mathbb{R}^{2d}\equiv\times_\text{$T$ times}\mathbb{R}^{2d}``:
```math
    \tilde{\mathbb{J}}(z^{(1)}\times\cdots\times{}z^{(T)}, \bar{z}^{(1)}\times\cdots\times{}\bar{z}^{(T)}) =: (z^{(1)})^T\mathbb{J}_{2d}\bar{z}^{(1)} + \cdots + (z^{(T)})^T\mathbb{J}_{2d}\bar{z}^{(T)}.
```

[^1]: In **???**, we mentioned that a symplectic vector space is an even-dimensional vector space endowed with a distinct non-degenerate skew-symmetric two-form. Thus ``\mathbb{R}^{2d}\times\cdots\times\mathbb{R}^{2d}`` can become a symplectic vector space by defining such a form.

The two-form ``\tilde{\mathbb{J}}:(\times_\text{($T$ times)}\mathbb{R}^{2d})\times(\times_\text{($T$ times)}\mathbb{R}^{2d})\to\mathbb{R}`` is easily seen to be skew-symmetric, linear and non-degenerate[^2].

[^2]: To proof non-degeneracy we have to show that for any vector ``z:=z^{(1)}\times\cdots\times{}z^{(T)}, \bar{z}^{(1)}\times\cdots\times{}\bar{z}^{(T)}\neq0`` on the product space we can find another vector $\hat{z}$ such that ``\tilde{\mathbb{J}}(z, \hat{z})\neq0``. But we know that at we have at least one integer $i$ for which we can find ``\hat{z}^{(i)}`` such that ``(z^{(i)})^T\mathbb{J}_{2d}\hat{z}^{(i)}\neq0``. Now we take for ``\hat{z}`` simply ``0\times\cdots\times\hat{z}^{(i)}\times\cdots\times0``.

To simplify the discussion of symplecticity on the product space ``\times_\text{($T$ times)}\mathbb{R}^{2d}`` we define an isomorphism ``\times_\text{($T$ times)}\mathbb{R}^{2d}\stackrel{\sim}{\longrightarrow}\mathbb{R}^{2dT}``; this way there will not be a difference in the definition of symplecticity to the one in **???**.
Specifically, the isomorphism ``\times_\text{($T$ times)}\mathbb{R}^{2d}\stackrel{\sim}{\longrightarrow}\mathbb{R}^{2dT}`` takes the following form:
```math
Z = \begin{pmatrix} Q_1\\ Q_2 \\ P_1 \\ P_2  \end{pmatrix} =
        \left[
        \begin{array}{cccc}
            q_1^{(1)} &  q_1^{(2)} & \quad\cdots\quad & q_1^{(T)} \\
            q_2^{(1)} &  q_2^{(2)} & \cdots & q_2^{(T)} \\
            \cdots &  \cdots & \cdots & \cdots \\
            q_n^{(1)} & q_d^{(2)} & \cdots & q_d^{(T)}\\
            p_1^{(1)} & p_1^{(2)} & \cdots & p_1^{(T)} \\
            p_2^{(1)} & p_2^{(2)} & \cdots & p_2^{(T)} \\
            \cdots &  \cdots & \cdots & \cdots \\
            p_d^{(1)} & p_d^{(2)} & \cdots & p_d^{(T)}\\
            \end{array}\right] \mapsto 
            \left[\begin{array}{c}  q_1^{(1)} \\ q_1^{(2)} \\ \cdots \\ q_1^{(T)} \\ q_2^{(1)} \\ \cdots \\ q_d^{(T)} \\ p_1^{(1)} \\  \cdots \\ p_1^{(T)} \\ p_2^{(1)} \\ \cdots \\ p_d^{(T)} \end{array}\right] =: Z_\mathrm{vec}.
            \label{eq:big_vector}
```

The main difficulty in adapting a transformer-like architecture to be symplectic or structure-preserving is to adapt the activation function. Indeed, the softmax acts vector-wise and cannot preserve symplecticity. We thus replace the softmax by a different activation function. This new activation function is a composition of a skew-symmetrization operation ``\Phi`` and a Cayley transform: 
```math
    \sigma(C) = \mathrm{Cayley}(\Phi(C)),
```

where we have:
```math
\Phi_{ij} = \begin{cases} c_{ij} & \text{if $i<j$}  \\ -c_{ji} & \text{if $i>j$} \\ 0 & \text{else.}\end{cases} \qquad\text{ and }\qquad \mathrm{Cayley}(Y) = \frac{1}{2}(\mathbb{I}_{T} - Y)(\mathbb{I}_{T} + Y)^{-1}.
```

The Cayley transform maps skew-symmetric matrices to orthonormal matrices[^3], and ``\Phi`` maps arbitrary matrices to skew-symmetric ones. This results in a new activation function for our attention mechanism which we denote by ``\Lambda(Z) = \sigma (Z^T A Z)``. Because ``\Lambda(Z)`` is orthonormal the entire mapping is equivalent to a multiplication by a symplectic matrix. To see this note that the attention layer has the following action:

```math
Z \mapsto Z\Lambda(Z).
\label{eq:LambdaRight}
```

In the transformed coordinate system (in terms of the vector ``Z_\mathrm{vec}`` from \\ldots), this is equivalent to multiplication by a sparse matrix ``\tilde\Lambda(Z)`` from the left:

[^3]: The orthonormal matrices ``\{B\in\mathbb{R}^{d\times{}d}:B^TB=\mathbb{I}_d\}`` form a Lie group under regular matrix multiplication. The associated Lie algebra is the vector space of skew-symmetric matrices ``\mathfrak{g}=\{C:C+C^T\}`` and the Lie algebra is mapped to the Lie group via the Cayley transform. More details on this can be found in e.g. [hairer2006geometric](@cite).

```math
    \tilde{\Lambda}(Z) Z_\mathrm{vec} :=
    \begin{pmatrix}
    \Lambda(Z) & \mathbb{O} & \cdots  & \mathbb{O} \\
    \mathbb{O} & \Lambda(Z) & \cdots & \mathbb{O} \\
    \cdots & \cdots & \ddots & \cdots \\ 
    \mathbb{O} & \mathbb{O} & \cdots & \Lambda(Z) \\
    \end{pmatrix}
    \left[\begin{array}{c}  q_1^{(1)} \\ q_1^{(2)} \\ \ldots \\ q_1^{(T)} \\ q_2^{(1)} \\ \ldots \\ q_d^{(T)} \\ p_1^{(1)} \\  \ldots \\ p_1^{(T)} \\ p_2^{(1)} \\ \ldots \\ p_d^{(T)} \end{array}\right] .
```

![Volume-Preserving Transformer](tikz/vp_transformer.png)

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

For the remaining parts of the transformer, the feedforward neural network is replaced by a SympNet and the first add connection is removed, as shown in figure \ref{fig:attention_transformer_architecture}. Removal of the add connection is necessary as the addition with the input is not a symplectic operation. 
It is important to note that the map in cref{eq:Lambda_right} is not actually symplectic either but only volume-preserving.
Even though ``\Lambda(Z)`` is a symplectic matrix it explicitly depends on ``Z``, leading to additional terms in the Jacobian of this transformation. As a result the modified transformer preserves phasespace volume, but is not symplectic, which is a stronger property. 
