```@raw latex 
\appendix
```

# The Rigid Body

In this appendix, we sketch the derivation of the rigid body equations used in [Experimental results](@ref).
The differential equation for the rigid body [hairer2006geometric, arnold1978mathematical](@cite) describes the dynamics of a solid object fixed at a point. M[fig:RigidBody]m(@latex) shows an example for a rigid body. 
Its dynamics can always be described in terms of an ellipsoid. To see this, let us consider the derivation of the rigid body equations. The motion of a point ``(x_1, x_2, x_3)^T`` in the rigid body ``\mathcal{B}`` can be described as follows: 
```math
v := \frac{d}{dt} \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
= \omega \times \begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix}
= \begin{pmatrix} \omega_2x_3 - \omega_3x_2 \\ \omega_3x_1 - \omega_1x_3 \\ \omega_1x_2 - \omega_2x_1 \end{pmatrix}
= \begin{pmatrix} 0 & - \omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix}\begin{bmatrix} x_1 \\ x_2 \\ x_3 \end{bmatrix},
```
where ``\omega`` is the angular velocity. 
The total kinetic energy of ``\mathcal{B}`` is obtained by integrating over the entire volume of the body:
```math
T  = \frac{1}{2} \int_\mathcal{B} ||v||^2 dm = \frac{1}{2} \int_\mathcal{B} || \omega \times x ||^2 dm = \frac{1}{2} \omega^T \Theta \omega,        
```
where 
```math
\Theta_{ij} = \begin{cases}
    \int_\mathcal{B}(x_k^2 + x_\ell^2)dm & \text{ if } i = j, \\  
    -  \int_\mathcal{B}x_ix_jdm & \text{else} ,
\end{cases}
```
where ``dm`` indicates an integral over the mass density of the rigid body, and we further have ``i\neq{}k\neq\ell\neq{}i`` for the first case.

The mathematical description of the motion of a rigid body hence does not require knowledge of the precise shape of the body, but only the coefficients ``\Theta_{ij}``. As ``\Theta`` is a symmetric matrix, we can write the kinetic energy as:
```math
T = \frac{1}{2}\omega^T\Theta\omega = \frac{1}{2}\omega^TU^TIU\omega,
\label{eq:RigidBodyKineticEnergy}
```
where ``U`` is an orthonormal matrix that diagonalizes ``\Theta.`` In M[eq:RigidBodyKineticEnergy]m(@latex) we called the eigenvalues of ``\Theta`` ``I = \mathrm{diag} (I_1, I_2, I_3)``.

This shows that it is sufficient to know the eigenvalues of the matrix ``\Theta`` which are called *moments of inertia* and denoted by ``I_k`` for ``k = 1, 2, 3`` to describe the motion of the rigid body (modulo a rotation). From this point of view every rigid body is equivalent to an ellipsoid as indicated in M[fig:RigidBody]m(@latex). 

```@raw latex
\begin{figure}
\includegraphics[width=.5\textwidth]{tikz/ellipsoid.png}
\caption{Any rigid body fixed at a point (left) can be described through an ellipsoid (right) through $I_1$, $I_2$ and $I_3$.}
\label{fig:RigidBody}
\end{figure}
```

## Formulation of the Equations of Motion in the Euler-Poincaré Framework

The dynamics of the rigid body can be described through a rotational matrix ``Q(t)``, i.e. each point of the rigid body ``x(0)\in\mathcal{B}`` can be described through ``x(t) = Q(t)x(0)`` where ``Q(t)^TQ(t) = \mathbb{I}``. We can therefore describe the evolution of the rigid body through a differential equation on the Lie group ``G := \{Q\in\mathbb{R}^{d\times{}d}:Q^TQ = \mathbb{I}\}``. The associated tangent vector ``\dot{Q}\in{}T_QG`` can be mapped to the Lie algebra[^1] ``\mathfrak{g}=T_\mathbb{I}G`` by:
```math
W := \dot{Q}Q^T = \begin{pmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix} \in \mathfrak{g}. 
\label{eq:LieAlgebraRepresentation}
```
[^1]: This is the Lie algebra of skew-symmetric matrices: ``\mathfrak{g} = \{W\in\mathbb{R}^{d\times{}d}:W^T = -W\}``.

As was indicated in equation M[eq:LieAlgebraRepresentation]m(@latex), the components of the skew-symmetric matrix ``W`` are equivalent to those of the angular velocity ``\omega`` as can easily be checked: 
```math
\dot{x} (t)
= \frac{d}{dt}Q(t)x(0) = \dot{Q}(t)x(0) = \dot{Q}(t)Q^T(t)x(t) = W(t)x(t)
% = Wx
% = \begin{pmatrix} 0 & -\omega_3 & \omega_2 \\ \omega_3 & 0 & -\omega_1 \\ -\omega_2 & \omega_1 & 0 \end{pmatrix} \begin{pmatrix} x_1 \\ x_2 \\ x_3 \end{pmatrix}
% = \\ \begin{pmatrix}  -\omega_3x_2 + \omega_2x_3 \\ \omega_3x_1 - \omega_1x_3 \\ -\omega_2x_1 + \omega_1x_2 \end{pmatrix}
= \omega (t) \times x (t) .
```

With this description, the kinetic energy can be written as: 
```math
T = \frac{1}{2}\mathrm{tr}(WDW^T),
\label{eq:KineticEnergyForLieGroup}
```
where ``D = \mathrm{diag}(d_1, d_2, d_3)`` is a diagonal matrix[^2] that satisfies ``I_1 = d_2 + d_3``, ``I_2 = d_3 + d_1`` and ``I_3 = d_1 + d_2``. We now write ``z := I^{-1}\omega`` and introduce the following notation[^3]:

[^3]: Note that the ``\hat{}`` operation used here is different from the hat operation used in [How is Structure Preserved?](@ref).

```math
\hat{z} = \widehat{\begin{bmatrix} z_1 \\ z_2 \\ z_3 \end{bmatrix}} = \widehat{\begin{bmatrix} \frac{\omega_1}{I_1} \\ \frac{\omega_2}{I_2} \\ \frac{\omega_3}{I_3} \end{bmatrix}} = \begin{pmatrix} 0 & -\frac{\omega_3}{I_3} & \frac{\omega_2}{I_2} \\ \frac{\omega_3}{I_3} & 0 & -\frac{\omega_1}{I_1} \\ -\frac{\omega_2}{I_2} & \frac{\omega_1}{I_1} & 0 \end{pmatrix}.
```

[^2]: This matrix is equivalent to the diagonal entries of the *coefficient of inertia matrix* in [holm2009geometric](@cite).

and obtain via the Euler-Poincaré equations[^4] for M[eq:KineticEnergyForLieGroup]m(@latex): 
```math
\widehat{I\dot{\omega}} = [\widehat{I\omega}, W],
```

[^4]: For the Euler-Poincaré equations we have to compute variations of M[eq:KineticEnergyForLieGroup]m(@latex) with respect to ``\delta{}W = \delta(\dot{Q}Q^{-1}) = \dot{\Sigma} - [W,\Sigma]`` where ``\Sigma := \delta{}QQ^{-1}``. For more details on this see [holm2009geometric](@cite).

or equivalently:
```math
\frac{d}{dt}\begin{bmatrix} z_1 \\  z_2 \\ z_3  \end{bmatrix} 
= \begin{bmatrix} \mathfrak{a}z_2z_3 \\ \mathfrak{b}z_1z_3 \\ \mathfrak{c}z_1z_2 \end{bmatrix}.
\label{eq:FinalRigidBodyEquations}
```

In the above equation, we defined ``\mathfrak{a} := I_3^{-1} - I_2^{-1}``, ``\mathfrak{b} := I_1^{-1} - I_3^{-1}`` and ``\mathfrak{c} := I_2^{-1} - I_1^{-1}``. In all of the examples, we set ``I_1 = 1``, ``I_2 = 2`` and ``I_3 = 2/3``, thus yielding ``\mathfrak{a} = 1``, ``\mathfrak{b} = -1/2`` and ``\mathfrak{c} = -1/2``.
