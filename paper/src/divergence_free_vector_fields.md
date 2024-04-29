# Divergence-free vector fields

The aim of this work is to adapt the transformer architecture to the setting of dynamical systems that are described by *divergence-free vector fields* on ``\mathbb{R}^{d}``, i.e. mappings ``f:\mathbb{R}^d\to\mathbb{R}^d`` for which ``\nabla\cdot{}f = \sum_i\partial_if_i = 0``. 
The *flow* of such a vector field ``f`` is the (unique) map ``\varphi_f^t:\mathbb{R}^d\to\mathbb{R}^d``, such that 
```math
\left. \frac{d}{dt} \right|_{t=t_0}\varphi^t(z) = f(\varphi^{t_0}(z)),
```
for ``z\in\mathbb{R}^d`` and ``t`` is a time step. For a divergence-free vector field ``f`` the flow ``\varphi_f^t`` is volume-preserving, i.e. the determinant ob the jacobian of ``\varphi_f^t`` is one: ``\mathrm{det}(\nabla\varphi_f^t) = 1``. This can easily be proved:
```math
\frac{d}{dt}\nabla\varphi^t(z) = \nabla{}f(\varphi^t(z))\nabla\varphi^t(z) \implies \mathrm{tr}\left( \left(\nabla\varphi^t(z) \right)^{-1} \frac{d}{dt}\nabla\varphi^t(z) \right) = \mathrm{tr}\left(\nabla{}f(\varphi^t(z))\right) = 0.
\label{eq:VolumePreservingFlows}
```
We further have 
```math
\mathrm{tr}(A^{-1}\dot{A}) = \frac{\frac{d}{dt}\mathrm{det}(A)}{\mathrm{det}(A)} 
```
and therefore
```math
\frac{d}{dt}\mathrm{det}\left( \nabla\varphi^t(z) \right) = 0.
```
In m[eq:VolumePreservingFlows]m(@latex) we have also used that the flow of the ODE is invertible.

Numerical integrators for ODEs now always aim at approximating the flow ``\varphi_f^t`` with an integration scheme ``\psi^h`` with a time step ``h``. Most of the time this time step ``h`` is fixed. If the flow ``\varphi_f^t`` exhibits certain properties (like volume preservation) it makes sense to also imbue ``\psi^h`` with these properties. The discipline of doing so is generally known as "geometric numerical integration" (see [hairer2006geometric, leimkuhler2004simulating](@cite)).

In recent years numerical integrators based on neural networks have emerged and it has proven crucial to also imbue these integrators with properties of the system such as symplecticity (see [jin2020sympnets, greydanus2019hamiltonian](@cite)) and volume preservation (see [bajars2023locally](@cite)). The neural network architecture presented in [Volume-preserving transformer](@ref) falls in this category. 

Let us note that all symplectic and Hamiltonian vector fields[^1] are also divergence-free. Symplecticity, however, is a much stronger property; so preserving this property is preferable to just preserving volume. If a symplectic scheme is however not available, a volume-preserving one usually also offers improvements over one that does not respect any of the properties of the vector field.

[^1]: Strictly speaking Hamiltonian vector fields form a subspace of the space of all symplectic vector fields (see [bishop1980tensor](@cite)).
