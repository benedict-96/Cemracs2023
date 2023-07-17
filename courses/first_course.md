# Linear and Nonlinear Schemes for Forward Model Reduction and Inverse Problems 

## first lecture

Consider Parametric PDEs: 
- $\mathcal{B}(u;\vartheta) = 0$
- $u\in{}V$ a Hilbert/Banach space, 
- $\vartheta\in\Theta$ is compact (parameter space).

Example: 

$$ -\vartheta\Delta{}u = f, $$

where $\vartheta\in[\vartheta_\mathrm{min}, \vartheta_\mathrm{max}]$ is a diffusion parameter. Then we have the "parameter-to-solution map":

$$ \Theta \to V, \vartheta \mapsto u(\vartheta)\in{}V $$ 

and the solution manifold: $\mathcal{M} = \{ u(\vartheta)\in{}V:\vartheta\in\Theta \}$.

### Approximation (with neural networks)

- linear approximation by a *finite-dimensional linear space*: $V_n = \mathrm{span}(v_i)_{i=1}^n$. 
- nonlinear approximation: $V_n = \left{ \sum_{i=1}^nc_iv_i : c_i\in\mathbb{R}, v_i\in\mathcal{D} \right}$
- nonlinear parametric manifold: $V_n = \{ D(c): c\in\mathbb{R}^n \}$ (a nonlinear approximation)

