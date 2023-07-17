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
- nonlinear approximation: $V_n = \{ \sum_{i=1}^nc_iv_i : c_i\in\mathbb{R}, v_i\in\mathcal{D} \}$
- nonlinear parametric manifold: $V_n = \{ D(c): c\in\mathbb{R}^n \}$ (a nonlinear approximation)

With encoder/decoder terminology: 
$\mathrm{decoder}: \mathbb{R}^n\to{}V$ and $\mathrm{encoder}: V\to\mathbb{R}^n$. The solution manifold is the image of the decoder.

Error of best approximation: 

$$ 
e_n(u) = \mathrm{inf}_{v\in{}V_n}|| u - v ||.
$$

Questions regarding the approximation: 
- **universality**: Does $e_n(u) \to 0$ as $n\to\infty \forall u\in{}V$.
- **expressivity**: 
- Characterize the class of functions for which a certain convergence is achieved (approximation class): $\mathcal{A}^\gamma = \{ u\in{}V: \mathrm{sup}_{n\geq{}1}\gamma(n)e_n(u) < \infty \}$ for some growth function $\gamma$. 
- **algorithm**: ????

Quality of encoding procedure is measures by $\mathcal{E}^{WC}(\mathcal{K};(E,D))$ (maximum over $v\in\mathcal{K}$) or $\mathcal{E}^{av}(\mathcal{K};(E,D))$.
In connection with this define the *n-width* (generalization of Kolmogorov n-width).

- Approximation number: everything is linear.
- Kolmogorov n-width: $D$ linear, $E$ arbitrary. 