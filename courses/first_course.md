# Linear and Nonlinear Schemes for Forward Model Reduction and Inverse Problems 

Consider Parametric PDEs: 
- $\mathcal{B}(u;\vartheta) = 0$
- $u\in{}V$ a Hilbert/Banach space, 
- $\vartheta\in\Theta$ is compact (parameter space).

Example: 
$$ -\vartheta\Delta{}u = f, $$
where $\vartheta\in[\vartheta_\mathrm{min}, \vartheta_\mathrm{max}]$ is a diffusion parameter. Then we have the "parameter-to-solution map":
$$ \Theta \to V, \vartheta \mapsto u(\vartheta)\in{}V $$ and the solution manifold: $\mathcal{M} = \{ u(\vartheta)\in{}V:\vartheta\in\Theta \}$.