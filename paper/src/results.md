# Experimental results

We now compare three different neural network architectures that we all train on the data coming from a [rigid body](rigid_body.md). Those architectures are ...
1. a volume-preserving feedforward neural network,
2. a volume-preserving transformer,
3. a regular transformer. 

## Loss functions 

For training the feedforward neural network and the transformer we pick similar loss functions. In both cases they take the form: 

```math 
L_{\mathcal{NN}}(\mathtt{input}, \mathtt{output}) = ||\mathtt{output} - \mathcal{NN}(\mathtt{input})||_2/||\mathtt{output}||_2,
```

where ``||\cdot||_2`` is the ``L_2``-norm. The only difference between the two losses (for the feedforward neural network and the transformer) is that the `input` and `output` are vectors ``\in\mathbb{R}^d`` in the first case and matrices ``\in\mathbb{R}^{d\times{}T}`` in the second. 

```@raw latex
\begin{figure}
\includegraphics[width = .5\textwidth]{simulations/vpt_Float64/validation_3.png}
\caption{The first component $x$ plotted for the time interval [0, 14].}
\label{fig:Validation}
\end{figure}
```

```@raw latex
\begin{figure}
\includegraphics[width = .5\textwidth]{simulations/vpt_Float64/training_loss_3.png}
\caption{Training loss for the different networks.}
\label{fig:TrainingLoss}
\end{figure}
```

```@raw latex
\begin{figure}
\includegraphics[width = .5\textwidth]{simulations/vpt_Float64/validation3d_3.png}
\caption{Validation plot in 3d.}
\label{fig:Validation3d}
\end{figure}
```

As is shown in figure m[fig:Validation]m(@latex) the volume-preserving feedforward network manages to predict the time evolution of the rigid body up to a certain point, but then drifts off. The volume-preserving feedforward transformer manages to stay close to the numerical solution much better. It also outperforms the regular transformer while using fewer parameters. 



## Why does regular attention fail? 

We can see in m[fig:Validation]m(@latex) that the regular transformer looks like a step function, i.e. it predicts a step and then stays there for another two steps before going to a different value again. 

To see how this can happen we look at the input of the attention layer: 

```math
\left[\begin{matrix}
(z^{(1)})^TAz^{(1)} & (z^{(1)})^TAz^{(2)} & (z^{(1)})^TAz^{(3)} \\ 
(z^{(2)})^TAz^{(1)} & (z^{(2)})^TAz^{(2)} & (z^{(2)})^TAz^{(3)} \\ 
(z^{(3)})^TAz^{(1)} & (z^{(3)})^TAz^{(2)} & (z^{(3)})^TAz^{(3)}
\end{matrix}\right] =: \left[\begin{matrix} p^{(1)} & p^{(2)} & p^{(3)} \end{matrix}\right].
\label{eq:ScalarProductResult}
```

The output of the attention layers then is: 

```math
\left[\begin{matrix} \mathrm{softmax}(p^{(1)}) & \mathrm{softmax}(p^{(2)}) & \mathrm{softmax}(p^{(3)}) \end{matrix}\right].
```

The results in figure m[fig:Validation]m(@latex) seem as if ``\mathrm{softmax}(p^{(i)})`` is the same regardless of the integer ``i = 1, 2, 3``. For the volume-preserving attention mechanism introduced in this work this can never happen as the three columns are not treated independently: the result is always a matrix with three independent columns that has determinant 1 or -1[^1]. 

[^1]: By investigating the weight matrix of the attention layer further, we see that the output of m[eq:ScalarProductResult]m(@latex) is a matrix whose entries are all roughly the same. 

__Put a few images to proof this!__

The minimum is achieved when all scalar products map to the same value. Maybe this is because there are not enough degrees of freedom to make more complicated mappings. 