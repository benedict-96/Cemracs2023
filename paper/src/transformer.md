# The Transformer

The transformer architecture [vaswani2017attention](@cite) was originally motivated by natural language processing (NLP) tasks and has quickly come to dominate that field. The ``T`` in chatGPT stands for "Transformer" and transformer models are the key element for generative AI. These models are a type of neural network architecture designed to process sequential input data, such as sentences or time-series. The transformer has replaced, or is in the process of replacing, earlier architectures such as long short term memory (LSTM) networks [graves2012long](@cite) and other recurrent neural networks (RNNs, see [rumelhart1985learning](@cite)).

The transformer architecture is essentially a combination of an attention layer and a residual net (ResNet[^1],see [he2016deep](@cite)). This architecture is visualized in figure **???**[^2].
The improvements compared to RNNs and LSTMs include the ability to better capture long-range dependencies and contextual information and its near-perfect parallelizability for computation on GPUs and modern hardware.

[^1]: The simplest form of a ResNet is a regular feed-forward neural network with an add connection: ``x \rightarrow x + \sigma (Wx + b)``.

[^2]: The three arrows going into the multihead attention module symbolize that the input is used three times: twice when computing the correlation matrix ``C`` and then again when the input is reweighted based on ``C``. In the NLP literature those inputs are referred to as "queries", "keys" and "values" [vaswani2017attention](@cite).

![](tikz/transformer.png)

The attention layer, the first part of a transformer layer, takes a series of vectors ``z^{(1)}_\mu, \ldots, z^{(T)}_\mu`` as input (the ``\mu`` indicates a specific time sequence) and outputs a *learned convex combination of these vectors*. So for a specific input: 
```math
input = [z_\mu^{(1)}, z_\mu^{(2)}, \ldots, z_\mu^{(T)}],
```

we get the following for the output of an attention layer:
```math
output = \left[ \sum_{i=1}^Ty^{(1)}_iz_\mu^{(i)}, \sum_{i=1}^Ty^{(2)}_iz_\mu^{(i)}, \ldots, \sum_{i=1}^Ty^{(T)}_iz_\mu^{(i)} \right] .
```

With all the coefficients satisfying ``\forall{}j=1,\ldots,T:\sum_{i=1}^Ty^{(j)}_i = 1``. It is important to note that the mapping 

```math
input \mapsto \left([y_i^{(j)}]_{i=1,\ldots,T,j=1,\ldots,T}\right)
```

is nonlinear. These coefficients are computed based on a correlation of the input data and involve learnable parameters that are changed during training.

The correlation in the input data is computed through a *correlation matrix* : ``Z \rightarrow Z^T A Z =: C``. The correlations in the input data are therefore determined by computing weighted scalar products of all possible combinations of two input vectors where the weighting is done with ``A``; any entry of the matrix ``c_{ij}=(z^{(\mu)}_i)^TAz^{(\mu)}_j`` is the result of computing the scalar product of two input vectors. So any relationship, short-term or long-term, is encoded in this matrix.

In the next step, a softmax function is applied column-wise to $C$ and returns the following output: 
```math
y_i^{(j)} = [\mathrm{softmax}(C)]_{ij} := e^{c_{ij}}/\left(\sum_{i'=1}^Te^{c_{i'j}}\right).
```
This softmax function maps the correlation matrix to the a sequence of *probability vectors*, i.e. vecors in the space ``\mathcal{P}:=\{\mathbf{y}\in[0,1]^d: \sum_{i=1}^dy_i = 1\}``. Every one of these $d$ probability vectors is then used to compute a convex combination of the input vectors ``[z_\mu^{(1)}, z_\mu^{(2)}, \ldots, z_\mu^{(T)}]``, i.e. we get ``\sum_{i=1}^Ty_i^{(j)}z_\mu^{(i)}`` for ``j=1,\ldots,T``.

<!--
\begin{remark}
\Cref{fig:transformer} indicates the use of a *multi-head attention layer* as opposed to a *single-head attention layer*. What we described in this section is single-head attention. A multi-head attention layer is slightly more complex: it is a concatenation of multiple single-head attention layers. This is useful for NLP tasks[^2] but introduces additional complexity that makes it harder to imbue the multi-head attention layer with structure-preserving properties.
\end{remark}

[^2]: Intuitively, multi-head attention layers allow for attending to different parts of the sequence in different ways (i.e. different heads in the multi-head attention layer *attend to* different parts of the input sequence) and can therefore extract richer contextual information.
-->