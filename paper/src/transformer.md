# The transformer

The transformer architecture [vaswani2017attention](@cite) was originally motivated by natural language processing (NLP) tasks and has quickly come to dominate that field. The "T" in ChatGPT (see e.g. [achiam2023gpt](@cite)) stands for "Transformer" and transformer models are the key element for generative AI. These models are a type of neural network architecture designed to process sequential input data, such as sentences or time-series data. The transformer has replaced, or is in the process of replacing, earlier architectures such as long short term memory (LSTM) networks [graves2012long](@cite) and other recurrent neural networks (RNNs, see [rumelhart1985learning](@cite)). The transformer architecture is visualized in M[fig:TransformerArchitecture]m(@latex)[^1]. As the output of the transformer is of the same dimension as the input to the transformer, we can stack *transformer units* on top of each other; the number of *stacked units* is described by the integer "L". In essence, the transformer consists of a residual network (ResNet[^2]) [he2016deep](@cite) and an attention layer. We describe these two core components in some detail.

[^1]: The three arrows going into the multihead attention module symbolize that the input is used three times: twice when computing the correlation matrix ``C`` and then again when the input is re-weighted based on ``C``. In the NLP literature those inputs are referred to as "queries", "keys" and "values" [vaswani2017attention](@cite).

[^2]: ResNets are often used because they improve stability in training [he2016deep](@cite) or make it possible to interpret a neural network as an ODE solver [chen2018neural](@cite).

## Residual Neural Networks

In its simplest form a ResNet is a standard feedforward neural network with an *add connection*:
```math
\mathrm{ResNet}: z \rightarrow z + \mathcal{NN}(z),
```
where ``\mathcal{NN}`` is any feedforward neural network. In this work we use a version where the ResNet step is repeated `n_blocks` times (also confer M[fig:VolumePreservingFeedForward]m(@latex)), i.e. we have
```math
    \mathrm{ResNet} = \mathrm{ResNet}_{\ell_\mathtt{n\_blocks}}\circ\cdots\circ\mathrm{ResNet}_{\ell_2}\circ\mathrm{ResNet}_{\ell_1}.
```
Further, one ResNet layer is simply ``\mathrm{ResNet}_{\ell_i}(z) = z + \mathcal{NN}_{\ell_i}(z) = z + \sigma(W_iz + b_i)`` where we pick tanh as activation function ``\sigma.`` 

## The Attention Layer

The attention layer, which can be seen as a preprocessing step to the ResNet, takes a series of vectors ``z^{(1)}_\mu, \ldots, z^{(T)}_\mu`` as input (the ``\mu`` indicates a specific time sequence) and outputs a *learned convex combination of these vectors*. So for a specific input: 
```math
input = Z = [z_\mu^{(1)}, z_\mu^{(2)}, \ldots, z_\mu^{(T)}],
```
the output of an attention layer becomes:
```math
output = \left[ \sum_{i=1}^Ty^{(1)}_iz_\mu^{(i)}, \sum_{i=1}^Ty^{(2)}_iz_\mu^{(i)}, \ldots, \sum_{i=1}^Ty^{(T)}_iz_\mu^{(i)} \right] ,
\label{eq:StandardTransformerOutput}
```
with the coefficients ``y_{i}^{(j)}`` satisfying ``\sum_{i=1}^Ty^{(j)}_i = 1 \; \forall{} j=1,\ldots,T``. It is important to note that the mapping 
```math
input \mapsto \left([y_i^{(j)}]_{i=1,\ldots,T,j=1,\ldots,T}\right)
```
is nonlinear. These coefficients are computed based on a correlation of the input data and involve learnable parameters that are adapted to the data during training.

The correlations in the input data are computed through a *correlation matrix* : ``Z \rightarrow Z^T A Z =: C``; they are therefore determined by computing weighted scalar products of all possible combinations of two input vectors where the weighting is done with ``A``; any entry of the matrix ``c_{ij}=(z^{(i)}_\mu)^TAz^{(j)}_\mu`` is the result of computing the scalar product of two input vectors. So any relationship, short-term or long-term, is encoded in this matrix.

After having obtained ``C``, a softmax function is applied column-wise to ``C`` and returns the following output: 
```math
y_i^{(j)} = [\mathrm{softmax}(C)]_{ij} := e^{c_{ij}}/\left(\sum_{i'=1}^Te^{c_{i'j}}\right).
```
This softmax function maps the correlation matrix to a sequence of *probability vectors*, i.e., vectors in the space ``\mathcal{P}:=\{\mathbf{y}\in[0,1]^d: \sum_{i=1}^dy_i = 1\}``. Every one of these ``d`` probability vectors is then used to compute a convex combination of the input vectors ``[z_\mu^{(1)}, z_\mu^{(2)}, \ldots, z_\mu^{(T)}]``, i.e., we get ``\sum_{i=1}^Ty_i^{(j)}z_\mu^{(i)}`` for ``j=1,\ldots,T``. Note that we can also write the convex combination of input vectors as:

```math
output = input\Lambda = Z\Lambda,
\label{eq:RightMultiplication}
```
where ``\Lambda = \mathrm{softmax}(C).`` So a linear recombination of input vectors can be seen as a multiplication by a matrix from the right.

Despite its simplicity, the transformer exhibits vast improvements compared to RNNs and LSTMs, including the ability to better capture long-range dependencies and contextual information and its near-perfect parallelizability for computation on GPUs and modern hardware.
Furthermore, the simplicity of the transformer architecture makes it possible to interpret all its constituent operations, which is not as easily accomplished when using LSTMs, for example. As the individual operations have a straight-forward mathematical interpretation, it is easier to imbue them with additional structure such as volume-preservation.

REMARK::
M[fig:TransformerArchitecture]m(@latex) indicates the use of a *multi-head attention layer* as opposed to a *single-head attention layer*. What we described in this section is single-head attention. A multi-head attention layer is slightly more complex: it is a concatenation of multiple single-head attention layers. This is useful for NLP tasks[^3], but introduces additional complexity that makes it harder to imbue the multi-head attention layer with structure-preserving properties. For this reason we stick to single-head attention in this work.::


[^3]: Intuitively, multi-head attention layers allow for attending to different parts of the sequence in different ways (i.e. different heads in the multi-head attention layer *attend to* different parts of the input sequence) and can therefore extract richer contextual information.

```@raw latex
\begin{figure}[h]
\includegraphics[width = .25\textwidth]{tikz/transformer.png}
\caption{Sketch of the transformer architecture. It is a composition of an attention layer and a feedforward neural network. The first \textit{add connection} is drawn in green to emphasize that this can be left out. The Integer ``L'' indicates how often a \textit{transformer unit} (i.e. what is enclosed within the big black borders) is repeated.}
\label{fig:TransformerArchitecture}
\end{figure}
```