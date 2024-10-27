# Introduction

This work is concerned with the development of accurate and robust neural network architectures for the solution of differential equations.
It is motivated by several trends that have determined the focus of research in scientific machine learning [baker2019workshop](@cite) in recent years.

First, machine learning techniques have been successfully applied to the identification of dynamical systems for which data are available, but the underlying differential equation is either (i) not known or (ii) too expensive to solve. The first problem (i) often occurs when dealing with experimental data [chen2021discovering, goyal2021lqresnet](@cite); the second one (ii) is crucial in *reduced-order modeling* as will be elaborated on below. Various machine learning models have been shown to be able to capture the behaviour of dynamical systems accurately and (within their regime of validity) make predictions at much lower computational costs than traditional numerical algorithms [baker2019workshop](@cite).

Second, in certain application areas, hitherto established neural network architectures have been gradually replaced by transformer neural networks [vaswani2017attention](@cite). Primarily, this concerns recurrent neural networks such as long short-term memory networks (LSTMs [hochreiter1997long](@cite)) that treat time series data, e.g., for natural language processing, but also convolutional neural networks (CNNs) for image recognition [dosovitskiy2020image](@cite). Transformer networks tend to capture long-range dependencies and contextual information better than other architectures and allow for much more efficient parallelization on modern hardware.

Lastly, the importance of including information about the physical system in scientific machine learning models has been recognized. This did not come as a surprise as it is a finding that had long been established for conventional numerical methods [hairer2006geometric](@cite).
In this work, the physical property we consider is *volume preservation* [arnold1978mathematical, hairer2006geometric](@cite), which is a property of the flow of divergence-free vector fields. There are essentially two approaches through which this property can be accounted for. The first one is the inclusion of terms in the loss function that penalize non-physical behaviour; these neural networks are known as physics-informed neural networks (PINNs[^1] [raissi2019physics](@cite)). The drawback of this approach is that a given property is enforced only weakly and therefore only approximately satisfied. The other approach, and the one taken here, is to encode physical properties into the network architecture; a related example of this are "symplectic neural networks" (SympNets [jin2020sympnets](@cite)). While this approach is often more challenging to implement, the resulting architectures adhere to a given property exactly.

[^1]: See [buchfink2023symplectic](@cite) for an application of PINNs where *non-physical behavior* is penalized.

An application, where all of these developments intersect, is *reduced-order modeling* [lassila2014model, lee2020model, fresca2021comprehensive](@cite). Reduced-order modeling is typically split into an *offline phase* and an *online phase* (see M[fig:OfflineOnlineSplit]m(@latex)).

```@raw latex
\begin{figure}[h]
\centering
\includestandalone[width=.5\textwidth]{tikz/offline_online}
\caption{Visualization of offline-online split in reduced order modeling. The volume-preserving transformer presented in this work is used for the online phase, i.e. it is used to model $\mathcal{NN}$ in this figure. Here FOM stands for \textit{full-order model} and ROM stands for \textit{reduced-order model}.}
\label{fig:OfflineOnlineSplit}
\end{figure}
```

For the online phase we face the following challenges:
1. in many cases we need to recover the dynamics of our system from data alone (also known as "non-intrusive reduced order modeling" [lee2020model, fresca2021comprehensive, yildiz2024structure](@cite)), 
2. if the big system exhibits specific structure (such as volume-preservation) it is often crucial to also respect this structure in the reduced model [tyranowski2019symplectic, brantner2023symplectic](@cite).  

Our aim in this work is to construct structure-preserving neural network models, that can be used to compute the dynamics of reduced models. Efforts to use neural networks for the online stage have been made in the past, for example using LSTMs [lee2020model, fresca2021comprehensive](@cite).[^2] Inspired by one of the trends mentioned above, namely the gradual replacement of such architectures, our work will be based on transformers instead.

[^2]: Apart from neural networks, there are also other approaches to alleviate the cost during the online stage, notably the "discrete empirical interpolation method" (DEIM [chaturantabut2010nonlinear](@cite)).

While previously, other authors applied transformers for model reduction, and volume-preserving neural networks have been developed as well, both aspects have not yet been considered together.
Thus to our knowledge, this work is the first that aims at imbuing a transformer with structure-preserving properties[^3] (namely volume-preservation) and applying it to a system described by a divergence-free vector field. 
In the previous work of other authors, transformer neural networks have been used for the online stage of reduced order modeling (i.e. have been applied to dynamical systems) in [hemmasian2023reduced, solera2023beta](@cite). The authors applied the vanilla transformer architecture without taking any physical properties of the system into account.
Volume-preserving feedforward neural networks have been developed in [bajars2023locally](@cite). The authors based the network design on a theorem introduced in [kang1995volume](@cite) for the design of traditional integrators for divergence-free vector fields [hairer2006geometric](@cite). 

[^3]: In [guo2022transformer](@cite) the transformer is used in a "structure-conforming" way. In this approach the special architecture of the transformer is leveraged for approximating a solution to a partial differential equation; however no "structure" is preserved in the way we describe it here, i.e. the neural network-based discretization does not share features with the analytic solution of the differential equation (like volume preservation for example).

The remainder of this paper is structured as follows: In [Divergence-free vector fields](@ref) we discuss the basic theory behind divergence-free vector fields and volume-preserving flows, in [The transformer](@ref) the (standard) transformer is introduced, in [Volume-Preserving ResNets](@ref) we introduce a new class of volume-preserving feedforward neural networks (that differ slightly from what is discussed in e.g. [bajars2023locally](@cite)), in [The Volume-Preserving Transformer](@ref) we introduce our new adapted transformer and in [Experimental results](@ref) we finally present results of applying the volume-preserving transformer to a rigid body, an example of a divergence-free vector field (see [The Rigid Body](@ref)). In [Conclusion and future work](@ref) we summarize our findings and state potential future investigations to extend the work presented here.
