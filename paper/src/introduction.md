# Introduction

This work addresses a problem in scientific machine learning [baker2019workshop](@cite) whose motivation comes from two trends and one observation: 
    
The first trend is using neural networks to identify dynamics of models for which data are available, but the underlying differential equation is either (i) not known or (ii) too expensive to solve. The first problem (i) often occurs when dealing with experimental data (see [chen2021discovering](@cite) and [goyal2021lqresnet](@cite)); the second one (ii) is crucial in *reduced-order modeling* (this will be elaborated on below). 

The second trend is a gradual replacement of hitherto established neural network architectures by transformer neural networks; the neural networks that are replaced are primarily recurrent neural networks such as long short-term memory networks (LSTMs, see [hochreiter1997long](@cite)) that treat time series data, but also convolutional neural networks (CNNs) for image recognition (see [dosovitskiy2020image](@cite)).

The observation mentioned at the beginning of this section is the importance of including information about the physical system into a machine learning model. In this paper the physical property we consider is *volume preservation* (see [arnold1978mathematical, bishop1980tensor, hairer2006geometric, leimkuhler2004simulating](@cite)), which is a property of the flow divergence-free vector fields. There are essentially two approaches through which this property can be accounted for: the first one is the inclusion of terms in the loss function that penalize non-physical behaviour; these neural networks are known as physics-informed neural network (PINNs[^1], see [raissi2019physics](@cite)). The other approach, and the one taken here, is to hard-code physical properties into the network architecture; a related example of this are ``symplectic neural networks'' (SympNets, see [jin2020sympnets](@cite)).  

[^1]: See [buchfink2023symplectic](@cite) for an application of PiNNs that considers symplecticity.

An application, where all of these trends and observations come into play, is *reduced-order modeling* (see [lee2020model, lassila2014model, fresca2021comprehensive](@cite)), which is used in the following setting: Suppose we are given a high-dimensional parametric ordinary differential equation (PODE), also referred to as the full order model (FOM), obtained from discretizing a parametric partial differential equation (PPDE). Typically the high-dimensional PODE  needs to be solved for many different parameter instances (see [lassila2014model, fresca2021comprehensive](@cite)), resulting in prohibitively expensive computational costs.
In order to alleviate the computational cost involved in solving the high-dimensional PODE many times, a reduced model is built based on training data. A typical reduced-order modeling framework consists of three stages: 
    
1. Solving the FOM for a limited range of parameter values to obtain *training data*. 
2. Constructing two maps, called *reduction* and *reconstruction*, that map from the FOM space to a reduced space of much smaller dimension and from this reduced space to the FOM space respectively. This is referred to as the *offline stage*.   
3. Solving the reduced model. This is a PODE of much smaller dimension. This step is referred as the *online stage*.

This work addresses the third step in this framework, the *online stage*, with a neural network based approach inspired by the trends and observations mentioned above. Efforts to use neural networks for the online stage have been made by other authors, for example using long short-term memory networks [lee2020model, fresca2021comprehensive](@cite).[^2] Above we described the gradual replacement of LSTMs by transformers as the second trend, which we also follow here.

[^2]: Apart from neural networks, there are also other approaches to alleviate the cost during the online stage, notably the "discrete empirical interpolation method" (DEIM, [chaturantabut2010nonlinear](@cite)).

We should note that transformer neural networks have also been used for the online stage in reduced order modeling, notably in [hemmasian2023reduced, solera2023beta](@cite), but in these cases the vanilla transformer was applied without taking physical properties of the system into account, which means that these approaches do not fall into one category with e.g. SympNets. Motivated by reduced-order modeling for Hamiltonian systems (see [peng2016symplectic, tyranowski2019symplectic, buchfink2023symplectic](@cite)) this work aims at imbuing a transformer with structure-preserving properties (i.e. volume-preservation) to be applied to a system described by a divergence-free vector field. 

The rest of this paper is structured as follows: ...