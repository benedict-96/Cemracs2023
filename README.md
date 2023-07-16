# CEMRACS 2023 project
@all feel free to change and amend these points throughout our project!

## Here's a rough outline (subject to change):

- The best outcome would be to beat LSTM on time series data! (If not we should at least significantly beat ResNets).

1. Make sure everyone can run transformer scripts on their laptops. Possibly on GPU. 
    
2. Experiment with hyperparameters on a simple data set (i.e. MNIST). Ideally the transformer should perform much better than a feedforward neural network (or even LSTM) while only needing a few layers. Experiment a bit with the hyperparameters - initial tests have shown that the result after one epoch should be around 50%, the specific set of hyperparameters may not be appropriate. Hyperparameters that should be changed are (i) the depth of the network $L$ (ii) the learning rate (iii) `use_bias` (iv) `add_connection` (v) `use_average` (vi) `use_softmax` (v) `patch_length`.
    
3. Are multihead-attention layers with weights on the Stiefel manifold universal approximators? (Probably not!) Invistigate this and find their approximation capabilities. Do we need to append another feedforward neural network or can we do without? (We probably need it as the multihead attention is a sort of preprocessing).

4. Investigate the reason for the *prioritization* of certain aspects of the input data by self-attention! For MNIST this is probably because certain features (curls, curve, etc.) appearing together imply a specific number. How can a transformer encode this correlation for the entire data set? 

5. Based on this we should think about what physical system to train transformers on. I.e. which features of the orbit(s) are particularly well-suited for being used with transformers?


6. Apply all of this to a physical system, should also be of dimension $\approx50$.(Maybe this also works with systems of lower dimension, i.e. 6 dimensions with 3 heads. An example would be $H = \sum_{i=1}^6\frac{p_i^2}{2} + U(q^1, \ldots, q^6)$ and take e.g. harmonic oscillators with mild coupling for $U$.) We could also try to integrate the same system with SympNets.


7. compare the new neural-network based <integrators> with traditional ones (i.e. ResNets).

### A multihead attention layer looks like this: 

$$ X \mapsto [P^V_1X\sigma((P^K_1X)^TP^Q_1X), \ldots, P^V_{N_h}X\sigma((P^K_{N_h}X)^TP^Q_{N_h}X)], $$

where $\sigma$ is the softmax function and is applied element-wise. The idea behind this is that the network can learn to prioritize certain aspects of the input data.
