# CEMRACS 2023 project

## Here's a rough outline (subject to change):

- The best outcome would be to beat LSTM on time series data! (If not we should at least significantly beat ResNets).

- Make sure everyone can run transformer scripts on their laptops. Possibly on GPU. 
    
- Experiment with hyperparameters on a simple data set (i.e. MNIST). Ideally the transformer should perform much better than a feedfofward neural network (or even LSTM) while only needing a few layers. 
        A big success at this stage would be if we could demonstrate these superior properties (maybe even compare with LSTMs) for only a few multihead attention layers and no normalization and no addition. 
    
- Apply all of this to a physical system, should also be of dimension $\approx50$.\footnote{Maybe this also works with systems of lower dimension, i.e. 6 dimensions with 3 heads.}

- Are multihead-attention layers with weights on the Stiefel manifold universal approximators? (Probably not!) Invistigate this and find their approximation capabilities. Do we need to append another feedforward neural network or can we do without?

- Investigate the reason for this \textit{prioritization} of the certain aspects of the input data by self-attention! For MNIST this is probably because certain features (curls, curve, etc.) appearing together imply a specific number.

- compare the new neural-network based ``integrators'' with traditional ones (i.e. ResNets).

### A multihead attention layer looks like this: 

$$ X \mapsto [P^V_1X\sigma((P^K_1X)^TP^Q_1X), \ldots, P^V_{N_h}X\sigma((P^K_{N_h}X)^TP^Q_{N_h}X)], $$

where $\sigma$ is the softmax function and is applied element-wise. The idea behind this is that the network can learn to prioritize certain aspects of the input data.
