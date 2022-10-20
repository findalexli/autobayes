Report(https://github.com/findalexli/autobayes/blob/main/report.pdf)


To increase access to causal learning without having to manually choose free
hyperparameters, we introduced Bayesian Optimization based techniques that treats each
causal model as a set of predictive models from each node’s Markov Blanket. We demonstrated
higher performance than out-of-the-box NOTEARS-MLP algorithm using stimulated datasets.
The code is available at https://github.com/findalexli/autobayes

Related Work
A key innovative in causal inference was transforming the discrete DAG constraints into a
smotth function proposed by Zheng 2018, namely the NOTEARS algorithm. The authors conver
the combinatorial acyclic constraints on the DAG into a constrained continuous optimization
problem. (Zheng et al. 2018). NOTEARS states a differentiable object function that can be
optimized via common optimization techniques. More recently, Lachapelle et al. proposed to
learn the DAG from the weights of a neural network while still considering the DAG constraints.
(Lachapelle et al. 2019). Zheng et al. further extended the DAG learning into a nonparametric
problem, proposing a generic formula of the DAG learning as E[X_j | X_parentj] = g_j (f_j ())
where f_j(u1, … uj) depends only on the set parents. As a example, they proposed a Multilayer
Layer Perception model.
While NOTEARS and its descendants proposed a wide range of algorithms, there are
regularization parameters and weight cutoff parameters left to be tuned. There have been
several techniques proposed to tune causal discovery networks, including StARs method (Liu et
al. 2010) introduced to tune the lambda penalty hyper-parameter for graphical lasso (Friedman
et al. 20080. Another framework to select model configuration based on in-sample fitting of the
data and model complexity has been introduced to causal disvoery tuning, namely the Bayesian
Information Criterion(BIC) (Maathuis et al. 2009). Most recently, a Out-of-Sample Causal Tuning
techniques (OCT) was introduced that treats a causal model as a set of predictive model, one
for each nodes given its Markov Blanket, then tune the choices using cross-validation (Biza et
al. 2011).


In this work, our main contribution is
- Extension of the OCT technique to suit a non-parametric optimization presented by
NOTEARS-MLP. We are able to tune four free parameters in the NOTEARS-MLP setup:
L1 regularization parameter, L2 regularization parameter, Number of Nodes in the
Hidden Layer and weight cutoff.
- While the OCT authors admitted limitation on their algorithm by inability to account for
false positives, we introduced additional penality term for weakly correlated features in
the Markov Blanket

We implemented the following feature based on NO-TEARS-MLP: There are a number of free
hyper parameters pending user-choice. Published NOTEARS-MLP paper notably used fixed
parameters across graph types of weight threshold of 0.5, Lamba 1 and Lambda 2 = 0.03 and
[d, 10, 1] fully connected layers.
1. Weight Threshold: threshold to transform learned adjacency matrix to estimated edges
2. Lamba 1: L1 regularization constant
3. Lambda 2: L2 regularization
4. Multilayer-perceptron neural structure.
We setup the tuning experiments using Bayesian Optimization as experiment configuration
generalization techniques. A prior probability capture observed behavior of the objective
function, and is being updated to estimate the posterior distribution over the objective function.
We update the prior when we pick a combination of hyperparameter values at each iteration, i.e.
P(metric | hyperparameter combination), and stop when we meet preset stopping criteria. An
acquisition function is made from the estimated posterior distribution that would infer the next
configuration setup. This choice is motivated by extensive computer time.
