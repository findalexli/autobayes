from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math

from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from timeit import default_timer as timer
import math
import notears.utils as ut
import cdt
from metric import MetricsDAG
from pgmpy.models import BayesianModel
import networkx as nx
from sklearn.model_selection import KFold
from catboost import Pool, CatBoostRegressor
from sklearn.metrics import mean_squared_error
from ax.service.managed_loop import optimize

torch.set_default_dtype(torch.double)
np.set_printoptions(precision=3)
ut.set_random_seed(123)

from mlp import NotearsMLP, notears_nonlinear

#data, graph = cdt.data.load_dataset('sachs')

def graph_estimator(train, parameterization):
    torch.set_default_dtype(torch.double)

    d= len(train.columns) # TODO solve the d issue 
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    global W_est
    W_est = notears_nonlinear(model,
                          train.to_numpy(), # Change to train
                          lambda1=parameterization['lambda1'],
                          lambda2=parameterization['lambda2'],
                          max_iter=30,
                          h_tol=1e-8,
                          rho_max=1e+16,
                          w_threshold=parameterization['w_threshold']
    )
    W_est_binary = W_est !=0
    assert ut.is_dag(W_est)
    return W_est_binary

def train_evaluate_given_parameter_config(parameterization: dict):
    kfold = KFold(n_splits=2, shuffle=True)
    folds_score = []
    for train_idx, test_idx in kfold.split(data):
        train, test = data.iloc[train_idx, :], data.iloc[test_idx, :]

        # 1. Find the estiamte dgraph using inner fold training data
        weight_estimated = graph_estimator(train, parameterization)
        sm = nx.convert_matrix.from_numpy_array(W_est != 0)

        bn = BayesianModel()
        bn.add_edges_from(sm.edges)
        nodes = list(bn.nodes())
        # 2. For each node, train a regression tree model and record its performance
        score_list = []
        for node in nodes:
            markov_blanket = bn.get_markov_blanket(node)
            train_x, train_y = train.loc[:, markov_blanket], train.loc[:, node]
            test_x, test_y = test.loc[:, markov_blanket], test.loc[:, node]
            train_pool = Pool(train_x, train_y)

            model = CatBoostRegressor(iterations=50, verbose=False)
            model.fit(train_pool)
            mse = mean_squared_error(model.predict(test_x), test_y)
            print(f'MSE at Node {node} is = {mse}')
            score_list.append(mse)
        # 3. Average fold score of average score across nodes
        fold_score = np.mean(score_list)
        folds_score.append(fold_score)
    avg_folds_score = np.mean(folds_score)
    return avg_folds_score


def bayesian_optimize(data):
    best_parameters, best_values, experiment, model = optimize(
            parameters=[
              {
                "name": "lambda1",
                "type": "range",
                "bounds": [0.01, 0.5],
                "log_scale": True
              },
              {
                 "name": "lambda2",
                 "type": "range",
                 "bounds": [0.01, 0.5],
                 "log_scale": True
              },
              {
                  "name": "w_threshold",
                  "type": "range",
                  "bounds": [0.01, 0.5],
                  "log_scale": True
               }
            ],
            # Booth function
            evaluation_function=train_evaluate_given_parameter_config,
            minimize=True,
        )
    return best_parameters, best_values, experiment, model

# def data_given_index(index):
#     # TODO data generator
#     setup = {'n': 200, 'graph_type': 'ER', 'sem_type': 'mlp'}
#     d_list = [10, 20, 40]
#     er2_s0_d = [(2*d, d) for d in d_list]
#
#     tuple = er2_s0_d[index]
#     setup['s0'] = tuple[0]
#     setup['d'] = tuple[1]
#     W_true = ut.simulate_dag(setup['d'], setup['s0'], setup['graph_type'])
#     data = pd.DataFrame(ut.simulate_nonlinear_sem(graph, setup['n'], setup['sem_type']))
#     return data, W_true

def data_given_setup(setup):
    W_true = ut.simulate_dag(setup['d'], setup['s0'], setup['graph_type'])
    data = pd.DataFrame(ut.simulate_nonlinear_sem(W_true, setup['n'], setup['sem_type']))
    return data, W_true

def main():
    setup = {'n': 200, 'graph_type': 'ER', 'sem_type':'mlp'} # Default
    d_list = [10, 20, 30, 40]
    er2_s0_d = [(2*d, d) for d in d_list]
    # This is for 10, 20, 30, 40
    metrics_list = []
    parameters_list = []
    for tuple in er2_s0_d:
        setup['s0'] = tuple[0]
        setup['d'] = tuple[1]
        global data 
        data, W_true = data_given_setup(setup)
        best_parameters, best_values, _, _ = bayesian_optimize(data)
        # Take best parameter, and retrain with all data
        W_estimated = graph_estimator(data, best_parameters)
        metrics_list.append(MetricsDAG(W_estimated, W_true).metrics)
        parameters_list.append(best_parameters)
        del data 
    df = pd.DataFrame({'metrics': metrics_list, 'parameters': parameters_list, 'd_list': d_list})
    df.to_csv('Records.csv')

if __name__ == "__main__":
    main()
