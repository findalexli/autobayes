from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math
import sys,os
sys.path.append(os.getcwd())
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
import json
import sys
torch.set_default_dtype(torch.double)
np.set_printoptions(precision=3)
ut.set_random_seed(123)

from mlp_model import NotearsMLP, notears_nonlinear
from sklearn.linear_model import LinearRegression

#data, graph = cdt.data.load_dataset('sachs')

def graph_estimator(train, parameterization):
    torch.set_default_dtype(torch.double)

    d= len(train.columns)
    hidden_number_of_nodes = parameterization['nodes']
    model = NotearsMLP(dims=[d, hidden_number_of_nodes, 1], bias=True)
    global W_est
    W_est = notears_nonlinear(model,
                          train.to_numpy(), # Change to train
                          lambda1=parameterization['lambda1'],
                          lambda2=parameterization['lambda2'],
                          max_iter=100,
                          h_tol=1e-8,
                          rho_max=1e+16,
                          w_threshold=parameterization['w_threshold']
    )
    W_est_binary = W_est !=0
    return W_est_binary

def train_evaluate_no_csv_given_parameter_config(parameterization: dict):
    train=data.sample(frac=0.8,random_state=200) #random state is a seed value
    test=data.drop(train.index)

            # 1. Find the estiamte dgraph using inner fold training data
    weight_estimated = graph_estimator(train, parameterization)
    sm = nx.convert_matrix.from_numpy_array(weight_estimated != 0)

    bn = BayesianModel()
    bn.add_edges_from(sm.edges)
    nodes = list(bn.nodes())
    # 2. For each node, train a regression tree model and record its performance
    score_list = []
    for node in nodes:
        markov_blanket = bn.get_markov_blanket(node)
        if len(markov_blanket) <1:
                continue
        train_x, train_y = train.loc[:, markov_blanket], train.loc[:, node]
        test_x, test_y = test.loc[:, markov_blanket], test.loc[:, node]
            
        model = LinearRegression()
        model.fit(train_x, train_y)
        print(f'Coefficients = {model.coef_}')
        rmse = mean_squared_error(model.predict(test_x), test_y, squared=False)
        # Get number of coefficients 
        weak_correlated_count = (np.abs(model.coef_) < 0.15).sum()
        adjusted_rmse = rmse + 0.3 * weak_correlated_count
        print(f'Weak correlated count {weak_correlated_count}')
        score_list.append(adjusted_rmse)
        print(f'MSE at Node {node} is = {rmse}')
    if len(score_list) <=1:
        return None
    avg_score = np.mean(score_list)
    print('Average Score across nodes {avg_score}')
    return avg_score

def train_evaluate_given_parameter_config(parameterization: dict):
    kfold = KFold(n_splits=3, shuffle=True)
    folds_score = []
    for train_idx, test_idx in kfold.split(data):
        train, test = data.iloc[train_idx, :], data.iloc[test_idx, :]

        # 1. Find the estiamte dgraph using inner fold training data
        weight_estimated = graph_estimator(train, parameterization)
        sm = nx.convert_matrix.from_numpy_array(weight_estimated != 0)

        bn = BayesianModel()
        bn.add_edges_from(sm.edges)
        nodes = list(bn.nodes())
        # 2. For each node, train a regression tree model and record its performance
        score_list = []
        for node in nodes:
            markov_blanket = bn.get_markov_blanket(node)
            if len(markov_blanket) <1:
                continue
            train_x, train_y = train.loc[:, markov_blanket], train.loc[:, node]
            test_x, test_y = test.loc[:, markov_blanket], test.loc[:, node]
            
            LinearRegression
            model = LinearRegression()
            model.fit(train_x, train_y)
            print(f'Coefficients = {model.coef_}')
            rmse = mean_squared_error(model.predict(test_x), test_y, squared=False)
            print(f'MSE at Node {node} is = {rmse}')
            
            # Get number of coefficients 
            weak_correlated_count = (np.abs(model.coef_) < 0.15).sum()
            adjusted_rmse = rmse + 0.3 * weak_correlated_count
            print(f'Weak correlated count {weak_correlated_count}')
            score_list.append(adjusted_rmse)
        # 3. Average fold score of average score across nodes
        fold_score = np.mean(score_list)
        folds_score.append(fold_score)
    avg_folds_score = np.mean(folds_score)
    print(f'Average fold scores {avg_folds_score}')
    return avg_folds_score


def bayesian_optimize(data, cross_validated):
    if cross_validated:
        eval_function = train_evaluate_given_parameter_config
    else:
        eval_function = train_evaluate_no_csv_given_parameter_config
    best_parameters, best_values, experiment, model = optimize(
            parameters=[
              {
                "name": "lambda1",
                "type": "fixed",
                "value": 0.03,
              },
              {
                 "name": "lambda2",
                 "type": "fixed",
                 "value": 0.03,
              },
              {
                  "name": "w_threshold",
                  "type": "range",
                  "bounds": [0.01, 0.5],
                  "log_scale": False
               }, 
               {
                  "name": "nodes",
                  "type": "fixed",
                  "value": 10,
               }
            ],
            # Booth function
            evaluation_function=eval_function,
            total_trials=10,
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

def main(stimulated =True, test=False, cross_validated=False):
    if stimulated: 

        setup = {'n': 200, 'graph_type': 'ER', 'sem_type':'gp'} # Default
        setup['sem_type'] = sys.argv[1]
        setup['er_multiplier'] = int(sys.argv[2])
        d_list = [10, 20, 30, 40]
        er_s0_d = [(setup['er_multiplier']*d, d) for d in d_list]
        # This is for 10, 20, 30, 40
        metrics_list = []
        parameters_list = []
        if test:
            er_s0_d = [(20, 10)]

        for tuple in er_s0_d:
            setup['s0'] = tuple[0]
            setup['d'] = int(tuple[1])
            global data
            data, W_true = data_given_setup(setup)
            print(data)
            best_parameters, best_values, _, _ = bayesian_optimize(data, cross_validated)
            print(f'Best Parameters {best_parameters}')
            # Take best parameter, and retrain with all data
            W_estimated = graph_estimator(data, best_parameters)
            metrics_list.append(MetricsDAG(W_estimated, W_true).metrics)
            parameters_list.append(best_parameters)
        outcome = {'metrics': metrics_list, 'parameters': parameters_list, 'd_list': d_list}
        with open(f"data_er{sys.argv[2]}_{sys.argv[1]}.json", 'w') as outfile:
            json.dump(outcome, outfile)
            
    else:
        data, graph = cdt.data.load_dataset('sachs')
        best_parameters, best_values, _, _ = bayesian_optimize(data, False)
        print(f'Best Parameters {best_parameters}')
        W_estimated = graph_estimator(data, best_parameters)
        log = {'W_estimated': W_estimated, 'best_parameters': best_parameters, 'best_values': best_values}
        with open('data_saches.json', 'w') as outfile:
            json.dump(log, outfile)

    
if __name__ == "__main__":
    main()
