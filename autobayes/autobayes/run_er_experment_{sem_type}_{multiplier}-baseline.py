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


def data_given_setup(setup):
    W_true = ut.simulate_dag(setup['d'], setup['s0'], setup['graph_type'])
    data = pd.DataFrame(ut.simulate_nonlinear_sem(W_true, setup['n'], setup['sem_type']))
    return data, W_true

def main(test=False):

    setup = {'n': 1000, 'graph_type': 'ER', 'sem_type':'gp'} # Default
    setup['sem_type'] = sys.argv[1]
    setup['er_multiplier'] = int(sys.argv[2])
    d_list = [10, 20, 30, 40]
    er_s0_d = [(setup['er_multiplier']*d, d) for d in d_list]
    # This is for 10, 20, 30, 40
    metrics_list = []
    parameters_list = []
    
    parameterization = {'lambda1': 0.01, 'lambda2': 0.01, 'w_threshold': 0.5, 'nodes': 10}
    if test:
        er_s0_d = [(20, 10)]

    for tuple in er_s0_d:
        setup['s0'] = tuple[0]
        setup['d'] = int(tuple[1])
        global data
        data, W_true = data_given_setup(setup)

        W_est_binary = graph_estimator(data, parameterization)
    
        print(data)
        # Take best parameter, and retrain with all data
        metrics_list.append(MetricsDAG(W_est_binary, W_true).metrics)
    outcome = {'metrics': metrics_list, 'd_list': d_list}
    with open(f"Baseline_1000_data_er{sys.argv[2]}_{sys.argv[1]}.json", 'w') as outfile:
        json.dump(outcome, outfile)
            
    
if __name__ == "__main__":
    main()
