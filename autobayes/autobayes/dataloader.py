from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import math
import notears.utils as ut


n, d, s0, graph_type, sem_type = 200, 10, 20, 'ER', 'mlp'

B_true = ut.simulate_dag(d, s0, graph_type)
# np.savetxt('W_true.csv', B_true, delimiter=',')
X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
