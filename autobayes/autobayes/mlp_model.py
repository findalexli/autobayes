

from notears.locally_connected import LocallyConnected
from notears.lbfgsb_scipy import LBFGSBScipy
from notears.trace_expm import trace_expm
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import notears.utils as ut
from ax.service.managed_loop import optimize

from metric import MetricsDAG

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('\nUsing device: {}\n'.format(device))

class NotearsMLP(nn.Module):
    def __init__(self, dims, bias=True):
        super(NotearsMLP, self).__init__()
        assert len(dims) >= 2
        assert dims[-1] == 1
        d = dims[0]
        self.dims = dims
        # fc1: variable splitting for l1
        self.fc1_pos = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_neg = nn.Linear(d, d * dims[1], bias=bias)
        self.fc1_pos.weight.bounds = self._bounds()
        self.fc1_neg.weight.bounds = self._bounds()
        # fc2: local linear layers
        layers = []
        for l in range(len(dims) - 2):
            layers.append(LocallyConnected(d, dims[l + 1], dims[l + 2], bias=bias))
        self.fc2 = nn.ModuleList(layers)

    def _bounds(self):
        d = self.dims[0]
        bounds = []
        for j in range(d):
            for m in range(self.dims[1]):
                for i in range(d):
                    if i == j:
                        bound = (0, 0)
                    else:
                        bound = (0, None)
                    bounds.append(bound)
        return bounds

    def forward(self, x, device):  # [n, d] -> [n, d]
        x = self.fc1_pos.to(device)(x) - self.fc1_neg.to(device)(x)  # [n, d * m1]
        x = x.view(-1, self.dims[0], self.dims[1])  # [n, d, m1]
        for fc in self.fc2.to(device):
            x = torch.sigmoid(x)  # [n, d, m1]
            x = fc(x)  # [n, d, m2]
        x = x.squeeze(dim=2)  # [n, d]
        return x

    def h_func(self):
        """Constrain 2-norm-squared of fc1 weights along m1 dim to be a DAG"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        A = A.cpu()
        h = trace_expm(A) - d  # (Zheng et al. 2018)
        # A different formulation, slightly faster at the cost of numerical stability
        # M = torch.eye(d) + A / d  # (Yu et al. 2019)
        # E = torch.matrix_power(M, d - 1)
        # h = (E.t() * M).sum() - d
        return h

    def l2_reg(self):
        """Take 2-norm-squared of all parameters"""
        reg = 0.
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        reg += torch.sum(fc1_weight ** 2)
        for fc in self.fc2:
            reg += torch.sum(fc.weight ** 2)
        return reg

    def fc1_l1_reg(self):
        """Take l1 norm of fc1 weight"""
        reg = torch.sum(self.fc1_pos.weight + self.fc1_neg.weight)
        return reg

    @torch.no_grad()
    def fc1_to_adj(self) -> np.ndarray:  # [j * m1, i] -> [i, j]
        """Get W from fc1 weights, take 2-norm over m1 dim"""
        d = self.dims[0]
        fc1_weight = self.fc1_pos.weight - self.fc1_neg.weight  # [j * m1, i]
        fc1_weight = fc1_weight.view(d, -1, d)  # [j, m1, i]
        A = torch.sum(fc1_weight * fc1_weight, dim=1).t()  # [i, j]
        W = torch.sqrt(A)  # [i, j]
        W = W.cpu().detach().numpy()  # [i, j]
        return W


def golem_likelihood_ev(output, target):
    n = target.shape[0]
    d = target.shape[1]
    log_likelihood_term = 0.5*d*torch.log(squared_loss(output, target))



def squared_loss(output, target):
    # By GOLEM, we do not compute the least squares, but the likelihood-Ev objective
    n = target.shape[0]
    loss = 0.5 / n * torch.sum((output - target) ** 2)
    return loss


def dual_ascent_step(model, X, lambda1, lambda2, rho, alpha, h, rho_max):
    """Perform one step of dual ascent in augmented Lagrangian."""
    h_new = None
    optimizer = LBFGSBScipy(model.parameters())
    X_torch = torch.from_numpy(X).to(device)
    while rho < rho_max:
        def closure():
            optimizer.zero_grad()
            X_hat = model(X_torch, device)
            loss = squared_loss(X_hat, X_torch)
            h_val = model.h_func()
            penalty = 0.5 * rho * h_val * h_val + alpha * h_val
            l2_reg = 0.5 * lambda2 * model.l2_reg()
            l1_reg = lambda1 * model.fc1_l1_reg()
            primal_obj = loss + penalty + l1_reg
            primal_obj.backward()
            return primal_obj
        optimizer.step(closure)  # NOTE: updates model in-place
        with torch.no_grad():
            h_new = model.h_func().item()
        if h_new > 0.25 * h:
            rho *= 10
        else:
            break
    alpha += rho * h_new
    return rho, alpha, h_new


def notears_nonlinear(model: nn.Module,
                      X: np.ndarray,
                      lambda1,
                      lambda2,
                      max_iter,
                      h_tol,
                      rho_max,
                      w_threshold):
    rho, alpha, h = 1.0, 0.0, np.inf
    for _ in range(max_iter):
        rho, alpha, h = dual_ascent_step(model, X, lambda1, lambda2,
                                         rho, alpha, h, rho_max)
        if h <= h_tol or rho >= rho_max:
            break
    W_est = model.fc1_to_adj()
    W_est[np.abs(W_est) < w_threshold] = 0
    return W_est

# def wrapper(setup, parameterization):
#     n = setup['n']
#     d = setup['d']
#     s0 = setup['s0']
#     graph_type = setup['graph_type']
#     sem_type = setup['mlp']
#     return train_evaluate(parameterization)

def train_evaluate(parameterization: dict):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)
    ut.set_random_seed(123)

    # Data
    """ Parameters
    d_list = [10, 20, 40]
    er1_s0_d = [(d, d) for d in d_list]
    er1_s0_d = [(2d, d) for d in d_list]
    er4_s0_d = [(4d, d) for d in d_list]
    n_list = [1000, 20]
    3） ANM with MLPs
    """
    n = parameterization['n']
    d = parameterization['d']
    s0 = parameterization['s0']
    graph_type = parameterization['graph_type']
    sem_type = parameterization['sem_type']

    B_true = ut.simulate_dag(d, s0, graph_type)
    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    # Setup fully connected layers
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    # lambda1: float = 0.,
    # lambda2: float = 0.,
    # max_iter: int = 100,
    # h_tol: float = 1e-8,
    # rho_max: float = 1e
    # w_threshold: float = 0.3):
    W_est = notears_nonlinear(model,
                              X,
                              lambda1=parameterization['lambda1'],
                              lambda2=parameterization['lambda2'],
                              max_iter=100,
                              h_tol=1e-8,
                              rho_max=1e+16,
                              w_threshold=parameterization['w_threshold']
    )
    assert ut.is_dag(W_est)
    # acc = ut.count_accuracy(B_true, W_est != 0)
    # print(acc)

    acc_castle = MetricsDAG(W_est != 0, B_true)
    print(acc_castle.metrics)
    return acc_castle.metrics['gscore']


def all_metrics_given_parameters(parameterization: dict):
    torch.set_default_dtype(torch.double)
    np.set_printoptions(precision=3)
    ut.set_random_seed(123)

    # Data
    """ Parameters
    d_list = [10, 20, 40]
    er1_s0_d = [(d, d) for d in d_list]
    er1_s0_d = [(2d, d) for d in d_list]
    er4_s0_d = [(4d, d) for d in d_list]
    n_list = [1000, 20]
    3） ANM with MLPs
    """
    n = parameterization['n']
    d = parameterization['d']
    s0 = parameterization['s0']
    graph_type = parameterization['graph_type']
    sem_type = parameterization['sem_type']

    #n, d, s0, graph_type, sem_type = 200, 10, 20, 'ER', 'mlp'

    B_true = ut.simulate_dag(d, s0, graph_type)
    X = ut.simulate_nonlinear_sem(B_true, n, sem_type)

    # Setup fully connected layers
    model = NotearsMLP(dims=[d, 10, 1], bias=True)
    W_est = notears_nonlinear(model,
                              X,
                              lambda1=parameterization['lambda1'],
                              lambda2=parameterization['lambda2'],
                              max_iter=100,
                              h_tol=1e-8,
                              rho_max=1e+16,
                              w_threshold=parameterization['w_threshold']
    )
    assert ut.is_dag(W_est)
    # acc = ut.count_accuracy(B_true, W_est != 0)
    # print(acc)

    acc_castle = MetricsDAG(W_est != 0, B_true)
    return acc_castle.metrics

def bayesian_optimize(setup):
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
               },
                {
                    "name": "n",
                    "type": "fixed",
                    "value": setup['n'],
                },
                {
                    "name": "d",
                    "type": "fixed",
                    "value": setup['d'],
                },
                {
                    "name": "s0",
                    "type": "fixed",
                    "value": setup['s0'],
                },
                {
                    "name": "graph_type",
                    "type": "fixed",
                    "value": setup['graph_type'],
                },
                {
                    "name": "sem_type",
                    "type": "fixed",
                    "value": setup['sem_type'],
                }
            ],
            # Booth function
            evaluation_function=train_evaluate,
            minimize=False,
        )
    return best_parameters, best_values, experiment, model



def run_experiements():
    # n, d, s0, graph_type, sem_type = 200, 10, 20, 'ER', 'mlp'
    setup = {'n': 200, 'graph_type': 'ER', 'sem_type':'mlp'}
    d_list = [10, 20, 30, 40]
    er2_s0_d = [(2*d, d) for d in d_list]
    # This is for 10, 20, 30, 40
    metrics_list = []
    parameters_list = []
    for tuple in er2_s0_d:
        setup['s0'] = tuple[0]
        setup['d'] = tuple[1]
        best_parameters, best_values, _, _ = bayesian_optimize(setup)
        metrics_list.append(all_metrics_given_parameters(best_parameters))
        parameters_list.append(best_parameters)
    df = pd.DataFrame({'metrics': metrics_list, 'parameters': parameters_list, 'd_list': d_list})
    df.to_csv('Records.csv')

def main():
    run_experiements()

def main_old():
    setup = {'n': 200, 'graph_type': 'ER', 'sem_type': 'mlp', 'd': 10, 's0': 20}
    best_parameters, best_values, experiment, model = bayesian_optimize(setup)
    print('--------------------------------------------')
    print('Exeriments results')
    print(best_parameters)
    print(best_values)


    # torch.set_default_dtype(torch.double)
    # np.set_printoptions(precision=3)
    #
    # import notears.utils as ut
    # ut.set_random_seed(123)
    #
    # n, d, s0, graph_type, sem_type = 2000, 50, 9, 'ER', 'mim'
    # B_true = ut.simulate_dag(d, s0, graph_type)
    # np.savetxt('W_true.csv', B_true, delimiter=',')
    #
    # X = ut.simulate_nonlinear_sem(B_true, n, sem_type)
    # np.savetxt('X.csv', X, delimiter=',')
    #
    # model = NotearsMLP(dims=[d, 10, 1], bias=True)
    # W_est = notears_nonlinear(model, X, lambda1=0.01, lambda2=0.01)
    # assert ut.is_dag(W_est)
    # np.savetxt('W_est.csv', W_est, delimiter=',')
    # acc = ut.count_accuracy(B_true, W_est != 0)
    # print(acc)


if __name__ == '__main__':
    main()
