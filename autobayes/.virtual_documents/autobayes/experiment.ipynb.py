# OPTIONAL: Load the "autoreload" extension so that code can change
get_ipython().run_line_magic("load_ext", " autoreload")

# OPTIONAL: always reload modules so that as you change code in src, it gets loaded
get_ipython().run_line_magic("autoreload", " 2")

from mlp_model import *


best_parameters, best_values, experiment, model = bayesian_optimize()


all_metrics_given_parameters(best_parameters)


best_values


??model


import numpy as np

from ax.plot.contour import plot_contour
from ax.plot.trace import optimization_trace_single_method
from ax.service.managed_loop import optimize
from ax.metrics.branin import branin
from ax.utils.measurement.synthetic_functions import hartmann6
from ax.utils.notebook.plotting import render, init_notebook_plotting

init_notebook_plotting()


render(plot_contour(model=model, param_x='lambda1', param_y='lambda2', metric_name='objective'))



render(plot_contour(model=model, param_x='lambda1', param_y='w_threshold', metric_name='objective'))



# `plot_single_method` expects a 2-d array of means, because it expects to average means from multiple 
# optimization runs, so we wrap out best objectives array in another array.
best_objectives = np.array([[trial.objective_mean for trial in experiment.trials.values()]])
best_objective_plot = optimization_trace_single_method(
    y=np.maximum.accumulate(best_objectives, axis=1),
    title="Model performance (G) vs. # of iterations",
    ylabel="G_score",
)
render(best_objective_plot)



