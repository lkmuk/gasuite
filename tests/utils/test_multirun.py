# what we want to test
from gasuite.utils.multirun import (
    run_multiple_times,
    extract_multirun_cost_vals_stats,
)

# dependencies for implementing the test
import numpy as np
from gasuite.brkga import (
    BRKGA_TSP,
    BRKGA_Population_size, 
    Termination_criteria,
)
from gasuite.problems.geometric_tsp import eval_tour_cost
from pytest import approx
import matplotlib.pyplot as plt
from functools import partial
from typing import OrderedDict
from time import perf_counter

# the same one used in the notebook
xy_points = np.array([
        [-2, 0.5],
        [1, 2],
        [3, 2],
        [2, 1],
        [-1, 3],
        [4, 1],
        [-1, -1],
        [4, 3],
        [1.5, -0.5],
        [3, -1],
        [0, 3]
    ], dtype = float
)

true_global_min = 19.5189

solver_BRKGA_TSP_prms = OrderedDict(
    num_cities = len(xy_points),
    individual_cost_fnc = partial(eval_tour_cost, point_cloud = xy_points),
    pop_sizes = BRKGA_Population_size(
        total = 200, elite=2, mutant=50),
    tc = Termination_criteria(
        max_num_gen=500, 
        earlyTerm_improvement_convergence=1e-6, 
        earlyTerm_patience=30), # in actual uses, you shall consider a larger value
    crossover_bias = 0.6,
)

def _test_run_multiple_times(num_process, plt_show = False):
    seeds = 5*np.arange(5) # for research purposes, use more seeds
    t0 = perf_counter()
    computed_results = run_multiple_times(BRKGA_TSP, seeds, num_proc=num_process, **solver_BRKGA_TSP_prms)
    t1 = perf_counter()
    print(f"{len(seeds)} runs using {num_process:d} process(es) took {(t1-t0):.3f} sec") 
    assert len(computed_results) == len(seeds)
    
    summary = extract_multirun_cost_vals_stats(computed_results)
    assert summary.num_runs == len(seeds)
    
    _, ax = plt.subplots()
    summary.visualize(ax, opacity_individual_run=0.2)
    if plt_show:
        ax.set_title("Results from multiple runs of BRKGA")
        plt.show()

def test_run_multiple_times_singleProc():
    _test_run_multiple_times(num_process=1, plt_show=False)
def test_run_multiple_times_4Proc():
    _test_run_multiple_times(num_process=4, plt_show=False)


if __name__ == "__main__":
    _test_run_multiple_times(num_process=4, plt_show=False)
    _test_run_multiple_times(num_process=1, plt_show=True)