from .optim import Cost_vals_stats, Solution_report
import numpy as np
from multiprocessing.pool import Pool
from functools import partial

from typing import Callable, OrderedDict, Any
import matplotlib.axes

def run_multiple_times(solver_fnc: Callable[[Any], Solution_report], seeds: list[int], num_proc = 1, **solver_main_prms: OrderedDict) -> list[Solution_report]:
    """Running the same solver configuration with different seeds
    
    For research and for some deployment, a stochastic solver, e.g., 
    a Genetic Algorithm, with the same configuration shall be 
    run several times. We can save the wall-clock execution time 
    by parallelize the runs.
      
    Quick tip: try to **avoid** any lambda functions in 
    `solver_fnc` and `solver_main_prms`.
    Use `functools.partial` instead. See `test_multirun.py`.
    
    :param solver_fnc: a stochastic optimization implementation
            we expect its last argument is a numpy random generator object.
            We also expect it to output a `Solution_report` object
    :param seeds: each run means one seed (each run should have a different seed)
    :param num_proc: number of parallel running process
            set it to 1 for single thread, 
            set it to higher numbers {2, 3, ..., num_CPU_cores} for multi-processing
    :param solver_main_prms: the parameters for `solver_fnc` (other than the rng)
            We expect this to be in the **same order** as the interface of `solver_fnc`.
            So you should use OrderDict
    :returns: a list of Solution_report
    """
    assert len(set(seeds)) == len(seeds), "detected duplicate seeds"
    rngs = [np.random.default_rng(s) for s in seeds]
    runner = partial(solver_fnc, *solver_main_prms.values())
    if num_proc == 1:
        results = [runner(g) for g in rngs] # single thread version
    else:
        with Pool(len(rngs)) as p:
            results = p.map(runner, rngs)
    return results

def _pad_by_holding_last_value(array1d: np.ndarray, total_output_length: int) -> np.ndarray:
    assert array1d.ndim == 1
    num_extrapolants = total_output_length - len(array1d)
    assert num_extrapolants >= 0, "You are trying to truncate the input array, which isn't what this function is for"
    return np.concatenate([array1d, [float(array1d[-1])]*num_extrapolants])

class Multirun_cost_vals_stats:
    def __init__(self, costs_multirun: list[Cost_vals_stats]):
        """Extract the `best_in_population[i][j]` from the results 
        
        where `i` is the index of the run (corresponding to an unique random generator seed)
        and `j` is the iteration index. 
        
        In each run, the iteration may terminate earlier at different `j`.
        This might complicate the computation of statistics, this helper class 
        aims to simplify this process --- shorter runs' best_in_population data will
        be extrapolated by holding the last value.
        
        
        Notes: 
        
        * The user is responsible to ensure the solver parameter is identical during each run.
        
        * The interface of this constructor is meant to be minimalistic.
          So it doesn't accept a list of solution reports.
          It is true that our solver API outputs a Solution_report.
          To create `Multirun_cost_vals_stats` from list[Solution_report],
          you can use the convenience function `extract_multirun_cost_vals_stats`
        """
        # include the initial generation
        self._num_iterations = max([this_run.num_completed_iterations for this_run in costs_multirun]) + 1
        
        # The best-in-population cost for each iteration and run
        self._bestInPop = np.array([
            _pad_by_holding_last_value(costs_this_run.best, self._num_iterations) for costs_this_run in costs_multirun
        ])
        self._precompute_statistics_across_runs()
    
    @property
    def num_runs(self) -> int:
        return len(self._bestInPop)
    
    def _precompute_statistics_across_runs(self):
        self._bestInPop_mean = np.mean(self._bestInPop, axis=0)
        
        bestInPop_sorted_by_cost = np.sort(self._bestInPop, axis=0)
        self._bestInPop_max = bestInPop_sorted_by_cost[-1,:]
        self._bestInPop_median = bestInPop_sorted_by_cost[self.num_runs//2,:]
        self._bestInPop_min = bestInPop_sorted_by_cost[0,:]
    
    @property
    def max_num_performed_iterations(self) -> int:
        """Some runs may terminate earlier"""
        return self._num_iterations - 1
    
    def visualize(self, ax: matplotlib.axes.Axes, opacity_individual_run = 0.2):
        xdata = np.arange(self._num_iterations)
        if opacity_individual_run > 0:
            for bestInPopOfRunXXX in self._bestInPop:
                ax.plot(xdata, bestInPopOfRunXXX, "-", c='gray', lw=1, alpha=opacity_individual_run, label=None)
        ax.plot(xdata, self._bestInPop_max, ":b", label="Max", lw=2)
        ax.plot(xdata, self._bestInPop_mean, "-.k", label="Mean", lw=2)
        ax.plot(xdata, self._bestInPop_median, "--g", label="Median", lw=2)
        ax.plot(xdata, self._bestInPop_min, "-r", label="Min", lw=2.5)
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Best-in-population cost")
        ax.legend()


def extract_multirun_cost_vals_stats(solver_outputs : list[Solution_report]) -> Multirun_cost_vals_stats:
    return Multirun_cost_vals_stats([report.cost_stats for report in solver_outputs])