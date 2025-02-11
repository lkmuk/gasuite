import  numpy as np
from typing import NamedTuple, Union
from enum import Enum
from dataclasses import dataclass
import warnings

from matplotlib.axes import Axes as Plt_Axes

class Cost_vals_stats:
    def __init__(self, max_iter: int):
        assert max_iter > 1
        self._dat = np.ones((max_iter, 6))*np.nan
        
        # number of iterations performed
        self._i : int = 0 
    
    @property
    def num_completed_iterations(self) -> int:
        return self._i - 1
    
    def update(self, sorted_cost_vals: np.ndarray) -> None:
        if self._i >= len(self._dat):
            raise BufferError("Too many updates, exceeding the originally planned size")
        assert sorted_cost_vals[0] <= sorted_cost_vals[-1], "your cost values are not sorted"
        pop_sz = len(sorted_cost_vals)
        self._dat[self._i] = np.array((
            sorted_cost_vals.mean(),
            sorted_cost_vals[0],
            sorted_cost_vals[-1],
            sorted_cost_vals[pop_sz//4],
            sorted_cost_vals[pop_sz//2],
            sorted_cost_vals[3*pop_sz//4]
        ))
        self._i += 1
    
    @property
    def mean(self):
        """mean-in-population for every generation
        """
        return self._dat[:self._i, 0]

    @property
    def best(self):
        """best-in-population for every generation
        """
        return self._dat[:self._i, 1]
    
    @property
    def worst(self):
        """worst-in-population for every generation
        """
        return self._dat[:self._i, 2]
    
    @property
    def percentile25(self):
        """for every generation, the best 25% percentile-in-population
        """
        return self._dat[:self._i, 3]

    @property
    def percentile50(self):
        """for every generation, the median-in-population 
        """
        return self._dat[:self._i, 4]
    
    @property
    def percentile75(self):
        """for every generation, the best 75% percentile-in-population
        """
        return self._dat[:self._i, 5]
    
    def get_all_time_best(self, skip=True):
        return self.best[-1] if skip else np.min(self.best) 

    def visualize_learning_curve(self, ax: Plt_Axes, true_optim = None, lw= 0.5, **kwargs):
        xdata = np.arange(self.num_completed_iterations+1)
        ax.plot(xdata, self.worst, label="max", c='blue', ls="-", lw=lw, **kwargs)
        ax.plot(xdata, self.percentile75, label="pct75", c='cyan', ls="-", lw=lw, **kwargs)
        ax.plot(xdata, self.mean, label="mean", c='green', ls="-", lw=lw, **kwargs)
        ax.plot(xdata, self.percentile25, label="pct25", c='orange', ls="-", lw=lw, **kwargs)
        if true_optim:
            ax.plot([0, self.num_completed_iterations], [true_optim]*2, c='gray', lw=2, ls="--", label="true optimum")
        ax.grid()
        ax.legend()
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Cost")


@dataclass
class Termination_criteria:
    max_num_gen: int
    earlyTerm_improvement_convergence: float # = latest_improvement / old_cost
    earlyTerm_patience: int = 0
    def __post_init__(self):
        assert self.max_num_gen >= 1
        assert 0. <= self.earlyTerm_improvement_convergence <= 1.
        assert self.earlyTerm_patience >= 0
        if self.earlyTerm_patience >= self.max_num_gen:
            warnings.warn("Erroneous termination criteria, the minimum number of iteration is at least `earlyTerm_patience`, which is somehow larger than  `max_num_gen`")
        
    
class Termination_decision(Enum):
    keep_on_running = 0
    stop_on_max_iterations = 1
    stop_on_sufficient_cost_convergence = 2

class Optimization_looper:
    def __init__(self, tc: Termination_criteria, best_known_cost: float):
        self._tc = tc
        self._best_known_cost = best_known_cost
        self.reset()
    
    def should_continue(self) -> bool:
        return self.state == Termination_decision.keep_on_running
    
    def get_num_iters(self):
        return self.iter
    
    def reset(self):
        self.iter: int = 0
        self.patience_count: int = 0
        self.state = Termination_decision.keep_on_running

    def _try_update_best_cost(self, best_cost_among_this_gen: float) -> tuple[bool, Union[float, None]]:
        # TODO bypass for RKGA!
        improvement = self._best_known_cost - best_cost_among_this_gen
        if improvement >= 0:
            self._best_known_cost = best_cost_among_this_gen
            return True, improvement
        else:
            return False, None
    
    def step(self, best_cost_among_this_gen: float) -> bool:
        """process the latest result and update the internal data
        return: is the best cost a new record? 
        
        Meant to be called at the end of each iteration
        """
        
        self.iter += 1
        if self.iter >= self._tc.max_num_gen:
            self.state = Termination_decision.stop_on_max_iterations
            return self._try_update_best_cost(best_cost_among_this_gen)[0]
                      
        # is the improvement significant?
        min_abs_improvement = self._tc.earlyTerm_improvement_convergence * self._best_known_cost
        
        is_new_best, improvement = self._try_update_best_cost(best_cost_among_this_gen)
        
        if improvement is None:
            # when there is no improvement, we don't count it as a failed generation
            self.state = Termination_decision.keep_on_running
            return False
        elif improvement < min_abs_improvement:
            self.patience_count += 1
            self.state = Termination_decision.keep_on_running \
                if self.patience_count < self._tc.earlyTerm_patience \
                else Termination_decision.stop_on_sufficient_cost_convergence
            return is_new_best
        else:
            self.patience_count = 0
            self.state = Termination_decision.keep_on_running
            return is_new_best
        
@dataclass
class Solution_report:
    optimizer: any
    cost_stats: Cost_vals_stats
    reason: Termination_decision