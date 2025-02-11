import numpy as np
from gasuite.problems.geometric_tsp import eval_tour_cost
from gasuite.brkga import BRKGA_TSP, BRKGA_Population_size

# only needed for the test implementation
from pytest import approx
from typing import NamedTuple, Union, Final


# the same one used in the notebook
pcd : Final = np.array([
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
    ], dtype = np.float32
)

class Solution_Oracle(NamedTuple):
    from gasuite.utils.optim import Termination_decision

    seed : Union[int, None]
    expected_tour : np.typing.NDArray[int]
    expected_num_iters : int
    expected_reason_term : Termination_decision

def test_solve_Euc2d_TSP_BRKGA():
    from gasuite.utils.optim import Termination_decision, Termination_criteria

    population_cfg = BRKGA_Population_size(total=200, elite=4, mutant=40)
    
    term_criteria = Termination_criteria(
        max_num_gen=800, 
        earlyTerm_improvement_convergence=1e-6, 
        earlyTerm_patience=50,
    )
    
    oracles = (
        Solution_Oracle(
            123, 
            [9, 8, 6, 0, 4, 10, 1, 3, 2, 7, 5],
            72,
            Termination_decision.stop_on_sufficient_cost_convergence
        ),
        Solution_Oracle(
            6565, 
            [9, 5, 7, 2, 3, 1, 10, 4, 0, 6, 8],
            55,
            Termination_decision.stop_on_sufficient_cost_convergence
        ),
        Solution_Oracle(
            8964, 
            [4, 0, 6, 8, 9, 5, 7, 2, 3, 1, 10],
            73,
            Termination_decision.stop_on_sufficient_cost_convergence
        ),
    )     
   
    # single threaded implementation            
    for oracle in oracles:
        computed = BRKGA_TSP(
            len(pcd), lambda rank : eval_tour_cost(rank, pcd), 
            population_cfg, term_criteria, crossover_bias=0.7, 
            rng=np.random.default_rng(oracle.seed))
        # print(computed.optimizer)
        # print(computed.cost_stats.num_completed_iterations)
        # print(computed.cost_stats.best)
        # print(computed.cost_stats.get_all_time_best())
        assert computed.reason == oracle.expected_reason_term
        assert computed.cost_stats.num_completed_iterations == oracle.expected_num_iters # -1 to disregard the initial generation
        np.testing.assert_equal(computed.optimizer, oracle.expected_tour)
        assert eval_tour_cost(computed.optimizer, pcd) == approx(19.5189, abs=1e-4)

if __name__ == "__main__":
    test_solve_Euc2d_TSP_BRKGA()