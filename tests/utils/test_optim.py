from gasuite.utils.optim import (
    Termination_criteria, 
    Termination_decision,
    Optimization_looper,
    Cost_vals_stats,
    )

import numpy as np
import matplotlib.pyplot as plt

def test_termination_on_max_iter():
    term_criteria = Termination_criteria(
        max_num_gen=5, 
        earlyTerm_improvement_convergence=1e-6, 
        earlyTerm_patience=0
    )
    best_cost = 1000.
    looper = Optimization_looper(
        term_criteria, best_cost
        )
    assert looper.get_num_iters() == 0
    while looper.should_continue():
        best_cost *= 0.8
        is_new_best = looper.step(best_cost)
        print("finished iteration: ", looper.get_num_iters(), " new cost:", looper._best_known_cost)
        if looper.get_num_iters() < term_criteria.max_num_gen:
            assert is_new_best == True
    assert looper.get_num_iters() == term_criteria.max_num_gen
    assert looper.state == Termination_decision.stop_on_max_iterations

def _test_early_termination(cost_seq, expected_num_iter, term_criteria):
    """ internal implemenation of the test suite for early termination"""
    looper = Optimization_looper(
        term_criteria, cost_seq[0]
        )
    assert looper.get_num_iters() == 0
    while looper.should_continue():
        best_cost = cost_seq[looper.get_num_iters()+1]
        if best_cost is None:
            raise NotImplementedError(
                "When implementing the test case, the pre-specified \
                    cost sequence doesn't anticipate running that many iterations \
                        --- so if the test case is implemented correctly, \
                            this means the code-under-test fails to terminate early")
        is_new_best = looper.step(best_cost)
        print(f"finished iteration: {looper.get_num_iters()}, new cost: {looper._best_known_cost: .2f}")
    assert looper.get_num_iters() == expected_num_iter
    assert looper.state == Termination_decision.stop_on_sufficient_cost_convergence

def test_dont_earlyTerm_dueTo_smallRiseInCost():
    _test_early_termination(
        cost_seq = [1000., 640, 320, 320.000001, 100, 95.5, None],
        expected_num_iter = 5,
        term_criteria = Termination_criteria(
            max_num_gen=10, 
            earlyTerm_improvement_convergence=0.05, 
            earlyTerm_patience=0
        )
    )
def test_earlyTerm_dueTo_smallImprovement():
    _test_early_termination(
        cost_seq = [1000., 64, 32, 32-1e-8],
        expected_num_iter = 3,
        term_criteria = Termination_criteria(
            max_num_gen=5, 
            earlyTerm_improvement_convergence=1e-6, 
            earlyTerm_patience=0
        )
    )


def test_earlyTerm_dueTo_NsuccessiveSmallImprovement():
    _test_early_termination(
        cost_seq = [1000, 991, 980, 980, 980, 980, None],
        expected_num_iter = 4,
        term_criteria = Termination_criteria(
            max_num_gen=20, 
            earlyTerm_improvement_convergence=1e-6, 
            earlyTerm_patience=2
        )
    )

def test_cost_stats_vis(plt_show = False):
    sut = Cost_vals_stats(max_iter=80)
    
    # some dummy data
    cost_data = np.array([
        [38.05, 28.03, 45.48, 35.47, 38.05, 40.58],
        [36.29, 27.11, 46.73, 33.51, 36.11, 39.51],
        [34.59, 27.11, 45.11, 32.  , 34.21, 37.28],
        [32.73, 25.84, 45.01, 28.23, 32.21, 35.99],
        [32.17, 25.65, 45.14, 26.86, 30.7 , 36.72],
        [30.62, 24.34, 43.46, 25.84, 28.1 , 34.71],
        [30.17, 24.09, 45.02, 25.65, 27.65, 34.01],
        [30.01, 21.92, 45.48, 24.34, 27.47, 35.86],
        [28.43, 19.52, 42.08, 24.09, 26.7 , 32.58],
        [28.47, 19.52, 45.5 , 21.92, 25.19, 33.97],
        [26.5 , 19.52, 45.62, 19.52, 24.3 , 32.44],
        [26.7 , 19.52, 44.25, 19.52, 22.86, 34.36],
        [26.57, 19.52, 45.2 , 19.52, 23.15, 35.49],
        [25.66, 19.52, 46.08, 19.52, 19.52, 33.89],
        [27.22, 19.52, 46.59, 19.52, 21.92, 36.13],
        [27.32, 19.52, 45.23, 19.52, 24.68, 34.02],
        [26.93, 19.52, 45.01, 19.52, 21.65, 35.92],
        [26.36, 19.52, 42.73, 19.52, 22.17, 34.58],
        [26.51, 19.52, 46.06, 19.52, 24.33, 34.03],
        [26.43, 19.52, 44.59, 19.52, 23.27, 34.85]])
    
    for cost_array in cost_data:
        sut.update(cost_array)
    _, ax = plt.subplots()
    sut.visualize_learning_curve(ax)
    if plt_show:
        plt.show()

if __name__ == "__main__":
    # test_earlyTerm_dueTo_NsuccessiveSmallImprovement()
    test_cost_stats_vis(plt_show=True)