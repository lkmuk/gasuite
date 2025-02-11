import numpy as np
from gasuite.problems.geometric_tsp import eval_tour_cost

# only needed for the test implementation
from pytest import approx


# the same one used in the notebook
pcd = np.array([
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

def test_eval_cost_single_tour_ordinalSoln():
    from math import hypot
    tour = np.arange(len(pcd))
    cost = eval_tour_cost(tour, pcd)
    expected_cost = hypot(1.5,3) + 2 + hypot(1,1) + hypot(2,3) + hypot(5,2) * 2 + hypot(5, 4) + hypot(2.5, 3.5) + hypot(0.5, 1.5) + hypot(3, 4) + hypot(2, 2.5)
    assert cost == approx(expected_cost)


def test_eval_cost_single_tour_notBadSoln():
    tour = np.array([2, 5, 6, 0, 4, 10, 7, 9, 8, 3, 1], dtype=np.int64) # np.arange(len(pcd))
    cost = eval_tour_cost(tour, pcd)
    expected_cost = 26.9943333
    assert cost == approx(expected_cost)