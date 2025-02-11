from . import *
import numpy as np
from typing import Callable

def eval_tour_cost(tour: np.ndarray, point_cloud: np.ndarray, norm_type = 2) -> float:
    """ evaluate the total L2 path length of a SINGLE Hamiltonian tour
    
    no checks on the argument, so ensure you know what you are doing when calling this
    """
    tour_xy = point_cloud[tour]
    cost = np.linalg.norm(tour_xy[-1]-tour_xy[0])
    cost += np.sum(np.linalg.norm(tour_xy[1:]-tour_xy[:-1], axis=1, ord=norm_type))
    return cost.item()