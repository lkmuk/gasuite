import numpy as np
import numba

"""
Performance-oriented code
"""
@numba.njit(boundscheck=True)
def _eval_tour_cost(cost_matrix: np.ndarray, tour: np.ndarray) -> float|int:
    """
    caller responsible to ensure the tour is valid
    """
    cost = cost_matrix[tour[-1]][tour[0]]
    for i in range(len(tour)-1):
        cost += cost_matrix[tour[i]][tour[i+1]]
    return cost

@numba.njit(boundscheck=True)
def _precompute_geom_edge_cost_matrix(points: np.ndarray, ord = 2) -> np.ndarray:
    """Pre-compute the symmetric edge cost matrix for the graph/points
    
    @param: points: N x D float (D typically 2 or 3)
    @param: Norm type: 1 or 2
    
    Remark: for large point set (of size about 100,000), 
    you cannot pre-compute the edge cost like this.
    Nonetheless, for many realistic "large-scale" Generalized TSP instances,
    we rarely reach 100,000 cities. So this is indeed applicable
    
    Remark: fastmath or not isn't that a big deal.
    
    You wrap this function with numba JIT because the penalty of JIT 
    is low (almost always within 1 second) compared to pure native Python 
    execution especially if the point set is large.
    
    TODO: simplify to integers
    TODO: remove boundscheck after dev
    """
    assert points.ndim == 2
    num_cities = len(points)
    cost_mat = np.zeros((num_cities, num_cities), dtype=float)
    for ii in range(num_cities):
        xy_from = points[ii]
        for jj in range(ii): # no need to evaluate the diagnoal which is always 0
            xy_to = points[jj]
            c = np.linalg.norm(xy_to-xy_from, ord=ord)
            cost_mat[ii][jj] = c
            cost_mat[jj][ii] = c # Symmetric
    return cost_mat


if __name__ == "__main__":
    def perf_test_eval_tour_cost():
        from time import perf_counter
        dim = 5000
        rng = np.random.default_rng(32423)
        cost_mat = rng.integers(low=0, high=10, size = (dim, dim))
        random_tour = rng.permutation(dim)
        
        def do_work(num_trials = 50):
            ts0 = perf_counter()
            for i in range(num_trials):
                _eval_tour_cost(cost_mat, random_tour)
            tsf = perf_counter()
            print(f"time taken: {float(tsf-ts0)/num_trials:.6f} s (averaged over {num_trials:d} runs)")
        
        do_work()
        print("note that the first call involves a runtime penalty due to JIT compilation")
        print("let's rerun without any JIT compile to get a more realistic estimation of the runtime")
        do_work()
    
    # perf_test_eval_tour_cost()
    
    def test_precompute_edge_costs(dim = 5):
        rng = np.random.default_rng(32423)
        xyz = rng.integers(low=0, high=10, size = (dim, 3)).astype(float)
        print(xyz)
        computed_cost_mat = _precompute_geom_edge_cost_matrix(xyz)
        print(computed_cost_mat)
    test_precompute_edge_costs()
    
    def perf_test_precompute_edge_costs(skip_jit = False):
        """ This test justifies always using numba
        Results for this performance test
        
        native Python execution needs about 90 s
        
        with JIT + execution about 5 s
        after JIT, each exec. also about 5 s
        
        """
        from time import perf_counter
        dim = 10000
        rng = np.random.default_rng(32423)
        xyz = rng.integers(low=0, high=10, size = (dim, 3)).astype(float)
        
        
        def _precompute_geom_edge_cost_matrix(points: np.ndarray, ord = 2):
            assert points.ndim == 2
            num_cities = len(points)
            cost_mat = np.zeros((num_cities, num_cities), dtype=float)
            for ii in range(num_cities):
                xy_from = points[ii]
                for jj in range(ii):
                    xy_to = points[jj]
                    c = np.linalg.norm(xy_to-xy_from, ord=ord)
                    cost_mat[ii][jj] = c
                    cost_mat[jj][ii] = c # Symmetric
            return cost_mat
        
        
        print(f"Testing the edge cost precomputation (JIT: {not skip_jit})")
        sut = _precompute_geom_edge_cost_matrix if skip_jit else numba.njit(_precompute_geom_edge_cost_matrix, boundscheck=True)
        
        def do_work(num_trials = 1):
            ts0 = perf_counter()
            for i in range(num_trials):
                cost_matrix = sut(xyz)
            tsf = perf_counter()
            print(f"time taken: {float(tsf-ts0)/num_trials:.6f} s (averaged over {num_trials:d} runs)")
        
        do_work()
        print("note that the first call involves a runtime penalty due to JIT compilation")
        print("let's rerun without any JIT compile to get a more realistic estimation of the runtime")
        do_work()
    
    perf_test_precompute_edge_costs(skip_jit=False)
    perf_test_precompute_edge_costs(skip_jit=True)