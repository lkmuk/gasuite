from dataclasses import dataclass
import numpy as np
from typing import Callable

from .utils.optim import Termination_criteria, Optimization_looper, Solution_report, Cost_vals_stats

@dataclass
class BRKGA_Population_size:
    total : int
    elite: int
    mutant: int # not really mutation in the usual sense, some call it "immigration" instead
    @property
    def xover(self) -> int:
        return self.total - self.elite - self.mutant

def generate_permutations(rng: np.random.Generator, N: int, num_cities: int) -> np.ndarray:
    return rng.random((N, num_cities))

def generate_whichCity(rng: np.random.Generator, N: int, nums_cities: list[int]) -> np.ndarray:
    return rng.integers(0, nums_cities, (N, len(nums_cities))) #.astype(float)

def crossover_uniform_biased(rng: np.random.Generator, parents_elite, parents_else, bias = 0.7):
    assert parents_elite.shape == parents_else.shape
    use_elite = rng.random(parents_elite.shape) < bias
    return parents_elite * use_elite + parents_else * ~use_elite

def decode_rank(population: np.ndarray) -> np.ndarray:
    assert population.ndim == 2
    # random-key population decoded into permutations
    return np.argsort(population, axis = 1) 

def decode_rankAndInt(population: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    assert population.ndim == 2
    fractional, integral = np.modf(population)
    return np.argsort(fractional, axis = -1), integral.astype(int)

def _BRKGA_evolve_perm(sorted_pop: np.ndarray, pop_sizes: BRKGA_Population_size, crossover_bias : int, rng: np.random.Generator) -> np.ndarray:
    """ create a new generation for permutation
        Notes: `sorted_pop` should be sorted in ascending cost
    """
    assert len(sorted_pop) == pop_sizes.total
    
    # 1. copy the elites
    new_pop = np.zeros_like(sorted_pop)
    new_pop[:pop_sizes.elite] = sorted_pop[:pop_sizes.elite]
    
    # 2. biased cross-over
    idx_mating_elites = rng.integers(0, pop_sizes.elite, size = (pop_sizes.xover,))
    idx_mating_else = rng.integers(pop_sizes.elite, pop_sizes.total, size = (pop_sizes.xover,))
    new_pop[pop_sizes.elite:-pop_sizes.mutant] = crossover_uniform_biased(
        rng, 
        sorted_pop[idx_mating_elites], 
        sorted_pop[idx_mating_else], 
        crossover_bias)
    
    # 3. introduce mutants
    num_cities = sorted_pop.shape[1]
    new_pop[-pop_sizes.mutant:] = generate_permutations(rng, pop_sizes.mutant, num_cities)
            
    return new_pop

def _BRKGA_evolve_GTSP(nums_cities: list[int], sorted_pop: np.ndarray, pop_sizes: BRKGA_Population_size, crossover_bias : int, rng: np.random.Generator) -> np.ndarray:
    assert len(nums_cities) == len(sorted_pop) == pop_sizes.total
    new_pop = _BRKGA_evolve_perm(sorted_pop, pop_sizes, crossover_bias, rng)
    new_pop[-pop_sizes.mutant:] += generate_whichCity(rng, pop_sizes.mutant, nums_cities)

def BRKGA_TSP(num_cities: int, individual_cost_fnc: Callable[[np.ndarray], float], 
              pop_sizes: BRKGA_Population_size, tc: Termination_criteria, 
              crossover_bias = 0.7, rng: np.random.Generator = np.random.default_rng()
              ) -> Solution_report:
    
    cost_vals_hist = Cost_vals_stats(tc.max_num_gen+1)
    
    # random-key
    pop = generate_permutations(rng, pop_sizes.total, num_cities)
    looper = Optimization_looper(tc, np.inf)
    
    soln = None
    while True:        
        ## evaluation and sorting
        pop_decoded = decode_rank(pop) # 0-based permutations
        # TODO avoid re-evaluating elites
        cost_vals = np.array([individual_cost_fnc(p) for p in pop_decoded])
        order = np.argsort(cost_vals)
        cost_vals = cost_vals[order]
        pop = pop[order]
        cost_vals_hist.update(cost_vals)
        is_new_best = looper.step(float(cost_vals[0]))
        if is_new_best:
            # copy to release the decoded data    
            soln = pop_decoded[order[0], :].copy()
        if not looper.should_continue():
            break
        
        pop = _BRKGA_evolve_perm(pop, pop_sizes, crossover_bias, rng)
        
    return Solution_report(soln, cost_vals_hist, looper.state)
    
            
def BRKGA_GTSP(nums_cities: list[int], individual_cost_fnc: Callable[[np.ndarray, np.ndarray], float], 
              pop_sizes: BRKGA_Population_size, tc: Termination_criteria, 
              crossover_bias = 0.7, rng: np.random.Generator = np.random.default_rng()
              ) -> Solution_report:
    cost_vals_hist = Cost_vals_stats(tc.max_num_gen+1)
    
    # random-key + which city <-- GTSP
    pop = generate_permutations(rng, pop_sizes.total, len(nums_cities))
    pop += generate_whichCity(rng, pop_sizes.total, nums_cities) # <-- GTSP
    looper = Optimization_looper(tc, np.inf)
    
    soln = None
    while True:        
        ## evaluation and sorting
        ranks, whichCities  = decode_rankAndInt(pop) # <-- GTSP
        # TODO avoid re-evaluating elites
        cost_vals = np.array([individual_cost_fnc(rank, w) for rank, w in zip(ranks, whichCities)])
        order = np.argsort(cost_vals)
        cost_vals = cost_vals[order]
        pop = pop[order]
        cost_vals_hist.update(cost_vals)
        is_new_best = looper.step(float(cost_vals[0]))
        if is_new_best:
            # copy to release the decoded data 
            soln = ranks[order[0], :].copy(), whichCities[order[0],:].copy() # <-- GTSP
        if not looper.should_continue():
            break
        
        pop = _BRKGA_evolve_GTSP(nums_cities, pop, pop_sizes, crossover_bias, rng) # <-- GTSP
        
    return Solution_report(soln, cost_vals_hist, looper.state)