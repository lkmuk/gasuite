"""Generalized TSP
"""
import numpy as np
import numba 

from dataclasses import dataclass
import matplotlib.axes
from enum import Enum


@dataclass
class Geometric_GTSP:
    """Geometric Generalized TSP
    
    Similar to GTSPLIB
    --------------------
    
    * City indices are "flattened".
    * Each cluster needs NOT occupy a contiguous city index space
    
    Unlike GTSPLIB
    ---------------------
    * we use 0-indexing instead of 1-indexing
    
    """
    xy: np.ndarray # all cities
    clusters: list[np.ndarray]
    @property
    def nums_cities_per_cluster(self) -> list[int]:
        return [len(c) for c in self.clusters]
    @property
    def num_clusters(self) -> int:
        return len(self.clusters)
    @property
    def num_cities(self) -> int:
        return len(self.xy)

    def _prepare_lut(self):
        """Prepare lookup table for mapping city index representations
        
        Pre-compute lookup tables for mapping a flatten city 
        index to a nested index {cluster_index}.{local_index}
        
        Why convert flattened city indices into the nested representation?
        
        1. checking if a tour is valid
        2. some algorithm, e.g., BRKGA loves nested representation
        """
        self.which_cluster : np.ndarray = np.zeros((self.num_cities,), dtype=int)
        self.which_localCity: np.ndarray = np.zeros((self.num_cities,), dtype=int)
        for cluster_id, members in enumerate(self.clusters):
            for local_city_index, flattened_city_index in enumerate(members):
                self.which_cluster[flattened_city_index] = cluster_id
                self.which_localCity[flattened_city_index] = local_city_index

    def __post_init__(self):        
        assert self.xy.ndim == 2

        # a quick test
        assert self.num_cities == sum(self.nums_cities_per_cluster)
        # a thorough test: are the clusters really disjoint? and covering [0,..., N-1]
        counter = np.zeros((self.num_cities,))
        for cluster_idx, members in enumerate(self.clusters):
            assert len(set(members)) == len(members), f"cluster {cluster_idx} contains duplicate city"
            for city_id in members:
                if city_id >= self.num_cities or city_id < 0:
                    raise ValueError(f"Invalid city ID {city_id} in cluster {cluster_idx}")
                if counter[city_id] == 1:
                    raise ValueError(f"City ID {city_id} already appeared in another cluster (of a lower index)")
                counter[city_id] = 1
        # technically we don't need this any more at this point but just to be clear
        assert np.all(counter==1), "some cities are not assigned to any clusters"
        
        self._prepare_lut()
        
    def extract_tour_clusterSequence(self, tour) -> np.ndarray:
        return self.which_cluster[tour]
    
    def extract_tour_localCityIndices_sortedByTourPosition(self, tour) -> np.ndarray:
        return self.which_localCity[tour]
    
    def convert_tour_asNestedIndices(self, tour, sort_localId = True) -> tuple[np.ndarray,np.ndarray]:
        """
        `sort_localId`: 
            If true, the second returned array of local (in-cluster) city indices 
            will be sorted by ascending cluster ID.
            Else, it will be sorted by the tour position instead
        """
        which_cluster = self.extract_tour_clusterSequence(tour)
        local_idx = self.extract_tour_localCityIndices_sortedByTourPosition(tour)
        if not sort_localId:
            return which_cluster, local_idx
        else:
            cluster_tour_pos = np.argsort(which_cluster)
            local_idx = local_idx[cluster_tour_pos] # rearrange it to be arranged by cluster index
            return which_cluster, local_idx
    
    def assert_tour_is_valid(self, tour):
        cluster_seq = self.extract_tour_clusterSequence(tour)
        assert len(set(cluster_seq)) == len(cluster_seq), "Some cluster is visited more than once by the tour"
        assert len(cluster_seq) == self.num_clusters, "Some cluster is missing from the tour"        
    
    
    def eval_tour_cost(self, tour, ord=2) -> float:
        """the tour shall exclude home position
        """
        pts = self.xy[[*tour, tour[0]]]
        return np.sum(np.linalg.norm(np.diff(pts, axis=0), axis=1, ord=ord)).item()
    
    def visualize_problem(self, ax: matplotlib.axes.Axes, txtOffset = (0.2, 0.2), c='k', marker='^', ls=' ', **kwargs):
        """
        Remarks
        --------------
        * All parameters after `txtOffset` concerns `matplotlib.pyplot.plot`, 
          so see the documentation there.
        * You can disable the text annotation by setting `txtOffset` as `None`.
        * We use text annotation instead of labels/legend to distinguish contours
          because we anticipate there will be many contours (well exceeding 6).
        """
        for cluster_idx, members in enumerate(self.clusters):
            ax.plot(*self.xy[members].T, c=c, marker=marker, ls=ls, **kwargs)
            startXY = members[0]
            if txtOffset is not None:
                ax.annotate(str(cluster_idx), xy=startXY, xytext=startXY + txtOffset, arrowprops=dict(arrowstyle="->"))

    def visualize_tour(self, ax: matplotlib.axes.Axes, tour, **plot_kwargs):
        tour = [*tour, tour[0]]
        ax.plot(*self.xy[tour].T, **plot_kwargs)

    def serialize_gtsplib(self, instance_name: str, comments: list[str] = []) -> str:
        assert self.xy.shape[1] in (2,3), "GTSPLIB support only 2D/3D"
        
         # first the metadata part 
        # (called the "specification part" in [Reinelt91])
        lines = []
        lines.append("NAME: " + instance_name)
        
        lines += ["COMMENT: " + str(d) for d in comments]
               
        lines +=[
            "TYPE: GTSP", # symmetric when there is no open paths
            f"DIMENSION: {self.num_cities:d}",
            f"GTSP_SETS : {self.num_clusters:d}",
            f"EDGE_WEIGHT_TYPE: EUC_{self.xy.shape[1]}D", 
            "EDGE_WEIGHT_FORMAT : FUNCTION",
        ]
        
        # now the data part
        # remember that TSPLIB...
        # * uses 1-based indexing
        # * requires a redundant numbering for each row...
        
        lines.append("NODE_COORD_SECTION:")
        lines += [
            f"{1+ii:8d} {float(xy[0]):10.3f} {float(xy[1]):10.3f}" 
            for ii, xy in enumerate(self.xy)
        ]

        lines.append("GTSP_SET_SECTION:")
        lines += [
            f"{cluster_id+1:d} " + " ".join(map(str, members+1))+" -1"
            for cluster_id, members in enumerate(self.clusters)
        ]
        
        lines.append("EOF")
        final = "\n".join(lines)
        return final
        
        


class _Tour_parse_state(Enum):
    seeking_type = 0
    seeking_dim = 1
    seeking_tour = 2
    

def read_tour_tsplib(filepath, ensure_no_revisits=True, generalized=True) -> np.ndarray:
    """parse a tour file in TSPLIB format and convert it to 0-based indexing

    Expected format of the file:
    ```
    ...
    TYPE : TOUR
    DIMENSION : 34523
    TOUR_SECTION
    ..
    ..
    -1
    EOF
    ```
    
    Note that (1) no preceding space in each line and 
    (2) the order matters    
    
    """   
    state = _Tour_parse_state.seeking_type
    tour_length = 0
    with open(filepath, 'r') as f:
        line = f.readline() 
        line = line.capitalize()
        while line:
            if state == _Tour_parse_state.seeking_type:
                if line.startswith("TYPE") and line.endswith("TOUR\n"):
                    state = _Tour_parse_state.seeking_dim
            elif state == _Tour_parse_state.seeking_dim:
                if line.startswith("DIMENSION"):
                    tour_length = int(line.split(' ')[-1])
                    state = _Tour_parse_state.seeking_tour
            elif state == _Tour_parse_state.seeking_tour:
                if line.startswith("TOUR_SECTION"):
                    tour_data = np.zeros((tour_length,), dtype=int)
                    for ii in range(tour_length):
                        d = int(f.readline()[:-1]) # get rid of the last character "\n"
                        # For TSP, the valid range of d is {1, 2, ..., tour_length}.
                        # For Generalized TSP, we won't check the upper bound here 
                        if d <= 0:
                            raise ValueError(f"Invalid TOUR_SECTION: city index {d} is too small")
                        if (not generalized) and d > tour_length:
                            raise ValueError(f"Invalid TOUR_SECTION: city index {d} is too large")
                        tour_data[ii] = d
                    # reinstate 0-indexing
                    tour_data -= 1
                    
                    # penultimate line
                    pline = f.readline()
                    if pline != "-1\n":
                        raise ValueError("Invalid TOUR_SECTION: longer than the declared dimension")
                    
                    if ensure_no_revisits and len(tour_data) != len(set(tour_data)):
                        raise ValueError("Invalid TOUR_SECTION: some city is visited more than once.")
                    
                    return tour_data.reshape(-1)
                
            line = f.readline()
    
    raise ValueError("Invalid tour file because it fails to progress while " + str(state.name))
    
