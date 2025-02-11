from gasuite.brkga import *
import numpy as np


def test_generate_permutation():
    rng = np.random.default_rng(34)
    expected_shape = (1000, 5)
    
    randKeys = generate_permutations(rng, *expected_shape)
    
    assert randKeys.shape == expected_shape
    assert (randKeys >= 0).all() is np.True_, "each random key must be in the range [0,1)"
    assert (randKeys < 1).all() is np.True_, "each random key must be in the range [0,1)"
    np.testing.assert_allclose(randKeys.reshape(-1).mean(axis=0), 0.5, err_msg="Bad sampling statistics", atol=0.03)
    
  
def test_generate_whichCity():
    rng = np.random.default_rng(34)
    sets_sizes = [3, 4, 2, 6, 10]
    expected_shape = (1000, len(sets_sizes))
    
    flags = generate_whichCity(rng, expected_shape[0], sets_sizes)
    
    assert flags.shape == expected_shape    
    assert flags.dtype is np.dtype(int)
    channelWiseMin = np.min(flags, axis=0)
    channelWiseMax = np.max(flags, axis=0)
    np.testing.assert_equal(channelWiseMin >= 0, True)
    np.testing.assert_equal(channelWiseMax < sets_sizes, True)
    # only check the mean for now
    # TODO: check if it's a uniform distribution
    expected_integral_portion_means = (np.array(sets_sizes) - 1)*0.5
    np.testing.assert_allclose(flags.mean(axis=0), expected_integral_portion_means, rtol=0.06)
    
def test_decode_rank():
    random_keys = np.array([
        [0.8, 0.41, 0.3, 0.4],
        [0.1, 0.2, 0.8, 0.3],
    ])
    expected_permutation = np.array([
        [2, 3, 1, 0],
        [0, 1, 3, 2]
    ])
    
    computed = decode_rank(random_keys)
    np.testing.assert_array_equal(computed, expected_permutation)
    
if __name__ == "__main__":
    test_generate_permutation()