from scipy.linalg import toeplitz
import numpy as np
import math

def bilateral_filter(j, t, y_j, y_t, delta1=1.0, delta2=1.0):
    idx1 = -1.0 * (math.fabs(j-t) **2.0)/(2.0*delta1**2)
    idx2 = -1.0 * (math.fabs(y_j-y_t) **2.0)/(2.0*delta2**2)
    weight = (math.exp(idx1)*math.exp(idx2))
    #print('args: ', j, t, y_j, y_t, weight,math.exp(idx1),math.exp(idx2) )
    return weight

def get_neighbor_idx(total_len, target_idx, H=3):
    '''
    Let i = target_idx.
    Then, return i-H, ..., i, ..., i+H, (i+H+1)
    '''
    return [np.max([0, target_idx-H]), np.min([total_len, target_idx+H+1])]

def get_neighbor_range(total_len, target_idx, H=3):
    start_idx, end_idx = get_neighbor_idx(total_len, target_idx, H)
    return np.arange(start_idx, end_idx)

def get_season_idx(total_len, target_idx, T=10, K=2, H=5):
    num_season = np.min([K, int(target_idx/T)])
    if target_idx < T:
        key_idxs = target_idx + np.arange(0, num_season+1)*(-1*T)
    else:        
        key_idxs = target_idx + np.arange(1, num_season+1)*(-1*T)
    
    idxs = list(map(lambda idx: get_neighbor_range(total_len, idx, H), key_idxs))
    season_idxs = []
    for item in idxs:
        season_idxs += list(item)
    season_idxs = np.array(season_idxs)
    return season_idxs

def get_relative_trends(delta_trends):
    init_value = np.array([0])
    idxs = np.arange(len(delta_trends))
    relative_trends = np.array(list(map(lambda idx: np.sum(delta_trends[:idx]), idxs)))
    relative_trends = np.concatenate([init_value, relative_trends])
    return relative_trends

def get_toeplitz(shape, entry):
    h, w = shape
    num_entry = len(entry)
    assert np.ndim(entry) < 2
    if num_entry < 1:
        return np.zeros(shape)
    row = np.concatenate([entry[:1], np.zeros(h-1)])
    col = np.concatenate([np.array(entry), np.zeros(w-num_entry)])
    return toeplitz(row, col)

if __name__ == '__main__':
    test_result = bilateral_filter(1,1,1,1)
    assert test_result == 1
    print(test_result)
