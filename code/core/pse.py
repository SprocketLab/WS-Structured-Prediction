import numpy as np

def pseudo_dist(u,v, tk):
    return np.sqrt(np.linalg.norm(u[:tk]-v[:tk])**2 - np.linalg.norm(u[tk:]-v[tk:])**2)

def pseudo_embedding(D, dim=0):
    n = D.shape[0]
    
    bilin_form = np.zeros((n-1, n-1))
    for i in range(1,n):
        for j in range(1,n):
             bilin_form[i-1,j-1] = 1/2 * (D[0,i]+ D[0,j]- D[i,j])
    w, v = np.linalg.eigh(bilin_form)
    w = np.real(w)

    idx_pos = np.argwhere((w > 0) & (w > 10e-12)).flatten()
    idx_neg = np.argwhere((w < 0) & (w < -10e-12)).flatten()
    idx_zero = np.argwhere((10e-12 >= w) & (-10e-12 <= w)).flatten()
    idx = np.concatenate([idx_pos, idx_neg])
    
    w = w[idx]
    # should be refactored
    idx_pos = np.argwhere((w > 0) & (w > 10e-12)).flatten()
    idx_neg = np.argwhere((w < 0) & (w < -10e-12)).flatten()
    v = v[:,idx]
    
    w_abs = np.abs(w)
    if dim != 0: # take some smaller number of dimensons than all
        idx_sorted = w_abs.argsort()[::-1]
        idx_top = idx_sorted[:dim]
        idx_pos = np.intersect1d(idx_top, idx_pos)
        idx_neg = np.intersect1d(idx_top, idx_neg)

    pdim, ndim = len(idx_pos), len(idx_neg)
    print(pdim,ndim) 

    X = v[:, :pdim+ndim] @ np.sqrt(np.abs(np.diag(w[:pdim+ndim])))
    return np.concatenate((np.expand_dims(np.zeros(pdim+ndim), axis=0), X), axis=0), pdim