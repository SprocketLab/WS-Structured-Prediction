import fire
import sys
import os
import core.pse as pse
import core.utils as utils
import numpy as np
import itertools
import math
import networkx as nx
import matplotlib.pyplot as plt

from core.ranking_utils import RankingUtils, perm2ranking
from core.mallows import Mallows
from core.ws_ranking import *
import numpy as np
from tensorly.decomposition import parafac


def generate_true_rankings(d, n, k, w):
    """Sample random centers
    d: number of items in the ranking
    n: number of samples
    k: number of centers
    w: true prior
    returns Y the actual centers, Y_inds the center indices
    """
    r_utils = RankingUtils(d)
    centers = [r_utils.get_random_ranking() for _ in range(k)]
    Y_inds = np.random.choice(a=np.arange(k), p=w, size=n)
    Y = [centers[Y_ind] for Y_ind in Y_inds]
    return Y, Y_inds


def sample_mallows_LFs(Y, m, thetas):
    """ """
    n = len(Y)
    d = len(Y[0])
    thetas = np.array(thetas)
    r_utils = RankingUtils(d)
    # TODO reimplement this using inverse CDF
    lst_mlw = [Mallows(r_utils, theta) for theta in thetas]
    L = [[mlw.sample(y)[0] for mlw in lst_mlw] for y in Y]
    return L


# Compute psuedo-euclidean embeddings
def compute_pse_space(d, dim=0):
    """
    d: number of items in the ranking
    dim:
    """
    # Compute all possible rankings
    Yspace = []
    for perm in itertools.permutations(np.arange(d)):
        Yspace.append(Ranking(perm))

    # Compute pairwise distances b/w unique objs
    map_obj_id = {str(list(k)): v for v, k in enumerate(Yspace)}
    map_id_obj = {v: k for v, k in enumerate(Yspace)}

    # All pairwise distances
    r_utils = RankingUtils(d)
    D = [[r_utils.kendall_tau_distance(r1, r2) ** 2 for r2 in Yspace] for r1 in Yspace]
    D = np.array(D)

    Yspace_emb, tk = pse.pseudo_embedding(D, dim)
    return Yspace, Yspace_emb, tk, map_obj_id, map_id_obj


def get_estimated_w_mu(L_emb, k):
    """
    L_emb: LF output embeddings
    k: rank of tensor decomposition/number of centers
    """
    T_hat = (
        L_emb[0, :, :][:, :, None, None]
        * L_emb[1, :, :][:, None, :, None]
        * L_emb[2, :, :][:, None, None, :]
    ).mean(axis=0)

    # Compute tensor decomposition
    # TODO try different things
    w_hat, mu_hat = parafac(T_hat, rank=k, normalize_factors=True)
    return np.array(w_hat), np.array(mu_hat)


def compute_mse(a, b):
    return np.mean((a - b) ** 2)


def compute_sign_mse(a, b):
    return np.min([compute_mse(a, b), compute_mse(a, -b)])


def main(k=2, d=4, n=100, max_m=3, debug=True):
    """ """

    w = np.ones(k) / k  # TODO de-uniformize the prior
    theta_star = np.array([0.8, 0.9, 1.0])  # TODO Change

    Y, Y_inds = generate_true_rankings(d, n, k, w)
    L = sample_mallows_LFs(Y, max_m, theta_star)

    # Embeds the entire label space
    Yspace, Yspace_emb, tk, map_obj_id, map_id_obj = compute_pse_space(d)
    Yspace_emb_pos = Yspace_emb[:, :tk]
    Yspace_emb_neg = Yspace_emb[:, tk:]

    # Get LF embeddings
    L_emb = np.array(
        [
            [Yspace_emb[map_obj_id[str(L[i][lf])], :] for i in range(n)]
            for lf in range(max_m)
        ]
    )
    L_emb_pos, L_emb_neg = L_emb[:, :, :tk], L_emb[:, :, tk:]

    # Get Y center embeddings
    Y_emb = np.array([Yspace_emb[map_obj_id[str(Y[i])], :] for i in range(n)])
    Y_emb_pos, Y_emb_neg = Y_emb[:, :tk], Y_emb[:, tk:]

    # Tensor decomposition for + -
    # TODO run this several times
    w_hat_pos, mu_hat_pos = get_estimated_w_mu(L_emb_pos, k)
    w_hat_neg, mu_hat_neg = get_estimated_w_mu(L_emb_neg, k)

    if debug:
        print()
        print(f"Number of permutations: {np.math.factorial(d)}")
        print(f"Yspace_emb_pos.shape: {Yspace_emb_pos.shape}")
        print(f"Yspace_emb_neg.shape: {Yspace_emb_neg.shape}")
        print()
        print(f"L_emb.shape: {L_emb.shape}")
        print(f"Y_emb.shape: {Y_emb.shape}")
        print()
        print(f"w_hat_pos.shape: {w_hat_pos.shape}")
        print(f"mu_hat_pos.shape: {mu_hat_pos.shape}")
        print()

    # Compute true mu using expectations


if __name__ == "__main__":
    fire.Fire(main)
