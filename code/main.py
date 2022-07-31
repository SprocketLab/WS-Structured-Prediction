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
import tensorly as tl


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


def sample_LFs(Yspace, Y_inds, probs, m, d):
    """ """
    return [
        [
            Yspace[
                np.random.choice(
                    np.arange(np.math.factorial(d)),
                    p=probs[lf_ind, center_ind, :],
                    size=1,
                )[0]
            ]
            for lf_ind in range(m)
        ]
        for center_ind in Y_inds
    ]


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

    # All pairwise distances
    r_utils = RankingUtils(d)
    D = [[r_utils.kendall_tau_distance(r1, r2) ** 2 for r2 in Yspace] for r1 in Yspace]
    D = np.array(D)

    Yspace_emb, tk = pse.pseudo_embedding(D, dim)
    return Yspace, Yspace_emb, tk, map_obj_id


def three_tensor_prod(w, x, y, z):
    """
    Three-way outer product
    """
    r = x.shape[0]
    M3 = np.zeros([r, r, r])
    if len(w.shape) == 0:
        for i in range(x.shape[0]):
            for j in range(y.shape[0]):
                for k in range(z.shape[0]):
                    M3[i, j, k] += w * x[i] * y[j] * z[k]
    else:
        for a in range(w.shape[0]):
            for i in range(x.shape[0]):
                for j in range(y.shape[0]):
                    for k in range(z.shape[0]):
                        M3[i, j, k] += w[a] * x[i, a] * y[j, a] * z[k, a]
    return M3


def get_estimated_w_mu(L_emb, k):
    """
    L_emb: LF output embeddings
    k: rank of tensor decomposition/number of centers
    """
    # T_hat = (
    #    L_emb[0, :, :][:, :, None, None]
    #    * L_emb[1, :, :][:, None, :, None]
    #    * L_emb[2, :, :][:, None, None, :]
    # ).mean(axis=0)

    print("tensor")
    T_hat = np.einsum("ij,ik,il->jkl", L_emb[0], L_emb[1], L_emb[2])
    print(L_emb)
    print(T_hat.shape)

    # Compute tensor decomposition
    # TODO try different things
    weights, factors = tl.decomposition.parafac_power_iteration(T_hat, rank=k)
    T_hat_cp = tl.cp_tensor.CPTensor((weights, factors))
    T_hat_cp_norm = T_hat_cp  # tl.cp_tensor.cp_normalize(T_hat_cp)
    w_hat, mu_hat = T_hat_cp_norm.weights, T_hat_cp_norm.factors
    print(w_hat / w_hat.sum())
    quit()

    # w_hat, mu_hat = parafac(T_hat, rank=k, normalize_factors=True)
    return np.array(w_hat), np.array(mu_hat)


def compute_mse(a, b):
    """"""
    return np.mean((a - b) ** 2)


def compute_sign_mse(a, b):
    """"""
    return np.min([compute_mse(a, b), compute_mse(a, -b)])


def get_probability_table(space, centers, theta, tk):
    """ """
    potentials = np.zeros((theta.shape[0], space.shape[0], centers.shape[0]))
    for i in range(space.shape[0]):  # All possible rankings
        for j in range(centers.shape[0]):
            lam = space[i, :]
            y = centers[j, :]
            # Compute un-normalized potentials
            potentials[:, i, j] = np.exp(-theta * pse.pseudo_dist(lam, y, tk=tk) ** 2)
    partition = potentials.sum(axis=1, keepdims=True)
    probabilities = potentials / partition
    # Indexing should be (LF indices, Center indices, indices of specific LF values)
    probabilities = probabilities.transpose((0, 2, 1))
    return probabilities


def main(k=2, d=5, n=1000, max_m=3, debug=True):
    """ """

    # w = np.ones(k) / k  # TODO de-uniformize the prior
    w = np.sort(np.array([0.5, 0.5]))  # Always sort
    theta_star = np.array([0.8, 0.9, 1.0])  # TODO Change

    ###### Embed the entire label space ######
    Yspace, Yspace_emb, tk, map_obj_id = compute_pse_space(d)
    Yspace_emb_pos = Yspace_emb[:, :tk]
    Yspace_emb_neg = Yspace_emb[:, tk:]

    ###### Compute the probability table ######
    Y, Y_inds = generate_true_rankings(d, n, k, w)
    # Get Y center embeddings
    Y_emb = np.array([Yspace_emb[map_obj_id[str(Y[i])], :] for i in range(n)])
    Y_emb_pos, Y_emb_neg = Y_emb[:, :tk], Y_emb[:, tk:]
    # Compute the full probability table exactly
    probs = get_probability_table(Yspace_emb, np.unique(Y_emb, axis=0), theta_star, tk)

    ###### Sample proportionally to the probabilities ######
    L = sample_LFs(Yspace, Y_inds, probs, max_m, d)

    ###### Get sampled LF embeddings ######
    L_emb = np.array(
        [
            [Yspace_emb[map_obj_id[str(list(L[i][lf]))], :] for i in range(n)]
            for lf in range(max_m)
        ]
    )
    L_emb_pos, L_emb_neg = L_emb[:, :, :tk], L_emb[:, :, tk:]

    ###### Get population-level vectors mu_a|y ######
    mu_pop = []
    for center_ind in range(k):
        L_probs = np.array(
            [
                [
                    probs[lf, center_ind, map_obj_id[str(list(L[i][lf]))]]
                    for i in range(n)
                ]
                for lf in range(max_m)
            ]
        )
        mu_pop.append((L_emb * L_probs[:, :, None]).mean(axis=1))
    # TODO Not sure if it makes sense to compute these in this way...
    mu_pop = np.stack(mu_pop).transpose((1, 2, 0))
    mu_pop_pos, mu_pop_neg = mu_pop[:, :tk, :], mu_pop[:, tk:, :]
    print(f"mu_pop_pos.shape: {mu_pop_pos.shape}")
    print(f"mu_pop_neg.shape: {mu_pop_neg.shape}")

    ###### Computing the tensor product and running ######
    ###### tensor decomposition as a sanity check ######
    T_pop = sum(
        [
            w[i]
            * (
                mu_pop_pos[0, :, i][:, None, None]
                * mu_pop_pos[1, :, i][None, :, None]
                * mu_pop_pos[2, :, i][None, None, :]
            )
            for i in range(k)
        ]
    )
    w_hat_pop, mu_hat_pop = parafac(T_pop, rank=k, normalize_factors=True)
    print(w_hat_pop / np.sum(w_hat_pop))

    ###### Tensor decomposition for + - ######
    ###### TODO run this several times ######
    w_hat_neg, mu_hat_neg = get_estimated_w_mu(L_emb_neg, k)
    w_hat_pos, mu_hat_pos = get_estimated_w_mu(L_emb_pos, k)

    print(w_hat_pos / np.sum(w_hat_pos))
    quit()

    print(mu_hat_pos.shape)
    print(mu_hat_neg.shape)

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
