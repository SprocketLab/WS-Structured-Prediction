import fire
import sys
import os
import core.pse as pse
import numpy as np
import itertools

from core.ranking_utils import RankingUtils
from core.ws_ranking import Ranking
import numpy as np

from tensor_decomp import mixture_tensor_decomp_full
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


def main(k=2, d=4, n=10000, max_m=3, debug=False):
    """ """

    # w = np.ones(k) / k  # TODO de-uniformize the prior
    w = np.sort(np.array([0.25, 0.75]))  # Always sort
    theta_star = np.array([0.99, 0.9, 1.0])  # TODO Change

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

    ###### Get population-level vectors mu_a|y ######
    print("probs.shape", probs.shape)
    print("Yspace_emb.shape", Yspace_emb.shape)
    mu_pop = np.einsum("ijk,kl->ijl", probs, Yspace_emb).transpose(0, 2, 1)
    mu_pop_pos, mu_pop_neg = mu_pop[:, :tk, :], mu_pop[:, tk:, :]
    print("mu_pop_pos.shape", mu_pop_pos.shape)
    print("mu_pop_neg.shape", mu_pop_neg.shape)

    if debug:
        # Run over all permutations
        # for each one get the prob for center y
        # get the embedding for this permutation
        # take the sum of probability * embedding for all of these
        # k x embedding_dim
        # Put these into M2, M3, try to recover
        # positive
        (
            w_rec,
            mu_pop_pos_rec1,
            mu_pop_pos_rec2,
            mu_pop_pos_rec3,
        ) = mixture_tensor_decomp_full(
            w, mu_pop_pos[0, :, :], mu_pop_pos[1, :, :], mu_pop_pos[2, :, :], debug=True
        )
        mu_pop_pos_rec = np.array([mu_pop_pos_rec1, mu_pop_pos_rec2, mu_pop_pos_rec3])
        # negative
        (
            w_rec,
            mu_pop_neg_rec1,
            mu_pop_neg_rec2,
            mu_pop_neg_rec3,
        ) = mixture_tensor_decomp_full(
            w, mu_pop_neg[0, :, :], mu_pop_neg[1, :, :], mu_pop_neg[2, :, :], debug=True
        )
        mu_pop_neg_rec = np.array([mu_pop_neg_rec1, mu_pop_neg_rec2, mu_pop_neg_rec3])
        # Ok, we get perfect recovery here. Moving on.

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

    ###### Tensor decomposition for + - ######
    ###### TODO run this several times ######
    # w_hat_neg, mu_hat_neg = get_estimated_w_mu(L_emb_neg, k)
    # w_hat_pos, mu_hat_pos = get_estimated_w_mu(L_emb_pos, k)

    print(L_emb_pos.shape)
    (w_rec, mu_hat_pos1, mu_hat_pos2, mu_hat_pos3,) = mixture_tensor_decomp_full(
        np.ones(n) / n,
        L_emb_pos[0, :, :].T,
        L_emb_pos[1, :, :].T,
        L_emb_pos[2, :, :].T,
        k=k,
        debug=debug,
    )
    print(mu_hat_pos1.shape)

    T_pos = np.einsum(
        "i,ji,ki,li->jkl",
        w,
        mu_pop_pos[0, :, :],
        mu_pop_pos[1, :, :],
        mu_pop_pos[2, :, :],
    )
    T_pos_hat = np.einsum(
        "i,ji,ki,li->jkl",
        np.ones(n) / n,
        mu_hat_pos1,
        mu_hat_pos2,
        mu_hat_pos3,
    )
    print(compute_sign_mse(T_pos, T_pos_hat))

    print(L_emb_neg.shape)
    (w_rec, mu_hat_neg1, mu_hat_neg2, mu_hat_neg3,) = mixture_tensor_decomp_full(
        np.ones(n) / n,
        L_emb_neg[0, :, :].T,
        L_emb_neg[1, :, :].T,
        L_emb_neg[2, :, :].T,
        k=k,
        debug=debug,
    )
    print(mu_hat_neg1.shape)

    T_neg = np.einsum(
        "i,ji,ki,li->jkl",
        w,
        mu_pop_neg[0, :, :],
        mu_pop_neg[1, :, :],
        mu_pop_neg[2, :, :],
    )
    T_neg_hat = np.einsum(
        "i,ji,ki,li->jkl",
        np.ones(n) / n,
        mu_hat_neg1,
        mu_hat_neg2,
        mu_hat_neg3,
    )
    print(compute_sign_mse(T_neg, T_neg_hat))

    quit()

    print(w_rec / w_rec.sum())
    mu_hat_pos = np.array([mu_hat_pos1, mu_hat_pos2, mu_hat_pos3])
    print(f"mse pos: {compute_sign_mse(mu_hat_pos, mu_pop_pos)}")

    print(L_emb_neg.shape)
    (w_rec, mu_hat_neg1, mu_hat_neg2, mu_hat_neg3,) = mixture_tensor_decomp_full(
        np.ones(n) / n,
        L_emb_neg[0, :, :].T,
        L_emb_neg[1, :, :].T,
        L_emb_neg[2, :, :].T,
        k=k,
        debug=debug,
    )
    print(w_rec / w_rec.sum())
    mu_hat_neg = np.array([mu_hat_neg1, mu_hat_neg2, mu_hat_neg3])
    print(f"mse neg: {compute_sign_mse(mu_hat_neg, mu_pop_neg)}")

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


if __name__ == "__main__":
    fire.Fire(main)
