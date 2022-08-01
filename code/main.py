import fire
import sys
import os
import core.pse as pse
import numpy as np
import itertools
import random

from core.ranking_utils import RankingUtils, Ranking
import numpy as np

from tensor_decomp import mixture_tensor_decomp_full, mse_perm, max_ae_perm, mse
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


def kendall_tau_distance(a, b):
    order_a = list(a.permutation)
    order_b = list(b.permutation)
    pairs = itertools.combinations(range(0, len(order_a)), 2)
    distance = 0
    for x, y in pairs:
        a = order_a.index(x) - order_a.index(y)
        b = order_b.index(x) - order_b.index(y)
        if a * b < 0:
            distance += 1
    return distance


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
    # r_utils = RankingUtils(d)
    D = [[kendall_tau_distance(r1, r2) ** 2 for r2 in Yspace] for r1 in Yspace]
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


def get_probability_table(space, centers, theta, tk):
    """ """
    potentials = np.zeros((theta.shape[0], space.shape[0], centers.shape[0]))
    for i in range(space.shape[0]):  # All possible rankings
        for j in range(centers.shape[0]):
            lam = space[i, :]
            y = centers[j, :]
            # Compute un-normalized potentials
            d = pse.pseudo_dist(lam, y, tk=tk)
            potentials[:, i, j] = np.exp(-theta * d**2)

    partition = potentials.sum(axis=1, keepdims=True)
    probabilities = potentials / partition
    # Indexing should be (LF indices, Center indices, indices of specific LF values)
    probabilities = probabilities.transpose((0, 2, 1))
    return probabilities


def get_probability_table_true(space, centers, theta):
    """ """
    # r_utils = RankingUtils(4)
    potentials = np.zeros((theta.shape[0], len(space), len(centers)))
    for i in range(len(space)):  # All possible rankings
        for j in range(len(centers)):
            lam = space[i]
            y = centers[j]
            # Compute un-normalized potentials
            d = kendall_tau_distance(lam, y)
            potentials[:, i, j] = np.exp(-theta * d**2)

    partition = potentials.sum(axis=1, keepdims=True)
    probabilities = potentials / partition
    # Indexing should be (LF indices, Center indices, indices of specific LF values)
    probabilities = probabilities.transpose((0, 2, 1))
    return probabilities


def main(k=2, d=4, n=10000, max_m=3, debug=False, seed=42):
    """ """
    random.seed(seed)
    np.random.seed(seed)

    # w = np.ones(k) / k  # TODO de-uniformize the prior
    w = np.sort(np.array([0.2, 0.8]))  # Always sort
    theta_star = np.array([0.6, 0.99, 0.8])  # TODO Change

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
    probs = get_probability_table_true(
        Yspace,
        [Y[np.where(Y_inds == 0)[0][0]], Y[np.where(Y_inds == 1)[0][0]]],
        theta_star,
    )

    ###### Get population-level vectors mu_a|y ######
    print("probs.shape", probs.shape)
    print("Yspace_emb.shape", Yspace_emb.shape)
    mu_pop = np.einsum("ijk,kl->ijl", probs, Yspace_emb).transpose(0, 2, 1)
    mu_pop_pos, mu_pop_neg = mu_pop[:, :tk, :], mu_pop[:, tk:, :]
    print("mu_pop_pos.shape", mu_pop_pos.shape)
    print("mu_pop_neg.shape", mu_pop_neg.shape)

    # if debug:
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
        w,
        mu_pop_pos[0, :, :],
        mu_pop_pos[1, :, :],
        mu_pop_pos[2, :, :],
        debug=True,
        savedir="T_pos_td",
    )
    mu_pop_pos_rec = np.array([mu_pop_pos_rec1, mu_pop_pos_rec2, mu_pop_pos_rec3])
    T_pos_td = np.einsum(
        "i,ji,ki,li->jkl",
        w_rec,
        mu_pop_pos_rec1,
        mu_pop_pos_rec2,
        mu_pop_pos_rec3,
    )

    # negative
    (
        w_rec,
        mu_pop_neg_rec1,
        mu_pop_neg_rec2,
        mu_pop_neg_rec3,
    ) = mixture_tensor_decomp_full(
        w,
        mu_pop_neg[0, :, :],
        mu_pop_neg[1, :, :],
        mu_pop_neg[2, :, :],
        debug=True,
        savedir="T_neg_td",
    )
    mu_pop_neg_rec = np.array([mu_pop_neg_rec1, mu_pop_neg_rec2, mu_pop_neg_rec3])
    T_neg_td = np.einsum(
        "i,ji,ki,li->jkl",
        w_rec,
        mu_pop_neg_rec1,
        mu_pop_neg_rec2,
        mu_pop_neg_rec3,
    )
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
    print(L_emb_pos.shape)
    (w_rec, mu_hat_pos1, mu_hat_pos2, mu_hat_pos3,) = mixture_tensor_decomp_full(
        np.ones(n) / n,
        L_emb_pos[0, :, :].T,
        L_emb_pos[1, :, :].T,
        L_emb_pos[2, :, :].T,
        k=k,
        debug=debug,
        savedir="T_pos_hat",
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
        w_rec,
        mu_hat_pos1,
        mu_hat_pos2,
        mu_hat_pos3,
    )
    T_pos_samples = np.einsum(
        "i,ji,ki,li->jkl",
        np.ones(n) / n,
        L_emb_pos[0, :, :].T,
        L_emb_pos[1, :, :].T,
        L_emb_pos[2, :, :].T,
    )

    print("positive")
    print("sampled -- TD(sampled) \t\t", mse_perm(T_pos_samples, T_pos_hat))
    print("population -- sampled \t\t", mse_perm(T_pos, T_pos_samples))
    print("population -- TD(population) \t", mse_perm(T_pos, T_pos_td))
    print("population -- TD(sampled) \t", mse_perm(T_pos, T_pos_hat))

    # NOTE assumes that the vectors are in order, for convenience...
    print("Checking error of recovered components (+)...")
    print(f"MSE(mu_pop_1, mu_hat_1) \t", mse(mu_pop_pos[0, :, :], mu_hat_pos1))
    print(f"MSE(mu_pop_2, mu_hat_2) \t", mse(mu_pop_pos[1, :, :], mu_hat_pos2))
    print(f"MSE(mu_pop_3, mu_hat_3) \t", mse(mu_pop_pos[2, :, :], mu_hat_pos3))
    print(w_rec)
    w_rec_pos = w_rec
    mu_hat_pos = np.array([mu_hat_pos1, mu_hat_pos2, mu_hat_pos3])

    print(L_emb_neg.shape)
    (w_rec, mu_hat_neg1, mu_hat_neg2, mu_hat_neg3,) = mixture_tensor_decomp_full(
        np.ones(n) / n,
        L_emb_neg[0, :, :].T,
        L_emb_neg[1, :, :].T,
        L_emb_neg[2, :, :].T,
        k=k,
        debug=debug,
        savedir="T_neg_hat",
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
        w_rec,
        mu_hat_neg1,
        mu_hat_neg2,
        mu_hat_neg3,
    )
    T_neg_samples = np.einsum(
        "i,ji,ki,li->jkl",
        np.ones(n) / n,
        L_emb_neg[0, :, :].T,
        L_emb_neg[1, :, :].T,
        L_emb_neg[2, :, :].T,
    )
    print("negative")
    print("sampled -- TD(sampled) \t\t", mse_perm(T_neg_samples, T_neg_hat))
    print("population -- sampled \t\t", mse_perm(T_neg, T_neg_samples))
    print("population -- TD(population) \t", mse_perm(T_neg, T_neg_td))
    print("population -- TD(sampled) \t", mse_perm(T_neg, T_neg_hat))

    # NOTE assumes that the vectors are in order, for convenience...
    print("Checking error of recovered components (-)...")
    print(f"MSE(mu_pop_1, mu_hat_1) \t", mse(mu_pop_neg[0, :, :], mu_hat_neg1))
    print(f"MSE(mu_pop_2, mu_hat_2) \t", mse(mu_pop_neg[1, :, :], mu_hat_neg2))
    print(f"MSE(mu_pop_3, mu_hat_3) \t", mse(mu_pop_neg[2, :, :], mu_hat_neg3))
    print(w_rec)
    w_rec_neg = w_rec
    mu_hat_neg = np.array([mu_hat_neg1, mu_hat_neg2, mu_hat_neg3])

    #
    # Compute expected squared distance
    # run over all permutations
    # sq. dist x probability
    #

    # print(Yspace_emb.shape)

    Y_emb_unique = np.array(
        [Y_emb[np.where(Y_inds == 0)[0][0]], Y_emb[np.where(Y_inds == 1)[0][0]]]
    )
    Y_emb_unique_pos, Y_emb_unique_neg = Y_emb_unique[:, :tk], Y_emb_unique[:, tk:]
    Yspace_emb_pos, Yspace_emb_neg = Yspace_emb[:, :tk], Yspace_emb[:, tk:]

    ########## Compute mu (expect. squared dist) using formula
    # Positive version, population
    exp_sq_dist_pos = np.zeros((max_m, k))
    for lf in range(max_m):
        for i in range(Yspace_emb_pos.shape[0]):
            exp_sq_dist_pos[lf] += probs[lf, :, i] * (
                np.linalg.norm(Yspace_emb_pos[i]) ** 2
                + np.linalg.norm(Y_emb_unique_pos, axis=1) ** 2
                - 2 * (Yspace_emb_pos[i] @ Y_emb_unique_pos.T)
            )
    print("pos")
    print(w @ exp_sq_dist_pos.T)

    # Negative version, population
    exp_sq_dist_neg = np.zeros((max_m, k))
    for lf in range(max_m):
        for i in range(Yspace_emb_neg.shape[0]):
            exp_sq_dist_neg[lf] += probs[lf, :, i] * (
                np.linalg.norm(Yspace_emb_neg[i]) ** 2
                + np.linalg.norm(Y_emb_unique_neg, axis=1) ** 2
                - 2 * (Yspace_emb_neg[i] @ Y_emb_unique_neg.T)
            )
    print("neg")
    print(w @ exp_sq_dist_neg.T)

    print("pos - neg")
    exp_sq_dist_population = w @ (exp_sq_dist_pos - exp_sq_dist_neg).T
    print(exp_sq_dist_population)

    ### Using TD ###
    exp_sq_dist_TD_pos = np.zeros((max_m, k))
    for lf in range(max_m):
        exp_sq_dist_TD_pos[lf] = (
            (np.linalg.norm(L_emb_pos[lf, :, :], axis=1) ** 2).mean()
            + np.linalg.norm(Y_emb_unique_pos, axis=1) ** 2
            - 2 * (mu_hat_pos[lf, :, :] * Y_emb_unique_pos.T).sum(axis=0)
        )
    print("using TD (+)")
    print(w_rec_pos @ exp_sq_dist_TD_pos.T)

    exp_sq_dist_TD_neg = np.zeros((max_m, k))
    for lf in range(max_m):
        exp_sq_dist_TD_neg[lf] = (
            (np.linalg.norm(L_emb_neg[lf, :, :], axis=1) ** 2).mean()
            + np.linalg.norm(Y_emb_unique_neg, axis=1) ** 2
            - 2 * (mu_hat_neg[lf, :, :] * Y_emb_unique_neg.T).sum(axis=0)
        )
    print("using TD (-)")
    print(w_rec_neg @ exp_sq_dist_TD_neg.T)

    print("pos - neg")
    exp_sq_dist_TD = (w_rec_pos @ exp_sq_dist_TD_pos.T) - (
        w_rec_neg @ exp_sq_dist_TD_neg.T
    )
    print(exp_sq_dist_TD)

    # exp_sq_dist_TD_neg = np.zeros((max_m, k))
    # for lf in range(max_m):
    #     exp_sq_dist_TD_neg[lf] = (
    #         np.linalg.norm(mu_hat_neg[lf, :, :], axis=0) ** 2
    #         + np.linalg.norm(Y_emb_unique_neg, axis=1) ** 2
    #         - 2 * (mu_hat_neg[lf, :, :] * Y_emb_unique_neg.T).sum(axis=0)
    #     )
    #     print(mu_hat_neg[lf, :, :].shape, Y_emb_unique_neg.T.shape)
    # print(exp_sq_dist_TD_neg)
    # print(exp_sq_dist_TD_pos - exp_sq_dist_TD_neg)
    ### Using TD ###


if __name__ == "__main__":
    fire.Fire(main)
