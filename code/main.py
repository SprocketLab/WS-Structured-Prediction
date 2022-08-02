import fire
import core.pse as pse
import numpy as np
import itertools
import random
import numpy as np

from sklearn.datasets import make_classification
from tensor_decomp import mixture_tensor_decomp_full, mse_perm, mse
from core.ranking_utils import RankingUtils, Ranking
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression


def mapped_make_classification(y, expansion=2):
    x_c, y_c = make_classification(n_samples=len(y) * expansion)
    x_c_0 = x_c[np.where(y_c == 0)]
    y_c_0 = y_c[np.where(y_c == 0)]
    x_c_1 = x_c[np.where(y_c == 1)]
    y_c_1 = y_c[np.where(y_c == 1)]
    x = np.zeros((len(y), x_c.shape[1]))
    y_recon = np.zeros(len(y))
    x[np.where(y == 0)] = x_c_0[: (sum(y == 0))]
    x[np.where(y == 1)] = x_c_1[: (sum(y == 1))]
    # Sanity check
    y_recon[np.where(y == 0)] = y_c_0[: (sum(y == 0))]
    y_recon[np.where(y == 1)] = y_c_1[: (sum(y == 1))]
    assert sum(y_recon - y) == 0.0
    return x, y


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


def main(k=2, d=4, n=10000, max_m=3, debug=False, seed=42, theta_1=None):
    """ """
    random.seed(seed)
    np.random.seed(seed)

    w = np.sort(np.array([0.2, 0.8]))
    print(f"using w \t {w}")

    theta_star = np.array([0.0, 0.0, 1.0])
    if theta_1 is not None:
        # Assume theta_1 is in [0, 1]
        theta_star = np.array([0.0, 1.0 - theta_1, theta_1])
    print(f"using theta_star \t {theta_star}")

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
        # savedir="T_pos_td",
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
        # savedir="T_neg_td",
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
        # savedir="T_pos_hat",
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
        "i,ji,ki,li->jkl", w_rec, mu_hat_pos1, mu_hat_pos2, mu_hat_pos3
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
        # savedir="T_neg_hat",
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
        "i,ji,ki,li->jkl", w_rec, mu_hat_neg1, mu_hat_neg2, mu_hat_neg3
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
    print("using population (+)")
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
    print("using population (-)")
    print(w @ exp_sq_dist_neg.T)

    print("using population (pos-neg)")
    exp_sq_dist_pop = w @ (exp_sq_dist_pos - exp_sq_dist_neg).T
    print(exp_sq_dist_pop)

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

    print("using TD (pos-neg)")
    exp_sq_dist_TD = (w_rec_pos @ exp_sq_dist_TD_pos.T) - (
        w_rec_neg @ exp_sq_dist_TD_neg.T
    )
    print(exp_sq_dist_TD)

    print(
        f"err(exp_sq_dist_pop, exp_sq_dist_TD) \t {np.abs(exp_sq_dist_pop - exp_sq_dist_TD)}"
    )

    thetas_td = np.ones_like(exp_sq_dist_TD) / exp_sq_dist_TD
    thetas_td /= thetas_td.sum()

    ######### Perform UWS distance estimation #########
    # print(L_emb.shape)
    # TODO check if this is right?
    L1 = L_emb[0]
    L2 = L_emb[1]
    L3 = L_emb[2]
    e_L1_L2 = np.mean([pse.pseudo_dist(L1[i], L2[i], tk=tk) for i in range(n)])
    e_L1_L3 = np.mean([pse.pseudo_dist(L1[i], L3[i], tk=tk) for i in range(n)])
    e_L2_L3 = np.mean([pse.pseudo_dist(L1[i], L3[i], tk=tk) for i in range(n)])
    # print(e_L1_L2, e_L1_L3, e_L2_L3)
    e_L1_y = 1.0 / 2 * (e_L1_L2 + e_L1_L3 - e_L2_L3)
    e_L2_y = 1.0 / 2 * (e_L1_L2 + e_L2_L3 - e_L1_L3)
    e_L3_y = 1.0 / 2 * (e_L1_L3 + e_L2_L3 - e_L1_L2)
    exp_sq_dist_UWS = np.array([e_L1_y, e_L2_y, e_L3_y])
    print(
        f"err(exp_sq_dist_pop, exp_sq_dist_UWS) \t {np.abs(exp_sq_dist_pop - exp_sq_dist_UWS)}"
    )

    thetas_uws = np.ones_like(exp_sq_dist_UWS) / exp_sq_dist_UWS
    thetas_uws /= thetas_uws.sum()

    ######### End model taining using TD and UWS parameter estimates #########

    print("### THETA COMPARISON ###")
    print("thetas*:\t", theta_star / theta_star.sum())
    print("thetas_td:\t", thetas_td)
    print("thetas_ws:\t", thetas_uws)

    def run_inference(thetas):
        """Assumes two centers"""
        # Compute the weighted Fréchet mean
        fréchets_td = np.einsum("i,ijk->jk", thetas, L_emb)
        centers = np.tile(Y_emb_unique, (n, 1, 1))
        dists_0 = [
            pse.pseudo_dist(fréchets_td[i], centers[i, 0, :], tk=tk) for i in range(n)
        ]
        dists_1 = [
            pse.pseudo_dist(fréchets_td[i], centers[i, 1, :], tk=tk) for i in range(n)
        ]
        dists_to_centers = np.array([dists_0, dists_1])
        preds = np.argmin(dists_to_centers, axis=0)
        return preds

    y_true = Y_inds
    preds_uws = run_inference(thetas_uws)
    preds_td = run_inference(thetas_td)
    lm_acc_uws = accuracy_score(y_true[n // 2 :], preds_uws[n // 2 :])
    lm_acc_td = accuracy_score(y_true[n // 2 :], preds_td[n // 2 :])
    print(f"TD LM accuracy: \t{lm_acc_td}")
    print(f"UWS LM accuracy:\t{lm_acc_uws}")

    # Sweep values of theta or low-n
    # train end-model
    # Dump these into the appendix and write a description
    # add imgur links
    x_true, _y = mapped_make_classification(y_true)
    assert (_y == y_true).all()  # make sure we've mapped properly

    # Train clf on
    # (x_true[:n//2], preds_uws[:n//2]) then on
    # (x_true[:n//2], preds_td[:n//2])
    # Evaluate both on
    # (x_true[n//2:], y_true[n//2:])
    ### TD end model training
    clf = LogisticRegression(random_state=seed)
    clf.fit(x_true[: n // 2], preds_td[: n // 2])
    em_acc_td = clf.score(x_true[n // 2 :], y_true[n // 2 :])
    ### UWS end model training
    clf = LogisticRegression(random_state=seed)
    clf.fit(x_true[: n // 2], preds_uws[: n // 2])
    em_acc_uws = clf.score(x_true[n // 2 :], y_true[n // 2 :])
    # Report scores
    print(f"TD EM accuracy: \t{em_acc_td}")
    print(f"UWS EM accuracy:\t{em_acc_uws}")


if __name__ == "__main__":
    fire.Fire(main)
