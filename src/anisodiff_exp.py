import numpy as np
from numba import njit, prange
from numba.typed import Dict
from numba import types


# Stopping criterion for linear anisotropic
# image diffusion: a fingerprint image
# enhancement case

# https://jivp-eurasipjournals.springeropen.com/articles/10.1186/s13640-016-0105-x

def spatial_entropy(L):
    """
    Compute spatial entropy H_t(L) as defined in Eq. (17) of the paper.

    Parameters:
        L: 2D numpy array (image at time t with real values)

    Returns:
        H: float (spatial entropy)
    """
    # Normalize to get p_t(x, y)
    C = np.sum(L)
    if C == 0:
        return 0.0  # avoid division by zero

    p = L / C
    # p = p[p > 0]  # exclude zeros to avoid log(0)

    H = -np.sum(p * np.log(p + 1e-12))
    return H


def compute_h_t(h_entropy, t_steps):
    if t_steps[0] > 0:
        dx1 = t_steps[1] - t_steps[0]
        dx2 = t_steps[2] - t_steps[1]
        a = -dx2 / (dx1 * (dx1 + dx2))
        b = (dx2 - dx1) / (dx1 * dx2)
        c = dx1 / (dx2 * (dx1 + dx2))
        return a * h_entropy[0] + b * h_entropy[1] + c * h_entropy[2]
    elif t_steps[1] > 0:
        return (h_entropy[2] - h_entropy[1]) / (t_steps[2] - t_steps[1])
    else:
        return np.inf


@njit(parallel=True)
def _anisotropic_diffusion_step(N, values, node_labels, predecessors, rows, cols, lifetimes_dict, maxdist_dict, alpha, spatial_sigma,
                                intensity_sigma):
    new_values = np.zeros(N, dtype=np.float64)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for node in prange(N):
        current_value = values[node]
        current_label = node_labels[node]
        xi, yi = divmod(node, cols)

        spatial_sigma = maxdist_dict[current_label]
        intensity_sigma = lifetimes_dict[current_label]

        # collect neighbors: first parent (if any), then 8-connected neighbors
        neighbor_idxs = []
        neighbor_idxs.append(predecessors[node])

        for dr, dc in directions:
            nr, nc = xi + dr, yi + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                j = int(nr * cols + nc)
                # if j < N and labels[j] == current_label:
                neighbor_idxs.append(j)

        # Compute bilateral update
        weight_sum = 0.0
        update = 0.0

        for k, n in enumerate(neighbor_idxs):
            n_value = values[n]
            xj, yj = divmod(n, cols)

            spatial_dist = np.sqrt((xi - xj) ** 2 + (yi - yj) ** 2)
            intensity_diff = n_value - current_value

            w_spatial = np.exp(- (spatial_dist ** 2) / (2 * spatial_sigma ** 2))
            w_intensity = np.exp(- (intensity_diff ** 2) / (2 * intensity_sigma ** 2))

            weight = w_intensity * w_spatial

            update += weight * intensity_diff
            weight_sum += weight

        # if weight_sum > 0:
        #    update /= weight_sum
        new_values[node] = current_value + alpha * update

    return new_values


def anisotropic_graph_diffusion(input_tree,
                                steps=500,
                                alpha=0.1,
                                spatial_sigma=1,
                                intensity_sigma=0.1,
                                gth=None):
    tree = input_tree.copy()

    rows = tree.image_info['rows']
    cols = tree.image_info['cols']
    node_labels = tree.get_node_labels()
    lifetimes_dict = tree.get_lifetimes('dict')
    maxdist_dict = tree.get_max_distances('dict')

    lifetimes_numba = Dict.empty(key_type=types.int64, value_type=types.float64)
    maxdist_numba = Dict.empty(key_type=types.int64, value_type=types.float64)
    for k, v in lifetimes_dict.items():
        lifetimes_numba[int(k)] = float(v)

    for k, v in maxdist_dict.items():
        maxdist_numba[int(k)] = float(v)

    N = len(node_labels)
    tree_predecessors = tree.get_predecessors()
    predecessors = -np.ones((N,), dtype=np.int32)
    for i in range(N):
        preds = tree_predecessors.get(i, [])
        if preds:
            predecessors[i] = preds[0]

    h_entropy = [0,0,0]
    t_steps = [0, 0, 0]
    h_change = [np.inf, np.inf, np.inf]
    prev_mse = np.inf
    score = np.inf
    for step in range(steps):
        values = tree.get_node_values()

        new_values = _anisotropic_diffusion_step(N, values, node_labels, predecessors, rows, cols,
                                                 lifetimes_numba, maxdist_numba, alpha, spatial_sigma, intensity_sigma)

        if gth is not None:
            mse = np.mean(np.square(new_values - gth.flatten()))
            if mse < prev_mse:
                prev_mse = mse
            else:
                score = prev_mse
                break
        else:
            entropy = spatial_entropy(new_values)
            h_entropy.pop(0)
            h_entropy.append(entropy)
            t_steps.pop(0)
            t_steps.append(np.log(step + 1))
            h_change.pop(0)
            h_change.append(compute_h_t(h_entropy, t_steps))

            if h_change[-2] < h_change[-1] and h_change[-1] != np.inf:
                score = -h_entropy[0]
                break

        tree.set_node_values(new_values.tolist())

    return tree, step, score

