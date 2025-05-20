import numpy as np
from numba import njit, prange, int32
from numba.typed import Dict



@njit(parallel=True)
def _anisotropic_diffusion_step(N, values, node_labels, predecessors, rows, cols, alpha, spatial_sigma, intensity_sigma):
    new_values = np.zeros(N, dtype=np.float64)

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    for node in prange(N):
        current_value = values[node]
        current_label = node_labels[node]
        xi, yi = divmod(node, cols)

        # collect neighbors: first parent (if any), then 8-connected neighbors
        neighbor_idxs = []
        #for pred in :
        #if pred != -1:
        neighbor_idxs.append(predecessors[node])

        for dr, dc in directions:
            nr, nc = xi + dr, yi + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                j = int(nr * cols + nc)
                #if j < N and labels[j] == current_label:
                neighbor_idxs.append(j)

        # Compute bilateral update
        weight_sum = 0.0
        update = 0.0

        for k, n in enumerate(neighbor_idxs):
            n_value = values[n]
            xj, yj = divmod(n, cols)

            spatial_dist = np.sqrt((xi - xj)**2 + (yi - yj)**2)
            intensity_diff = n_value - current_value

            w_spatial = np.exp(- (spatial_dist**2) / (2 * spatial_sigma**2))
            w_intensity = np.exp(- (intensity_diff**2) / (2 * intensity_sigma**2))

            weight =  w_intensity * w_spatial

            update += weight * intensity_diff
            weight_sum += weight

        #if weight_sum > 0:
        #    update /= weight_sum
        new_values[node] = current_value + alpha * update

    return new_values



def anisotropic_graph_diffusion(input_tree,
                                steps=100,
                                alpha=0.1,
                                spatial_sigma=0.1,
                                intensity_sigma=0.1,
                                gth=None):

    tree = input_tree.copy()

    rows = tree.image_info['rows']
    cols = tree.image_info['cols']
    node_labels = tree.get_node_labels()

    N = len(node_labels)
    tree_predecessors = tree.get_predecessors()
    predecessors = -np.ones((N,), dtype=np.int32)
    for i in range(N):
        preds = tree_predecessors.get(i, [])
        if preds:
            predecessors[i] = preds[0]

    prev_mse = np.inf
    for step in range(steps):
        values = tree.get_node_values()

        new_values = _anisotropic_diffusion_step(N, values, node_labels, predecessors, rows, cols,
                                                 alpha, spatial_sigma, intensity_sigma)
        if gth is not None:
            mse = np.mean(np.square(new_values - gth.flatten()))
            if mse < prev_mse:
                prev_mse = mse
                #print(mse)
            else:
                break
        tree.set_node_values(new_values.tolist())

    return tree
