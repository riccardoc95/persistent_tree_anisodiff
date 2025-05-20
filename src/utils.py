import numpy as np
from scipy import ndimage
from pixhomology.exp import image_to_graph
from collections import defaultdict
from scipy.spatial import ConvexHull, distance


def check_neighbors(values):
    center = values[4]
    neighbors = np.delete(values, 4)
    if np.all(neighbors == neighbors[0]):
        return neighbors[0]
    else:
        return center


def image_to_tree(image):
    rows, cols =  image.shape
    min_ = image.min()
    max_ = image.max()

    image = (image - min_) / (max_ - min_)
    edges, _ = image_to_graph(image)
    labels = ndimage.generic_filter(edges, check_neighbors, size=3, mode='constant', cval=np.nan)
    node_labels = labels.flatten().tolist()
    list_edges = edges.flatten().tolist()
    node_values = image.flatten().tolist()
    root = np.argmax(node_values)

    unique_labels = np.unique(node_labels)

    # Compute min and max for each label
    mins = ndimage.minimum(node_values, node_labels, index=unique_labels)
    maxs = ndimage.maximum(node_values, node_labels, index=unique_labels)
    label_to_death = dict(zip(unique_labels, mins.tolist()))
    label_to_birth = dict(zip(unique_labels, maxs.tolist()))

    image_info = {
        'rows': rows,
        'cols': cols,
        'min': min_,
        'max': max_
    }
    tree_info = {
        'root': root,
        'node_labels': node_labels,
        'list_edges': list_edges,
        'node_values': node_values,
        'label_to_birth': label_to_birth,
        'label_to_death': label_to_death,
        'components': unique_labels
    }
    return image_info, tree_info


def max_jump_threshold(lifetimes):
    """
    from: https://www.frontiersin.org/journals/applied-mathematics-and-statistics/articles/10.3389/fams.2024.1260828/full
    :param lifetimes: array of lifetimes
    :return: threshold value
    """
    L = lifetimes
    SL = np.sort(L)
    J = SL[1:] - SL[:-1]
    i_max = np.argmax(J)
    thr = (SL[i_max-2] + SL[i_max-1]) / 2

    return thr


def compute_max_distances(labels, M):
    max_distances = {}
    for idx, label in enumerate(labels):
        xc, yc = divmod(label, M)
        x, y = divmod(idx, M)
        d = np.sqrt((xc - x) ** 2 + (yc - y) ** 2)
        if label in max_distances:
            d_back = max_distances[label]
            if d > d_back:
                max_distances[label] = d_back
        else:
            max_distances[label] = d
    return max_distances