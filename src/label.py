import numpy as np
from skimage.measure import label


def relabel_disconnected_components_skimage(segmentation):
    new_segmentation = np.zeros_like(segmentation)
    current_label = 1
    unique_ids = np.unique(segmentation)

    for uid in unique_ids:
        if uid == 0:
            continue
        mask = segmentation == uid
        labeled = label(mask, connectivity=1)
        for i in range(1, labeled.max() + 1):
            new_segmentation[labeled == i] = current_label
            current_label += 1

    return new_segmentation