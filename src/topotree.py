import numpy as np
import copy

from scipy.spatial import cKDTree
from scipy import ndimage

from .utils import image_to_tree
from .utils import max_jump_threshold
from .utils import compute_max_distances


class Tree:
    def __init__(self):
        self.image_info = {'rows': 0, 'cols': 0, 'min': 0, 'max': 0}

        self.predecessors = {}
        self.successors = {}
        self.node_values = []
        self.node_labels = []

        self.components = []
        self.label_to_birth = {}
        self.label_to_death = {}

        self.root = None

    def __len__(self):
        return len(self.node_values)

    def __getitem__(self, index):
        value = self.node_values[index]
        label = self.node_labels[index]
        birth = self.label_to_birth[label]
        death = self.label_to_death[label]
        if index in self.predecessors:
            predecessors = self.predecessors[index]
        else:
            predecessors = []
        if index in self.successors:
            successors = self.successors[index]
        else:
            successors = []

        return {'value': value,
                'label': label,
                'birth': birth,
                'death': death,
                'predecessors': predecessors,
                'successors': successors}

    def get_predecessors(self):
        return self.predecessors

    def get_successors(self):
        return self.successors

    def get_node_labels(self):
        return self.node_labels

    def get_node_values(self):
        return self.node_values

    def set_node_values(self, values):
        self.node_values = values

    def add_edge(self, u, v):
        if u not in self.successors:
            self.successors[u] = []
        self.successors[u].append(v)
        if v not in self.predecessors:
            self.predecessors[v] = []
        self.predecessors[v].append(u)

    def add_edge_from_list(self, edges):
        for v in range(len(edges)):
            u = edges[v]
            if u != v:
                self.add_edge(u, v)

    def from_image(self, image):
        self.image_info, tree_info = image_to_tree(image)

        self.node_labels = tree_info['node_labels']
        self.add_edge_from_list(tree_info['list_edges'])
        self.node_values = tree_info['node_values']
        self.root = tree_info['root']
        self.label_to_death = tree_info['label_to_death']
        self.label_to_birth = tree_info['label_to_birth']

        self.components = tree_info['components']

    def to_image(self, clip=False):
        image = np.array(self.node_values).reshape(self.image_info['rows'], self.image_info['cols'])
        image = image * (self.image_info['max'] - self.image_info['min']) + self.image_info['min']
        if clip:
            image = np.clip(image, 0, None)
        return image

    def segmentation(self, bg_value=0):
        bg_value = (bg_value - self.image_info['min']) / (self.image_info['max'] - self.image_info['min'])
        segm = np.array(self.node_labels) + 1
        segm[np.array(self.node_values) < bg_value] = 0
        return segm.reshape(self.image_info['rows'], self.image_info['cols'])

    def get_lifetimes(self, mode='list'):
        if mode == 'list':
            return [self.label_to_birth[l] - self.label_to_death[l] for l in self.components]
        elif mode == 'dict':
            return {l: self.label_to_birth[l] - self.label_to_death[l] for l in self.components}
        else:
            raise ValueError

    def get_max_distances(self, mode='list'):
        if mode == 'list':
            return list(compute_max_distances(self.node_labels, self.image_info['cols']).values())
        elif mode == 'dict':
            return compute_max_distances(self.node_labels, self.image_info['cols'])
        else:
            raise ValueError

    def copy(self):
        new_tree = Tree()

        # Copia profonda dei dizionari e liste
        new_tree.image_info = copy.deepcopy(self.image_info)

        new_tree.predecessors = copy.deepcopy(self.predecessors)
        new_tree.successors = copy.deepcopy(self.successors)
        new_tree.node_values = copy.deepcopy(self.node_values)
        new_tree.node_labels = copy.deepcopy(self.node_labels)

        new_tree.components = copy.deepcopy(self.components)
        new_tree.label_to_birth = copy.deepcopy(self.label_to_birth)
        new_tree.label_to_death = copy.deepcopy(self.label_to_death)

        new_tree.root = self.root

        return new_tree


class CutTree(Tree):
    def __init__(self):
        super().__init__()
        self.predecessors_cut = {}
        self.successors_cut = {}
        self.node_labels_cut = {}
        self.components_cut = []

    def __getitem__(self, index):
        value = self.node_values[index]
        label = self.node_labels_cut[index]
        birth = self.label_to_birth[label]
        death = self.label_to_death[label]
        if index in self.predecessors_cut:
            predecessors = self.predecessors_cut[index]
        else:
            predecessors = []
        if index in self.successors_cut:
            successors = self.successors_cut[index]
        else:
            successors = []

        return {'value': value,
                'label': label,
                'birth': birth,
                'death': death,
                'predecessors': predecessors,
                'successors': successors}

    def get_predecessors(self):
        if self.predecessors_cut:
            return self.predecessors_cut
        return super().get_predecessors()

    def get_successors(self):
        if self.successors_cut:
            return self.successors_cut
        return super().get_successors()

    def get_node_labels(self):
        if self.node_labels_cut:
            return self.node_labels_cut
        return super().get_node_labels()

    def add_edge_cut(self, u, v):
        if u not in self.successors_cut:
            self.successors_cut[u] = []
        self.successors_cut[u].append(v)
        if v not in self.predecessors_cut:
            self.predecessors_cut[v] = []
        self.predecessors_cut[v].append(u)

    def cut(self, level=None):
        self.node_labels_cut = self.node_labels.copy()
        if level is None:
            level = max_jump_threshold(self.get_lifetimes())
        else:
            level = (level - self.image_info['min']) / (self.image_info['max'] - self.image_info['min'])

        above_cut = {i for i in range(len(self))
                     if self.label_to_birth[self.node_labels[i]] - \
                     self.label_to_death[self.node_labels[i]] >= level}
        below_cut = set(range(len(self))) - above_cut

        above_points = []
        above_nodes = []
        below_points = []

        for u in above_cut:
            ai, aj = divmod(u, self.image_info['cols'])
            avalue = self.node_values[u]
            above_points.append((aj, ai, avalue))
            above_nodes.append(u)
            for v in self.successors.get(u, []):
                if v in above_cut:
                    self.add_edge_cut(u, v)

        ckdtree = cKDTree(above_points)
        for node in below_cut:
            bi, bj = divmod(node, self.image_info['cols'])
            avalue = self.node_values[node]
            below_points.append((bj, bi, avalue))
        distances, indices = ckdtree.query(below_points)
        for best_candidate, node in zip(indices, below_cut):
            actual_u = above_nodes[best_candidate.item()]
            while actual_u not in self.label_to_birth:
                actual_u = self.predecessors[actual_u][0]
            if self.label_to_birth[actual_u] == self.label_to_death[actual_u]:
                actual_u = self.predecessors[actual_u][0]

            self.add_edge_cut(actual_u, node)
            self.node_labels_cut[node] = actual_u

        self.components_cut = np.unique(self.node_labels_cut)

        # PATCH 1
        all_pred_cut = np.unique([x[0] for x in self.predecessors_cut.values()])
        conversion = {}
        for n in all_pred_cut:
            p = n
            while n not in self.components_cut and n in self.predecessors_cut:
                n = self.predecessors_cut[n][0]
            conversion[p] = n

        for i in range(len(self.predecessors_cut)):
            if i in self.predecessors_cut:
                self.predecessors_cut[i][0] = conversion[self.predecessors_cut[i][0]]

        # PATCH 2
        mins = ndimage.minimum(self.node_values, self.node_labels_cut, index=self.components_cut)
        maxs = ndimage.maximum(self.node_values, self.node_labels_cut, index=self.components_cut)
        label_to_death = dict(zip(self.components_cut, mins.tolist()))
        label_to_birth = dict(zip(self.components_cut, maxs.tolist()))
        self.label_to_birth = label_to_birth
        self.label_to_death = label_to_death


        # PATCH 3
        lifetime_dict = self.get_lifetimes('dict')
        for x in set(range(len(self))) - set(self.predecessors_cut):
            if x != self.root:
                xl = self.node_labels_cut[x]
                l = self.label_to_birth[xl] - self.label_to_death[xl]
                k = self.root
                for k, v in lifetime_dict.items():
                    if l < v:
                        break
                self.add_edge_cut(x, k)

        return level * (self.image_info['max'] - self.image_info['min']) + self.image_info['min']

    def get_lifetimes(self, mode='list'):
        if mode == 'list':
            if len(self.components_cut) > 0:
                return [self.label_to_birth[l] - self.label_to_death[l] for l in self.components_cut]
            else:
                return [self.label_to_birth[l] - self.label_to_death[l] for l in self.components]
        elif mode == 'dict':
            if len(self.components_cut) > 0:
                unsorted_dict = {l:self.label_to_birth[l] - self.label_to_death[l] for l in self.components_cut}
            else:
                unsorted_dict = {l: self.label_to_birth[l] - self.label_to_death[l] for l in self.components}
            return dict(sorted(unsorted_dict.items(), key=lambda item: item[1], reverse=False))
        else:
            raise ValueError

    def get_max_distances(self, mode='list'):
        if mode == 'list':
            if len(self.node_labels_cut) > 0:
                return list(compute_max_distances(self.node_labels_cut, self.image_info['cols']).values())
            else:
                return list(compute_max_distances(self.node_labels, self.image_info['cols']).values())
        elif mode == 'dict':
            if len(self.node_labels_cut) > 0:
                return compute_max_distances(self.node_labels_cut, self.image_info['cols'])
            else:
                return compute_max_distances(self.node_labels, self.image_info['cols'])
        else:
            raise ValueError

    def segmentation(self, bg_value=0):
        bg_value = (bg_value - self.image_info['min']) / (self.image_info['max'] - self.image_info['min'])
        segm = np.array(self.node_labels_cut) + 1
        segm[np.array(self.node_values) < bg_value] = 0
        return segm.reshape(self.image_info['rows'], self.image_info['cols'])

    def copy(self):
        new_tree = CutTree()

        # Copia profonda dei dizionari e liste
        new_tree.image_info = copy.deepcopy(self.image_info)

        new_tree.predecessors = copy.deepcopy(self.predecessors)
        new_tree.successors = copy.deepcopy(self.successors)
        new_tree.node_values = copy.deepcopy(self.node_values)
        new_tree.node_labels = copy.deepcopy(self.node_labels)

        new_tree.components = copy.deepcopy(self.components)
        new_tree.label_to_birth = copy.deepcopy(self.label_to_birth)
        new_tree.label_to_death = copy.deepcopy(self.label_to_death)

        new_tree.root = self.root

        new_tree.predecessors_cut = copy.deepcopy(self.predecessors_cut)
        new_tree.successors_cut = copy.deepcopy(self.successors_cut)
        new_tree.node_labels_cut = copy.deepcopy(self.node_labels_cut)
        new_tree.components_cut = copy.deepcopy(self.components_cut)

        return new_tree
