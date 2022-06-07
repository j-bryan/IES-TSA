class Node:
    def __init__(self) -> None:
        self.parents = []
        self.children = []
        self.clusterable_features = []
    
    def add_child(self, n):
        self.children.append(n)
    
    def add_parent(self, n):
        self.parents.append(n)


class Transformer(Node):
    """ A node which performs a transformation in both direcitons. Must have only one parent and one child. """
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X, y=None, **fit_params):
        raise NotImplementedError

    def transform(self, X):
        raise NotImplementedError
    
    def fit_transform(self, X, y=None, **fit_params):
        return self.fit(X, y, **fit_params).transform(X)
    
    def inverse_transform(self, Xt):
        raise NotImplementedError


class Branch(Node):
    def __init__(self, groups) -> None:
        super().__init__()
    
    def fit(self, X, y=None, **fit_params):
        pass


class Model(Node):
    def __init__(self) -> None:
        super().__init__()
    
    def fit(self, X, y=None, **fit_params):
        pass

    def evaluate(self, X):
        pass


class ROMTree:
    """
    Uses a tree structure to form arbitrary data pre- and post-processing, modeling, and evaluation workflows.
    Operations (transformers, segmentation, clustering, models, etc.) comprise the nodes. Nodes are connected
    bidirectionally for operations which must fetch 
    """
    def __init__(self):
        self.root = None
        self.leaf
    
    def add_node(self, n):
        pass

    def add_all(self, nodes):
        """ Adds all nodes to tree in the order specified. Automatically duplicates nodes when segmenting. """
        pass
