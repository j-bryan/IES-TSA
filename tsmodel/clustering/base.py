class ClusteringBase:
    """
    Clustering if difficult here because we could possibly be wanting to cluster over a model space
    or over transformed data. How can these be handled equivalently? Seems dependent just on what is
    passed to the clustering algorithm...
    - How could we get an arbitrary set of parameters for clustering from upstream transformers?
        - Each model has a _clusterable_params attribute
        - Can walk back up the tree/list/whatever to collect the clusterable params from previous nodes
        - Adds all of these clusterable params to a feature vector for the segment
    """
    def fit(self, X, y=None, **fit_params):
        pass
