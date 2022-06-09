from sklearn.cluster import KMeans


class KMeansCluster(KMeans):
    def __init__(self, n_clusters=8, *, init='k-means++', n_init=10, max_iter=300, tol=0.0001, precompute_distances='deprecated', verbose=0, random_state=None, copy_x=True, n_jobs='deprecated', algorithm='auto'):
        super().__init__(n_clusters, init, n_init, max_iter, tol, precompute_distances, verbose, random_state, copy_x, n_jobs, algorithm)
    
    def sample(self, method):
        m = method.lower()
        if m == 'first':
            pass
        elif m == 'random':
            pass
        elif m == 'centroid':
            pass
        else:
            raise ValueError
        