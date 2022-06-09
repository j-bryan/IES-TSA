from copy import deepcopy
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from tsmodel.base import Clusterable


class SegmentTransformer(BaseEstimator, TransformerMixin):
    """ Fits the a series of transformers to each segment and returns a list of feature vectors to use for clustering. """
    def __init__(self, steps) -> None:
        super().__init__()
        self._steps = steps
        self._pipes = []
    
    def fit(self, X, y=None, **fit_params):
        for i, X_seg in enumerate(X):
            pipe = Pipeline(deepcopy(self._steps))
            pipe.fit(X_seg)
            self._pipes.append(pipe)
        return self
    
    def transform(self, X):
        Xt = []
        for i, X_seg in enumerate(X):
            Xt.append(self._pipes[i].transform(X_seg))
        return Xt
    
    def inverse_transform(self, Xt):
        X = []
        for i, Xt_seg in enumerate(Xt):
            X.append(self._pipes[i].inverse_transform(Xt_seg))
        return X


class ClusteringSegmentTransformer(BaseEstimator, TransformerMixin):
    """ Fits the a series of transformers to each segment and returns a list of feature vectors to use for clustering. """
    def __init__(self, steps, cluster_sampling='None') -> None:
        super().__init__()
        self._steps = steps
        self._pipes = []
        self._clusterable_features = []
        self._cluster_sampling_method = cluster_sampling
    
    def fit(self, X, y=None, **fit_params):
        for i, X_seg in enumerate(X):
            pipe = Pipeline(deepcopy(self._steps))
            X_seg_t = pipe.fit_transform(X_seg)
            self._pipes.append(pipe)
            self._clusterable_features.append(self._collect_clusterable_features(pipe))
        return self
    
    def transform(self, X):
        return np.array(self._clusterable_features)  # clusterable features need to be all the same length, so we can go ahead and make this a numpy array
    
    def inverse_transform(self, Xt):
        """ Xt is an array of segments from which to generate synthetic segments """
        # Need to handle time stuff here
        X = [self._pipes[seg_i].inverse_transform()]

    def _collect_clusterable_features(self, trans):
        feature_vector = []
        if isinstance(trans, Pipeline):
            for step in trans:
                if isinstance(step, Clusterable):
                    feature_vector.extend(step.get_clusterable_features())
        elif isinstance(trans, Clusterable):
            feature_vector = trans.get_clusterable_features()
        else:
            raise TypeError  # TODO: add a more descriptive error message
        return feature_vector
