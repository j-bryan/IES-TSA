import numpy as np
from sklearn.cluster import KMeans


class ROM:
    def __init__(self, preprocessor, model, postprocessor=None):
        """
        Parameters
        ----------
        preprocessor : sklearn.Pipeline or sklearn.base.TransformerMixin
            Applies one or more transformations to a data set before the model is fit. Also used for applying
            the inverse transformation when generating synthetic histories
        model : TimeSeriesBase object
            The time series model to be fit to the historical data.
        postprocessor : sklearn.Pipeline or sklearn.base.TransformerMixin
            Applies extra transformations to synthetic series after preprocessor inverse_transform is applied
        """
        self._preprocessor = preprocessor  # gets you from input data to model and back
        self._postprocessor = postprocessor  # applies extra transformations when coming back through the pipeline
        self._model = model  # time series model
        self._fit_model = None
        self._input_shape = None

        # self._has_preproc_estimator = True if 

    def fit(self, X):
        self._input_shape = X.shape
        Xt = self._preprocessor.fit_transform(X)  # TODO: use fit_predict if the last step is something like KMeans()
        self._fit_model = fit_res = self._model.fit(Xt)
        if self._postprocessor:
            self._postprocessor.fit(X)
    
    def evaluate(self, **sim_params):
        Xst = self._fit_model.simulate(**sim_params).reshape(self._input_shape)
        Xs = self._preprocessor.inverse_transform(Xst)
        if self._postprocessor:
            Xs = self._postprocessor.transform(Xs)
        return Xs


class KMeansRom:
    def __init__(self, n_clusters, preprocessor=None, postprocessor=None):
        self.n_clusters = n_clusters
        self.preproc = preprocessor
        self.postproc = postprocessor
        self.kmeans = None
        self.segment_labels = []
        self._nsegs_orig = 0
    
    def fit(self, X, y=None, **fit_params):
        self._nsegs_orig = len(X)

        if self.preproc is not None:
            features = self.preproc.fit_transform(X, y, **fit_params)
        else:
            features = X

        self.kmeans = KMeans(n_clusters=self.n_clusters, **fit_params).fit(features)
        self.segment_labels = self.kmeans.predict(X)

        if self.postproc is not None:
            self.postproc.fit(X, y, **fit_params)
        return self
        
    def evaluate(self, nobs=None, sample_method='none'):
        """ Creates a realization of length nobs """
        # Need to create an array of which segment to use and how many observations to generate from that segment. We need to also keep
        ## track of what time we're starting from as we generate the Fourier stuff for each segment
        ## Either we need a number of observations to create (nobs) AND a pivot length (pivot_length), OR we need an array of segment lengths.
        if nobs is None and pivot_length is None and seglens is None:
            raise ValueError
        elif nobs is not None and pivot_length is not None:
            # use nobs & pivot length
            segment_lengths = [pivot_length for i in range(nobs // pivot_length)]
            remainder = nobs % pivot_length
            if remainder != 0:
                segment_lengths.append(remainder)
        elif seglens is not None:
            # use seglens given
            segment_lengths = seglens
        else:
            raise ValueError
        
        sm = sample_method.lower()
        if sm == 'none':  # use the segments in the same order they were in in the original data
            pass
        elif sm == 'random':  # use a random model from the same cluster
            pass
        elif sm == 'first':  # use the first model from the cluster
            pass
        elif sm == 'centroid':  # use the centroid of the cluster to build a new model
            pass
        else:
            raise ValueError(f'Sample method {sample_method} must be in the allowed methods ["random", "first", "centroid", "none"].')
    
    def _evaluate_none(self, seglens, t0):
        t = [t0]
        for i in range(len(seglens) - 1):
            t.append(t[-1] + seglens[i])
        return zip(seglens, [i % self._nsegs_orig for i in range(len(seglens))], t)

    def _evaluate_random(self, seglens, t0):
        pass

    def evaluate_centroid(self, seglens, t0):
        pass
