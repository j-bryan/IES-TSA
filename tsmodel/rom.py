from tsmodel.models.base import TimeSeriesBase
from sklearn.base import TransformerMixin


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

        # TODO: add ability to detect segmentation and clustering, detecting when multiple pipelines should be built
        ## Have two base classes: TransformerBase, SegmentedTransformerBase
        ## Need to add Segmenter class that will split up the transformed time series
        ##      - inverse_transform() joins the segments back together
        ## After segmentation, each subsequent model needs to be applied to each segment individually
        ## 

    def fit(self, X):
        Xt = self._preprocessor.fit_transform(X)
        print('fit preprocessor')
        fit_res = self._model.fit(Xt)
        print('fit model')
        if self._postprocessor:
            self._postprocessor.fit(X)
            print('fit postprocessor')
        return fit_res
    
    def simulate(self, **sim_params):
        Xst = self._model.simulate(**sim_params)
        Xs = self._preprocessor.inverse_transform(Xst)
        if self._postprocessor:
            Xs = self._postprocessor.transform(Xs)
        return Xs


class ROMTree:
    """
    Uses a tree structure to form arbitrary data pre- and post-processing, modeling, and evaluation workflows.
    Operations (transformers, segmentation, clustering, models, etc.) comprise the nodes. Nodes are connected
    bidirectionally for operations which must fetch 
    """
    def __init__(self):
        pass
