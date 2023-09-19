"""Module to provide random layers for ELM and OS-ELM models"""

# ===================================================
# Acknowledgement:
# Author: David C. Lambert [dcl -at- panix -dot- com]
# Copyright(c) 2013
# License: Simple BSD
# ===================================================
from abc import ABCMeta, abstractmethod
from math import sqrt

import numpy as np
import scipy.sparse as sp
from scipy.spatial.distance import cdist, pdist, squareform
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.metrics import pairwise_distances
from sklearn.utils import check_random_state, check_array
from sklearn.utils.extmath import safe_sparse_dot




__all__ = [
    "RandomLayer",
    "MLPRandomLayer",
    "RBFRandomLayer",
    "GRBFRandomLayer",
]

use_know = False

def set_use_know(s_use_know):
  use_know = s_use_know


class BaseRandomLayer(BaseEstimator, TransformerMixin):
    """Abstract Base class for random layers"""
    __metaclass__ = ABCMeta

    _internal_activation_funcs = dict()

    @classmethod
    def activation_func_names(cls):
        """Get list of internal activation function names"""
        return cls._internal_activation_funcs.keys()

    def __init__(self,
                 n_hidden=20,
                 random_state=0,
                 activation_func=None,
                 activation_args=None,):

        self.n_hidden = n_hidden
        self.random_state = random_state
        self.activation_func = activation_func
        self.activation_args = activation_args

        self.components_ = dict()
        self.input_activations_ = None

        # keyword args for internally defined funcs
        self._extra_args = dict()

    @abstractmethod
    def _generate_components(self, X):
        """Generate components of hidden layer given X"""

    @abstractmethod
    def _compute_input_activations(self, X, non_selected_features=None):
        """Compute input activations given X"""

    def _compute_hidden_activations(self, X, non_selected_features=None):
        """Compute hidden activations given X"""
        # compute input activations and pass them
        # through the hidden layer transfer functions
        # to compute the transform

        self._compute_input_activations(X, non_selected_features)

        acts = self.input_activations_

        if callable(self.activation_func):
            args_dict = self.activation_args if self.activation_args else {}
            X_new = self.activation_func(acts, **args_dict)
        else:
            func_name = self.activation_func
            func = self._internal_activation_funcs[func_name]
            X_new = func(acts, **self._extra_args)

        return X_new

    def fit(self, X, y=None):
        """Generate a random hidden layer.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training set: only the shape is used to generate random component
            values for hidden units

        y : not used: placeholder to allow for usage in a Pipeline.

        Returns
        -------
        self
        """
        # perform fit by generating random components based
        # on the input array
        X = check_array(X, accept_sparse=True)
        self._generate_components(X)

        return self

    def transform(self, X, non_selected_features=None, y=None):
        """Generate the random hidden layer's activations given X as input.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape [n_samples, n_features]
            Data to transform

        y : not used: placeholder to allow for usage in a Pipeline.

        Returns
        -------
        X_new : numpy array of shape [n_samples, n_components]
        """
        # perform transformation by calling compute_hidden_activations
        # (which will normally call compute_input_activations first)
        X = check_array(X, accept_sparse=True)

        if len(self.components_) == 0:
            raise ValueError('No components initialized')

        return self._compute_hidden_activations(X, non_selected_features)

  
class RandomLayer(BaseRandomLayer):
    """RandomLayer is a transformer that creates a feature mapping of the
    inputs that corresponds to a layer of hidden units with randomly
    generated components.

    The transformed values are a specified function of input activations
    that are a weighted combination of dot product (multilayer perceptron)
    and distance (rbf) activations:

      input_activation = alpha * mlp_activation + (1-alpha) * rbf_activation

      mlp_activation(x) = dot(x, weights) + bias
      rbf_activation(x) = rbf_width * ||x - center||/radius

      alpha and rbf_width are specified by the user

      weights and biases are taken from normal distribution of
      mean 0 and sd of 1

      centers are taken uniformly from the bounding hyperrectangle
      of the inputs, and radii are max(||x-c||)/sqrt(n_centers*2)

    The input activation is transformed by a transfer function that defaults
    to numpy.tanh if not specified, but can be any callable that returns an
    array of the same shape as its argument (the input activation array, of
    shape [n_samples, n_hidden]).  Functions provided are 'sine', 'tanh',
    'tribas', 'inv_tribas', 'sigmoid', 'hardlim', 'softlim', 'gaussian',
    'multiquadric', or 'inv_multiquadric'.

    Parameters
    ----------
    `n_hidden` : int, optional (default=20)
        Number of units to generate

    `alpha` : float, optional (default=0.5)
        Mixing coefficient for distance and dot product input activations:
        activation = alpha*mlp_activation + (1-alpha)*rbf_width*rbf_activation

    `rbf_width` : float, optional (default=1.0)
        multiplier on rbf_activation

    `user_components`: dictionary, optional (default=None)
        dictionary containing values for components that would otherwise be
        randomly generated.  Valid key/value pairs are as follows:
           'radii'  : array-like of shape [n_hidden]
           'centers': array-like of shape [n_hidden, n_features]
           'biases' : array-like of shape [n_hidden]
           'weights': array-like of shape [n_features, n_hidden]

    `activation_func` : {callable, string} optional (default='tanh')
        Function used to transform input activation

        It must be one of 'tanh', 'sine', 'tribas', 'inv_tribas',
        'sigmoid', 'hardlim', 'softlim', 'gaussian', 'multiquadric',
        'inv_multiquadric' or a callable.  If None is given, 'tanh'
        will be used.

        If a callable is given, it will be used to compute the activations.

    `activation_args` : dictionary, optional (default=None)
        Supplies keyword arguments for a callable activation_func

    `random_state`  : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        hidden unit weights at fit time.

    Attributes
    ----------
    `input_activations_` : numpy array of shape [n_samples, n_hidden]
        Array containing dot(x, hidden_weights) + bias for all samples

    `components_` : dictionary containing two keys:
        `bias_weights_`   : numpy array of shape [n_hidden]
        `hidden_weights_` : numpy array of shape [n_features, n_hidden]

    See Also
    --------
    """

    # triangular activation function
    _tribas = lambda x: np.clip(1.0 - np.fabs(x), 0.0, 1.0)

    # inverse triangular activation function
    _inv_tribas = lambda x: np.clip(np.fabs(x), 0.0, 1.0)

    # sigmoid activation function
    _sigmoid = lambda x: 1.0/(1.0 + np.exp(-x))

    # hard limit activation function
    _hardlim = lambda x: np.array(x > 0.0, dtype=float)

    _softlim = lambda x: np.clip(x, 0.0, 1.0)

    # identity or linear activation function
    _linear = lambda x: x

    # ReLU
    _relu = lambda x: np.maximum(x, 0)

    # Softplus activation function
    _softplus = lambda x: np.log(1.0 + np.exp(x))

    # gaussian RBF
    _gaussian = lambda x: np.exp(-pow(x, 2.0))

    # multiquadric RBF
    _multiquadric = lambda x: np.sqrt(1.0 + pow(x, 2.0))

    # inverse multiquadric RBF
    _inv_multiquadric = lambda x: 1.0/(np.sqrt(1.0 + pow(x, 2.0)))

    # internal activation function table
    _internal_activation_funcs = {
        'sine': np.sin,
        'tanh': np.tanh,
        'tribas': _tribas,
        'inv_tribas': _inv_tribas,
        'linear': _linear,
        'relu': _relu,
        'softplus': _softplus,
        'sigmoid': _sigmoid,
        'softlim': _softlim,
        'hardlim': _hardlim,
        'gaussian': _gaussian,
        'multiquadric': _multiquadric,
        'inv_multiquadric': _inv_multiquadric,
    }

    def __init__(self,
                 n_hidden=20,
                 alpha=0.5,
                 random_state=None,
                 activation_func='tanh',
                 activation_args=None,
                 user_components=None,
                 rbf_width=1.0,):

        super(RandomLayer, self).__init__(
            n_hidden=n_hidden,
            random_state=random_state,
            activation_func=activation_func,
            activation_args=activation_args
        )

        if isinstance(self.activation_func, str):
            func_names = self._internal_activation_funcs.keys()
            if self.activation_func not in func_names:
                msg = "Unknown activation function '%s'" % self.activation_func
                raise ValueError(msg)

        self.alpha = alpha
        self.rbf_width = rbf_width
        self.user_components = user_components

        self._use_mlp_input = (self.alpha != 0.0)
        self._use_rbf_input = (self.alpha != 1.0)
        self.original_weights = None

    def _get_user_components(self, key):
        """Look for given user component"""
        try:
            return self.user_components[key]
        except (TypeError, KeyError):
            return None

    def _compute_radii(self):
        """Generate RBF radii"""

        # use supplied radii if present
        radii = self._get_user_components('radii')

        # compute radii
        if radii is None:
            centers = self.components_['centers']

            n_centers = centers.shape[0]
            max_dist = np.max(pairwise_distances(centers))
            radii = np.ones(n_centers) * max_dist/sqrt(2.0 * n_centers)

        self.components_['radii'] = radii

    def _compute_centers(self, X, sparse, rs):
        """Generate RBF centers"""

        # use supplied centers if present
        centers = self._get_user_components('centers')

        # use points taken uniformly from the bounding hyperrectangle
        if centers is None:
            n_features = X.shape[1]

            if sparse:
                cols = [X.getcol(i) for i in range(n_features)]

                min_dtype = X.dtype.type(1.0e10)
                sp_min = lambda col: np.minimum(min_dtype, np.min(col.data))
                min_Xs = np.array(list(map(sp_min, cols)))

                max_dtype = X.dtype.type(-1.0e10)
                sp_max = lambda col: np.maximum(max_dtype, np.max(col.data))
                max_Xs = np.array(list(map(sp_max, cols)))
            else:
                min_Xs = X.min(axis=0)
                max_Xs = X.max(axis=0)

            spans = max_Xs - min_Xs
            ctrs_size = (self.n_hidden, n_features)
            centers = min_Xs + spans * rs.uniform(0.0, 1.0, ctrs_size)

        self.components_['centers'] = centers

    def _compute_biases(self, rs):
        """Generate MLP biases"""

        # use supplied biases if present
        biases = self._get_user_components('biases')
        if biases is None:
            b_size = self.n_hidden
            biases = rs.normal(size=b_size)

        self.components_['biases'] = biases

    def _compute_weights(self, X, rs):
        """Generate MLP weights"""

        # use supplied weights if present
        weights = self._get_user_components('weights')
        if weights is None:
            n_features = X.shape[1]
            hw_size = (n_features, self.n_hidden)
            weights = rs.normal(size=hw_size)
            self.original_weights = weights.copy()

        self.components_['weights'] = weights

    def _generate_components(self, X):
        """Generate components of hidden layer given X"""

        rs = check_random_state(self.random_state)
        if self._use_mlp_input:
            self._compute_biases(rs)
            self._compute_weights(X, rs)

        if self._use_rbf_input:
            self._compute_centers(X, sp.issparse(X), rs)
            self._compute_radii()

    def _compute_input_activations(self, X, non_selected_features=None):
        """Compute input activations given X"""
        global use_know
        n_samples = X.shape[0]
        rs = check_random_state(self.random_state)
        mlp_acts = np.zeros((n_samples, self.n_hidden))
        if self._use_mlp_input:
            b = self.components_['biases']
            w = self.components_['weights'].copy()
            zero_rows = np.where(~w.any(axis=1))[0].tolist()
            if len(zero_rows) > 0 and use_know:
                #w[zero_rows, :] = self.original_weights[zero_rows, :].copy()
                w[zero_rows, :] = self.components_['weights'][zero_rows, :].copy()
            elif len(zero_rows) > 0 and not use_know:
                for i in zero_rows:
                    w[i, :] = rs.normal(size=(w.shape[1]))
            #############################################
            
            if non_selected_features is not None:
                w[non_selected_features, :] = 0
            mlp_acts = self.alpha * (safe_sparse_dot(X, w) + b)
            self.components_['weights'] = w.copy()

        rbf_acts = np.zeros((n_samples, self.n_hidden))
        if self._use_rbf_input:
            radii = self.components_['radii']
            centers = self.components_['centers']
            scale = self.rbf_width * (1.0 - self.alpha)

            if sp.issparse(X):
                X = X.todense()

            rbf_acts = scale * cdist(X, centers)/radii

        self.input_activations_ = mlp_acts + rbf_acts


class MLPRandomLayer(RandomLayer):
    """Wrapper for RandomLayer with alpha (mixing coefficient) set
       to 1.0 for MLP activations only"""

    def __init__(self, n_hidden=20, random_state=None,
                 activation_func='tanh', activation_args=None,
                 weights=None, biases=None):

        user_components = {'weights': weights, 'biases': biases}
        super(MLPRandomLayer, self).__init__(
            n_hidden=n_hidden,
            random_state=random_state,
            activation_func=activation_func,
            activation_args=activation_args,
            user_components=user_components,
            alpha=1.0
        )


class RBFRandomLayer(RandomLayer):
    """Wrapper for RandomLayer with alpha (mixing coefficient) set
       to 0.0 for RBF activations only"""

    def __init__(self, n_hidden=20, random_state=None,
                 activation_func='gaussian', activation_args=None,
                 centers=None, radii=None, rbf_width=1.0):

        user_components = {'centers': centers, 'radii': radii}
        super(RBFRandomLayer, self).__init__(
            n_hidden=n_hidden,
            random_state=random_state,
            activation_func=activation_func,
            activation_args=activation_args,
            user_components=user_components,
            rbf_width=rbf_width,
            alpha=0.0
        )


class GRBFRandomLayer(RBFRandomLayer):
    """Random Generalized RBF Hidden Layer transformer

    Creates a layer of radial basis function units where:

       f(a), s.t. a = ||x-c||/r

    with c the unit center
    and f() is exp(-gamma * a^tau) where tau and r are computed
    based on [1]

    Parameters
    ----------
    `n_hidden` : int, optional (default=20)
        Number of units to generate, ignored if centers are provided

    `grbf_lambda` : float, optional (default=0.05)
        GRBF shape parameter

    `gamma` : {int, float} optional (default=1.0)
        Width multiplier for GRBF distance argument

    `centers` : array of shape (n_hidden, n_features), optional (default=None)
        If provided, overrides internal computation of the centers

    `radii` : array of shape (n_hidden),  optional (default=None)
        If provided, overrides internal computation of the radii

    `use_exemplars` : bool, optional (default=False)
        If True, uses random examples from the input to determine the RBF
        centers, ignored if centers are provided

    `random_state`  : int or RandomState instance, optional (default=None)
        Control the pseudo random number generator used to generate the
        centers at fit time, ignored if centers are provided

    Attributes
    ----------
    `components_` : dictionary containing two keys:
        `radii_`   : numpy array of shape [n_hidden]
        `centers_` : numpy array of shape [n_hidden, n_features]

    `input_activations_` : numpy array of shape [n_samples, n_hidden]
        Array containing ||x-c||/r for all samples

    See Also
    --------
    ELMRegressor, ELMClassifier, SimpleELMRegressor, SimpleELMClassifier,
    SimpleRandomLayer

    References
    ----------
    .. [1] Fernandez-Navarro, et al, "MELM-GRBF: a modified version of the
              extreme learning machine for generalized radial basis function
              neural networks", Neurocomputing 74 (2011), 2502-2510

    """

    def __init__(self, n_hidden=20, grbf_lambda=0.001,
                 centers=None, radii=None, random_state=None):

        self._internal_activation_funcs = {'grbf': self._grbf}

        super(GRBFRandomLayer, self).__init__(
            n_hidden=n_hidden,
            activation_func='grbf',
            centers=centers, radii=radii,
            random_state=random_state
        )

        self.grbf_lambda = grbf_lambda
        self.dN_vals = None
        self.dF_vals = None
        self.tau_vals = None

    @staticmethod
    def _grbf(acts, taus):
        """GRBF activation function"""
        return np.exp(np.exp(-pow(acts, taus)))

    def _compute_centers(self, X, sparse, rs):
        """Generate centers, then compute tau, dF and dN vals"""
        # get centers from superclass, then calculate tau_vals
        # according to ref [1]
        super(GRBFRandomLayer, self)._compute_centers(X, sparse, rs)

        centers = self.components_['centers']
        sorted_distances = np.sort(squareform(pdist(centers)))
        self.dF_vals = sorted_distances[:, -1]
        self.dN_vals = sorted_distances[:, 1]/100.0

        tauNum = np.log(np.log(self.grbf_lambda) /
                        np.log(1.0 - self.grbf_lambda))

        tauDenom = np.log(self.dF_vals/self.dN_vals)

        self.tau_vals = tauNum/tauDenom

        self._extra_args['taus'] = self.tau_vals

    def _compute_radii(self):
        """Generate radii"""
        # according to ref [1]
        denom = pow(-np.log(self.grbf_lambda), 1.0/self.tau_vals)
        self.components_['radii'] = self.dF_vals/denom


"""Module to build Online Sequential Extreme Learning Machine (OS-ELM) models"""

# ===================================================
# Author: Leandro Ferrado
# Copyright(c) 2018
# License: Apache License 2.0
# ===================================================

import warnings

import numpy as np
from scipy.linalg import pinv
#from scipy.linalg import pinv2
from scipy.sparse import eye
from scipy.special import softmax
from sklearn.base import RegressorMixin, BaseEstimator
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import LabelBinarizer
from sklearn.utils import as_float_array
from sklearn.utils.extmath import safe_sparse_dot



__all__ = [
    "OSELMRegressor",
    "OSELMClassifier",
]


def multiple_safe_sparse_dot(*matrices):
    """
    Make safe_sparse_dot() calls over multiple matrices

    Parameters
    ----------
    matrices: iterable of matrices

    Returns
    -------
    dot_product : array or sparse matrix
    """
    if len(matrices) < 2:
        raise ValueError("Argument 'matrices' must have at least 2 matrices")

    r = matrices[0]
    for m in matrices[1:]:
        r = safe_sparse_dot(r, m)

    return r


class OSELMRegressor(BaseEstimator, RegressorMixin):
    """
    OSELMRegressor is a regressor based on Online Sequential
    Extreme Learning Machine (OS-ELM).

    This type of model is an ELM that....   ...
    [1][2]

    Parameters
    ----------
    `n_hidden` : int, optional (default=20)
        Number of units to generate in the SimpleRandomLayer

    `activation_func` : {callable, string} optional (default='sigmoid')
        Function used to transform input activation

        It must be one of 'tanh', 'sine', 'tribas', 'inv_tribase', 'sigmoid',
        'hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric' or
        a callable.  If none is given, 'tanh' will be used. If a callable
        is given, it will be used to compute the hidden unit activations.

    `activation_args` : dictionary, optional (default=None)
        Supplies keyword arguments for a callable activation_func

    `use_woodbury`  : bool, optional (default=False)
        Flag to indicate if Woodbury formula should be used for the fit
        step, or just the traditional iterative procedure. Not recommended if
        handling large datasets.

    `random_state`  : int, RandomState instance or None (default=None)
        Control the pseudo random number generator used to generate the
        hidden unit weights at fit time.

    Attributes
    ----------
    `P` : np.array
        ...

    `beta` : np.array
    ...

    See Also
    --------
    ELMRegressor, MLPRandomLayer

    References
    ----------
    .. [1] http://www.extreme-learning-machines.org
    .. [2] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
          Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
              2006.

    """
    def __init__(self,
                 n_hidden=20,
                 activation_func='sigmoid',
                 activation_args=None,
                 use_woodbury=False,
                 random_state=123,):
        self.n_hidden = n_hidden
        self.random_state = random_state
        self.activation_func = activation_func
        self.activation_args = activation_args
        self.use_woodbury = use_woodbury
        self.random_layer = None
        

        self.P = None
        self.beta = None

    def _create_random_layer(self):
        """Pass init params to MLPRandomLayer"""
        if self.random_layer is None:
            self.random_layer=  MLPRandomLayer(n_hidden=self.n_hidden,
                                              random_state=self.random_state,
                                              activation_func=self.activation_func,
                                              activation_args=self.activation_args)
        return self.random_layer

    def _fit_woodbury(self, X, y):
        """Compute learning step using Woodbury formula"""
        # fit random hidden layer and compute the hidden layer activations
        H = self._create_random_layer().fit_transform(X)
        y = as_float_array(y, copy=True)

        if self.beta is None:
            # this is the first time the model is fitted
            if len(X) < self.n_hidden:
                raise ValueError("The first time the model is fitted, "
                                 "X must have at least equal number of "
                                 "samples than n_hidden value!")
            self.P = pinv(safe_sparse_dot(H.T, H))
            #self.P = pinv2(safe_sparse_dot(H.T, H))
            self.beta = multiple_safe_sparse_dot(self.P, H.T, y)
        else:
            if len(H) > 10e3:
                warnings.warn("Large input of %i rows and use_woodbury=True "\
                              "may throw OOM errors" % len(H))

            M = eye(len(H)) + multiple_safe_sparse_dot(H, self.P, H.T)
            self.P -= multiple_safe_sparse_dot(self.P, H.T, pinv(M), H, self.P)
            #self.P -= multiple_safe_sparse_dot(self.P, H.T, pinv2(M), H, self.P)
            e = y - safe_sparse_dot(H, self.beta)
            self.beta += multiple_safe_sparse_dot(self.P, H.T, e)

    def _fit_iterative(self, X, y, non_selected_features=None):
        """Compute learning step using iterative procedure"""
        # fit random hidden layer and compute the hidden layer activations
        model = self._create_random_layer().fit(X)
        H = model.transform(X, non_selected_features)
        y = as_float_array(y, copy=True)

        if self.beta is None:
            # this is the first time the model is fitted
            if len(X) < self.n_hidden:
                raise ValueError("The first time the model is fitted, "
                                 "X must have at least equal number of "
                                 "samples than n_hidden value!")

            self.P = safe_sparse_dot(H.T, H)
            P_inv = pinv(self.P)
            #P_inv = pinv2(self.P)
            self.beta = multiple_safe_sparse_dot(P_inv, H.T, y)
        else:
            self.P += safe_sparse_dot(H.T, H)
            P_inv = pinv(self.P)
            #P_inv = pinv2(self.P)
            e = y - safe_sparse_dot(H, self.beta)
            self.beta = self.beta + multiple_safe_sparse_dot(P_inv, H.T, e)

    def fit(self, X, y, non_selected_features=None):
        """
        Fit the model using X, y as training data.

        Notice that this function could be used for n_samples==1 (online learning),
        except for the first time the model is fitted, where it needs at least as 
        many rows as 'n_hidden' configured.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """
        if self.use_woodbury:
            self._fit_woodbury(X, y)
        else:
            self._fit_iterative(X, y, non_selected_features)

        return self

    def partial_fit(self, X, y):
        """
        Fit the model using X, y as training data. Alias for fit() method.

        Notice that this function could be used for n_samples==1 (online learning),
        except for the first time the model is fitted, where it needs at least as
        many rows as 'n_hidden' configured.

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]
            Training vectors, where n_samples is the number of samples
            and n_features is the number of features.

        y : array-like of shape [n_samples, n_outputs]
            Target values (class labels in classification, real numbers in
            regression)

        Returns
        -------
        self : object

            Returns an instance of self.
        """
        return self.fit(X, y)

    @property
    def is_fitted(self):
        """Check if model was fitted

        Returns
        -------
            boolean, True if model is fitted
        """
        return self.beta is not None

    def predict(self, X):
        """
        Predict values using the model

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape [n_samples, n_features]

        Returns
        -------
        C : numpy array of shape [n_samples, n_outputs]
            Predicted values.
        """
        if not self.is_fitted:
            raise ValueError("OSELMRegressor not fitted")

        # compute hidden layer activations
        #random_layer = self._create_random_layer()
        self.random_layer.fit(X)
        
        #H = self._create_random_layer().fit_transform(X)
        H = self.random_layer.transform(X)

        # compute output predictions for new hidden activations
        predictions = safe_sparse_dot(H, self.beta)

        return predictions


class OSELMClassifier(OSELMRegressor):
      """
      OSELMClassifier is a classifier based on the Extreme Learning Machine.

      An Extreme Learning Machine (ELM) is a single layer feedforward
      network with a random hidden layer components and ordinary linear
      least squares fitting of the hidden->output weights by default.
      [1][2]

      OSELMClassifier is an OSELMRegressor subclass that first binarizes the
      data, then uses the superclass to compute the decision function that
      is then unbinarized to yield the prediction.

      The params for the RandomLayer used in the input transform are
      exposed in the ELMClassifier constructor.

      Parameters
      ----------
      `n_hidden` : int, optional (default=20)
          Number of units to generate in the SimpleRandomLayer

      `activation_func` : {callable, string} optional (default='sigmoid')
          Function used to transform input activation

          It must be one of 'tanh', 'sine', 'tribas', 'inv_tribase', 'sigmoid',
          'hardlim', 'softlim', 'gaussian', 'multiquadric', 'inv_multiquadric' or
          a callable.  If none is given, 'tanh' will be used. If a callable
          is given, it will be used to compute the hidden unit activations.

      `activation_args` : dictionary, optional (default=None)
          Supplies keyword arguments for a callable activation_func

      `random_state`  : int, RandomState instance or None (default=None)
          Control the pseudo random number generator used to generate the
          hidden unit weights at fit time.

      Attributes
      ----------
      `classes_` : numpy array of shape [n_classes]
          Array of class labels

      See Also
      --------
      ELMRegressor, OSELMRegressor, MLPRandomLayer

      References
      ----------
      .. [1] http://www.extreme-learning-machines.org
      .. [2] G.-B. Huang, Q.-Y. Zhu and C.-K. Siew, "Extreme Learning Machine:
            Theory and Applications", Neurocomputing, vol. 70, pp. 489-501,
                2006.
      """

      def __init__(self,
                  n_hidden=20,
                  activation_func='sigmoid',
                  activation_args=None,
                  binarizer= None,#LabelBinarizer(neg_label=-1, pos_label=1),
                  use_woodbury=False,
                  random_state=123):

          super(OSELMClassifier, self).__init__(n_hidden=n_hidden,
                                                random_state=random_state,
                                                activation_func=activation_func,
                                                activation_args=activation_args,
                                                use_woodbury=use_woodbury)
          self.classes_ = None
          self.binarizer = binarizer

      def decision_function(self, X):
          """
          This function return the decision function values related to each
          class on an array of test vectors X.

          Parameters
          ----------
          X : array-like of shape [n_samples, n_features]

          Returns
          -------
          C : array of shape [n_samples, n_classes] or [n_samples,]
              Decision function values related to each class, per sample.
              In the two-class case, the shape is [n_samples,]
          """
          return super(OSELMClassifier, self).predict(X)

      def fit(self, X, y, non_selected_features=None):
          """
          Fit the model using X, y as training data.

          Parameters
          ----------
          X : {array-like, sparse matrix} of shape [n_samples, n_features]
              Training vectors, where n_samples is the number of samples
              and n_features is the number of features.

          y : array-like of shape [n_samples, n_outputs]
              Target values (class labels in classification, real numbers in
              regression)

          Returns
          -------
          self : object

              Returns an instance of self.
          """
          if not self.is_fitted:
              self.classes_ = np.unique(y)
              y_bin = self.binarizer.fit_transform(y)
          else:
              y_bin = self.binarizer.transform(y)

          super(OSELMClassifier, self).fit(X, y_bin, non_selected_features)

          return self

      def predict(self, X):
          """
          Predict class values using the model

          Parameters
          ----------
          X : {array-like, sparse matrix} of shape [n_samples, n_features]

          Returns
          -------
          C : numpy array of shape [n_samples, n_outputs]
              Predicted class values.
          """
          if not self.is_fitted:
              raise ValueError("OSELMClassifier not fitted")

          proba_predictions = self.predict_proba(X)
          class_predictions = np.argmax(proba_predictions, axis=1)#self.binarizer.inverse_transform(raw_predictions)

          return class_predictions
      
      def predict_proba(self, X):
          """
          Predict probability values using the model

          Parameters
          ----------
          X : {array-like, sparse matrix} of shape [n_samples, n_features]

          Returns
          -------
          P : numpy array of shape [n_samples, n_outputs]
              Predicted probability values.
          """
          if not self.is_fitted:
              raise ValueError("OSELMClassifier not fitted")

          raw_predictions = self.decision_function(X)
          #print(raw_predictions)
          # using softmax to translate raw predictions into probability values
          proba_predictions = softmax(raw_predictions)

          return proba_predictions

      def score(self, X, y, **kwargs):
          """Force use of accuracy score since
          it doesn't inherit from ClassifierMixin"""
          return accuracy_score(y, self.predict(X))
