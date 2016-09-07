import mdp
import Oger
import numpy as np
import numpy.linalg as la
import os
import cPickle as pickle
import copy

class RidgeRegressionNode(mdp.Node):

    '''
    Ridge Regression Node that also optimizes the regularization parameter
    '''
    def __init__(self, ridge_param=mdp.numx.power(10, mdp.numx.arange(-10,5,0.5)), eq_noise_var=0, other_error_measure=None, cross_validate_function=None, low_memory=False, verbose=False, plot_errors=False, with_bias=True, clear_memory=True, input_dim=None, output_dim=None, dtype=None, *args, **kwargs):
        '''

        ridge_params contains the list of regularization parameters to be tested. If it is set to 0 no regularization
        or the eq_noise_var is used. Default 10^[-15:5:0.2].

        It is also possible to define an equivalent noise variance: the ridge parameter is set such that a
        regularization equal to a given added noise variance is achieved. Note that setting the ridge_param has
        precedence to the eq_noise_var and that optimizing the eq_noise_var is not yet supported.

        If an other_error_measure is used processing is slower! For classification for example one can use:
        other_error_measure = Oger.utils.threshold_before_error(Oger.utils.loss_01). Default None.

        cross_validation_function can be any function that returns a list of containing the cross validation sequence.
        The arguments for this function can be set using args and kwargs. n_samples is automatically set to the number
        of training examples given. Default Oger.evaluation.leave_one_out.

        low_memory=True Limits memory use to twice the size of the covariance matrix. It saves data in files instead of
        keeping them in memory, this is possible for up to 128 training examples. Default False

        verbose=True gives additional information about the optimization progress

        plot_errors=True gives a plot of the validation errors in function of log10(ridge_param). Default False

        with_bias=True adds an additional bias term. Default True.
        '''
        super(RidgeRegressionNode, self).__init__(input_dim=input_dim, output_dim=output_dim, dtype=dtype)
        self.ridge_param = ridge_param
        self.eq_noise_var = eq_noise_var
        self.other_error_measure = other_error_measure
        self.low_memory = low_memory
        self.verbose = verbose
        self.plot_errors = plot_errors
        self.with_bias = with_bias
        if cross_validate_function == None:
            cross_validate_function = Oger.evaluation.leave_one_out
        self.cross_validate_function = cross_validate_function
        self._args, self._kwargs = args, kwargs
        self.clear_memory = clear_memory

        self._xTx_list, self._xTy_list, self._yTy_list, self._len_list = [], [], [], []
        if other_error_measure:
            self._x_list = []
            self._y_list = []

    def _get_supported_dtypes(self):
        return ['float32', 'float64']

    def _train(self, x, y):
        y = y.astype(self.dtype) #avoid True + True != 2
        if self.with_bias:
            x = np.concatenate((x, np.ones((x.shape[0], 1), dtype=self.dtype)), axis=1)
        if not self._output_dim == y.shape[1]:
            self._output_dim = y.shape[1]
        #Calculate the covariance matrices
        self._set('xTx', np.dot(x.T, x))
        self._set('xTy', np.dot(x.T, y))
        self._yTy_list.append(np.sum(y**2, axis=0))
        self._len_list.append(len(y))
        if self.other_error_measure:
            self._set('x', x)
            self._set('y', y)

    def _stop_training(self):
        if (type(self.ridge_param) is list or type(self.ridge_param) is np.ndarray) and len(self._xTx_list)>1:
            self._ridge_params = self.ridge_param
            if self.other_error_measure:
                calc_error = self._calc_other_error
            else:
                calc_error = self._calc_mse

            import time
            t_start = time.time()
            train_samples, val_samples = self.cross_validate_function(n_samples=len(self._xTx_list), *self._args, **self._kwargs)
            errors = np.zeros((len(self._ridge_params), self._output_dim))
            val_sets = range(len(train_samples))
            if self.verbose:
                val_sets = mdp.utils.progressinfo(val_sets, style='timer')
            for k in val_sets:
                errors += calc_error(train_samples[k], val_samples[k])
            errors /= len(train_samples)

            self.val_error, self.ridge_param = [], []
            for o in range(self._output_dim):
                r = mdp.numx.where(errors == np.nanmin(errors[:,o]))[0][-1]
                self.val_error.append(errors[r,o])
                self.ridge_param.append(self._ridge_params[r])
                if r==0 or r==len(self._ridge_params)-1:
                    import warnings
                    #warnings.warn('The ridge parameter selected for output ' + str(o) + ' is ' + str(self.ridge_param[-1]) + '. This is the largest or smallest possible value from the list provided. Use larger or smaller ridge parameters to avoid this warning!')

            if self.verbose:
                print 'Total time:', time.time()-t_start, 's'
                print 'Found the ridge_param(s) =', self.ridge_param, 'with a validation error(s) of:', self.val_error
            if self.plot_errors:
                import pylab
                pylab.plot(np.log10(self._ridge_params),errors)
                pylab.show()
        else:
            if len(self._xTx_list)==1 and (type(self.ridge_param) is list or type(self.ridge_param) is np.ndarray) and len(self.ridge_param)>1:
                import warnings
                warnings.warn('Only one fold found, optimization is not supported. Instead no regularization or eq_noise_var is used!')
                self.ridge_param = 0
            elif self.ridge_param == 0:
                self.ridge_param = self.eq_noise_var**2 * np.sum(self._len_list)
            self.ridge_param = self.ridge_param * np.ones((self._output_dim,))

        self._final_training()
        self._clear_memory()

    def _execute(self, x):
        return np.dot(x, self.w).reshape((-1,self._output_dim)) + self.b

    def _get(self,name,l=None):
        l = list(l)
        ret = copy.copy(self._get_one(name, l.pop()))
        for i in l:
            if name.count('T') or name.count('len'):
                ret += self._get_one(name, i)
            else:
                ret = np.concatenate((ret, self._get_one(name, i)), axis=0)
        return ret

    def _get_one(self, name, i):
        t = getattr(self, '_' + name + '_list')
        if self.low_memory and not name.count('yTy') and not name.count('len'):
            t[i].seek(0)
            return pickle.load(t[i])
        else:
            return t[i]

    def _set(self,name,t,i=None):
        if self.low_memory and not name.count('yTy') and not name.count('len'):
            f = os.tmpfile()
            pickle.dump(t, f, protocol=-1)
            t = f
        if not i==None:
            getattr(self, '_' + name + '_list')[i] = t
        else:
            getattr(self, '_' + name + '_list').append(t)

    def _calc_mse(self, train, val, s=None):
        # Calculate the MSE for this validation set
        if s==None:
            s=range(self._input_dim + self.with_bias)
        errors = np.zeros((len(self._ridge_params), self._output_dim))
        D_t, C_t = la.eigh(self._get('xTx', train)[s,:][:,s] +  np.eye(len(s))) #reduce condition number
        D_t -= 1
        D_t[np.where(D_t<0)] = 0 #eigenvalues can only be positive
        D_t = 1 / (np.atleast_2d(D_t).T + np.atleast_2d(self._ridge_params))
        xTy_Ct = np.dot(C_t.T, self._get('xTy', train)[s,:])
        xTx_Cv = np.dot(C_t.T, np.dot(self._get('xTx', val)[s,:][:,s], C_t))
        xTy_Cv = 2 * np.dot(C_t.T, self._get('xTy', val)[s,:])
        # calculate error for all ridge params at once
        for o in range(self._output_dim):
            W_Ct = xTy_Ct[:,o:o+1] * D_t
            errors[:,o] += np.sum(W_Ct * (np.dot(xTx_Cv, W_Ct) - xTy_Cv[:,o:o+1]), axis=0)
        return (errors + self._get('yTy', val)) / self._get('len', val)

    def _calc_other_error(self, train, val, s=None):
        # Calculate error for this validation set using other error measure
        if s==None:
            s=range(self._input_dim + self.with_bias)
        errors = np.zeros((len(self._ridge_params), self._output_dim))
        xTx_t = self._get('xTx', train)[s,:][:,s]
        xTy_t = self._get('xTy', train)[s,:]
        x = self._get('x', val)[:,s]
        y = self._get('y', val)
        for r in range(len(self._ridge_params)):
            output = np.dot(x, la.solve(xTx_t + self._ridge_params[r] * np.eye(len(xTx_t)), xTy_t))
            for o in range(self._output_dim):
                errors[r,o] = self.other_error_measure(output[:,o], y[:,o])
        return errors

    def _final_training(self):
        # Calculate final weights
        xTx = self._get('xTx', range(len(self._xTx_list)))
        xTy = self._get('xTy', range(len(self._xTx_list)))
        W = np.zeros(xTy.shape)
        for o in range(self._output_dim):
            W[:,o] = la.solve(xTx + self.ridge_param[o] * np.eye(len(xTx)), xTy[:,o])
        if self.with_bias:
            self.b = W[-1, :]
            self.w = W[:-1, :]
        else:
            self.b = 0
            self.w = W

    def _clear_memory(self):
        if self.clear_memory:
            self._xTx_list, self._xTy_list, self._yTy_list, self._len_list = [],[],[],[]