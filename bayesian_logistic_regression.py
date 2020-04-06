from __future__ import print_function, division, absolute_import
import numpy as np

'''
    We base on the code of SVGD released by its original author (Qiang Liu and Dilin Wang). Note that we also keep some
    of their comments in the code.

    Bayesian Logistic Regression (the same setting as Gershman et al. 2012):
    The observed data D = {X, y} consist of N binary class labels, 
    y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
    The hidden variables \theta = {w, \alpha} consist of d regression coefficients w_k \in R,
    and a precision parameter \alpha \in R_+. We assume the following model:
        p(\alpha) = Gamma(\alpha; a, b)
        p(w_k | a) = N(w_k; 0, \alpha^-1)
        p(y_t = 1| x_t, w) = 1 / (1+exp(-w^T x_t))
'''


class BayesianLR_VR:
    def __init__(self, X, Y, batchsize=100, a0=1, b0=0.01):
        self.X, self.Y = X, Y
        self.batchsize = min(batchsize, X.shape[0])
        self.a0, self.b0 = a0, b0

        self.N = X.shape[0]
        self.permutation = np.random.permutation(self.N)
        self.iter = 0

    def grad_ln_posterior(self, theta, data_idx=None):
        '''
        M: number of particles (i.e. theta)
        n: number of data points at which we compute the posterior (i.e. the number of elements in data_idx)
        N: number of all data points in the dataset
        D: dimension of each data point
        '''

        if data_idx is None:
            if self.batchsize > 0:
                batch = [i % self.N for i in range(self.iter * self.batchsize, (self.iter + 1) * self.batchsize)]
                ridx = self.permutation[batch]
                self.iter += 1
            else:
                ridx = np.random.permutation(self.X.shape[0])
        else:
            ridx = np.copy(data_idx)

        Xs = self.X[ridx, :]  # n x D
        Ys = self.Y[ridx]  # n

        w = theta[:, :-1]  # logistic weights: M x D
        alpha = np.exp(theta[:, -1])  # the last column is logalpha
        d = w.shape[1]  # D

        wt = np.multiply((alpha / 2), np.sum(w ** 2, axis=1))  # (M,)
        dalpha = d / 2.0 - wt + (self.a0 - 1) - self.b0 * alpha + 1  # (M,); the last term is the jacobian term

        coff = np.matmul(Xs, w.T)  # n x M
        y_hat = 1.0 / (1.0 + np.exp(-1 * coff))  # n x M

        dw_data = np.matmul(((np.broadcast_to(np.vstack(Ys), (len(Ys), theta.shape[0])) + 1) / 2.0 - y_hat).T,
                            Xs)  # Y \in {-1,1}
        dw_prior = -np.multiply(np.broadcast_to(np.vstack(alpha), (len(alpha), d)), w)  # M x D
        dw = dw_data * 1.0 * self.N / Xs.shape[0] + dw_prior  # re-scale to estimate the likelihood of the full
        # dataset from the likelihood of a minibatch

        return np.hstack([dw, np.vstack(dalpha)])  # first order derivative: M x (D + 1)

    def grad_ln_likelihood(self, theta, data_idx=None):
        '''
        M: number of particles (i.e. theta)
        n: number of data points at which we compute the likelihoods (i.e. the number of elements in data_idx)
        D: dimension of each data point
        '''

        if data_idx is None:  # compute the likelihood of every data point in the full dataset
            data_idx = np.arange(self.N)

        w = theta[:, :-1]  # logistic weights: M x D
        alpha = np.exp(theta[:, -1])  # the last column is logalpha: (M,)
        d = w.shape[1]  # D

        wt = np.multiply((alpha / 2), np.sum(w ** 2, axis=1))  # (M,)
        dalpha_data = (d / 2.0 - wt + 1) * len(data_idx) / self.N  # (M,)

        Xs = self.X[data_idx, :]  # n x D
        Ys = self.Y[data_idx]  # n

        coff = np.matmul(Xs, w.T)  # n x M
        y_hat = 1.0 / (1.0 + np.exp(-1 * coff))  # n x M

        dw_data = np.matmul(((np.broadcast_to(np.vstack(Ys), (len(Ys), theta.shape[0])) + 1) / 2.0 - y_hat).T,
                            Xs)  # Y \in {-1,1}

        return np.hstack([dw_data, np.vstack(dalpha_data)])  # M x (D + 1)

    def evaluation(self, theta, X_test, y_test):
        theta = theta[:, :-1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
        for t in range(M):
            coff = np.multiply(y_test, np.sum(
                -1 * np.multiply(np.broadcast_to(theta[t, :], (n_test, len(theta[t, :]))), X_test), axis=1))
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coff)))

        prob = np.mean(prob, axis=1)
        acc = np.mean(prob > 0.5)
        llh = np.mean(np.log(prob))
        return [acc, llh]

    def predict_in_dataset(self, theta, data_idx):
        '''
        Predict the labels given observations in the input dataset.

            theta: weights of the Bayesian logistic regression.

            data_idx: indices of the observations in the given dataset.
        '''
        return self.predict(theta, self.X[data_idx, :], self.Y[data_idx])

    def predict(self, theta, X_test, y_test=None):
        '''
        Predict the labels given observations.

            theta: weights of the Bayesian logistic regression.

            X_test: observations. Size N x D, where N is the number of observations and D is the dimension of each observation.

            y_test: corresponding true labels.
        '''
        theta = theta[:, :-1]
        M, n_test = theta.shape[0], len(y_test)

        prob = np.zeros([n_test, M])
        for t in np.arange(M):
            coeff = -1.0 * np.matmul(X_test, theta[t, :])
            prob[:, t] = np.divide(np.ones(n_test), (1 + np.exp(coeff)))

        prob = np.mean(prob, axis=1)
        y_pred = np.ones(n_test)
        y_pred[prob <= 0.5] = -1

        if y_test is None:
            return y_pred
        return y_pred, np.sum(y_pred == y_test)

    def eval_est_grad_var(self, theta, batchsize=None, shuffled_idx=None, variance_reduction=False, mu=None,
                          theta_ss=None):
        '''
        Evaluate the standard deviation of the estimator of the full-batch gradient.
            shuffled_idx: the list of the shuffled indices of all data points in the dataset. If this list is not
                provided, the current permutation stored in this object will be used. By default, shuffled_idx = None.

            variance_reduction: if True, SVRG estimator if used. If False, the traditional mini-batch estimator is used.
                By default, variance_reduction = False.

            mu: used in SVRG (i.e. when variance_reduction = True). This is the gradient of log likelihood of full
                dataset evaluated with the snapshot value of the parameters. By default, mu = None.

            theta_ss: (snapshot theta) used in SVRG. This is a snapshot of parameters. By default, theta_ss = None.
        '''
        batchsize = self.batchsize if batchsize is None else batchsize
        shuffled_idx = self.permutation if shuffled_idx is None else shuffled_idx
        all_est_grad = None
        true_grad = self.grad_ln_posterior(theta, shuffled_idx)  # (num_particles, particle_dim)
        num_batches = int(np.ceil(self.N / batchsize))
        if not variance_reduction:
            for it in np.arange(num_batches):
                batch = np.arange(it * batchsize, (it + 1) * batchsize) % self.N
                ridx = shuffled_idx[batch]
                rho = self.grad_ln_posterior(theta, ridx)  # (num_particles, particle_dim)
                if all_est_grad is None:
                    all_est_grad = rho
                else:
                    all_est_grad = np.hstack((all_est_grad, rho))  # (num_particles, particle_dim * num_batches)
        else:
            for it in np.arange(num_batches):
                batch = np.arange(it * batchsize, (it + 1) * batchsize) % self.N
                ridx = shuffled_idx[batch]
                gradlnp_hat = self.grad_ln_posterior(theta, ridx)  # (num_particles, particle_dim)
                rho = gradlnp_hat - self.grad_ln_likelihood(theta_ss,
                                                            ridx) * self.N / batchsize + mu  # (num_particles, particle_dim)
                if all_est_grad is None:
                    all_est_grad = rho
                else:
                    all_est_grad = np.hstack((all_est_grad, rho))  # (num_particles, particle_dim * num_batches)
        tmp = np.reshape((all_est_grad - np.tile(true_grad, (1, num_batches))) ** 2, (theta.shape[0], num_batches, -1))
        # (num_particles, num_batches, particle_dim)
        tmp = np.sqrt(np.mean(tmp, axis=1))  # (num_particles, particle_dim)
        stddev_est_grad = np.mean(tmp, axis=0)  # (particle_dim,)
        return stddev_est_grad
