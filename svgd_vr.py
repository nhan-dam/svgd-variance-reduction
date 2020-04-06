from __future__ import print_function, division, absolute_import
import numpy as np
from scipy.spatial.distance import pdist, squareform


class SVGD_VR():
    '''
    Stein Variational Gradient Descent with Variance Reduction.
    '''

    def __init__(self):
        pass

    def svgd_kernel(self, theta, h=-1):
        sq_dist = pdist(theta)
        pairwise_dists = squareform(sq_dist) ** 2
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(theta.shape[0] + 1))

        # compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)

        dxkxy = -np.matmul(Kxy, theta)
        sumkxy = np.sum(Kxy, axis=1)
        for i in range(theta.shape[1]):
            dxkxy[:, i] = dxkxy[:, i] + np.multiply(theta[:, i], sumkxy)
        dxkxy = dxkxy / (h ** 2)
        return (Kxy, dxkxy)

    def _update_online(self, x0, grad_ln_posterior, num_data, eval_fn, eval_freq=1, grad_ln_likelihood=None,
                       n_vr_updates=0, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False, decay_stepsize=0):
        '''
        The update procedure of Stein Variational Gradient Descent with Variance Reduction in online learning setting.

            num_data: the number of data points in the training set.

            n_vr_updates: the number of updates of the inner loop corresponding to variance reduction technique. If
                n_vr_updates = 0, it is computed from the dataset size by the formular n_vr_updates <- 5 * num_data
                as suggested by the original paper of the variance reduction technique. By default, n_vr_updates = 0.

            eval_fn: the function to evaluate the performance of the model with the current parameters.

            eval_freq: the frequency (in terms of number of data points) to record the intermediate performance.
                This value will be set to be greater than 1 for large datasets. By default, eval_freq = 1.

            decay_stepsize: the hyperparameters that control the decay of step size in SVGD and SVGD-VR. By default,
                decay_stepsize = 0.

            M: num of particles x0; N: num of observed data points; D: dimension of each particle.
        '''
        theta = np.copy(x0)
        n_svgd_updates = num_data

        decay_stepsize = max(decay_stepsize, 0)

        # adagrad with momentum (aka RMSProp)
        fudge_factor = 1e-6
        historical_grad = 0.0
        loss_log = np.zeros(int(np.ceil(num_data / eval_freq)) + 1)
        cnt = 0
        accumulative_loss = 0.0
        if grad_ln_likelihood is None:  # update parameters using SVGD
            data_idx_perm = np.random.permutation(num_data)
            for it in range(n_svgd_updates):
                if debug and (it + 1) % 1000 == 0:
                    print('iter ' + str(it + 1))

                data_idx = data_idx_perm[[it % num_data]]
                _, current_loss = eval_fn(theta, data_idx)
                accumulative_loss += current_loss  # we only evaluate the performance at one current data point
                if it % eval_freq == 0 or (it + 1) == num_data:
                    loss_log[cnt] = accumulative_loss * 1.0 / (it + 1)
                    cnt += 1

                lnpgrad = grad_ln_posterior(theta, data_idx)
                # calculating the kernel matrix
                kxy, dxkxy = self.svgd_kernel(theta, h=-1)
                grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

                # adagrad 
                if it == 0:
                    historical_grad = grad_theta ** 2
                else:
                    historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
                adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
                theta = theta + stepsize * np.exp(-decay_stepsize * it) * adj_grad
        else:  # update parameters using SVGD-VR
            if n_vr_updates == 0:
                n_vr_updates = 5 * num_data  # following the formula in SVRG paper

            data_idx_perm = np.random.permutation(num_data)
            for l in range(int(np.ceil(n_svgd_updates / n_vr_updates))):
                if debug and (l + 1) % 100 == 0:
                    print('iter ' + str(l + 1))

                mu = grad_ln_likelihood(theta, np.arange(num_data))  # M x D

                theta_hat = np.copy(theta)
                for t in range(n_vr_updates):
                    if l * n_vr_updates + t >= num_data:
                        break
                    # calculating the kernel matrix
                    data_idx = data_idx_perm[[(l * n_vr_updates + t) % num_data]]

                    _, current_loss = eval_fn(theta_hat, data_idx)
                    accumulative_loss += current_loss  # we only evaluate the performance at one current data point
                    if (l * n_vr_updates + t) % eval_freq == 0 or (l * n_vr_updates + t + 1) == num_data:
                        loss_log[cnt] = accumulative_loss * 1.0 / (l * n_vr_updates + t + 1)
                        cnt += 1

                    kxy, dxkxy = self.svgd_kernel(theta_hat, h=-1)  # M x M, M x D
                    gradlnp_hat = grad_ln_posterior(theta_hat, data_idx)  # M x D
                    rho = gradlnp_hat - grad_ln_likelihood(theta, data_idx) * num_data + mu  # M x D
                    grad_theta_hat = (np.matmul(kxy, rho) + dxkxy) / x0.shape[0]

                    # adagrad 
                    if t == 0:
                        historical_grad = grad_theta_hat ** 2
                    else:
                        historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta_hat ** 2)
                    adj_grad = np.divide(grad_theta_hat, fudge_factor + np.sqrt(historical_grad))
                    theta_hat = theta_hat + stepsize * np.exp(-decay_stepsize * l) * adj_grad

                theta = np.copy(theta_hat)

        return theta, loss_log

    def update(self, x0, grad_ln_posterior, grad_ln_likelihood=None, num_data=0, n_svgd_updates=1000,
               n_vr_updates=0, batchsize_vr=1, stepsize=1e-3, bandwidth=-1, alpha=0.9, debug=False,
               eval_fn=None, eval_freq=0, decay_stepsize=0, online=False, eval_est_grad_var=None):
        '''
        The update procedure of Stein Variational Gradient Descent with Variance Reduction.

            num_data: the number of data points in the training set. If num_data = 0, the full dataset is used
                to compute grad(log p(x)) (i.e. similar to the original SVGD). By default, num_data = 0.

            n_vr_updates: the number of updates of the inner loop corresponding to variance reduction technique. If
                n_vr_updates = 0, it is computed from the dataset size by the formular n_vr_updates <- 5 * num_data
                as suggested by the original paper of the variance reduction technique. By default, n_vr_updates = 0.

            batchsize_vr: the batch size when computing likelihood in variance reduction method.

            eval_fn: the function to evaluate the performance of the model with the current parameters.
                By default, eval_fn = None.

            eval_freq: the frequency (in iterations) to evaluate the performance of the model with the current
                parameters. By default, eval_freq = 0. If eval_fn is not None and eval_freq is not positive, eval_freq
                will be assigned 0.

            decay_stepsize: the hyperparameters that control the decay of step size in SVGD and SVGD-VR. By default,
                decay_stepsize = 0.

            online: boolean flag for online learning setting. If online = True, we follow online learning to update
                particles. That means for each training data point, we predict its label before using it for training.
                We compute the accumulated accuracy of prediction. By default, online = False.

            eval_est_grad_var: a function to evaluate the variance of the estimator of the full-batch gradient. By
                default, eval_est_grad_var = None.

            M: num of particles x0; N: num of observed data points; D: dimension of each particle.
        '''
        # Check input
        if x0 is None or grad_ln_posterior is None:
            raise ValueError('x0 or grad_ln_posterior cannot be None!')
        if online and num_data == 0:
            raise ValueError('The number of training data points demanded to use online learning!')
        if num_data != 0 and not online and grad_ln_likelihood is None:
            raise ValueError('Likelihood function demanded to use variance reduction method!')

        # online learning
        if online:
            return self._update_online(x0, grad_ln_posterior, num_data, eval_fn, eval_freq=max(eval_freq, 1),
                                       grad_ln_likelihood=grad_ln_likelihood, n_vr_updates=n_vr_updates,
                                       stepsize=stepsize,
                                       bandwidth=bandwidth, alpha=alpha, debug=debug, decay_stepsize=decay_stepsize)

        # not online learning
        theta = np.copy(x0)
        if eval_fn is not None:
            eval_log = []
            eval_freq = max(eval_freq, 1)
        if eval_est_grad_var is not None:
            grad_var_log = []
            eval_freq = max(eval_freq, 1)

        decay_stepsize = max(decay_stepsize, 0)

        # adagrad with momentum (aka RMSProp)
        fudge_factor = 1e-6
        historical_grad = 0.0
        if num_data == 0:  # original SVGD
            for it in range(n_svgd_updates):
                if debug and (it + 1) % 100 == 0:
                    print('iter ' + str(it + 1))

                # evaluate the variance of the estimator for the gradient
                if eval_est_grad_var is not None and (it + 1) % eval_freq == 0:
                    tmp = eval_est_grad_var(theta)
                    if len(grad_var_log) == 0:
                        grad_var_log = tmp
                    else:
                        grad_var_log = np.vstack((grad_var_log, tmp))

                # update the parameters \theta
                lnpgrad = grad_ln_posterior(theta)
                # calculating the kernel matrix
                kxy, dxkxy = self.svgd_kernel(theta, h=-1)
                grad_theta = (np.matmul(kxy, lnpgrad) + dxkxy) / x0.shape[0]

                # adagrad 
                if it == 0:
                    historical_grad = grad_theta ** 2
                else:
                    historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta ** 2)
                adj_grad = np.divide(grad_theta, fudge_factor + np.sqrt(historical_grad))
                theta = theta + stepsize * np.exp(-decay_stepsize * it) * adj_grad

                if eval_fn is not None and (it + 1) % eval_freq == 0:
                    if len(eval_log) == 0:
                        eval_log = eval_fn(theta)
                    else:
                        eval_log = np.vstack((eval_log, eval_fn(theta)))

        else:  # SVGD-VR
            it = 0  # used to determine when to evaluate the performance of the current model
            if n_vr_updates == 0:
                n_vr_updates = 5 * num_data

            data_idx_perm = np.random.permutation(num_data)
            for l in range(n_svgd_updates):
                if debug and (l + 1) % 1 == 0:
                    print('iter ' + str(l + 1))

                mu = grad_ln_likelihood(theta, np.arange(num_data))  # M x D

                theta_hat = np.copy(theta)
                for t in range(n_vr_updates):
                    # evaluate the variance of the estimator for the gradient
                    if eval_est_grad_var is not None and (it + 1) % eval_freq == 0:
                        tmp = eval_est_grad_var(theta_hat, batchsize_vr, data_idx_perm, True, mu, theta)
                        if len(grad_var_log) == 0:
                            grad_var_log = tmp
                        else:
                            grad_var_log = np.vstack((grad_var_log, tmp))

                    # calculating the kernel matrix
                    data_idx = data_idx_perm[np.arange((l * n_vr_updates + t) * batchsize_vr,
                                                       (l * n_vr_updates + t + 1) * batchsize_vr) % num_data]
                    kxy, dxkxy = self.svgd_kernel(theta_hat, h=-1)  # M x M, M x D
                    gradlnp_hat = grad_ln_posterior(theta_hat, data_idx)  # M x D
                    rho = gradlnp_hat - grad_ln_likelihood(theta, data_idx) * num_data / batchsize_vr + mu  # M x D
                    grad_theta_hat = (np.matmul(kxy, rho) + dxkxy) / x0.shape[0]

                    # adagrad 
                    if t == 0:
                        historical_grad = grad_theta_hat ** 2
                    else:
                        historical_grad = alpha * historical_grad + (1 - alpha) * (grad_theta_hat ** 2)
                    adj_grad = np.divide(grad_theta_hat, fudge_factor + np.sqrt(historical_grad))
                    theta_hat = theta_hat + stepsize * np.exp(-decay_stepsize * l) * adj_grad

                    if eval_fn is not None and (it + 1) % eval_freq == 0:
                        if len(eval_log) == 0:
                            eval_log = eval_fn(theta_hat)
                        else:
                            eval_log = np.vstack((eval_log, eval_fn(theta_hat)))

                    it += 1

                theta = np.copy(theta_hat)

        if eval_est_grad_var is not None:
            if eval_fn is not None:
                return theta, eval_log, grad_var_log
            return theta, grad_var_log
        if eval_fn is not None:
            return theta, eval_log
        return theta
