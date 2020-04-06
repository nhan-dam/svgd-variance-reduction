from __future__ import print_function, division, absolute_import
import numpy as np
import tensorflow as tf
from scipy.spatial.distance import pdist, squareform

'''
    Source code for learning 1 hidden layer Bayesian neural net with Stein variational gradient descent with variance reduction.
    We base on the code of SVGD released by its original authors (Qiang Liu and Dilin Wang). Note that we also keep some
    of their comments in the code.

    The generative process of our Bayesian neural net is:
    
        p(y | W, X, \gamma) = \prod_i^N  N(y_i | f(x_i; W), \gamma^{-1})
        p(W | \lambda) = \prod_i N(w_i | 0, \lambda^{-1})
        p(\gamma) = Gamma(\gamma | a0, b0)
        p(\lambda) = Gamma(\lambda | a0, b0)
    
    Then, the unnormalised posterior distribution is:
        p~(W, \gamma, \lambda) = p(y | W, X, \gamma) p(W | \lambda) p(\gamma) p(\lambda) 
    Following Liu and Wang, we also update loggamma and loglambda to avoid the negative values of \gamma and \lambda.
'''


class svgdvr_bayesnn:
    '''
        We define a one-hidden-layer-neural-network specifically.
        
        Input
            -- X_train: training dataset, features
            -- y_train: training labels
            -- num_samples: number of training samples (note that 'num_data' is used to refer to the number of data
                points in self.x that we feed to the model when call tf session run, which can be equal to num_samples,
                batch_size or another value).
            -- data_dim: dimension of data points
            -- batch_size: sub-sampling batch size
            -- n_svgd_updates: maximum iterations for SVGD updates
            -- num_particles: number of particles are used to fit the posterior distribution
            -- num_hidden: number of hidden units
            -- a0, b0: hyper-parameters of Gamma distribution
            -- svgd_stepsize, auto_corr: parameters of adgrad
            -- 'init_particles': initial values for the particles
            -- 'preprocess_option': 0 - no data preprocessing, 1 - normalising, 2 - standardising.
    '''

    def __init__(self, X_train, y_train, batch_size=100, n_svgd_updates=1000, num_particles=20, num_hidden=50,
                 a0=1, b0=0.1, svgd_stepsize=1e-3, auto_corr=0.9, float_type=64, init_particles=None,
                 preprocess_option=0, n_vr_updates=0):
        # declare variables
        [self.num_samples, self.data_dim] = X_train.shape
        self.batch_size = batch_size
        self.n_svgd_updates = n_svgd_updates
        self.num_particles = num_particles
        self.num_hidden = num_hidden
        self.a0, self.b0 = a0, b0
        self.svgd_stepsize = svgd_stepsize
        self.auto_corr = auto_corr
        if float_type == 32:
            self.float_type = tf.float32
        elif float_type == 64:
            self.float_type = tf.float64
        else:  # undefined float type
            self.float_type = None
        self.init_particles = init_particles
        self.preprocess_option = preprocess_option
        self.n_vr_updates = 5 * self.num_samples if n_vr_updates == 0 else n_vr_updates
        if self.preprocess_option == 1:  # normalise
            self.min_X_train, self.max_X_train = np.min(X_train, axis=0), np.max(X_train, axis=0)
            self.min_y_train, self.max_y_train = np.min(y_train), np.max(y_train)
        elif self.preprocess_option == 2:  # standardise
            self.mean_X_train, self.std_X_train = np.mean(X_train, axis=0), np.std(X_train, axis=0)
            self.std_X_train[self.std_X_train == 0] = 1.
            self.mean_y_train, self.std_y_train = np.mean(y_train), np.std(y_train)
        X_train, y_train = self._preprocess_data(X_train, y_train)
        self.particle_dim = self.data_dim * self.num_hidden + self.num_hidden * 2 + 3

        # build the model
        self._build_model(X_train, y_train)

        # initialise variables
        self.init_variables_op = tf.global_variables_initializer()

        # start tf session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        self.tf_session = tf.Session(config=config)
        self.tf_session.run(self.init_variables_op)

    '''
        Initial values of all particles
    '''

    def _init_one_particle(self, a0, b0):
        w1 = 1.0 / np.sqrt(self.data_dim + 1) * np.random.randn(self.data_dim, self.num_hidden)
        b1 = np.zeros(self.num_hidden)
        w2 = 1.0 / np.sqrt(self.num_hidden + 1) * np.random.randn(self.num_hidden)
        b2 = 0.
        loggamma = np.log(np.random.gamma(a0, b0))
        loglambda = np.log(np.random.gamma(a0, b0))
        return w1, b1, w2, b2, loggamma, loglambda

    def _init_all_particles(self, X_train, y_train):
        if self.init_particles is None:
            init_w1 = np.zeros(
                (self.num_particles, self.data_dim, self.num_hidden))  # (num_particles, data_dim, num_hidden)
            init_b1 = np.zeros((self.num_particles, self.num_hidden))  # (num_particles, num_hidden)
            init_w2 = np.zeros((self.num_particles, self.num_hidden))  # (num_particles, num_hidden)
            init_b2 = np.zeros((self.num_particles, 1))  # (num_particles, 1)
            init_loggamma = np.zeros((self.num_particles, 1))  # (num_particles, 1)
            init_loglambda = np.zeros((self.num_particles, 1))  # (num_particles, 1)
            for it in np.arange(self.num_particles):
                init_w1[it, :, :], init_b1[it, :], init_w2[it, :], init_b2[it], init_loggamma[it], init_loglambda[it] = \
                    self._init_one_particle(self.a0, self.b0)
                ridx = np.random.choice(self.num_samples, np.min([self.num_samples, 1000]), replace=False)
                tmp = np.maximum(np.einsum('ij,jk->ik', X_train[ridx, :], init_w1[it, :, :]) + init_b1[it, :],
                                 0) * init_w2[it, :]
                nn_output = np.sum(tmp, axis=1) + init_b2[it]
                init_loggamma[it] = -np.log(np.mean((nn_output - y_train[ridx]) ** 2))
            self.init_particles = np.concatenate([init_w1.reshape([self.num_particles, -1]), init_b1, init_w2, init_b2,
                                                  init_loggamma, init_loglambda],
                                                 axis=1)  # (num_particles, particle_dim)
            if len(self.init_particles.shape) < 2:  # when there is only 1 particle
                self.init_particles = self.init_particles.reshape([1, self.particle_dim])
        return self.init_particles

    def _build_model(self, X_train, y_train):
        # self.preprocess_option = tf.placeholder(dtype=tf.int32)
        self.x = tf.placeholder(self.float_type, shape=[None, self.data_dim])
        self.y = tf.placeholder(self.float_type, shape=[None])

        self.w1_idx_end = self.data_dim * self.num_hidden
        self.b1_idx_start = self.w1_idx_end
        self.b1_idx_end = self.b1_idx_start + self.num_hidden
        self.w2_idx_start = self.b1_idx_end
        self.w2_idx_end = self.w2_idx_start + self.num_hidden

        self.init_particles = self._init_all_particles(X_train, y_train)
        self.particles = tf.Variable(self.init_particles, dtype=self.float_type)  # (num_particles, particle_dim)
        self.particles_hat = tf.placeholder(self.float_type, shape=[self.num_particles, self.particle_dim])

        grad_log_posterior_list = []
        log_posterior_list = []
        predicted_y_list = []
        nn_output_list = []
        probability_list = []
        grad_log_likelihood_list = []
        for it in np.arange(self.num_particles):
            grad_logpost, logpost, pred, nn_output, prob = self._grad_log_posterior(self.particles[it, :], True)
            grad_loglik = self._grad_log_likelihood(self.particles_hat[it, :], True)
            grad_log_posterior_list.append(grad_logpost)
            log_posterior_list.append(logpost)
            predicted_y_list.append(pred)
            nn_output_list.append(nn_output)
            probability_list.append(prob)
            grad_log_likelihood_list.append(grad_loglik)
        self.grad_log_posterior = tf.stack(grad_log_posterior_list, axis=0)  # (num_particles, particle_dim)
        self.log_posterior = tf.stack(log_posterior_list, axis=0)  # (num_particles,)
        self.predicted_y = tf.stack(predicted_y_list, axis=0)  # (num_particles, num_data)
        self.nn_output = tf.stack(nn_output_list, axis=0)
        self.probability = tf.stack(probability_list, axis=0)  # (num_particles, num_data)
        self.grad_log_likelihood = tf.stack(grad_log_likelihood_list, axis=0)  # (num_particles, particle_dim)
        self.new_particles = tf.placeholder(self.float_type, shape=[self.num_particles, self.particle_dim])
        # (num_particles, particle_dim)
        self.update_particles = tf.assign(self.particles, self.new_particles)

    '''
        Calculate kernel matrix and its gradient: K, \nabla_x k
    '''

    def _svgd_kernel(self, particles, h=-1):
        sq_dist = pdist(particles)
        pairwise_dists = squareform(sq_dist) ** 2  # (num_particles, num_particles)
        if h < 0:  # if h < 0, using median trick
            h = np.median(pairwise_dists)
            h = np.sqrt(0.5 * h / np.log(particles.shape[0] + 1))  # () - scalar

        # Compute the rbf kernel
        Kxy = np.exp(-pairwise_dists / h ** 2 / 2)  # (num_particles, num_particles)

        dxKxy = -np.einsum('ij,jk->ik', Kxy, particles)  # (num_particles, particle_dim)
        sumkxy = np.sum(Kxy, axis=1)  # (num_particles,)
        for i in range(particles.shape[1]):
            dxKxy[:, i] = dxKxy[:, i] + np.multiply(particles[:, i], sumkxy)
        dxKxy = dxKxy / (h ** 2)  # (num_particles, particle_dim)
        return (Kxy, dxKxy)

    '''
        Pack all parameters in our model
    '''

    def _pack_weights(self, w1, b1, w2, b2, loggamma, loglambda, use_tensorflow=False):
        if not use_tensorflow:
            params = np.concatenate([w1.flatten(), b1, w2, [b2], [loggamma], [loglambda]])
        else:
            params = tf.concat([tf.reshape(w1, shape=[self.data_dim * self.num_hidden]), b1, w2, [b2],
                                [loggamma], [loglambda]], axis=0)
        return params

    '''
        Unpack all parameters in our model
    '''

    def _unpack_weights(self, particles, use_tensorflow=False):
        w = particles
        if not use_tensorflow:
            if len(w.shape) > 1:
                w1 = np.reshape(w[:, :self.w1_idx_end], [self.num_particles, self.data_dim, self.num_hidden])
                # (num_particles, data_dim, num_hidden)
                b1 = w[:, self.b1_idx_start: self.b1_idx_end]  # (num_particles, num_hidden)
                w2 = w[:, self.w2_idx_start: self.w2_idx_end]  # (num_particles, num_hidden)
                b2 = w[:, -3]  # (num_particles,)
                loggamma = w[:, -2]  # (num_particles,)
                loglambda = w[:, -1]  # (num_particles,)
            else:
                w1 = np.reshape(w[:self.w1_idx_end], [self.data_dim, self.num_hidden])  # (num_hidden, data_dim)
                b1 = w[self.b1_idx_start: self.b1_idx_end]  # (num_hidden,)
                w2 = w[self.w2_idx_start: self.w2_idx_end]  # (num_hidden,)
                b2 = w[-3]  # () - scalar
                loggamma = w[-2]  # ()
                loglambda = w[-1]  # ()
        else:
            if len(w.get_shape().as_list()) > 1:
                w1 = tf.reshape(w[:, :self.w1_idx_end], shape=[self.num_particles, self.data_dim, self.num_hidden])
                # (num_particles, num_hidden, data_dim)
                b1 = w[:, self.b1_idx_start: self.b1_idx_end]  # (num_particles, num_hidden)
                w2 = w[:, self.w2_idx_start: self.w2_idx_end]  # (num_particles, num_hidden)
                b2 = w[:, -3]  # (num_particles,)
                loggamma = w[:, -2]  # (num_particles,)
                loglambda = w[:, -1]  # (num_particles,)
            else:
                w1 = tf.reshape(w[:self.w1_idx_end], shape=[self.data_dim, self.num_hidden])  # (data_dim, num_hidden)
                b1 = w[self.b1_idx_start: self.b1_idx_end]  # (num_hidden,)
                w2 = w[self.w2_idx_start: self.w2_idx_end]  # (num_hidden,)
                b2 = w[-3]  # () - scalar
                loggamma = w[-2]  # ()
                loglambda = w[-1]  # ()

        return w1, b1, w2, b2, loggamma, loglambda

    '''
        Compute the gradient of the log of posterior
    '''

    def _grad_log_posterior(self, particle, use_tensorflow=False):
        # IMPORTANT: Note that the gradient of log posterior and log posterior are used in training so they utilise
        # preprocessed y in computation. Whereas prediction and probability are used in test so they take the raw y.
        with tf.variable_scope('posterior'):
            current_batchsize = tf.cast(tf.shape(self.x)[0], dtype=self.float_type)
            w1, b1, w2, b2, loggamma, loglambda = self._unpack_weights(particle, use_tensorflow)
            nn_output = tf.reduce_sum(tf.nn.relu(tf.einsum('ij,jk->ik', self.x, w1) + b1) * w2, axis=1) + b2
            # (num_data,)

            log_likelihood_data = - 0.5 * current_batchsize * (
                        tf.cast(tf.log(2 * np.pi), dtype=self.float_type) - loggamma) \
                                  - (tf.exp(loggamma) / 2.) * tf.reduce_sum((nn_output - self.y) ** 2)  # () - scalar
            log_prior_data = (self.a0 - 1) * loggamma - self.b0 * tf.exp(loggamma) + loggamma  # () - scalar
            log_prior_w = - 0.5 * (self.particle_dim - 2) * (
                        tf.cast(tf.log(2 * np.pi), dtype=self.float_type) - loglambda) \
                          - (tf.exp(loglambda) / 2.) * (tf.reduce_sum(w1 ** 2) + tf.reduce_sum(w2 ** 2)
                                                        + tf.reduce_sum(b1 ** 2) + b2 ** 2) \
                          + (self.a0 - 1) * loglambda - self.b0 * tf.exp(loglambda) + loglambda  # () - scalar
            log_posterior = log_likelihood_data / current_batchsize * self.num_samples + log_prior_data + log_prior_w
            # () - scalar

            d_w1, d_b1, d_w2, d_b2, d_loggamma, d_loglambda = tf.gradients(log_posterior, [w1, b1, w2, b2,
                                                                                           loggamma, loglambda])
            grad_all = self._pack_weights(d_w1, d_b1, d_w2, d_b2, d_loggamma, d_loglambda, True)  # (particle_dim,)

            if self.preprocess_option == 0:  # no data preprocessing
                predicted_y = nn_output
            elif self.preprocess_option == 1:  # normalise data
                predicted_y = nn_output * (self.max_y_train - self.min_y_train) + self.min_y_train
            elif self.preprocess_option == 2:  # standardise data
                predicted_y = nn_output * self.std_y_train + self.mean_y_train
            prob = tf.sqrt(tf.exp(loggamma)) / tf.cast(tf.sqrt(2 * np.pi), dtype=self.float_type) \
                   * tf.exp(-tf.exp(loggamma) / 2.0 * (predicted_y - self.y) ** 2)  # (num_data,)

        return grad_all, log_posterior, predicted_y, nn_output, prob

    '''
        Compute the gradient of the log of likelihood
    '''

    def _grad_log_likelihood(self, particle, use_tensorflow=False):
        with tf.variable_scope('likelihood'):
            current_batchsize = tf.cast(tf.shape(self.x)[0], dtype=self.float_type)
            w1, b1, w2, b2, loggamma, loglambda = self._unpack_weights(particle, use_tensorflow)
            nn_output = tf.reduce_sum(tf.nn.relu(tf.einsum('ij,jk->ik', self.x, w1) + b1) * w2, axis=1) + b2
            # (num_data,)

            log_likelihood_data = - 0.5 * current_batchsize * (
                        tf.cast(tf.log(2 * np.pi), dtype=self.float_type) - loggamma) \
                                  - (tf.exp(loggamma) / 2.) * tf.reduce_sum((nn_output - self.y) ** 2)  # () - scalar
            log_likelihood_w = - 0.5 * (self.particle_dim - 2) * (
                        tf.cast(tf.log(2 * np.pi), dtype=self.float_type) - loglambda) \
                               - (tf.exp(loglambda) / 2.) * (tf.reduce_sum(w1 ** 2) + tf.reduce_sum(w2 ** 2)
                                                             + tf.reduce_sum(b1 ** 2) + b2 ** 2)  # () - scalar
            log_likelihood_w = log_likelihood_w * current_batchsize / self.num_samples  # to cancel the scaling effect
            # when the final output of this function is used

            d_w1, d_b1, d_w2, d_b2, d_loggamma = tf.gradients(log_likelihood_data, [w1, b1, w2, b2, loggamma])
            d_loglambda = tf.gradients(log_likelihood_w, [loglambda])[0]
            grad_all = self._pack_weights(d_w1, d_b1, d_w2, d_b2, d_loggamma, d_loglambda, True)  # (particle_dim,)

        return grad_all

    def _f_log_lik(self, loggamma, pred_y, y):
        return np.sum((loggamma - np.log(2 * np.pi)) / 2. - np.exp(loggamma) / 2 * (pred_y - y) ** 2)

    '''
        The update procedure of SVGD-VR
    '''

    def _svgdvr_update(self, X_train, y_train, X_dev, y_dev, variance_reduction=False, debug=False,
                       eval_freq=0, X_test=None, y_test=None, eval_est_stddev=False):
        rmse_list = []
        loglik_list = []
        est_stddev_list = []
        fudge_factor = 1e-6
        historical_grad = 0
        if not variance_reduction:
            for it in np.arange(self.n_svgd_updates):
                if eval_est_stddev and eval_freq > 0 and (it + 1) % eval_freq == 0:
                    est_stddev_list.append(self._eval_est_grad_stddev(X_train, y_train))

                # sub-sampling
                idx_batch = np.arange(it * self.batch_size, (it + 1) * self.batch_size) % self.num_samples
                feed_data = {self.x: X_train[idx_batch, :], self.y: y_train[idx_batch]}
                grad_particles, particles_val, logpost = self.tf_session.run([self.grad_log_posterior, self.particles,
                                                                              self.log_posterior], feed_dict=feed_data)
                if debug and (it + 1) % 100 == 0:
                    print('Iter %d: log posterior %.6f.' % (it + 1, np.mean(logpost)))
                # Calculating the kernel matrix
                Kxy, dxKxy = self._svgd_kernel(particles_val, h=-1)
                grad_particles = (np.einsum('ij,jk->ik', Kxy, grad_particles) + dxKxy) / self.num_particles
                # \Phi(x) - (num_particles, particle_dim)

                # AdaGrad with momentum
                if it == 0:
                    historical_grad = grad_particles ** 2
                else:
                    historical_grad = self.auto_corr * historical_grad + (1 - self.auto_corr) * grad_particles ** 2
                adj_grad = np.divide(grad_particles, fudge_factor + np.sqrt(historical_grad))
                particles_val = particles_val + self.svgd_stepsize * adj_grad
                self.tf_session.run(self.update_particles, feed_dict={self.new_particles: particles_val})

                # Test the interim model
                if eval_freq > 0 and X_test is not None and (it + 1) % eval_freq == 0:
                    current_particles_val = self.tf_session.run(self.particles)
                    self._tweak_gamma(X_dev, y_dev)
                    rmse, loglik = self.evaluation(X_test, y_test)
                    rmse_list.append(rmse)
                    loglik_list.append(loglik)
                    self.tf_session.run(self.update_particles, feed_dict={self.new_particles: current_particles_val})
        else:
            for l in np.arange(self.n_svgd_updates):
                particles_hat_val = self.tf_session.run(self.particles, feed_dict={})  # (num_particles, particle_dim)
                feed_data = {self.x: X_train, self.y: y_train, self.particles_hat: particles_hat_val}
                mu = self.tf_session.run(self.grad_log_likelihood, feed_dict=feed_data)  # (num_particles, particle_dim)

                for t in np.arange(self.n_vr_updates):
                    if eval_est_stddev and eval_freq > 0 and (l * self.n_vr_updates + t + 1) % eval_freq == 0:
                        est_stddev_list.append(
                            self._eval_est_grad_stddev(X_train, y_train, True, mu, particles_hat_val))

                    idx_batch = np.arange((l * self.n_vr_updates + t) * self.batch_size,
                                          (l * self.n_vr_updates + t + 1) * self.batch_size) % self.num_samples
                    feed_data = {self.x: X_train[idx_batch, :], self.y: y_train[idx_batch],
                                 self.particles_hat: particles_hat_val}
                    grad_posterior, grad_likelihood, particles_val, logpost = \
                        self.tf_session.run([self.grad_log_posterior, self.grad_log_likelihood, self.particles,
                                             self.log_posterior], feed_dict=feed_data)
                    if debug and (l * self.n_vr_updates + t + 1) % 100 == 0:
                        print('Iter %d: log posterior %.6f.' % (l + 1, np.mean(logpost)))
                    rho = grad_posterior - grad_likelihood * self.num_samples / self.batch_size + mu
                    Kxy, dxKxy = self._svgd_kernel(particles_val, h=-1)
                    grad_particles = (np.einsum('ij,jk->ik', Kxy, rho) + dxKxy) / self.num_particles
                    # \Phi(x) - (num_particles, particle_dim)

                    # AdaGrad with momentum
                    if t == 0:
                        historical_grad = grad_particles ** 2
                    else:
                        historical_grad = self.auto_corr * historical_grad + (1 - self.auto_corr) * grad_particles ** 2
                    adj_grad = np.divide(grad_particles, fudge_factor + np.sqrt(historical_grad))
                    particles_val = particles_val + self.svgd_stepsize * adj_grad
                    self.tf_session.run(self.update_particles, feed_dict={self.new_particles: particles_val})

                    # Test the interim model
                    if eval_freq > 0 and X_test is not None and (l * self.n_vr_updates + t + 1) % eval_freq == 0:
                        current_particles_val = self.tf_session.run(self.particles)
                        self._tweak_gamma(X_dev, y_dev)
                        rmse, loglik = self.evaluation(X_test, y_test)
                        rmse_list.append(rmse)
                        loglik_list.append(loglik)
                        self.tf_session.run(self.update_particles,
                                            feed_dict={self.new_particles: current_particles_val})

        if eval_freq > 0 and X_test is not None:
            rmse_list = np.stack(rmse_list, axis=0)
            loglik_list = np.stack(loglik_list, axis=0)
        if eval_est_stddev and eval_freq > 0:
            est_stddev_list = np.stack(est_stddev_list, axis=0)  # (..., particle_dim)
        return rmse_list, loglik_list, est_stddev_list

    '''
        Evaluate the standard deviation of the estimator of the full-batch gradient
    '''

    def _eval_est_grad_stddev(self, X_train, y_train, variance_reduction=False, mu=None, particles_hat=None):
        true_grad = self.tf_session.run(self.grad_log_posterior, feed_dict={self.x: X_train, self.y: y_train})
        # (num_particles, particle_dim)
        num_batches = int(np.ceil(self.num_samples / self.batch_size))
        all_est_grad = []
        if not variance_reduction:
            for it in np.arange(num_batches):
                idx_batch = np.arange(it * self.batch_size, (it + 1) * self.batch_size) % self.num_samples
                est_grad = self.tf_session.run(self.grad_log_posterior,
                                               feed_dict={self.x: X_train[idx_batch, :], self.y: y_train[idx_batch]})
                # (num_particles, particle_dim)
                all_est_grad.append(est_grad)
        else:
            for it in np.arange(num_batches):
                idx_batch = np.arange(it * self.batch_size, (it + 1) * self.batch_size) % self.num_samples
                feed_data = {self.x: X_train[idx_batch, :], self.y: y_train[idx_batch],
                             self.particles_hat: particles_hat}
                grad_posterior, grad_likelihood = \
                    self.tf_session.run([self.grad_log_posterior, self.grad_log_likelihood], feed_dict=feed_data)
                est_grad = grad_posterior - grad_likelihood * self.num_samples / self.batch_size + mu
                # (num_particles, particle_dim)
                all_est_grad.append(est_grad)
        all_est_grad = np.stack(all_est_grad, axis=0)  # (num_batches, num_particles, particle_dim)
        est_grad_stddev = np.mean(np.sqrt(np.mean((all_est_grad - true_grad) ** 2, axis=0)), axis=0)  # (particle_dim,)
        return est_grad_stddev

    def _tweak_gamma(self, X_dev, y_dev):
        X_dev = self._preprocess_data(X_dev)
        particles_val, pred_y_dev = self.tf_session.run([self.particles, self.predicted_y],
                                                        feed_dict={self.x: X_dev,
                                                                   self.y: y_dev})  # (num_particles, num_data)
        for it_particle in np.arange(self.num_particles):
            lik1 = self._f_log_lik(particles_val[it_particle, -2], pred_y_dev[it_particle, :], y_dev)
            loggamma = -np.log(np.mean((pred_y_dev[it_particle, :] - y_dev) ** 2))
            lik2 = self._f_log_lik(loggamma, pred_y_dev[it_particle, :], y_dev)
            if lik2 > lik1:
                particles_val[it_particle, -2] = loggamma
        self.tf_session.run(self.update_particles, feed_dict={self.new_particles: particles_val})

    '''
        Train the model with given training set
    '''

    def fit(self, X_train, y_train, X_dev, y_dev, variance_reduction=False, debug=False,
            eval_freq=0, X_test=None, y_test=None, eval_est_stddev=False):
        X_train, y_train = self._preprocess_data(X_train, y_train)
        rmse_list, loglik_list, est_stddev_list = self._svgdvr_update(X_train, y_train, X_dev, y_dev,
                                                                      variance_reduction=variance_reduction,
                                                                      debug=debug,
                                                                      eval_freq=eval_freq, X_test=X_test, y_test=y_test,
                                                                      eval_est_stddev=eval_est_stddev)

        # Model development
        self._tweak_gamma(X_dev, y_dev)

        return rmse_list, loglik_list, est_stddev_list

    '''
        Evaluating testing rmse and log-likelihood, which is the same as in PBP 
        Input:
            -- X_test: unnormalized testing feature set
            -- y_test: unnormalized testing labels
    '''

    def evaluation(self, X_test, y_test):
        X_test = self._preprocess_data(X_test)
        pred_y_test = np.zeros((self.num_particles, len(y_test)))
        prob = np.zeros((self.num_particles, len(y_test)))

        for it in np.arange(0, len(y_test), self.batch_size):
            idx_batch = np.arange(it, min(it + self.batch_size, len(y_test)))
            feed_data = {self.x: X_test[idx_batch, :], self.y: y_test[idx_batch]}
            pred_y_test[:, idx_batch], prob[:, idx_batch] = self.tf_session.run([self.predicted_y, self.probability],
                                                                                feed_dict=feed_data)

        pred = np.mean(pred_y_test, axis=0)

        svgd_rmse = np.sqrt(np.mean((pred - y_test) ** 2))
        svgd_ll = np.mean(np.log(np.mean(prob, axis=0)))

        return svgd_rmse, svgd_ll

    def _preprocess_data(self, X, y=None):
        '''
        Preprocess the data.
            - 'option': the preprocessing option. If option = 0, no preprocessing. If option = 1, normalise the data.
                If option = 2, standardise the data.
            - 'training': a boolean flag. If True, the input dataset is the training set, so we will compute and save
                the required statistics (i.e. min, max, mean, standard deviation). If False, the input dataset is the test
                set, so we use the pre-computed statistics. By default, 'training' is False.
            Dimension of X: (num_samples, data_dim)
        '''
        if self.preprocess_option == 0:
            print('No data preprocessing.')
        elif self.preprocess_option == 1:  # normalise the data
            # print('Normalise the data.')
            X = np.divide(X - np.broadcast_to(self.min_X_train, X.shape),
                          np.broadcast_to(self.max_X_train - self.min_X_train, X.shape))
            if y is not None:
                y = (y - self.min_y_train) / (self.max_y_train - self.min_y_train)
        elif self.preprocess_option == 2:  # standardise the data
            # print('Standardise the data.')
            X = np.divide(X - np.broadcast_to(self.mean_X_train, X.shape), np.broadcast_to(self.std_X_train, X.shape))
            if y is not None:
                y = (y - self.mean_y_train) / self.std_y_train
        else:
            print('Unsupported preprocessing option: %d.' % (self.preprocess_option))

        if y is not None:
            return X, y
        return X

    def reset_all_params(self, X_train, y_train):
        self.init_particles = None
        X_train, y_train = self._preprocess_data(X_train, y_train)
        self.init_particles = self._init_all_particles(X_train, y_train)
        self.tf_session.run(self.update_particles, feed_dict={self.new_particles: self.init_particles})


if __name__ == '__main__':
    print('TensorFlow version ', tf.__version__)

    np.random.seed(1)
    ''' load data file '''
    data_dir = './data'

    dataset = 'energy.csv'
    data = np.loadtxt('%s/%s' % (data_dir, dataset), delimiter=',')

    print('Dataset:', dataset)
    # The last column is the label and the other columns are features
    X_input = data[:, :-1]
    y_input = data[:, -1]

    ''' Build the training and testing data set'''
    train_ratio = 0.9  # We create the train and test sets with 90% and 10% of the data
    perm_idx = np.random.permutation(X_input.shape[0])
    num_train = int(np.floor(X_input.shape[0] * train_ratio))
    idx_train = perm_idx[:num_train]
    idx_test = perm_idx[num_train:]

    X_train, y_train = X_input[idx_train, :], y_input[idx_train]
    X_test, y_test = X_input[idx_test, :], y_input[idx_test]

    dev_ratio = 0.1
    num_dev = min(int(np.floor(num_train * dev_ratio)), 500)
    X_dev, y_dev = X_train[-num_dev:, :], y_train[-num_dev:]
    X_train, y_train = X_train[:-num_dev, :], y_train[:-num_dev]

    preprocess_type = 2

    batch_size, num_hidden, n_svgd_updates = 128, 64, 2048  # n_svgd_updates is a trade-off between running time and performance
    num_particles = 32
    eval_freq = n_svgd_updates // 256
    model = svgdvr_bayesnn(X_train, y_train, batch_size=batch_size, num_hidden=num_hidden, num_particles=num_particles,
                           n_svgd_updates=n_svgd_updates, preprocess_option=preprocess_type, float_type=64)

    batch_size_vr = batch_size
    n_vr_updates = 8
    n_svgd_updates_vr = n_svgd_updates // n_vr_updates
    model_vr = svgdvr_bayesnn(X_train, y_train, batch_size=batch_size_vr, num_hidden=num_hidden,
                              num_particles=num_particles,
                              n_svgd_updates=n_svgd_updates_vr, preprocess_option=preprocess_type,
                              n_vr_updates=n_vr_updates, float_type=64)

    perm_idx = np.random.permutation(X_train.shape[0])
    X_train, y_train = X_train[perm_idx, :], y_train[perm_idx]
    perm_idx = np.random.permutation(X_dev.shape[0])
    X_dev, y_dev = X_dev[perm_idx, :], y_dev[perm_idx]

    ''' Training Bayesian neural network with SVGD-VR '''
    # SVGD
    print('Train Bayesian neural network with SVGD...')
    model.reset_all_params(X_train, y_train)
    svgd_rmse, svgd_ll, svgd_est_stddev = model.fit(X_train, y_train, X_dev, y_dev, debug=False, eval_freq=eval_freq,
                                                    X_test=X_test, y_test=y_test, eval_est_stddev=True)

    # SVGD-VR
    print('Train Bayesian neural network with SVGD-VR...')
    model_vr.reset_all_params(X_train, y_train)
    svgdvr_rmse, svgdvr_ll, svgdvr_est_stddev = model_vr.fit(X_train, y_train, X_dev, y_dev, variance_reduction=True,
                                                             debug=False, eval_freq=eval_freq, X_test=X_test,
                                                             y_test=y_test, eval_est_stddev=True)

    np.save('./results/%s_svgd_rmse' % (dataset), svgd_rmse)
    np.save('./results/%s_svgd_ll' % (dataset), svgd_ll)
    np.save('./results/%s_svgd_est_stddev' % (dataset), svgd_est_stddev)
    np.save('./results/%s_svgdvr_rmse' % (dataset), svgdvr_rmse)
    np.save('./results/%s_svgdvr_ll' % (dataset), svgdvr_ll)
    np.save('./results/%s_svgdvr_est_stddev' % (dataset), svgdvr_est_stddev)

    svgd_norm = np.linalg.norm(svgd_est_stddev, axis=1)
    svgdvr_norm = np.linalg.norm(svgdvr_est_stddev, axis=1)
    ratio = svgdvr_norm / svgd_norm * 100

    print('SVGD: [RMSE, Log Likelihood] = %.4f, %.4f' % (svgd_rmse[-1], svgd_ll[-1]))
    print('SVGD-VR: [RMSE, Log Likelihood] = %.4f, %.4f' % (svgdvr_rmse[-1], svgdvr_ll[-1]))
    print('Ratio of norm of standard deviation of gradient estimators (SVGD-VR / SVGD):')
    print('Min: %.2f%%. Median: %.2f%%. Max: %.2f%%' % (np.min(ratio), np.median(ratio), np.max(ratio)))
