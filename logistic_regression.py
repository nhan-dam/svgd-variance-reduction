from __future__ import print_function, division, absolute_import
import numpy as np

'''
    Logistic Regression:
        The observed data D = {X, y} consist of N binary class labels, 
        y_t \in {-1,+1}, and d covariates for each datapoint, X_t \in R^d.
        p(y_t = 1| x_t, w) = 1 / (1 + exp(-w^T x_t))
'''


class LogisticRegression:
    def __init__(self, W, solver='sgd', batchsize=128):
        '''
        Initialise a Logistic Regression model.

            solver: name of the solver. Currently, this function supports 4 solvers: 'sgd', 'adagrad', 'rmsprop',
                'svrg-sgd', 'svrg-adagrad' and 'svrg-rmsprop'. By default, solver = 'sgd'.
        '''
        self.W = np.copy(W)
        self.solver = solver
        self.batchsize = batchsize

    def _sigmoid(self, X, W):
        '''
        Compute the sigmoid function given weights and inputs.

            X: N x D, where N is the number of data points and D is the dimension of each data point.

            W: (D,) array.
        '''
        coeff = -1.0 * np.matmul(X, W)
        return np.divide(np.ones(X.shape[0]), 1 + np.exp(coeff))

    def _fn_J(self, w, x, y, reg):
        loss_term = np.sum(-np.multiply(np.dot(x, w), y) + np.log(1 + np.exp(np.dot(x, w))))
        reg_term = reg / 2.0 * np.sum(w ** 2)
        return 1.0 / len(y) * loss_term + reg_term

    def fit(self, X, y, n_updates=128, learning_rate=0.01, regularisation_factor=0.1, n_svrg_updates=128,
            online=False, eval_freq=0, eval_fn=None, momentum_factor=0.9, decay_lr=0, debug=False):
        '''
        Train the model.

            n_updates: number of training iterations. By default, n_updates = 100.

            learning_rate: the learning rate. By default, learning_rate = 0.01.

            regularisation_factor: regularisation parameter used in L2 penalty.

            n_svrg_updates: number of training iterations in the inner loop of SVRG solver. By default,
                n_svrg_updates = 100.

            online: boolean flag for online learning setting. If online = True, we follow online learning to update
                particles. That means for each training data point, we predict its label before using it for training.
                We compute the accumulated accuracy of prediction. By default, online = False.

            eval_freq: the frequency that the performance of the model with current parameters is evaluated.
                If online = True, eval_freq is automatically set to -1, that means the evaluation is executed before
                training with each data point. Otherwise, if eval_freq <= 0, no evaluation will be executed during
                training and if eval_freq > 0 the evaluation will be executed after every eval_freq data points trained.

            eval_fn: the function to evaluate the performance of the model with the current parameters.
                By default, eval_fn = None.

            momentum_factor: momentum parameter used in RMSProp. By default, momentum_factor = 0.9.

            decay_lr: the hyperparameters that control the decay of learning rate. By default, decay_stepsize = 0,
                that means there is no decay.

            debug: boolean flag to determine the mode of this function. In debug mode, the function will print more
                information to the standard output during training. By default, debug = False.
        '''
        X_train = np.copy(X)
        y_train = np.copy(y)
        y_train[y_train == -1] = 0  # in this function, we use labels 0 and 1.
        num_data = X_train.shape[0]
        n_svrg_updates = 1 if self.solver != 'svrg-sgd' and self.solver != 'svrg-adagrad' and \
                              self.solver != 'svrg-rmsprop' else n_svrg_updates

        if online:
            batchsize = 1
            eval_freq = min(-eval_freq, -1)
            n_updates = int(np.ceil(num_data / n_svrg_updates))
        else:
            batchsize = min(self.batchsize, X_train.shape[0])
            eval_freq = n_updates * n_svrg_updates + 1 if eval_freq <= 0 else eval_freq

        data_idx_perm = np.random.permutation(num_data)
        if eval_freq < 0:
            loss_log = np.zeros(int(np.ceil(num_data / (-eval_freq))) + 1)
            accumulated_loss = 0
            cnt = 0
        elif eval_freq > 0:
            eval_log = []
        if self.solver == 'sgd':
            print('Train Logistic Regression with SGD solver.')
            for it in np.arange(n_updates):
                if debug and (it + 1) * batchsize % 1000 == 0:
                    print('iter %d' % (it + 1))

                data_idx = data_idx_perm[np.arange(it * batchsize, (it + 1) * batchsize) % num_data]
                x_batch = X_train[data_idx, :]
                y_batch = y_train[data_idx]

                if eval_freq < 0:
                    current_loss = self.predict(np.copy(X[data_idx]), np.copy(y[data_idx]), get_label=False,
                                                get_prob=False)
                    accumulated_loss += (current_loss * len(y_batch))
                    if it % (-eval_freq) == 0 or (it + 1) == num_data:
                        loss_log[cnt] = accumulated_loss * 1.0 / (it + 1)
                        cnt += 1

                grad_J = -np.sum(
                    np.multiply(x_batch, np.broadcast_to(np.vstack(y_batch - self._sigmoid(x_batch, self.W)),
                                                         (batchsize, x_batch.shape[1]))), axis=0) * 1.0 / batchsize \
                         + regularisation_factor * self.W

                # W_prime = self.W + 1.0 / 10000 * np.random.normal(0, 1, len(self.W))
                # numerical_grad_J = (self._fn_J(W_prime, x_batch, y_batch, regularisation_factor) - self._fn_J(self.W, x_batch, y_batch, regularisation_factor)) / (W_prime - self.W)
                # diff = numerical_grad_J - grad_J

                self.W = self.W - learning_rate * np.exp(-decay_lr * it) * grad_J
                # self.W = self.W - learning_rate * np.exp(-decay_lr * it) * numerical_grad_J

                if eval_freq > 0 and (it + 1) % eval_freq == 0:
                    if (it + 1) // eval_freq == 1:
                        eval_log = eval_fn()
                    else:
                        eval_log = np.vstack((eval_log, eval_fn()))
        elif self.solver == 'adagrad':
            print('Train Logistic Regression with AdaGrad solver.')
            fudge_factor = 1e-6
            historical_grad = 0.0
            for it in np.arange(n_updates):
                if debug and (it + 1) * batchsize % 1000 == 0:
                    print('iter %d' % (it + 1))

                data_idx = data_idx_perm[np.arange(it * batchsize, (it + 1) * batchsize) % num_data]
                x_batch = X_train[data_idx, :]
                y_batch = y_train[data_idx]

                if eval_freq < 0:
                    current_loss = self.predict(np.copy(X[data_idx]), np.copy(y[data_idx]), get_label=False,
                                                get_prob=False)
                    accumulated_loss += (current_loss * len(y_batch))
                    if it % (-eval_freq) == 0 or (it + 1) == num_data:
                        loss_log[cnt] = accumulated_loss * 1.0 / (it + 1)
                        cnt += 1

                grad_J = -np.sum(
                    np.multiply(x_batch, np.broadcast_to(np.vstack(y_batch - self._sigmoid(x_batch, self.W)),
                                                         (batchsize, x_batch.shape[1]))), axis=0) * 1.0 / batchsize \
                         + regularisation_factor * self.W

                historical_grad += (grad_J ** 2)
                adj_grad = np.divide(grad_J, fudge_factor + np.sqrt(historical_grad))
                self.W = self.W - learning_rate * np.exp(-decay_lr * it) * adj_grad

                if eval_freq > 0 and (it + 1) % eval_freq == 0:
                    if (it + 1) // eval_freq == 1:
                        eval_log = eval_fn()
                    else:
                        eval_log = np.vstack((eval_log, eval_fn()))
        elif self.solver == 'rmsprop':
            print('Train Logistic Regression with RMSProp solver.')
            fudge_factor = 1e-6
            historical_grad = 0.0
            for it in np.arange(n_updates):
                if debug and (it + 1) * batchsize % 1000 == 0:
                    print('iter %d' % (it + 1))

                data_idx = data_idx_perm[np.arange(it * batchsize, (it + 1) * batchsize) % num_data]
                x_batch = X_train[data_idx, :]
                y_batch = y_train[data_idx]

                if eval_freq < 0:
                    current_loss = self.predict(np.copy(X[data_idx]), np.copy(y[data_idx]), get_label=False,
                                                get_prob=False)
                    accumulated_loss += (current_loss * len(y_batch))
                    if it % (-eval_freq) == 0 or (it + 1) == num_data:
                        loss_log[cnt] = accumulated_loss * 1.0 / (it + 1)
                        cnt += 1

                grad_J = -np.sum(
                    np.multiply(x_batch, np.broadcast_to(np.vstack(y_batch - self._sigmoid(x_batch, self.W)),
                                                         (batchsize, x_batch.shape[1]))), axis=0) * 1.0 / batchsize \
                         + regularisation_factor * self.W

                if it == 0:
                    historical_grad = grad_J ** 2
                else:
                    historical_grad = momentum_factor * historical_grad + (1 - momentum_factor) * (grad_J ** 2)
                adj_grad = np.divide(grad_J, fudge_factor + np.sqrt(historical_grad))
                self.W = self.W - learning_rate * np.exp(-decay_lr * it) * adj_grad

                if eval_freq > 0 and (it + 1) % eval_freq == 0:
                    if (it + 1) // eval_freq == 1:
                        eval_log = eval_fn()
                    else:
                        eval_log = np.vstack((eval_log, eval_fn()))
        elif self.solver == 'svrg-sgd':
            print('Train Logistic Regression with SVRG-SGD solver.')
            for it in np.arange(n_updates):
                if debug and (it + 1) * batchsize % 1000 == 0:
                    print('iter %d' % (it + 1))

                mu = -np.sum(np.multiply(X_train, np.broadcast_to(np.vstack(y_train - self._sigmoid(X_train, self.W)),
                                                                  (len(y_train), X_train.shape[1]))),
                             axis=0) * 1.0 / len(y_train) \
                     + regularisation_factor * self.W
                w_hat = np.copy(self.W)
                for it_svrg in np.arange(n_svrg_updates):
                    data_idx = data_idx_perm[np.arange((it * n_svrg_updates + it_svrg) * batchsize,
                                                       (it * n_svrg_updates + it_svrg + 1) * batchsize) % num_data]
                    x_batch = X_train[data_idx, :]
                    y_batch = y_train[data_idx]

                    if eval_freq < 0:
                        if it * n_svrg_updates + it_svrg >= num_data:
                            break
                        self.W, w_hat = w_hat, self.W
                        current_loss = self.predict(np.copy(X[data_idx]), np.copy(y[data_idx]), get_label=False,
                                                    get_prob=False)
                        accumulated_loss += (current_loss * len(y_batch))
                        if (it * n_svrg_updates + it_svrg) % (-eval_freq) == 0 or (
                                it * n_svrg_updates + it_svrg + 1) == num_data:
                            loss_log[cnt] = accumulated_loss * 1.0 / (it * n_svrg_updates + it_svrg + 1)
                            cnt += 1
                        self.W, w_hat = w_hat, self.W

                    grad_J_hat = -np.sum(
                        np.multiply(x_batch, np.broadcast_to(np.vstack(y_batch - self._sigmoid(x_batch, w_hat)),
                                                             (batchsize, x_batch.shape[1]))), axis=0) * 1.0 / batchsize \
                                 + regularisation_factor * w_hat
                    grad_J = -np.sum(
                        np.multiply(x_batch, np.broadcast_to(np.vstack(y_batch - self._sigmoid(x_batch, self.W)),
                                                             (batchsize, x_batch.shape[1]))), axis=0) * 1.0 / batchsize \
                             + regularisation_factor * self.W
                    w_hat = w_hat - learning_rate * np.exp(-decay_lr * it) * (grad_J_hat - grad_J + mu)

                    if eval_freq > 0 and (it * n_svrg_updates + it_svrg + 1) % eval_freq == 0:
                        self.W, w_hat = w_hat, self.W
                        if (it * n_svrg_updates + it_svrg + 1) // eval_freq == 1:
                            eval_log = eval_fn()
                        else:
                            eval_log = np.vstack((eval_log, eval_fn()))
                        self.W, w_hat = w_hat, self.W

                self.W = np.copy(w_hat)
        elif self.solver == 'svrg-adagrad':
            print('Train Logistic Regression with SVRG-AdaGrad solver.')
            for it in np.arange(n_updates):
                if debug and (it + 1) * batchsize % 1000 == 0:
                    print('iter %d' % (it + 1))

                fudge_factor = 1e-6
                historical_grad = 0.0
                mu = -np.sum(np.multiply(X_train, np.broadcast_to(np.vstack(y_train - self._sigmoid(X_train, self.W)),
                                                                  (len(y_train), X_train.shape[1]))),
                             axis=0) * 1.0 / len(y_train) \
                     + regularisation_factor * self.W
                w_hat = np.copy(self.W)
                for it_svrg in np.arange(n_svrg_updates):
                    data_idx = data_idx_perm[np.arange((it * n_svrg_updates + it_svrg) * batchsize,
                                                       (it * n_svrg_updates + it_svrg + 1) * batchsize) % num_data]
                    x_batch = X_train[data_idx, :]
                    y_batch = y_train[data_idx]

                    if eval_freq < 0:
                        if it * n_svrg_updates + it_svrg >= num_data:
                            break
                        self.W, w_hat = w_hat, self.W
                        current_loss = self.predict(np.copy(X[data_idx]), np.copy(y[data_idx]), get_label=False,
                                                    get_prob=False)
                        accumulated_loss += (current_loss * len(y_batch))
                        if (it * n_svrg_updates + it_svrg) % (-eval_freq) == 0 or (
                                it * n_svrg_updates + it_svrg + 1) == num_data:
                            loss_log[cnt] = accumulated_loss * 1.0 / (it * n_svrg_updates + it_svrg + 1)
                            cnt += 1
                        self.W, w_hat = w_hat, self.W

                    grad_J_hat = -np.sum(
                        np.multiply(x_batch, np.broadcast_to(np.vstack(y_batch - self._sigmoid(x_batch, w_hat)),
                                                             (batchsize, x_batch.shape[1]))), axis=0) * 1.0 / batchsize \
                                 + regularisation_factor * w_hat
                    grad_J = -np.sum(
                        np.multiply(x_batch, np.broadcast_to(np.vstack(y_batch - self._sigmoid(x_batch, self.W)),
                                                             (batchsize, x_batch.shape[1]))), axis=0) * 1.0 / batchsize \
                             + regularisation_factor * self.W
                    grad_J_svrg = grad_J_hat - grad_J + mu
                    historical_grad += (grad_J_svrg ** 2)
                    adj_grad = np.divide(grad_J_svrg, fudge_factor + np.sqrt(historical_grad))
                    w_hat = w_hat - learning_rate * np.exp(-decay_lr * it) * adj_grad

                    if eval_freq > 0 and (it * n_svrg_updates + it_svrg + 1) % eval_freq == 0:
                        self.W, w_hat = w_hat, self.W
                        if (it * n_svrg_updates + it_svrg + 1) // eval_freq == 1:
                            eval_log = eval_fn()
                        else:
                            eval_log = np.vstack((eval_log, eval_fn()))
                        self.W, w_hat = w_hat, self.W

                self.W = np.copy(w_hat)
        elif self.solver == 'svrg-rmsprop':
            print('Train Logistic Regression with SVRG-RMSProp solver.')
            for it in np.arange(n_updates):
                if debug and (it + 1) * batchsize % 1000 == 0:
                    print('iter %d' % (it + 1))

                fudge_factor = 1e-6
                historical_grad = 0.0
                mu = -np.sum(np.multiply(X_train, np.broadcast_to(np.vstack(y_train - self._sigmoid(X_train, self.W)),
                                                                  (len(y_train), X_train.shape[1]))),
                             axis=0) * 1.0 / len(y_train) \
                     + regularisation_factor * self.W
                w_hat = np.copy(self.W)
                for it_svrg in np.arange(n_svrg_updates):
                    data_idx = data_idx_perm[np.arange((it * n_svrg_updates + it_svrg) * batchsize,
                                                       (it * n_svrg_updates + it_svrg + 1) * batchsize) % num_data]
                    x_batch = X_train[data_idx, :]
                    y_batch = y_train[data_idx]

                    if eval_freq < 0:
                        if it * n_svrg_updates + it_svrg >= num_data:
                            break
                        self.W, w_hat = w_hat, self.W
                        current_loss = self.predict(np.copy(X[data_idx]), np.copy(y[data_idx]), get_label=False,
                                                    get_prob=False)
                        accumulated_loss += (current_loss * len(y_batch))
                        if (it * n_svrg_updates + it_svrg) % (-eval_freq) == 0 or (
                                it * n_svrg_updates + it_svrg + 1) == num_data:
                            loss_log[cnt] = accumulated_loss * 1.0 / (it * n_svrg_updates + it_svrg + 1)
                            cnt += 1
                        self.W, w_hat = w_hat, self.W

                    grad_J_hat = -np.sum(
                        np.multiply(x_batch, np.broadcast_to(np.vstack(y_batch - self._sigmoid(x_batch, w_hat)),
                                                             (batchsize, x_batch.shape[1]))), axis=0) * 1.0 / batchsize \
                                 + regularisation_factor * w_hat
                    grad_J = -np.sum(
                        np.multiply(x_batch, np.broadcast_to(np.vstack(y_batch - self._sigmoid(x_batch, self.W)),
                                                             (batchsize, x_batch.shape[1]))), axis=0) * 1.0 / batchsize \
                             + regularisation_factor * self.W
                    grad_J_svrg = grad_J_hat - grad_J + mu
                    if it_svrg == 0:
                        historical_grad = grad_J_svrg ** 2
                    else:
                        historical_grad = momentum_factor * historical_grad + (1 - momentum_factor) * (grad_J_svrg ** 2)
                    adj_grad = np.divide(grad_J_svrg, fudge_factor + np.sqrt(historical_grad))
                    w_hat = w_hat - learning_rate * np.exp(-decay_lr * it) * adj_grad

                    if eval_freq > 0 and (it * n_svrg_updates + it_svrg + 1) % eval_freq == 0:
                        self.W, w_hat = w_hat, self.W
                        if (it * n_svrg_updates + it_svrg + 1) // eval_freq == 1:
                            eval_log = eval_fn()
                        else:
                            eval_log = np.vstack((eval_log, eval_fn()))
                        self.W, w_hat = w_hat, self.W

                self.W = np.copy(w_hat)
        else:
            raise ValueError('The requested solver %s is currently not supported.' % (self.solver))

        if eval_freq < 0:
            return self, loss_log
        if eval_fn is not None:
            return self, eval_log
        return self

    def predict(self, X_test, y_test=None, get_label=True, get_prob=False):
        '''
        Predict the labels given observations.

            x: one or many new observations. The dimensions of the matrix of observations are N x D, where N is the number
                of observations and D is the dimension of each observation.

            y: corresponding labels of the observations x. If we pass in y_test, the return values of this function
                will include the accuracy. By default, y is None.

            get_label: a boolean flag to determine if we return the predicted labels. get_label has higher precedence
                than get_prob. That means if y_test = None and get_label = False, then get_prob is automatically True.
                By default, get_label = True.

            get_prob: a boolean flag to determine if we return the uncertainty of the prediction. By default, get_prob = False.

            Return: (predicted labels, probabilities, accuracy)
        '''
        prob = self._sigmoid(X_test, self.W)
        y_pred = np.ones(len(prob))
        y_pred[prob <= 0.5] = -1

        if y_test is None:
            if not get_label:  # get_prob is automatically True
                return 0.5 + np.abs(prob - 0.5)
            if not get_prob:
                return y_pred
            return y_pred, 0.5 + np.abs(prob - 0.5)
        if not get_label:
            if not get_prob:
                return np.sum(y_pred == y_test) * 1.0 / len(y_test)
            return 0.5 + np.abs(prob - 0.5), np.sum(y_pred == y_test) * 1.0 / len(y_test)
        return y_pred, 0.5 + np.abs(prob - 0.5), np.sum(y_pred == y_test) * 1.0 / len(y_test)
