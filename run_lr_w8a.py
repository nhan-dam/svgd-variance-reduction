from __future__ import print_function, division, absolute_import
import numpy as np
from sklearn.datasets import load_svmlight_file
from functools import partial

from logistic_regression import LogisticRegression

'''
    Logistic Regression on 'w8a' dataset (downloaded from LIBSVM repository)
'''

print("Logistic regression on 'w8a' dataset")
data_dir = './data'
dataset = 'w8a'
X_train, y_train = load_svmlight_file('%s/%s.txt' % (data_dir, dataset))
X_test, y_test = load_svmlight_file('%s/%s.t.txt' % (data_dir, dataset))
X_train = X_train.toarray()
X_test = X_test.toarray()

n_train, n_test = X_train.shape[0], X_test.shape[0]
X_train = np.hstack([X_train, np.ones([n_train, 1])])
X_test = np.hstack([X_test, np.ones([n_test, 1])])
d = X_train.shape[1]
D = d + 1

print('Train: %d. Test: %d. Dimensions: %d (including bias)' % (X_train.shape[0], X_test.shape[0], d))

'''
    Logistic regression
'''
# initialise the model
solver = 'svrg-sgd'
batchsize = 128
n_svrg_updates = 128 if solver == 'svrg-sgd' or solver == 'svrg-adagrad' or solver == 'svrg-rmsprop' else 1
n_epochs = 20
n_updates = int(np.ceil(n_epochs * n_train / batchsize / n_svrg_updates))
eval_freq = n_svrg_updates
learning_rate = 2 ** (-5)

# train the model
print('Train Logistic Regression using %s with the optimal learning rate of %f.' % (solver, learning_rate))
model = LogisticRegression(np.random.normal(0, 1, X_train.shape[1]), solver=solver, batchsize=batchsize)
_, eval_log = model.fit(X_train, y_train, n_updates=n_updates, learning_rate=learning_rate,
                        n_svrg_updates=n_svrg_updates, eval_freq=eval_freq,
                        eval_fn=partial(model.predict, X_test, y_test, False, False), debug=False)

eval_log_filename = './results/lr_eval_%s_lr%f' % (dataset, learning_rate)

# save results to files
np.save(eval_log_filename, eval_log)
print('Logistic regression results saved to %s.' % (eval_log_filename))

print('Max accuracy %.2f%% (after %d * %d * %d data points).' %
      (np.max(eval_log) * 100, batchsize, n_svrg_updates, np.argmax(eval_log) + 1))
