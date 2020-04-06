from __future__ import print_function, division, absolute_import
import numpy as np
from sklearn.datasets import load_svmlight_file
from functools import partial
import matplotlib.pyplot as plt

from svgd_vr import SVGD_VR
from bayesian_logistic_regression import BayesianLR_VR

'''
    Bayesian Logistic Regression on 'w8a' dataset (downloaded from LIBSVM repository)
'''

print("Bayesian logistic regression on 'w8a' dataset")
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

# initialise the model
a0, b0 = 1, 0.01  # hyper-parameters
batchsize = 128
batchsize_vr = batchsize
n_vr_updates = 128
eval_freq_vr = n_vr_updates
eval_freq = eval_freq_vr * batchsize_vr // batchsize
n_epochs = 20
n_svgd_updates_vr = int(np.ceil(n_epochs * X_train.shape[0] / (batchsize_vr * n_vr_updates)))
n_svgd_updates = int(np.ceil(n_svgd_updates_vr * n_vr_updates * batchsize_vr / batchsize))
model = BayesianLR_VR(X_train, y_train, batchsize, a0, b0)

# initialise the variables
M = 100  # number of particles
theta0 = np.zeros([M, D])
alpha0 = np.random.gamma(a0, b0, M)
for i in range(M):
    theta0[i, :] = np.hstack([np.random.normal(0, np.sqrt(1 / alpha0[i]), d), np.log(alpha0[i])])

# train the model
stepsize = 2 ** (-5)
stepsize_vr = 2 ** (-5)

print('\nTrain SVGD and SVGD-VR with the optimal step size.')
print('Running SVGD')
theta, eval_log, grad_var_log = SVGD_VR().update(x0=theta0, grad_ln_posterior=model.grad_ln_posterior, bandwidth=-1,
                                                 n_svgd_updates=n_svgd_updates, stepsize=stepsize, alpha=0.9,
                                                 debug=True,
                                                 eval_fn=partial(model.evaluation, X_test=X_test, y_test=y_test),
                                                 eval_freq=eval_freq,
                                                 eval_est_grad_var=model.eval_est_grad_var)
print('Running SVGD-VR')
theta_vr, eval_log_vr, grad_var_log_vr = SVGD_VR().update(x0=theta0, grad_ln_posterior=model.grad_ln_posterior,
                                                          grad_ln_likelihood=model.grad_ln_likelihood,
                                                          num_data=X_train.shape[0],
                                                          bandwidth=-1, n_svgd_updates=n_svgd_updates_vr,
                                                          n_vr_updates=n_vr_updates,
                                                          batchsize_vr=batchsize_vr, stepsize=stepsize_vr, alpha=0.9,
                                                          debug=True,
                                                          eval_fn=partial(model.evaluation, X_test=X_test,
                                                                          y_test=y_test),
                                                          eval_freq=eval_freq_vr,
                                                          eval_est_grad_var=model.eval_est_grad_var)

eval_log_filename = './results/bayesian_lr_eval_%s_step%f_svgd' % (dataset, stepsize)
eval_log_vr_filename = './results/bayesian_lr_eval_%s_step%f_svgdvr' % (dataset, stepsize_vr)
grad_var_log_filename = './results/bayesian_lr_gradvar_%s_step%f_svgd' % (dataset, stepsize)
grad_var_log_vr_filename = './results/bayesian_lr_gradvar_%s_step%f_svgdvr' % (dataset, stepsize_vr)

# save results to files
np.save(eval_log_filename, eval_log)
np.save(eval_log_vr_filename, eval_log_vr)
np.save(grad_var_log_filename, grad_var_log)
np.save(grad_var_log_vr_filename, grad_var_log_vr)
print('Classification results saved to %s and %s.' % (eval_log_filename, eval_log_vr_filename))
print('Standard deviation of estimators saved to %s and %s.' % (grad_var_log_filename, grad_var_log_vr_filename))

# compute some statistics of the results
print('Max and min performance of SVGD: %.2f%% (after %d * %d * %d data points), %.2f%%.'
      % (
      np.max(eval_log[:, 0]) * 100, batchsize, eval_freq, np.argmax(eval_log[:, 0]) + 1, np.min(eval_log[:, 0]) * 100))
print('Max and min performance of SVGD-VR: %.2f%% (after %d * %d * %d data points), %.2f%%.'
      % (np.max(eval_log_vr[:, 0]) * 100, batchsize_vr, eval_freq_vr, np.argmax(eval_log_vr[:, 0]) + 1,
         np.min(eval_log_vr[:, 0]) * 100))

chosen_idx = 59  # best iteration chosen from cross-validation
print('After %d * %d * %d data points, accuracy of SVGD is %.2f%%.' %
      (batchsize, eval_freq, chosen_idx + 1, eval_log[chosen_idx, 0] * 100))
chosen_idx = 13  # best iteration chosen from cross-validation
print('After %d * %d * %d data points, accuracy of SVGD-VR is %.2f%%.' %
      (batchsize, eval_freq, chosen_idx + 1, eval_log_vr[chosen_idx, 0] * 100))

# plot the ratio
svgd_norm = np.linalg.norm(grad_var_log, axis=1)
svgdvr_norm = np.linalg.norm(grad_var_log_vr, axis=1)
ratio = svgdvr_norm / svgd_norm * 100

plt.figure(1)
plt.plot(np.arange(len(ratio)), ratio)
plt.xlabel('epochs')
plt.ylabel('%')
plt.xticks([0, len(ratio) - 1], ['0', str(n_epochs)])
plt.title('Ratio of Norm of Std Dev of Gradient Estimators in SVGD-VR and SVGD')
plt.show()
