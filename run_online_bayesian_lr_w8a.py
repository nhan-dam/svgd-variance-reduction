from __future__ import print_function, division, absolute_import
import numpy as np
from sklearn.datasets import load_svmlight_file
import matplotlib.pyplot as plt

from svgd_vr import SVGD_VR
from bayesian_logistic_regression import BayesianLR_VR

'''
    Online Learning Bayesian Logistic Regression on 'w8a' dataset (downloaded from LIBSVM repository)
'''

print("Online learning with Bayesian logistic regression on 'w8a' dataset")
data_dir = './data'
dataset = 'w8a'
X_train, y_train = load_svmlight_file('%s/%s.txt' % (data_dir, dataset))
X_test, y_test = load_svmlight_file('%s/%s.t.txt' % (data_dir, dataset))
X_train = X_train.toarray()
X_test = X_test.toarray()
X_input = np.vstack((X_train, X_test))
y_input = np.hstack((y_train, y_test))

n_input, n_train, n_test = X_input.shape[0], X_train.shape[0], X_test.shape[0]
X_input = np.hstack([X_input, np.ones([n_input, 1])])
d = X_input.shape[1]
D = d + 1

print('Train: %d. Test: %d. Full: %d, %d, %d. Dimensions: %d (including bias)' % (X_train.shape[0], X_test.shape[0],
                                                                                  n_input, X_input.shape[0],
                                                                                  len(y_input), d))

# initialise the model
a0, b0 = 1, 0.01  # hyper-parameters
batchsize = 1
batchsize_vr = batchsize
n_vr_updates = 128
eval_freq_vr = n_vr_updates
eval_freq = eval_freq_vr * batchsize_vr // batchsize
model = BayesianLR_VR(X_train, y_train, batchsize, a0, b0)

# initialise the variables
M = 100  # number of particles
theta0 = np.zeros([M, D])
alpha0 = np.random.gamma(a0, b0, M)
for i in range(M):
    theta0[i, :] = np.hstack([np.random.normal(0, np.sqrt(1 / alpha0[i]), d), np.log(alpha0[i])])

# train the model
stepsize = 2 ** (-8)
stepsize_vr = 2 ** (-8)

model = BayesianLR_VR(X_input, y_input, batchsize, a0, b0)
print('Running SVGD')
_, eval_log = SVGD_VR().update(x0=theta0, grad_ln_posterior=model.grad_ln_posterior, num_data=n_input, bandwidth=-1,
                               stepsize=stepsize, alpha=0.9, debug=True,
                               eval_fn=model.predict_in_dataset, eval_freq=eval_freq, online=True)
print('Running SVGD-VR')
_, eval_log_vr = SVGD_VR().update(x0=theta0, grad_ln_posterior=model.grad_ln_posterior,
                                  grad_ln_likelihood=model.grad_ln_likelihood, num_data=n_input,
                                  bandwidth=-1, n_vr_updates=n_vr_updates,
                                  batchsize_vr=batchsize_vr, stepsize=stepsize_vr, alpha=0.9, debug=True,
                                  eval_fn=model.predict_in_dataset, eval_freq=eval_freq_vr,
                                  online=True)

# save results to files
eval_log_filename = './results/online_bayesian_lr_eval_%s_step%f_svgd' % (dataset, stepsize)
eval_log_vr_filename = './results/online_bayesian_lr_eval_%s_step%f_svgdvr' % (dataset, stepsize_vr)
np.save(eval_log_filename, eval_log)
np.save(eval_log_vr_filename, eval_log_vr)
print('Online classification results saved to %s and %s.' % (eval_log_filename, eval_log_vr_filename))

# compute some statistics of the results
print('Accuracy of SVGD: %.2f%%.' % (eval_log[-1] * 100))
print('Accuracy of SVGD-VR: %.2f%%.' % (eval_log_vr[-1] * 100))

# plot the results
n_plot_points = len(eval_log)
interval = n_plot_points // 2
xticks_labels = ['1'] + [str((idx + 1) * eval_freq) for idx in np.arange(n_plot_points) if (idx + 1) % interval == 0]

plt.figure(2)
plt.plot(np.arange(n_plot_points), eval_log * 100, 'r', linewidth=0.5)
plt.plot(np.arange(n_plot_points), eval_log_vr * 100, 'b', linewidth=0.5)
plt.xticks([0] + list(range(interval - 1, n_plot_points, interval)), xticks_labels)
plt.xlabel('num of training data points')
plt.ylabel('accuracy (%)')
plt.legend(['SVGD', 'SVGD-VR'])
plt.title('Performance vs Amount of Online Bayesian Logistic Regression with SVGD and SVGD-VR')
plt.show()
