from __future__ import print_function, division, absolute_import
import numpy as np
from sklearn.datasets import load_svmlight_file

from logistic_regression import LogisticRegression
import matplotlib.pyplot as plt

'''
    Online Learning Logistic Regression on 'w8a' dataset (downloaded from LIBSVM repository)
'''

print("Logistic regression on 'w8a' dataset")
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

print('Train: %d. Test: %d. Full: %d, %d, %d. Dimensions: %d (including bias)' %
      (X_train.shape[0], X_test.shape[0], n_input, X_input.shape[0], len(y_input), d))

'''
    Logistic regression
'''
# initialise the model
solver = 'svrg-sgd'
batchsize = 1
n_svrg_updates = 128 if solver == 'svrg-sgd' or solver == 'svrg-adagrad' or solver == 'svrg-rmsprop' else 1
n_updates = int(np.ceil(n_input / batchsize / n_svrg_updates))
eval_freq = 128
learning_rate = 2 ** (-8)

# train the model
print('Run Online learning Logistic regression (using %s) with the learning rate of %f.' % (solver, learning_rate))
model = LogisticRegression(np.random.normal(0, 1, X_input.shape[1]), solver=solver, batchsize=batchsize)
_, eval_log = model.fit(X_input, y_input, n_updates=n_updates, learning_rate=learning_rate, eval_freq=eval_freq,
                        n_svrg_updates=n_svrg_updates, online=True, debug=True)

eval_log_filename = './results/online_lr_eval_%s_lr%f' % (dataset, learning_rate)

# save results to files
np.save(eval_log_filename, eval_log)
print('Logistic regression results saved to %s.' % (eval_log_filename))
print('Accuracy: %.2f%%.' % (eval_log[-1] * 100))

# plot the results
n_plot_points = len(eval_log)
interval = n_plot_points // 2
xticks_labels = ['1'] + [str((idx + 1) * eval_freq) for idx in np.arange(n_plot_points) if (idx + 1) % interval == 0]

plt.figure(3)
plt.plot(np.arange(n_plot_points), eval_log * 100, 'b', linewidth=0.5)
plt.xticks([0] + list(range(interval - 1, n_plot_points, interval)), xticks_labels)
plt.xlabel('num of training data points')
plt.ylabel('accuracy (%)')
plt.title('Performance vs Amount of Training Data of Online Learning Logistic Regression using %s' % (solver))
plt.show()
