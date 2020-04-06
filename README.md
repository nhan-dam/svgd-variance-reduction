# Stein Variational Gradient Descent with Variance Reduction (SVGD-VR)

Code to reproduce the results in the paper 'Stein variational gradient descent with variance reduction', IJCNN 2019.

Tested with Python 3.6 and TensorFlow 1.13.

Run the model with the recommended values for hyperparameters:
-------------------------------------
    1. Binary classification on 'w8a' dataset:
        a. SVGD and SVGD-VR: python run_bayesian_lr_w8a.py
        b. SVRG: python run_lr_w8a.py

    2. Online binary classification on 'w8a' dataset:
        a. SVGD and SVGD-VR: python run_online_bayesian_lr_w8a.py
        b. SVRG: python run_online_lr_w8a.py

    3. Regression on 'energy' dataset:
        SVGD and SVGD-VR: python bayesian_nn.py

Citation
--------
    Nhan Dam, Trung Le, Viet Huynh, and Dinh Phung. 'Stein variational gradient descent with variance reduction'. To appear in Proceedings of the International Joint Conference on Neural Networks (IJCNN), pages ?-?, Glasgow, UK, July 2020.

Bibtex
------
```
@InProceedings{dam_etal_ijcnn20_svgdvr,
  author    = {Nhan Dam and Trung Le and Viet Huynh and Dinh Phung},
  title     = {{S}tein variational gradient descent with variance reduction},
  booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN)},           
  pages     = {?--?},
  year      = {2020},
  month     = {July},
  address   = {Glasgow, UK}
}
 ```
