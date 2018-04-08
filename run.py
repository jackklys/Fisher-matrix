import tensorflow as tf
from nn import NN
from train import run_train
import numpy as np
from numpy.linalg import inv, eig
import os
from six.moves import cPickle
import time
from operator import sub
import experiments

if __name__ == '__main__':
    data_dir = os.path.join(os.getcwd(), 'MNIST_data')

    '''make and train model'''
    h_dim = 20
    y_dim = 10
    x_dim = 256
    layers = [x_dim] + 4 * [h_dim] + [y_dim]
    model = NN(layers)

    sess = run_train(model, steps=10, lr=0.001)


    with open(data_dir + '/test_images_16x16.pkl', 'rb') as f:
        test_images = cPickle.load(f)
    num_examples = 10

    experiments.compute_fisher(sess, model, test_images, num_examples)

