import tensorflow as tf
from nn import NN
import numpy as np
import os
from six.moves import cPickle as pkl
import time


def variable_summaries(var):
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
          stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)


def run_train(model, steps, lr):
    data_dir = os.path.join(os.getcwd(), 'MNIST_data')
    with open(data_dir + '/train_images_16x16.pkl', 'rb') as f:
        train_images = pkl.load(f)
    with open(data_dir + '/test_images_16x16.pkl', 'rb') as f:
        test_images = pkl.load(f)
    with open(data_dir + '/train_labels.pkl', 'rb') as f:
        train_labels = pkl.load(f)
    with open(data_dir + '/test_labels.pkl', 'rb') as f:
        test_labels = pkl.load(f)

    batch_size = 100

    x = tf.placeholder(tf.float32, [None, train_images.shape[1]])
    y_ = tf.placeholder(tf.float32, [None, train_labels.shape[1]])

    loss = model.mse_loss(y_, x)
    step = tf.train.AdamOptimizer(lr).minimize(loss)

    sess = tf.Session()

    '''summary stuff'''
    summaries_dir = os.path.join(os.getcwd(), 'summary')
    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter(summaries_dir + '/train',
                                         sess.graph)
    test_writer = tf.summary.FileWriter(summaries_dir + '/test', sess.graph)

    sess.run(tf.global_variables_initializer())

    print('training..')
    for i in range(steps):
        xs, ys = train_images[i:batch_size*i], train_labels[i:batch_size*i]
        _, summary = sess.run([step, merged], {x: xs, y_: ys})
    print('done training')

    correct_prediction = tf.equal(tf.argmax(model.forward(x), 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    acc, summary = sess.run([accuracy, merged], {x: test_images, y_: test_labels})

    print('test accuracy: ' + str(acc))

    return sess

def train_standard_model():
    '''make and train model'''
    h_dim = 20
    y_dim = 10
    x_dim = 256
    layers = [x_dim] + 3 * [h_dim] + [y_dim]
    model = NN(layers)

    sess = run_train(model, steps=500, lr=0.001)

    return sess, model

if __name__ == '__main__':
    sess, model = train_standard_model()















