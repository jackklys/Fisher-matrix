import tensorflow as tf
import numpy as np
from operator import truediv, add


class NN():
    def __init__(self, layers):
        self.layers = layers
        self.W = [tf.Variable(tf.truncated_normal([layers[i], layers[i + 1]], stddev=0.1)) for i in
                  range(len(layers) - 1)]
        self.b = [tf.Variable(tf.constant(0.1, shape=[layers[i + 1]])) for i in
                  range(len(layers) - 1)]

    def forward(self, x, with_activations=False):
        y = [x]
        s = []
        for W, b in zip(self.W[:-1], self.b[:-1]):
            s.append(tf.add(tf.matmul(y[-1],  W), b, name='preactivation'))
            y.append(tf.tanh(s[-1]))
        z = tf.add(tf.matmul(y[-1],  self.W[-1]), self.b[-1], name='preactivation')
        s.append(z)
        y.append(s[-1])

        for yy, ss, i in zip(y[1:], s, range(len(s))):
            tf.summary.histogram('y' + str(i), yy)
            tf.summary.histogram('s' + str(i), ss)

        if with_activations==True:
            return y, s
        else:
            return y[-1]

    def cross_entropy_loss(self, y_, x):
        cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(tf.nn.softmax(self.forward(x))), axis=1))
        return cross_entropy

    def mse_loss(self, y_, x):
        mse = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(y_, self.forward(x)), axis=1))
        return mse

    def exact_fisher(self, sess, test_images, block_diag=False):
        n = test_images.shape[0]
        x_feed = tf.placeholder(tf.float32, [None, self.layers[0]])
        y_feed = tf.placeholder(tf.float32, [None, self.layers[-1]])
        grads_ys, grads_yw, grads_l = self.loss_grads(x_feed, y_feed, block_diag)
        L = [tf.Variable( np.zeros(g.get_shape().as_list(), dtype=np.float32)) for g in grads_yw]

        A = []
        for l, g in zip(L, grads_yw):
            A.append(tf.assign(l, l + g))
        sess.run(tf.variables_initializer(L))

        for i in range(n):
            sess.run(A, feed_dict={x_feed: test_images[i].reshape(1, -1)})
        L = list(map(truediv, sess.run(L), [n] * len(L)))
        if block_diag:
            return L
        else:
            return L[0]

    def exact_fisher_with_samples(self, sess, test_images, num_samples, block_diag=False):
        n = test_images.shape[0] * num_samples
        x_feed = tf.placeholder(tf.float32, [None, self.layers[0]])
        y_feed = tf.placeholder(tf.float32, [None, self.layers[-1]])
        grads_ys, grads_yw, grads_l = self.loss_grads(x_feed, y_feed, block_diag)
        L = [tf.Variable(np.zeros(g.get_shape().as_list(), dtype=np.float32)) for g in grads_l]

        mu, s = self.forward(x_feed, with_activations=True)
        mu = tf.tile(mu[-1], [num_samples, 1])
        y = sess.run(tf.contrib.distributions.Normal(mu, 1.).sample(), {x_feed: test_images})
        images = np.tile(test_images, [num_samples, 1])

        A = []
        for l, g in zip(L, grads_l):
            A.append(tf.assign(l, l + g))
        sess.run(tf.variables_initializer(L))

        for i in range(n):
                sess.run(A, feed_dict={x_feed: images[i].reshape(1, -1), y_feed: y[i].reshape(1, -1)})
        L =  list(map(truediv, sess.run(L), [n] * len(L)))
        return L

    def a_blocks(self, x):
        y, _ = self.forward(x, with_activations=True)
        # A = [tf.matmul(tf.transpose(a), a) / tf.cast(tf.shape(x)[0], tf.float32) for a in y[:-1]]
        l = y[:-1]
        A = []
        for i in range(len(l)):
            C = []
            for j in range(len(l)):
                    C.append(tf.matmul(tf.transpose(l[i]), l[j]) / tf.cast(tf.shape(x)[0], tf.float32))
            A.append(C)
        return A

    def g_blocks_exact(self, sess, grads, x_feed, test_images):
        n = test_images.shape[0]
        l = grads
        A = []
        L = []
        block_sizes = self.layers[1:]
        for i in range(len(l)):
            B = []
            C = []
            for j in range(len(l)):
                B.append(tf.Variable(np.zeros((block_sizes[i], block_sizes[j]), dtype=np.float32)))
                C.append(tf.assign(B[j], B[j] + (1. / n) * tf.matmul(l[i], tf.transpose(l[j]))))
            L.append(B)
            sess.run(tf.variables_initializer(B))
            A.append(C)

        for i in range(n):
            sess.run(A, feed_dict={x_feed: test_images[i].reshape(1, -1)})
        return L

    def a_covariance(self, sess, test_images, blocks=False):
        x_feed = tf.placeholder(tf.float32, [None, self.layers[0]])
        Ab = self.a_blocks(x_feed)
        Ab = sess.run(Ab, feed_dict={x_feed: test_images})

        y, _ = self.forward(x_feed, with_activations=True)
        l = sess.run([tf.reduce_mean(a, axis=0, keep_dims=True) for a in y[:-1]], feed_dict={x_feed: test_images})
        A = []
        for i in range(len(l)):
            C = []
            for j in range(len(l)):
                    C.append(np.transpose(l[i]).dot(l[j]))
            A.append(C)
        if blocks:
            return A
        Ab = np.asarray(np.bmat(Ab))
        A = np.asarray(np.bmat(A))
        cov =  Ab - A
        return cov, Ab, A

    def g_covariance(self, sess, test_images, block_diag=False):
        x_feed = tf.placeholder(tf.float32, [None, self.layers[0]])
        y_feed = tf.placeholder(tf.float32, [None, self.layers[-1]])
        grads_ys, grads_yw, grads_l = self.loss_grads(x_feed, y_feed, block_diag)
        G = self.g_blocks_exact(sess, grads_ys, x_feed, test_images)
        G = sess.run(G, feed_dict={x_feed: test_images})

        n = test_images.shape[0]
        l = sess.run(grads_ys, feed_dict={x_feed: test_images})
        l = [np.mean(a.reshape(-1, int(a.shape[0]/n), int(a.shape[1])), axis=0) for a in l]

        A = []
        for i in range(len(l)):
            C = []
            for j in range(len(l)):
                C.append(l[i].dot(np.transpose(l[j])))
            A.append(C)
        G = np.asarray(np.bmat(G))
        A = np.asarray(np.bmat(A))
        cov =  G - A
        return cov, G, A


    def kfac_fisher(self, sess, test_images, block_diag=False):
        x_feed = tf.placeholder(tf.float32, [None, self.layers[0]])
        y_feed = tf.placeholder(tf.float32, [None, self.layers[-1]])
        A = self.a_blocks(x_feed)
        grads_ys, grads_yw, grads_l = self.loss_grads(x_feed, y_feed, block_diag)
        G = self.g_blocks_exact(sess, grads_ys, x_feed, test_images)

        if block_diag:
            L = sess.run([self.kronecker_product(a, g) for a, g in zip(A, G)], {x_feed: test_images})
            return L
        else:
            B = []
            for i in range(len(G)):
                C = []
                for j in range(len(G)):
                        C.append(sess.run(self.kronecker_product(A[i][j], G[i][j]), {x_feed: test_images}))
                B.append(C)

            return np.asarray(np.bmat(B))

    def kfac_fisher2(self, sess, test_images, block_diag=False):
        x_feed = tf.placeholder(tf.float32, [None, self.layers[0]])
        y_feed = tf.placeholder(tf.float32, [None, self.layers[-1]])
        A = self.a_covariance(sess, test_images, blocks=True)
        grads_ys, grads_yw, grads_l = self.loss_grads(x_feed, y_feed, block_diag)
        G = self.g_blocks_exact(sess, grads_ys, x_feed, test_images)

        if block_diag:
            L = sess.run([self.kronecker_product(a, g) for a, g in zip(A, G)], {x_feed: test_images})
            return L
        else:
            B = []
            for i in range(len(G)):
                C = []
                for j in range(len(G)):
                        C.append(sess.run(self.kronecker_product(A[i][j], G[i][j]), {x_feed: test_images}))
                B.append(C)

            return np.asarray(np.bmat(B))

    def kfac_fisher_rev(self, sess, test_images, block_diag=False):
        x_feed = tf.placeholder(tf.float32, [None, self.layers[0]])
        y_feed = tf.placeholder(tf.float32, [None, self.layers[-1]])
        A = self.a_blocks(x_feed)
        grads_ys, grads_yw, grads_l = self.loss_grads(x_feed, y_feed, block_diag)
        G = self.g_blocks_exact(sess, grads_ys, x_feed, test_images)

        if block_diag:
            L = sess.run([self.kronecker_product(g, a) for a, g in zip(A, G)], {x_feed: test_images})
            return L
        else:
            B = []
            for i in range(len(G)):
                C = []
                for j in range(len(G)):
                    C.append(sess.run(self.kronecker_product(G[i][j], A[i][j]), {x_feed: test_images}))
                B.append(C)

            return np.asarray(np.bmat(B))

    def loss_grads(self, x, y_, block_diag=False):
        z, s = self.forward(x, with_activations=True)
        z = z[-1]
        grads_ys = [tf.concat([tf.reshape(tf.gradients(z[:, i], ss), [-1, 1]) for i in range(self.layers[-1])], axis=1) for
                   ss in s]
        grads_yw = [tf.concat([tf.reshape(tf.gradients(z[:, i], W), [-1, 1]) for i in range(self.layers[-1])], axis=1)
                    for W in self.W]
        loss = 0.5 * tf.reduce_mean(tf.reduce_sum(tf.squared_difference(z, y_), axis=1), axis=0, keep_dims=True)
        grads_l = [tf.reshape(tf.gradients(loss, w), [-1, 1]) for w in self.W]
        if block_diag:
            grads_yw = [tf.matmul(g, tf.transpose(g)) for g in grads_yw]
            grads_l = [tf.matmul(g, tf.transpose(g)) for g in grads_l]
        else:
            grads_yw = tf.concat([g for g in grads_yw], axis=0)
            grads_yw = [tf.matmul(grads_yw, tf.transpose(grads_yw))]

            grads_l = tf.concat([g for g in grads_l], axis=0)
            grads_l = [tf.matmul(grads_l, tf.transpose(grads_l))]

        return grads_ys, grads_yw, grads_l

    def kronecker_product(self,A,B):
        a_shape = tf.shape(A)
        b_shape = tf.shape(B)

        x = tf.reshape(A, [a_shape[0], a_shape[1], 1])
        x = tf.tile(x, [1, 1, b_shape[1]])
        x = tf.reshape(x, [a_shape[0], a_shape[1] * b_shape[1]])

        x = tf.reshape(x, [a_shape[0], a_shape[1] * b_shape[1], 1])
        x = tf.tile(x, [1, 1, b_shape[0]])
        x = tf.transpose(x, [0, 2, 1])

        x = tf.reshape(x, [a_shape[0] * b_shape[0], a_shape[1] * b_shape[1]])

        y = tf.tile(B, [a_shape[0], a_shape[1]])

        return tf.multiply(x, y)


    # def approximate_fisher_with_samples(self, x, num_examples, num_samples):
    #     A = self.a_blocks(x)
    #     G = self.g_blocks_approximate(x, num_examples, num_samples)
    #     return [self.kronecker_product(a,g) for a, g in zip(A,G)]
    #
    # def g_blocks_approximate(self, x, num_examples, num_samples):
    #     n = num_examples * num_samples
    #     mu, s = self.forward(x, with_activations=True)
    #     mu = tf.tile(mu[-1], [num_samples, 1])
    #     y = tf.contrib.distributions.Normal(mu, 1.).sample()
    #     loss = tf.reshape(tf.reduce_sum(tf.squared_difference(y, mu), axis=1), [1, -1])
    #     G = [tf.concat([tf.reshape(tf.gradients(loss[:, i], ss)[0], [-1, 1])
    #                     for i in range(0, n)], axis=1) for ss in s]
    #     G = [tf.matmul(g, tf.transpose(g)) / n for g in G]
    #     return G

    # this computes zero
    # def expectation_squared_mse(self, x):
    #     y = self.forward(x)
    #     F = [tf.concat([tf.reshape(tf.gradients(y[:, i], W)[0], [-1, 1]) for W in self.W],
    #                    axis=0) for i in range(0, self.layers[-1])]
    #     F = tf.concat(F, axis=1) / tf.cast(tf.shape(x)[0], tf.float32)
    #     F = tf.matmul(F, tf.transpose(F))
    #     return F

    # OLD INEFFICIENT FUNCTIONS

    # def g_blocks_exact(self, x, num_examples):
    #     y, s = self.forward(x, with_activations=True)
    #     y = tf.reshape(y[-1], [1, -1])
    #     G = [tf.concat([tf.reshape(tf.gradients(y[:, i], ss)[0], [-1, 1])
    #                     for i in range(0, num_examples * self.layers[-1])], axis=1)
    #          for ss in s]
    #     G = [tf.matmul(g, tf.transpose(g)) / self.layers[-1] for g in G]
    #     return G
    #
    # def exact_fisher2(self, x, num_samples=1000):
    #     y = tf.reshape(self.forward(x), [1, -1])
    #     F = [ tf.concat([tf.reshape(tf.gradients(y[:, i], W)[0], [-1, 1]) for W in self.W],
    #                     axis=0) for i in range(0, num_samples) ]
    #     F = tf.concat(F, axis=1)
    #     F = tf.matmul(F, tf.transpose(F)) / (num_samples/self.layers[-1])
    #     return F









