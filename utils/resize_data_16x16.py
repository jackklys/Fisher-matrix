import numpy as np
from scipy import misc
import os
import cPickle

data_dir = os.path.join(os.getcwd(), 'MNIST_data')
with open(data_dir + '/train_images.pkl', 'rb') as f:
    train_set = cPickle.load(f)
with open(data_dir + '/test_images.pkl', 'rb') as f:
    test_set = cPickle.load(f)

train_set = [ misc.imresize(train_set[i].reshape((28, 28)), (16, 16)).astype(np.float32)
              for i in range(0, train_set.shape[0]) ]
train_set = [ x/x.max() for x in train_set ]
train_set = [ x.reshape((16*16)) for x in train_set ]
train_set = np.array(train_set)

print(train_set.shape)

with open(data_dir + '/train_images_16x16.pkl', 'wb') as f:
    cPickle.dump(train_set, f)

test_set = [misc.imresize(test_set[i].reshape((28, 28)), (16, 16)).astype(np.float32)
             for i in range(0, test_set.shape[0])]
test_set = [x / x.max() for x in test_set]
test_set = [x.reshape((16 * 16)) for x in test_set]
test_set = np.array(test_set)

print(test_set.shape)

with open(data_dir + '/test_images_16x16.pkl', 'wb') as f:
    cPickle.dump(test_set, f)




