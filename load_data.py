import numpy
np = numpy
import theano

from neurobricks.preprocess import ZCA, SubtractMeanAndNormalizeH

import argparse
import os
import time
def unhot(arr):
    return np.argmax(arr, axis=-1)

def load_dataset(dataset):
    if dataset == "CIFAR10":
        input_shape = (32, 32, 3)
        try:
            train_x = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/train_x_preprocessed.npy'))
            train_y = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/train_y_preprocessed.npy'))
            test_x = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/test_x_preprocessed.npy'))
            test_y = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/test_y_preprocessed.npy'))
        except:
            train_x = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/train_x.npy'))
            train_y = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/train_y.npy'))
            test_x = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/test_x.npy'))
            test_y = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/test_y.npy'))

            preprocess_model = SubtractMeanAndNormalizeH(train_x.shape[1])
            map_fun = theano.function([preprocess_model.varin], preprocess_model.output())

            zca_obj = ZCA()
            zca_obj.fit(map_fun(train_x), retain=zca_retain, whiten=True)
            preprocess_model = preprocess_model + zca_obj.forward_layer
            preprocess_function = theano.function([preprocess_model.varin], preprocess_model.output())
            train_x = preprocess_function(train_x)
            test_x = preprocess_function(test_x)

            np.save(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/train_x_preprocessed.npy'), train_x)
            np.save(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/train_y_preprocessed.npy'), train_y)
            np.save(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/test_x_preprocessed.npy'), test_x)
            np.save(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/test_y_preprocessed.npy'), test_y)

            nex = 50000
            ntest = 10000

    elif dataset == "MNIST":
        input_shape = (28, 28, 1)
        try:
            train = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'mnist/mnist-python/train_combined.npy'))
            test = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'mnist/mnist-python/valid_combined.npy'))
            train_x = train[:, :784]
            test_x = test[:, :784]
            train_y = unhot(train[:, 784:])
            test_y = unhot(test[:, 784:])
        except:
            raise

            from datasets.load_mnist import load_mnist
            path = '/Users/david/datasets/'
            train_x, train_y = load_mnist(path=path)
            test_x, test_y = load_mnist('testing', path=path)
            #train_y = train_y.astype("int64")
            #test_y = test_y.astype("int64")
            train_x = train_x.reshape((-1,784)).astype("float32")
            test_x = test_x.reshape((-1,784)).astype("float32")
            train_x /= 256.
            test_x /= 256.

        nex = 50000
        ntest = 10000

    return train_x, train_y, test_x, test_y, input_shape
