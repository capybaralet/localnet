import os
import numpy
np = numpy
npy_rng = numpy.random.RandomState(123)
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

#from neurobricks.dataset import CIFAR10
from neurobricks.classifier import LogisticRegression
from neurobricks.model import ReluAutoencoder
from neurobricks.preprocess import ZCA, SubtractMeanAndNormalizeH
from neurobricks.train import GraddescentMinibatch
from neurobricks.params import save_params, load_params, set_params, get_params

import pdb

import argparse
import os
import time

from theano.tensor.signal.pool import pool_2d, Pool
from pylearn2.packaged_dependencies.theano_linear.unshared_conv.unshared_conv import FilterActs
#outputs = locally_connected(inputs, filters)
locally_connected = FilterActs(1)


verbose = 0


"""
WIP:
time to make a class for params??
"""

class TieableParameter(object):
    """
    An object which represents a parameter which may be tied or untied.
    We maintain the untied and tied valued.

    When we "tie" the parameter, we

    """
    def __init__(self, untied_shape, tied_dimensions):
        self.__dict__.update(locals)
        self.untied_dimensions = []
        self.tied_shape = []
        for n in range(len(self.untied_shape)):
            if n not in tied_dimensions:
                self.untied_dimensions.append(n)
                self.tied_shape.append(self.untied_shape[n])
            else:
                self.tied_shape.append(1)
        self.untied_value = np.nan * np.ones(self.untied_shape)
        self.tied_value = np.nan * np.ones(self.untied_shape)

    def set_untied_value(self, untied_value): # TODO
        pass

    def is_tied(self): # TODO
        pass

    def tie(self): # TODO
        pass


# returns param_shape with tied_dims replaced with 1s
def get_tied_shape(param_shape, tied_dims):
    rval = []
    for dim in range(len(param_shape)):
        if dim in tied_dims:
            rval.append(1)
        else:
            rval.append(param_shape[dim])
    return rval

# takes a reference param and returns an array expanded along tied dims to have the untied_param_shape.
def get_untied(numpy_reference_param, untied_param_shape):
    tile_shape = tuple(np.array(untied_param_shape) / np.array(numpy_reference_param.shape))
    return np.tile(numpy_reference_param, tile_shape)

shapes_str = """
The shapes:
filters: nfilters_per_row, nfilters_per_col, nchannels_per_filter, nrows_per_filter, ncolumns_per_filter, filter_groups, nfilters_per_group
biases: filter_groups, nfilters_per_group, nfilters_per_row, nfilters_per_col
inputs: input_groups, channels_per_group, nrows, ncols, batch_size
outputs: groups, nfilters_per_group, nfilters_per_row, nfilters_per_column, batch_size
"""

# infer the shapes of parameters/activations in a convnet
# FIXME: in AlexNet, the padding is just enough to preserve the size of the input; in LeNet, there is extra padding, and this is not longer the case!!!!
def infer_shapes(input_shape, filter_sizes, nchannels, pool_sizes, pads):
    weights_shapes = []
    biases_shapes = []
    activation_shapes = [input_shape]
    for sz, chan, pool, pad in zip(filter_sizes, nchannels, pool_sizes, pads):
        activation_shape = activation_shapes[-1] # FIXME?
        weights_shapes.append([activation_shape[2],
                     activation_shape[3],
                     activation_shape[1],
                     sz,
                     sz,
                     1,
                     chan])
        biases_shapes.append([1,
                              chan,
                              activation_shape[2],
                              activation_shape[3]])
        # add padding
        activation_shapes[-1] = [activation_shapes[-1][n] + (0, 0, 2*pad, 2*pad, 0)[n] for n in range(5)]
        activation_shape = activation_shapes[-1]
        # infer next shape (post-pooling)
        activation_shape = [1,
                            chan,
                            (activation_shape[2] - sz + 1) / pool,
                            (activation_shape[3] - sz + 1) / pool,
                            activation_shape[4]]
        activation_shapes.append(activation_shape)
    return weights_shapes, biases_shapes, activation_shapes


# Like tie, but takes a reference param, which is used to extract the updates
# This allows us to easily SUM instead of AVERAGE the updates!
# we assume reference param is already in the right format...
# TODO: output or update reference_param
def tie(param, tied_dims, untied_dims, reference_param=None):
    ndim = len(tied_dims) + len(untied_dims)
    assert sorted(tied_dims + untied_dims) == range(ndim)
    # make lists for reshaping
    tile_shape = []
    for dd in range(ndim):
        if dd in tied_dims:
            tile_shape.append(param.shape[dd])
        else:
            tile_shape.append(1)
        #print tile_shape
    if reference_param is not None: # use the reference param to recover the updates
        sum_of_updates = T.sum(param - reference_param, tied_dims, keepdims=1)
        updated_reference_param = sum_of_updates + reference_param
    else: # just take the mean value of the parameters (i.e. averaging instead of summing the updates)
        updated_reference_param = T.mean(param, tied_dims, keepdims=1)
    updated_param = T.tile(updated_reference_param, tile_shape, ndim=ndim)
    return updated_param, updated_reference_param



###############
# BUILD MODEL #
###############

# Make Params
weights_shapes, biases_shapes, activations_shapes = infer_shapes(input_shape, filter_sizes, nchannels, pool_sizes, pads)
print shapes_str
print "weights_shapes =", weights_shapes
print "biases_shapes =", biases_shapes
print "activations_shapes =", activations_shapes
weights_sharing = (range(2), range(2,7)) # tied / untied
biases_sharing = (range(2,4), range(2))
print "tied, untied dims for weight/biases are:", weights_sharing, biases_sharing

# We'll make numpy arrays of the tied params, then use these to construct both the reference params and the (untied) params
numpy_weights = [np.random.uniform(-init_scale, init_scale, get_tied_shape(shp, weights_sharing[0])).astype("float32") for
              n,shp in enumerate(weights_shapes)]
numpy_biases = [np.zeros(get_tied_shape(shp, biases_sharing[0])).astype("float32") for
              n,shp in enumerate(biases_shapes)]
reference_weights = [theano.shared(nw, name='ref_w' + str(n)) for n, nw in enumerate numpy_weights]
reference_biases = [theano.shared(nb, name='ref_b' + str(n)) for n, nb in enumerate numpy_biases]
weights = [theano.shared(get_untied(nw, weights_shapes[n]), name='w' + str(n)) for n, nw in enumerate numpy_weights]
biases = [theano.shared(get_untied(nb, biases_shapes[n]), name='b' + str(n)) for n, nb in enumerate numpy_biases]
# create dict to look up reference params by their corresponding (untied) params
reference_dict = {}
reference_dict.update({pp: ref_pp for pp,ref_pp in zip(reference_weights, weights)})
reference_dict.update({pp: ref_pp for pp,ref_pp in zip(reference_biases, biases)})
# these are fully connected, so no weight sharing here!
output_weight = theano.shared(np.random.uniform(-init_scale, init_scale, (np.prod(activations_shapes[-1]) / batchsize, 10)).astype("float32"), 'w_out')
output_bias = theano.shared(np.zeros(10).astype("float32"), 'b_out')
params = weights + biases + [output_weight, output_bias]


# set-up fprop
varin = T.matrix('varin')
truth = T.lvector('truth')
varin.tag.test_value = train_x[:batchsize].eval()
truth.tag.test_value = train_y[:batchsize].eval()
targets = truth
#inputs: input_groups, channels_per_group, nrows, ncols, batch_size
activations = [varin.reshape(input_shape)]
for weight, bias, pool_size, activation_shape, pad in zip(weights, biases, pool_sizes, activations_shapes, pads):
    # pad with zeros
    activations[-1] = T.set_subtensor(
                        T.zeros(activation_shape)[:, :, pad:-pad, pad:-pad, :],
                        activations[-1])
    preactivations = locally_connected(activations[-1], weight) + bias.dimshuffle(0, 1, 2, 3, 'x')
    activations.append(preactivations * (preactivations > 0))
    # pool_2d pools over the last 2 dims
    activations.append(pool_2d(activations[-1].dimshuffle(0,1,4,2,3),
                               (pool_size, pool_size),
                               #st=pool_size,
                               mode='max', padding=(0,0),
                               ignore_border=1).dimshuffle(0,1,3,4,2))
preoutputs = T.dot(activations[-1].reshape((batchsize, -1)), output_weight) + output_bias
outputs = T.nnet.softmax(preoutputs)

# set-up costs
nll_cost = T.mean(-T.log(outputs[T.arange(targets.shape[0]), targets]))
predictions = T.argmax(outputs, axis=1)
error_rate  = 1 - T.mean(T.eq(predictions, targets))

# set-up tie_fn
for param in reference_dict.keys():
    param_update, reference_param_update = tie(param, 
    tie_updates[param: 
tie_updates = {}


tie_list = [pp_update, ref_pp_update
tie_updates = {ww: tie(ww, range(2), range(2,7)) for ww in weights}
tie_updates.update({bb: T.unbroadcast(tie(bb, range(2,4), range(2)), 0,1) for bb in biases})
tie_fn = theano.function([], [], updates=tie_updates)

