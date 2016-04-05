import os
import numpy
np = numpy
npy_rng = numpy.random.RandomState(123)
import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import cPickle

from neurobricks.dataset import CIFAR10
from neurobricks.classifier import LogisticRegression
from neurobricks.model import ReluAutoencoder
from neurobricks.preprocess import ZCA, SubtractMeanAndNormalizeH
from neurobricks.train import GraddescentMinibatch
from neurobricks.params import save_params, load_params, set_params, get_params

import pdb


"""
Pylearn2 wrapper on cudaconvnet

The shapes:
inputs: input_groups, channels_per_group, nrows, ncols, batch_size
filters: nfilters_per_row, nfilters_per_col, nchannels_per_filter, nrows_per_filter, ncolumns_per_filter, filter_groups, nfilters_per_group
biases: nfilters_per_row, nfilters_per_col, filter_groups, nfilters_per_group
outputs: groups, nfilters_per_group, nfilters_per_row, nfilters_per_column, batch_size


MUST PERFORM PADDING MANUALLY!
"""
from theano.tensor.signal.pool import pool_2d, Pool
from pylearn2.packaged_dependencies.theano_linear.unshared_conv.unshared_conv import FilterActs
locally_connected = FilterActs(1)
#outputs = locally_connected(inputs, filters)

# infer the shapes of parameters in a convnet
def params_shapes(input_shape, filter_sizes, nchannels, pool_sizes, pads):
    for sz, chan, pool, pad in zip(filter_sizes, nchannels, pool_sizes, pads):
        input_shape = [input_shape[n] + (2*pad, 2*pad, 0)[n] for n in range(3)]
        weights_shapes.append([input_shape[0] - sz,
                     input_shape[1] - sz,
                     input_shape[2],
                     sz,
                     sz,
                     1,
                     chan])
        biases_shapes.append([input_shape[0] - sz,
                     input_shape[1] - sz,
                     chan])
        input_shape = [(input_shape[0] - sz) / pool, (input_shape[1] - sz) / pool, chan]
    return weights_shapes, biases_shapes

# replace local params with their average (can be used to enforce CNN weight/bias sharing)
def sharify(params, shared_dims, unshared_dims):
    tile_shape = []
    ndim = len(shared_dims) + len(unshared_dims)
    assert sorted(shared_dims + unshared_dims) == range(ndim) # all dims accounted for?
    for dd in range(ndim):
        if dd in shared_dims:
            tile_shape.append(params.shape[dd])
        else:
            tile_shape.append(1)
    return T.tile(T.mean(params, shared_dims), tile_shape, ndim=ndim)

###################
# HYPERPARAMETERS #
###################
import argparse
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--sharify_every_n_batches", type=int, destination='sharify_every_n_batches', default=1)
args_dict = vars(parser.parse_args())
locals().update(args_dict)
settings_str = '_'.join([arg + "=" + str(args_dict[arg]) for arg in sorted(args_dict.keys())])
print "settings_str=", settings_str


zca_retain = 0.99
batchsize = 100
momentum = 0.9
#weightdecay = 0.01
finetune_lr = 5e-3
finetune_epc = 1000

###############
# BUILD MODEL #
###############

# AlexNet
input_shape = (32, 32, 3)
filter_sizes = [5,5,5]
nchannels = [32,32,64]
pool_sizes = [2,2,2]
pads = [2,2,2] # TODO!

# Make Params
weights_shapes, biases_shapes = params_shapes(input_shape, filter_sizes, nchannels, pool_sizes, pads)
weights = [theano.shared(np.random.uniform(-.01, .01, shp).astype("float32"), name='w'+str(n)) for 
              n,shp in enumerate(weights_shapes)]
biases = [theano.shared(np.zeros(shp).astype("float32"), name='b'+str(n)) for 
              n,shp in enumerate(biases_shapes)]
last_layer_dims = [weights[-1].shape.eval()[n] for n in [0,1,-1]]
output_weight = theano.shared(np.random.uniform(-.01, .01, (np.prod(last_layer_dims), 10)).astype("float32"), 'w_out')
output_bias = theano.shared(np.zeros(10).astype("float32"), 'b_out')

# set-up fprop
varin = T.matrix('varin')
truth = T.lvector('truth')
targets = truth
activations = [varin]
for weight, bias, pool_size in zip(weights, biases, pool_sizes): # TODO: padding
    preactivations = locally_connected(weight, activations[-1]) + bias
    activations.append(preactivations * (preactivations > 0))
    activations.append(pool_2d(activations, 
                               pool_size, 
                               st=pool_size,
                               mode='max', padding=(0,0),
                               ignore_border=1))
preoutputs = T.dot(activations[-1], output_weight) + output_bias
outputs = T.nnet.softmax(preoutputs)

# set-up costs
nll_cost = T.mean(-T.log(outputs[T.arange(targets.shape[0]), targets]))
predictions = T.argmax(outputs, axis=1)
error_rate  = 1 - T.mean(T.eq(predictions, targets))

# set-up sharify_fn
sharify_updates = {ww: sharify(ww, range(2), range(2,7)) for ww in weights} +\
                  {bb: sharify(bb, range(2), range(2,3)) for bb in biases}
sharify_fn = theano.function([], [], updates=sharify_updates)


#############
# LOAD DATA #
#############
cifar10_data = CIFAR10()
train_x, train_y = cifar10_data.get_train_set()
test_x, test_y = cifar10_data.get_test_set()

print "\n... pre-processing"
preprocess_model = SubtractMeanAndNormalizeH(train_x.shape[1])
map_fun = theano.function([preprocess_model.varin], preprocess_model.output())

zca_obj = ZCA()
zca_obj.fit(map_fun(train_x), retain=zca_retain, whiten=True)
preprocess_model = preprocess_model + zca_obj.forward_layer
preprocess_function = theano.function([preprocess_model.varin], preprocess_model.output())
train_x = preprocess_function(train_x)
test_x = preprocess_function(test_x)

feature_num = train_x.shape[0] * train_x.shape[1]

train_x = theano.shared(value = train_x, name = 'train_x', borrow = True)
train_y = theano.shared(value = train_y, name = 'train_y', borrow = True)
test_x = theano.shared(value = test_x,   name = 'test_x',  borrow = True)
test_y = theano.shared(value = test_y,   name = 'test_y',  borrow = True)
print "Done."

# compile error rate counters (TODO)
index = T.lscalar()
train_set_error_rate = theano.function(
    [index],
    error_rate,
    givens = {varin : train_x[index * batchsize: (index + 1) * batchsize],
              truth : train_y[index * batchsize: (index + 1) * batchsize]},
)
def train_error():
    return numpy.mean([train_set_error_rate(i) for i in xrange(50000/batchsize)])

test_set_error_rate = theano.function(
    [index],
    error_rate,
    givens = {varin : test_x[index * batchsize: (index + 1) * batchsize],
              truth : test_y[index * batchsize: (index + 1) * batchsize]},
)
def test_error():
    return numpy.mean([test_set_error_rate(i) for i in xrange(10000/batchsize)])
print "Done."

#############
# FINE-TUNE #
#############

print "\n\n... fine-tuning the whole network"
trainer = GraddescentMinibatch(
    varin=varin, 
    data=train_x, 
    truth=truth,
    truth_data=train_y,
    supervised=True,
    cost=prediction_error,
    params=params, 
    batchsize=batchsize, 
    learningrate=finetune_lr, 
    momentum=momentum,
    rng=npy_rng
)

init_lr = trainer.learningrate
prev_cost = numpy.inf
epc_cost = 0.
patience = 0
avg = 50
crnt_avg = [numpy.inf, ] * avg
hist_avg = [numpy.inf, ] * avg
for step in xrange(finetune_epc * 50000 / batchsize):
    # learn
    cost = trainer.step_fast(verbose_stride=500)
    epc_cost += cost

    # sharify
    if step % sharify_every_n_batches == 0:
        [sharify(W) for W in weights]

    if step % (50000 / batchsize) == 0 and step > 0:
        # set stop rule
        ind = (step / (50000 / batchsize)) % avg
        hist_avg[ind] = crnt_avg[ind]
        crnt_avg[ind] = epc_cost
        if sum(hist_avg) < sum(crnt_avg):
            break
    
        # adjust learning rate
        if prev_cost <= epc_cost:
            patience += 1
        if patience > 10:
            trainer.set_learningrate(0.9 * trainer.learningrate)
            patience = 0
        prev_cost = epc_cost

        # evaluate
        print "***error rate: train: %f, test: %f" % (
            train_error(), test_error())
        
        epc_cost = 0.
print "Done."
print "***FINAL error rate, train: %f, test: %f" % (
    train_error(), test_error()
)
save_params(model, 'dk_alexnet_' + settings_str + '_.npy')

pdb.set_trace()
