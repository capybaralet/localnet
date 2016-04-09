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

from load_data import load_dataset

from theano.tensor.signal.pool import pool_2d, Pool
from pylearn2.packaged_dependencies.theano_linear.unshared_conv.unshared_conv import FilterActs
#outputs = locally_connected(inputs, filters)
locally_connected = FilterActs(1)


verbose = 1
use_10percent_of_dataset = 1


"""
TODO:
    debug? doesn't seem to train...
    why do we need to unbroadcast??


TODO: replace share -> tie in all naming (sounds more active!)
"reference params" are the params at the time of the last tying
    reference params -> untiled_params
    params -> tiled_params

"""


# PARSE ARGS
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--tie_every_n_batches", type=int, dest='tie_every_n_batches', default=1)
parser.add_argument("--lr", type=float, dest='lr', default=.01)
parser.add_argument("--init_scale", type=float, dest='init_scale', default=.01)
parser.add_argument("--net", type=str, dest='net', default='AlexNet')
parser.add_argument("--dataset", type=str, dest='dataset', default='MNIST')
args_dict = vars(parser.parse_args())
locals().update(args_dict)
settings_str = '_'.join([arg + "=" + str(args_dict[arg]) for arg in sorted(args_dict.keys())])
print "settings_str=", settings_str


###################
# HYPERPARAMETERS #
###################
zca_retain = 0.99
batchsize = 100
momentum = 0.9
batchsize = 100
finetune_lr = lr
finetune_epc = 1000

# Network architecture
if net == 'LeNet': # TODO: top_mlp
    filter_sizes = [5,5]
    nchannels = [20, 50]
    pool_sizes = [2,2]
    pads = [2,2]
    #pads = [4,4]
elif net == 'AlexNet':
    filter_sizes = [5,5,5]
    nchannels = [32,32,64]
    pool_sizes = [2,2,2]
    pads = [2,2,2]

# SETUP SAVEPATH
script_dir = os.path.join(os.environ['SAVE_PATH'], os.path.basename(__file__)[:-3])
print script_dir
if not os.path.exists(script_dir): # make sure script_dir exists
    print "making directory:", script_dir
    try:
        os.makedirs(script_dir)
    except:
        pass
# Each run of the script has its own subdir
savepath = os.path.join(script_dir, settings_str)


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


# uses reference_param to extract the updates
# This allows us to easily SUM instead of AVERAGE the updates!
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


#############
# LOAD DATA #
#############
train_x, train_y, test_x, test_y, input_shape = load_dataset(dataset)
input_shape = (1, input_shape[2], input_shape[0], input_shape[1], batchsize) # reshaped for locally_connected layers

if use_10percent_of_dataset:
    train_x *= 256
    test_x *= 256
    nex = 5000
    ntest = nex / 10
    print "training on " + str(nex) + " examples"
else:
    nex = 50000
    ntest = 10000

train_x = train_x[:nex]
test_x = test_x[:ntest]
train_y = train_y[:nex]
test_y = test_y[:ntest]
train_x = theano.shared(value = train_x, name = 'train_x', borrow = True)
train_y = theano.shared(value = train_y, name = 'train_y', borrow = True)
test_x = theano.shared(value = test_x,   name = 'test_x',  borrow = True)
test_y = theano.shared(value = test_y,   name = 'test_y',  borrow = True)
print "Done."


###############
# BUILD MODEL #
###############
weights_shapes, biases_shapes, activations_shapes = infer_shapes(input_shape, filter_sizes, nchannels, pool_sizes, pads)
print shapes_str
print "weights_shapes =", weights_shapes
print "biases_shapes =", biases_shapes
print "activations_shapes =", activations_shapes
# TODO: replace _sharing with _dims_shared
weights_dims_shared = [1,1,0,0,0,0,0]
biases_dims_shared = [0,0,1,1]
weights_sharing = (range(2), range(2,7)) # tied / untied
biases_sharing = (range(2,4), range(2))
print "tied, untied dims for weight/biases are:", weights_sharing, biases_sharing

# Make Params:
# We make numpy arrays of the tied params, then use these to construct both the reference params and the (untied) params
numpy_weights = [np.random.uniform(-init_scale, init_scale, get_tied_shape(shp, weights_sharing[0])).astype("float32") for
              n,shp in enumerate(weights_shapes)]
numpy_biases = [np.zeros(get_tied_shape(shp, biases_sharing[0])).astype("float32") for
              n,shp in enumerate(biases_shapes)]
reference_weights = [theano.shared(nw, name='ref_w' + str(n), broadcastable=weights_dims_shared) for n, nw in enumerate(numpy_weights)]
reference_biases = [theano.shared(nb, name='ref_b' + str(n), broadcastable=biases_dims_shared) for n, nb in enumerate(numpy_biases)]
weights = [theano.shared(get_untied(nw, weights_shapes[n]), name='w' + str(n)) for n, nw in enumerate(numpy_weights)]
biases = [theano.shared(get_untied(nb, biases_shapes[n]), name='b' + str(n)) for n, nb in enumerate(numpy_biases)]
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

# set-up tie_fn (for manually retying parameters)
tie_updates = {}
for pp, ref_pp in zip(weights, reference_weights):
    pp_update, ref_pp_update = tie(pp, weights_sharing[0], weights_sharing[1], ref_pp)
    tie_updates[pp] = pp_update
    tie_updates[ref_pp] = ref_pp_update
for pp, ref_pp in zip(biases, reference_biases):
    pp_update, ref_pp_update = tie(pp, biases_sharing[0], biases_sharing[1], ref_pp)
    tie_updates[pp] = pp_update
    tie_updates[ref_pp] = ref_pp_update
tie_fn = theano.function([], [], updates=tie_updates)


###############################
# compile error rate counters #
###############################
index = T.lscalar()
train_set_error_rate = theano.function(
    [index],
    error_rate,
    givens = {varin : train_x[index * batchsize: (index + 1) * batchsize],
              truth : train_y[index * batchsize: (index + 1) * batchsize]},
)
def train_error():
    return numpy.mean([train_set_error_rate(i) for i in xrange(nex/batchsize)])

test_set_error_rate = theano.function(
    [index],
    error_rate,
    givens = {varin : test_x[index * batchsize: (index + 1) * batchsize],
              truth : test_y[index * batchsize: (index + 1) * batchsize]},
)
def test_error():
    return numpy.mean([test_set_error_rate(i) for i in xrange(ntest/batchsize)])

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
    cost=nll_cost,
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



grads = T.grad(nll_cost, params)
updates = {pp : pp - lr * grads[n] for n, pp in enumerate(params)}
train_fn = theano.function([index], 
        nll_cost,
        givens = {varin : train_x[index * batchsize: (index + 1) * batchsize],
                  truth : train_y[index * batchsize: (index + 1) * batchsize]},
        )


for step in range(100):
    print train_fn(step % 50)

grads_fn = theano.function([index], 
        grads,
        givens = {varin : train_x[index * batchsize: (index + 1) * batchsize],
                  truth : train_y[index * batchsize: (index + 1) * batchsize]},
        )


learning_curves = [list(), list()]
ttime = time.time()
for step in xrange(finetune_epc * nex / batchsize):
    # learn
    cost = trainer.step_fast(verbose_stride=500)
    epc_cost += cost

    # tie
    if step % tie_every_n_batches == 0:
        tie_fn()
        if verbose:
            print "Done tieing, step", step, time.time() - ttime
            print "batch cost =", cost
            print "epc_cost =", epc_cost / ((step + 1) % (nex / batchsize))

    if step % (nex / batchsize) == 0 and step > 0:
        # set stop rule
        ind = (step / (nex / batchsize)) % avg
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
        learning_curves[0].append(train_error())
        learning_curves[1].append(test_error())
        np.save(savepath + '__learning_curves.npy', np.array(learning_curves))
        print "***error rate: train: %f, test: %f" % (
                learning_curves[0][-1],
                learning_curves[1][-1])

        epc_cost = 0.
print "Done."
print "***FINAL error rate, train: %f, test: %f" % (
    train_error(), test_error()
)
save_params(model, savepath + '_.npy')

pdb.set_trace()
