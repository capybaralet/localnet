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

# DK imports (below)
import argparse
import os
import time

from load_data import load_dataset

from theano.tensor.signal.pool import pool_2d, Pool
from pylearn2.packaged_dependencies.theano_linear.unshared_conv.unshared_conv import FilterActs
#outputs = locally_connected(inputs, filters)
locally_connected = FilterActs(1)

verbose = 0
use_10percent_of_dataset = 0
load_init_params = 0
compare_blocks = 0
hardwire_cnn = 0

nesterov = 0

# PARSE ARGS
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--tie_every_n_batches", type=int, dest='tie_every_n_batches', default=1)
parser.add_argument("--lr", type=float, dest='lr', default=.01)
parser.add_argument("--momentum", type=float, dest='momentum', default=.0)
parser.add_argument("--dataset", type=str, dest='dataset', default='MNIST')
parser.add_argument("--init_scale", type=float, dest='init_scale', default=.01)
parser.add_argument("--net", type=str, dest='net', default='LeNet')
parser.add_argument("--which_trainer", type=str, dest='which_trainer', default='zhouhan')
args_dict = vars(parser.parse_args())
locals().update(args_dict)
settings_str = '_'.join([arg + "=" + str(args_dict[arg]) for arg in sorted(args_dict.keys())])
print "settings_str=", settings_str

"""
TODO:
    why does hardwire_cnn do so poorly??

NTS: the "width" in blocks is the *total width* of the distribution, so it should be double what I use here...
TODO: params -> tiled_params
TODO: refactor for untied biases
TODO: clean-up testing code...
"""


# FIXME: momentum is probably fucking with everything!
#           I need to tie the momentum as well!!!
# FIXME: apparently my attempts to fix the momentum were unsuccessful and may have just fucked everything up even worse!


###################
# HYPERPARAMETERS #
###################
zca_retain = 0.99
batchsize = 100
batchsize = 100
finetune_lr = lr
finetune_epc = 1000

# Network architecture
if net == 'LeNet': # TODO: top_mlp
    filter_sizes = [5,5]
    nchannels = [20, 50]
    pool_sizes = [2,2]
    pads = [4,4]
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
def infer_shapes(input_shape, filter_sizes, nchannels, pool_sizes, pads):
    weights_shapes = []
    biases_shapes = []
    activation_shapes = [input_shape] # include padding!
    for sz, chan, pool, pad in zip(filter_sizes, nchannels, pool_sizes, pads):
        # add padding to activation_shape
        activation_shapes[-1] = [activation_shapes[-1][n] + (0, 0, 2*pad, 2*pad, 0)[n] for n in range(5)]
        activation_shape = activation_shapes[-1]
        weights_shapes.append([activation_shape[2] - sz + 1,
                     activation_shape[3] - sz + 1,
                     activation_shape[1],
                     sz,
                     sz,
                     1,
                     chan])
        biases_shapes.append([1,
                              chan,
                              activation_shape[2] - sz + 1,
                              activation_shape[3] - sz + 1])
        # infer next shape (post-pooling)
        activation_shape = [1,
                            chan,
                            (activation_shape[2] - sz + 1) / pool,
                            (activation_shape[3] - sz + 1) / pool,
                            activation_shape[4]]
        activation_shapes.append(activation_shape)
    return weights_shapes, biases_shapes, activation_shapes

# returns param_shape with shared dims replaced with 1s
def get_untiled_shape(param_shape, dims_shared):
    return [1 if shared else param_shape[n] for n, shared in enumerate(dims_shared)]

# takes a untiled param and returns an array expanded along tied dims to have the tiled_param_shape.
def get_tiled(numpy_untiled_param, tiled_param_shape):
    tile_shape = tuple(np.array(tiled_param_shape) / np.array(numpy_untiled_param.shape))
    return np.tile(numpy_untiled_param, tile_shape)

def get_tiled_theano(theano_untiled_param, tiled_param_shape):
    tile_shape = tuple(np.array(tiled_param_shape) / np.array(theano_untiled_param.shape.eval()))
    return T.tile(theano_untiled_param, tile_shape)


# make this more general! (untiled_param is annoying...)
def tie(param, dims_shared, untiled_param=None):
    """
    (re)-tie shared (tiled) parameters along dims_shared
    uses untiled_param to extract the updates
    This allows us to easily SUM instead of AVERAGE the updates!
    """
    # make lists for reshaping
    tile_shape = []
    tying_dims = [n for n, dim in enumerate(dims_shared) if dim]
    for ndim, is_shared in enumerate(dims_shared):
        if is_shared:
            tile_shape.append(param.shape[ndim])
        else:
            tile_shape.append(1)
        #print tile_shape
    if untiled_param is not None: # use the untiled param to recover the updates
        sum_of_updates = T.sum(param - untiled_param, tying_dims, keepdims=1)
        updated_untiled_param = sum_of_updates + untiled_param
    else: # just take the mean value of the parameters (i.e. averaging instead of summing the updates)
        updated_untiled_param = T.mean(param, tying_dims, keepdims=1)
    updated_param = T.tile(updated_untiled_param, tile_shape, ndim=len(dims_shared))
    return updated_param, updated_untiled_param


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
else:
    nex = 50000
    ntest = 10000
print "training on " + str(nex) + " examples"

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
# these tell us which dims will be tied
weights_dims_shared = [1,1,0,0,0,0,0]
biases_dims_shared = [0,0,1,1]

if load_init_params:
    output_weight, output_bias, numpy_weights, numpy_biases = np.load('/u/kruegerd/local_cnn_test_params.npy')
else:
    # Make numpy params:
    output_weight = np.random.uniform(-init_scale, init_scale, (np.prod(activations_shapes[-1]) / batchsize, 10)).astype("float32")
    output_bias = np.zeros(10).astype("float32")
    # We make numpy arrays of the untiled params, then use these to construct both the untiled params and the tiled params
    numpy_weights = [np.random.uniform(-init_scale, init_scale, get_untiled_shape(shp, weights_dims_shared)).astype("float32") for
                  n,shp in enumerate(weights_shapes)]
    # TODO: rm
    numpy_biases = [np.zeros(get_untiled_shape(shp, biases_dims_shared)).astype("float32") for
                  n,shp in enumerate(biases_shapes)]
    if 0:
        np.save('/u/kruegerd/local_cnn_test_params.npy', [output_weight, output_bias, numpy_weights, numpy_biases])
# Make theano.shared params
# these are fully connected, so no weight sharing here!
output_weight = theano.shared(output_weight, 'w_out')
output_bias = theano.shared(output_bias, 'b_out')
biases = [theano.shared(get_tiled(nb, biases_shapes[n]), name='b' + str(n)) for n, nb in enumerate(numpy_biases)]
# the untiled filters
untiled_weights = [theano.shared(nw, name='ref_w' + str(n), broadcastable=weights_dims_shared) for n, nw in enumerate(numpy_weights)]
if hardwire_cnn:
    weights = [get_tiled_theano(rw, weights_shapes[n]) for n, rw in enumerate(untiled_weights)]
    params = untiled_weights + biases + [output_weight, output_bias]
else:
    weights = [theano.shared(get_tiled(nw, weights_shapes[n]), name='w' + str(n)) for n, nw in enumerate(numpy_weights)]
    params = weights + biases + [output_weight, output_bias]


# set-up fprop
varin = T.matrix('varin')
truth = T.lvector('truth')
#varin.tag.test_value = train_x[:batchsize].eval()
#truth.tag.test_value = train_y[:batchsize].eval()
varin.tag.test_value = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'mnist/mnist-python/100examples/train100_x.npy'))
truth.tag.test_value = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'mnist/mnist-python/100examples/train100_y.npy'))
targets = truth
#inputs: input_groups, channels_per_group, nrows, ncols, batch_size
activations = [varin.transpose().reshape(input_shape)]

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
# TODO: regularization?
cost = nll_cost

if compare_blocks:

    outputs = activations # Not the same as blocks (missing some?)
    grads = T.grad(cost, params)
    ovl = [var.tag.test_value for var in outputs]
    gvl = [var.tag.test_value for var in grads]
    pvl = [var.tag.test_value for var in [predictions]]
    print truth.tag.test_value.squeeze()
    print pvl

    grads = T.grad(cost, params)
    monitor = []
    monitored = activations + [preoutputs, outputs,] + grads + [cost, predictions, error_rate]
    monitored_str = "activations + [preoutputs, outputs,] + grads + [cost, predictions, error_rate]"
    assert False

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

train_set_cost = theano.function(
    [index],
    cost,
    givens = {varin : train_x[index * batchsize: (index + 1) * batchsize],
              truth : train_y[index * batchsize: (index + 1) * batchsize]},
)
def train_cost():
    return numpy.mean([train_set_cost(i) for i in xrange(nex/batchsize)])

################################
# training and tying functions #
################################
# FIXME!
grads = T.grad(cost, params)
updates = {}
velocities = {pp: theano.shared(0. * pp.get_value()) for pp in params}
velocity_updates = {velocities[pp]: momentum * velocities[pp] - lr * gg for pp, gg in zip(params, grads)}
# we use the updated velocities for the parameter updates (to avoid having to call a separate function "update_incs"
if nesterov:
    updates.update({pp: pp + momentum * velocity_updates[velocities[pp]] - lr * gg for pp, gg in zip(params, grads)})
else:
    updates.update({pp: pp + velocity_updates[velocities[pp]] for pp, gg in zip(params, grads)})
updates.update(velocity_updates)
train_fn = theano.function(
    [index],
    [cost, error_rate],
    givens = {varin : train_x[index * batchsize: (index + 1) * batchsize],
              truth : train_y[index * batchsize: (index + 1) * batchsize]},
    updates = updates,
)

if not hardwire_cnn:
    # set-up tie_fn (for manually retying parameters)
    tie_updates = {}
    for pp, ref_pp in zip(weights, untiled_weights):
        pp_update, ref_pp_update = tie(pp, weights_dims_shared, ref_pp)
        tie_updates[pp] = pp_update
        tie_updates[ref_pp] = ref_pp_update
    # velocity updates (also uses untiled ("reference") velocities)
    untiled_velocities = {pp: theano.shared(0. * pp.get_value(), broadcastable=weights_dims_shared) for pp in untiled_weights}
    for weight, untiled_weight in zip(weights, untiled_weights):
        vv = velocities[weight]
        ref_vv = untiled_velocities[untiled_weight]
        vv_update, ref_vv_update = tie(vv, weights_dims_shared, ref_vv)
        tie_updates[vv] = vv_update
        tie_updates[ref_vv] = ref_vv_update
    tie_fn = theano.function([], [], updates=tie_updates)

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
    cost=cost,
    params=params, 
    batchsize=batchsize, 
    learningrate=finetune_lr, 
    momentum=momentum,
    rng=npy_rng
)

prev_cost = numpy.inf
epc_cost = 0.
patience = 0
avg = 50
crnt_avg = [numpy.inf, ] * avg
hist_avg = [numpy.inf, ] * avg


learning_curves = [list(), list(), list()]
ttime = time.time()
for step in xrange(finetune_epc * nex / batchsize):

    batch_n = step % (nex / batchsize)

    # learn
    if which_trainer == 'zhouhan':
        cost = trainer.step_fast(verbose_stride=500)
        epc_cost += cost
    else:
        costs = train_fn(batch_n)
        epc_cost += costs[0]

    # tie
    if not hardwire_cnn and step % tie_every_n_batches == 0:
        tie_fn()

    if verbose:
        print "Done tying, step", step, time.time() - ttime
        print "batch cost =", cost
        #print "epc_cost =", epc_cost / ((step + 1) % (nex / batchsize))

    if batch_n == 0 and step > 0:
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
            lr.set_value(0.9 * lr.get_value())
            patience = 0
        prev_cost = epc_cost

        # evaluate
        learning_curves[0].append(train_error())
        learning_curves[1].append(test_error())
        learning_curves[2].append(train_cost())
        np.save(savepath + '__learning_curves.npy', np.array(learning_curves))
        print "***error rate -- train: %f, test: %f   cost -- train: %f" % (
                learning_curves[0][-1],
                learning_curves[1][-1],
                learning_curves[2][-1])

        epc_cost = 0.
print "Done."
print "***FINAL RESULTS:   error rate -- train: %f, test: %f   cost -- train: %f" % (
        learning_curves[0][-1],
        learning_curves[1][-1],
        learning_curves[2][-1])

save_params(model, savepath + '_.npy')

pdb.set_trace()
