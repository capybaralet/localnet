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

from utils import unhot

verbose = 1
test1ex = 0
hardwire_cnn = 1


"""
TODO: 
    debug?

    So the learningrate is affected by this averaging, so we need to make it larger,
      and to compare with traditional CNN, we also need to make it per-layer!

"""


# PARSE ARGS
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--sharify_every_n_batches", type=int, dest='sharify_every_n_batches', default=1)
parser.add_argument("--lr", type=float, dest='lr', default=.01)
parser.add_argument("--init_scale", type=float, dest='init_scale', default=.01)
parser.add_argument("--net", type=str, dest='net', default='AlexNet')
parser.add_argument("--dataset", type=str, dest='dataset', default='CIFAR10')
parser.add_argument("--L2", type=float, dest='L2', default=0)
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
finetune_epc = 1000


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
# FIXME: in AlexNet, the padding is just enough to preserve the size of the input;
#        in LeNet, there is extra padding, and this is no longer the case!!!!
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

# aka "re-tie"
# replace local params with their average (can be used to enforce CNN weight/bias sharing)
def sharify(param, shared_dims, unshared_dims):
    tile_shape = []
    ndim = len(shared_dims) + len(unshared_dims)
    assert sorted(shared_dims + unshared_dims) == range(ndim) # all dims accounted for?
    for dd in range(ndim):
        if dd in shared_dims:
            tile_shape.append(param.shape[dd])
        else:
            tile_shape.append(1)
        #print tile_shape
    return T.tile(T.mean(param, shared_dims, keepdims=1), tile_shape, ndim=ndim)

# Like sharify, but takes a reference param, which is used to extract the updates
# This allows us to easily SUM instead of AVERAGE the updates!
# TODO: output or update reference_param
def sharify(param, shared_dims, unshared_dims, reference_param=None):
    ndim = len(shared_dims) + len(unshared_dims)
    assert sorted(shared_dims + unshared_dims) == range(ndim)
    # make lists for reshaping
    tile_shape = []
    reference_dimshuffle = []
    reference_dim = 0
    for dd in range(ndim):
        if dd in shared_dims:
            tile_shape.append(param.shape[dd])
            reference_dimshuffle.append('x')
        else:
            tile_shape.append(1)
            reference_dimshuffle.append(reference_dim)
            reference_dim += 1
        #print tile_shape
    if reference_param is not None:
        sum_of_updates = T.sum(param - reference_param.dimshuffle(*reference_dimshuffle), shared_dims, keepdims=1)
        updated_param = sum_of_updates + reference_param.dimshuffle(*reference_dimshuffle)
    else:
        updated_param = T.mean(param, shared_dims, keepdims=1)
    return T.tile(updated_param, tile_shape, ndim=ndim)



#############
# LOAD DATA #
#############
print "\n... pre-processing"
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

    if 0: # why doesn't this work??? (need to decay LR more aggresively???)
        nex = 1000
        ntest = nex
        print "training on " + str(nex) + " examples"
    else:
        nex = 50000
        ntest = 10000
elif dataset == "MNIST":
    input_shape = (28, 28, 1)
    train = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'mnist/mnist-python/train_combined.npy'))
    test = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'mnist/mnist-python/valid_combined.npy'))
    train_x = train[:, :784]
    test_x = test[:, :784]
    train_y = unhot(train[:, 784:])
    test_y = unhot(test[:, 784:])
    nex = 50000
    ntest = 10000

if test1ex:
    batchsize = 10
    nex = 100
    ntest = 20


train_x = train_x[:nex]
test_x = test_x[:ntest]
train_y = train_y[:nex]
test_y = test_y[:ntest]
train_x = theano.shared(value = train_x, name = 'train_x', borrow = True)
train_y = theano.shared(value = train_y, name = 'train_y', borrow = True)
test_x = theano.shared(value = test_x,   name = 'test_x',  borrow = True)
test_y = theano.shared(value = test_y,   name = 'test_y',  borrow = True)


input_shape = (1, input_shape[2], input_shape[0], input_shape[1], batchsize) # reshaped for locally_connected layers

print "Done."


###############
# BUILD MODEL #
###############

# Make Params
weights_shapes, biases_shapes, activations_shapes = infer_shapes(input_shape, filter_sizes, nchannels, pool_sizes, pads)
print shapes_str
print "weights_shapes =", weights_shapes
print "biases_shapes =", biases_shapes
print "activations_shapes =", activations_shapes
w_out = theano.shared(np.random.uniform(-init_scale, init_scale, (np.prod(activations_shapes[-1]) / batchsize, 10)).astype("float32"), 'w_out')
b_out = theano.shared(np.zeros(10).astype("float32"), 'b_out')

if hardwire_cnn:
    untiled_weights = [theano.shared(np.random.uniform(-init_scale, init_scale, shp[2:]).astype("float32").reshape([1,1,] + shp[2:]), name='w' + str(n))
                for n,shp in enumerate(weights_shapes)]
    weights = [T.tile(w, shp[:2] + list(np.ones_like(shp)[2:])) for w,shp in zip(untiled_weights,weights_shapes)]
    # biases are still separate for each location...
    biases = [theano.shared(np.zeros(shp).astype("float32"), name='b'+str(n)) for 
                  n,shp in enumerate(biases_shapes)]
    params = untiled_weights + biases + [w_out, b_out]

else:
    if 1: # start with shared weights:
        weights = [theano.shared(np.tile(np.random.uniform(-init_scale, init_scale, shp[2:]).astype("float32").reshape([1,1,] + shp[2:]),
            (shp[:2] + list(np.ones_like(shp)[2:]))), name='w'+str(n)) for 
                    n,shp in enumerate(weights_shapes)]
    else:
        weights = [theano.shared(np.random.uniform(-init_scale, init_scale, shp).astype("float32"), name='w'+str(n)) for 
                  n,shp in enumerate(weights_shapes)]
    biases = [theano.shared(np.zeros(shp).astype("float32"), name='b'+str(n)) for 
                  n,shp in enumerate(biases_shapes)]
    params = weights + biases + [w_out, b_out]



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
preoutputs = T.dot(activations[-1].reshape((batchsize, -1)), w_out) + b_out
outputs = T.nnet.softmax(preoutputs)



# set-up costs
nll_cost = T.mean(-T.log(outputs[T.arange(targets.shape[0]), targets]))
L2_cost = 0
L2_cost = L2 * T.sum([(W**2).sum()**.5 for W in weights +[w_out,]])
train_cost = nll_cost + L2_cost
predictions = T.argmax(outputs, axis=1)
error_rate  = 1 - T.mean(T.eq(predictions, targets))

# set-up sharify_fn
if not hardwire_cnn:
    sharify_updates = {ww: sharify(ww, range(2), range(2,7)) for ww in weights}
    sharify_updates.update({bb: T.unbroadcast(sharify(bb, range(2,4), range(2)), 0,1) for bb in biases})
    sharify_fn = theano.function([], [], updates=sharify_updates)


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

monitor_fn_ = theano.function(
    [varin, truth],
    activations + [preoutputs, outputs, nll_cost, predictions],
)

monitor_fn = theano.function(
    [index],
    activations + [preoutputs, outputs, nll_cost, predictions],
    givens = {varin : train_x[index * batchsize: (index + 1) * batchsize],
              truth : train_y[index * batchsize: (index + 1) * batchsize]},
)

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
    cost=train_cost,
    params=params, 
    batchsize=batchsize, 
    learningrate=lr, 
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


monitored = []
learning_curves = [list(), list(), list()]
ttime = time.time()
if 0: # start with shared weights!
    sharify_fn()
for step in xrange(finetune_epc * nex / batchsize):

    if verbose:
        monitored.append(monitor_fn(step % (nex / batchsize)))
        #print monitored[-1][-3]

    # learn
    cost = trainer.step_fast(verbose_stride=500)
    epc_cost += cost

    # sharify
    if not hardwire_cnn and step % sharify_every_n_batches == 0:
        sharify_fn()
        if verbose:
            print "Done sharifying, step", step, time.time() - ttime
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
        learning_curves[2].append(epc_cost)
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
