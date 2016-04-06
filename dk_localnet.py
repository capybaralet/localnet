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

import argparse
import os
import time

from theano.tensor.signal.pool import pool_2d, Pool
from pylearn2.packaged_dependencies.theano_linear.unshared_conv.unshared_conv import FilterActs
#outputs = locally_connected(inputs, filters)
locally_connected = FilterActs(1)


verbose = 1


"""
TODO: 
    debug?
    Adam
"""


# PARSE ARGS
parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--sharify_every_n_batches", type=int, dest='sharify_every_n_batches', default=1)
parser.add_argument("--lr", type=float, dest='lr', default=.01)
parser.add_argument("--init_scale", type=float, dest='init_scale', default=.01)
parser.add_argument("--architecture", type=str, dest='architecture', default='LeNet')
args_dict = vars(parser.parse_args())
locals().update(args_dict)
settings_str = '_'.join([arg + "=" + str(args_dict[arg]) for arg in sorted(args_dict.keys())])
print "settings_str=", settings_str


batchsize = 100
input_shape = (32, 32, 3)
activation_shape = (1, input_shape[2], input_shape[0], input_shape[1], batchsize) # reshaped for locally_connected layers

if architecture == 'LeNet': # TODO: top_mlp
    filter_sizes = [5,5]
    nchannels = [20, 50]
    pool_sizes = [2,2]
    pads = [4,4]
elif architecture == 'AlexNet':
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
def infer_shapes(activation_shape, filter_sizes, nchannels, pool_sizes, pads):
    weights_shapes = []
    biases_shapes = []
    activation_shapes = [activation_shape]
    for sz, chan, pool, pad in zip(filter_sizes, nchannels, pool_sizes, pads):
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
zca_retain = 0.99
batchsize = 100
momentum = 0.9
#weightdecay = 0.01
finetune_lr = lr
finetune_epc = 1000


#############
# LOAD DATA #
#############
train_x = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/train_x.npy'))
train_y = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/train_y.npy'))
test_x = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/test_x.npy'))
test_y = np.load(os.path.join(os.environ["FUEL_DATA_PATH"], 'cifar10/npy/test_y.npy'))

if 1:
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
else:
    train_x = train_x.astype("float32")
    test_x = test_x.astype("float32")

train_x = theano.shared(value = train_x, name = 'train_x', borrow = True)
train_y = theano.shared(value = train_y, name = 'train_y', borrow = True)
test_x = theano.shared(value = test_x,   name = 'test_x',  borrow = True)
test_y = theano.shared(value = test_y,   name = 'test_y',  borrow = True)
print "Done."


###############
# BUILD MODEL #
###############

# Make Params
weights_shapes, biases_shapes, activations_shapes = infer_shapes(activation_shape, filter_sizes, nchannels, pool_sizes, pads)
print shapes_str
print "weights_shapes =", weights_shapes
print "biases_shapes =", biases_shapes
print "activations_shapes =", activations_shapes
weights = [theano.shared(np.random.uniform(-init_scale, init_scale, shp).astype("float32"), name='w'+str(n)) for 
              n,shp in enumerate(weights_shapes)]
biases = [theano.shared(np.zeros(shp).astype("float32"), name='b'+str(n)) for 
              n,shp in enumerate(biases_shapes)]
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
activations = [varin.reshape(activation_shape)]
for weight, bias, pool_size, activation_shape in zip(weights, biases, pool_sizes, activations_shapes):
    # pad with zeros
    activations[-1] = T.set_subtensor(
                        T.zeros(activation_shape)[:, :, pool_size:-pool_size, pool_size:-pool_size, :],
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

# set-up sharify_fn
sharify_updates = {ww: sharify(ww, range(2), range(2,7)) for ww in weights}.update(
                  {bb: sharify(bb, range(2,4), range(2)) for bb in biases})
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
learning_curves = [list(), list()]
ttime = time.time()
for step in xrange(finetune_epc * 50000 / batchsize):
    # learn
    cost = trainer.step_fast(verbose_stride=500)
    epc_cost += cost

    # sharify
    if step % sharify_every_n_batches == 0:
        sharify_fn()
        if verbose:
            print "Done sharifying, step", step, time.time() - ttime

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
