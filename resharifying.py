

from pylearn2.packaged_dependencies.theano_linear.unshared_conv.unshared_conv import FilterActs

"""
The shapes:
    inputs: groups, channels_per_group, nrows, ncols, batch_size
    filters: nfilters_per_row, nfilters_per_col, nchannels_per_filter, nrows_per_filter, ncolumns_per_filter, groups, nfilters_per_group
"""

locally_connected = FilterActs(1)
#outputs = locally_connected(inputs, filters)

def resharify(filters):
    return T.tile(T.mean(filters, (0,1)), (filters.shape[0], filters.shape[1], 1,1,1,1,1), ndim=7)

resharifying_updates = {filts:resharify(filts) for filts in filters}



#T.matrix()



#############################
resharify_every_n_batches = 1


