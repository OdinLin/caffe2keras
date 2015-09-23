import os
import sys
import h5py
from keras.layers.convolutional import ZeroPadding2D, Convolution2D, MaxPooling2D
from keras.layers.core import Dropout, Flatten, Activation, Dense
from keras.layers.normalization import LRN2D
from keras.layers.advanced_activations import LeakyReLU
import numpy as np
import warnings

import google.protobuf

from keras.models import Sequential

sys.path.append('/home/odin/Documents/caffe/python')

import caffe
import caffe.proto.caffe_pb2 as caffe_pb2


# The Keras equivalent to "xavier" initialization is "glorot_uniform". You may want to use this instead.
# Alternatively, you can also try "he_uniform" and "lecun_uniform".
# said by keras's creator from https://groups.google.com/forum/#!msg/keras-users/cEB-qJgNgRU/g9lKP99GFMMJ
# because default init is glorot_uniform aka xavier

debug = True

if sys.version_info < (3, 0, 0):
    _type_to_method = {}
    _oldname_to_method = {}


    def _layer(typ, oldname):
        def decorator(meth):
            global _type_to_method
            _type_to_method[typ] = meth
            typevalue = getattr(caffe_pb2.V1LayerParameter, oldname)
            _oldname_to_method[typevalue] = meth
            return meth

        return decorator

    available = True
else:
    available = False


class CaffeLoader(object):
    def __init__(self, model=None, prototxt_path=None, caffemodel_path=None):
        if not available:
            raise RuntimeError('CaffeFunction is not supported on Python 3')

        self.model = model
        self.prototxt_path = prototxt_path
        self.caffemodel_path = caffemodel_path

    def load(self):
        if not self.prototxt_path:
            raise RuntimeError('protxt file must be specified')

        if self.model is None:
            print 'model is None, so generate model first'
            self._load_prototxt()

        if self.caffemodel_path is not None:
            print 'loading weights now ...'
            self._load_caffe_params()

        return self.model

    def save(self, h5py_save_dir=None):
        if not self.caffemodel_path:
            raise RuntimeError('caffe model file must be specified')

        if h5py_save_dir is None:
            h5py_save_dir = 'weights.h5'

        if self.model is None:
            print 'model is None, so generate model first'
            self._load_prototxt()

        if self.caffemodel_path is not None:
            print 'loading weights now ...'
            self._save_weights(h5py_save_dir)

    def _save_weights(self, h5py_save_dir):

        net = caffe.Net(self.prototxt_path, self.prototxt_path, caffe.TEST)
        f = h5py.File(h5py_save_dir, 'w')

        params_index = 0
        params = net.params.values()
        f.attrs['nb_layers'] = len(self.model.layers)
        for i, layer in enumerate(self.model.layers):
            # print layer.get_config()['name']
            g = f.create_group('layer_{}'.format(i))
            if layer.get_config()['name'] == 'Convolution2D':
                g.attrs['nb_params'] = 2
                # w weights
                w_param_dset = g.create_dataset('param_0', params[params_index][0].data.shape,
                                                dtype=params[params_index][0].data.dtype)
                w_weights = params[params_index][0].data
                for w_i in xrange(w_weights.shape[0]):
                    for w_j in xrange(w_weights.shape[1]):
                        w_weights[w_i][w_j] = np.flipud(np.fliplr(w_weights[w_i][w_j]))
                w_param_dset[:] = w_weights
                # b weihts
                b_param_dset = g.create_dataset('param_1', params[params_index][1].data.shape,
                                                dtype=params[params_index][1].data.dtype)
                b_param_dset[:] = params[params_index][1].data

                params_index += 1
                print 'params_index: ', params_index
            elif layer.get_config()['name'] == 'Dense':
                g.attrs['nb_params'] = 2
                # w
                w_param_dset = g.create_dataset('param_0', params[params_index][0].data.T.shape,
                                                dtype=params[params_index][0].data.T.dtype)
                w_param_dset[:] = params[params_index][0].data.T
                # b
                b_param_dset = g.create_dataset('param_1', params[params_index][1].data.shape,
                                                dtype=params[params_index][1].data.dtype)
                b_param_dset[:] = params[params_index][1].data

                params_index += 1
                print 'params_index: ', params_index
            else:
                g.attrs['nb_params'] = 0

        f.flush()
        f.close()

    def _load_caffe_params(self):
        net = caffe.Net(self.prototxt_path, self.caffemodel_path, caffe.TEST)

        weights_layers = []
        params = net.params.values()
        for i, layer in enumerate(self.model.layers):
            if layer.get_config()['name'] in ['Convolution2D', 'Dense']:
                weights_layers.append(layer)

        if len(weights_layers) == len(params):
            for i in range(len(weights_layers)):
                print 'model shapes: ', params[i][0].data.shape, 'and', params[i][1].data.shape
                if weights_layers[i].get_config()['name'] is 'Dense':
                    # print 'weights: ', params[i][0].data.T[0][0]  # for debug
                    weights_layers[i].set_weights([params[i][0].data.T, params[i][1].data])
                elif weights_layers[i].get_config()['name'] is 'Convolution2D':
                    w_weights = params[i][0].data
                    for w_i in xrange(w_weights.shape[0]):
                        for w_j in xrange(w_weights.shape[1]):
                            w_weights[w_i][w_j] = np.flipud(np.fliplr(w_weights[w_i][w_j]))
                    weights_layers[i].set_weights([w_weights, params[i][1].data])

    def _load_prototxt(self):
        self.model = Sequential()
        net = caffe_pb2.NetParameter()

        google.protobuf.text_format.Merge(open(self.prototxt_path).read(), net)

        # self.stack_size = net.input_dim[1]

        if net.layer:  # v2
            self.layer_output_dim = [net.input_shape[0].dim[1], net.input_shape[0].dim[2], net.input_shape[0].dim[3]]
            for layer in net.layer:
                meth = _type_to_method.get(layer.type)
                if meth:
                    meth(self, layer)
                else:
                    warnings.warn(
                        'Skip the layer "%s", since CaffeFunction does not'
                        'support %s layer' % (layer.name, layer.type))
        else:  # v1
            self.layer_output_dim = [net.input_dim[1], net.input_dim[2], net.input_dim[3]]
            for layer in net.layers:
                meth = _oldname_to_method.get(layer.type)
                if meth:
                    meth(self, layer)
                else:
                    warnings.warn(
                        'Skip the layer "%s", since CaffeFunction does not'
                        'support it' % layer.name)

        return self.model

    @_layer('Convolution', 'CONVOLUTION')
    def _setup_convolution(self, layer):
        if layer.blobs:
            blobs = layer.blobs
            nb_filter, temp_stack_size, nb_col, nb_row = \
                blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width

            # model parallel network
            group = layer.convolution_param.group
            stack_size = temp_stack_size * group

            # maybe not all synapses are existant
            weights_p = np.zeros((nb_filter, stack_size, nb_col, nb_row))
            weights_b = np.array(blobs[1].data)

            chunk_data_size = len(blobs[0].data) // group
            stacks_size_per_chunk = stack_size // group
            nb_filter_per_chunk = nb_filter // group

            for i in range(group):
                chunk_weights = weights_p[i * nb_filter_per_chunk: (i + 1) * nb_filter_per_chunk,
                                i * stacks_size_per_chunk: (i + 1) * stacks_size_per_chunk, :, :]
                chunk_weights[:] = np.array(blobs[0].data[i * chunk_data_size:(i + 1) * chunk_data_size]).reshape(
                    chunk_weights.shape)

            weights = [weights_p.astype(dtype=np.float32), weights_b.astype(dtype=np.float32)]
        else:
            weights = None

        param = layer.convolution_param
        ksize = _get_ksize(param)
        stride = _get_stride(param)
        pad = _get_pad(param)
        nb_filter = param.num_output

        if pad[0] + pad[1] > 0:
            self.model.add(ZeroPadding2D(pad=(pad[0], pad[1])))
            if debug:
                print 'self.model.add(ZeroPadding2D(pad=({0}, {1})))'.format(pad[0], pad[1])

        self.model.add(Convolution2D(nb_filter, self.layer_output_dim[0], ksize[0], ksize[1],
                                     subsample=(stride[0], stride[1]), weights=weights))
        if debug:
            print 'self.model.add(Convolution2D({0}, {1}, {2}, {3}, subsample=({4}, {5}), weights={6}))'\
                .format(nb_filter, self.layer_output_dim[0], ksize[0], ksize[1], stride[0], stride[1], weights)
        self.layer_output_dim[0] = nb_filter
        self.layer_output_dim[1] = ((self.layer_output_dim[1] + 2 * pad[0]) - ksize[0]) / stride[0] + 1
        self.layer_output_dim[2] = ((self.layer_output_dim[2] + 2 * pad[1]) - ksize[1]) / stride[1] + 1

    @_layer('Pooling', 'POOLING')
    def _setup_pooling(self, layer):
        param = layer.pooling_param
        ksize = _get_ksize(param)
        stride = _get_stride(param)
        pad = _get_pad(param)

        if pad[0] + pad[1] > 0:
            self.model.add(ZeroPadding2D(pad=pad))
            if debug:
                print 'self.model.add(ZeroPadding2D(pad={0}))'.format(pad)

        if param.pool == param.MAX:
            self.model.add(MaxPooling2D(poolsize=ksize, stride=stride))
            if debug:
                print 'self.model.add(MaxPooling2D(poolsize={0}, stride={1}))'.format(ksize, stride)
        else:
            raise RuntimeError('Stochastic pooling is not supported')

        self.layer_output_dim[1] = ((self.layer_output_dim[1] + 2 * pad[0]) - ksize[0]) / stride[0] + 1
        self.layer_output_dim[2] = ((self.layer_output_dim[2] + 2 * pad[1]) - ksize[1]) / stride[1] + 1
        # print self.layer_output_dim

    @_layer('Dropout', 'DROPOUT')
    def _setup_dropout(self, layer):
        prob = layer.dropout_param.dropout_ratio
        self.model.add(Dropout(prob))
        if debug:
            print 'self.model.add(Dropout({0}))'.format(prob)

    @_layer('Flatten', 'FLATTEN')
    def _setup_flatten(self, layer):
        self.model.add(Flatten())
        if debug:
            print 'self.model.add(Flatten())'

    @_layer('LRN', 'LRN')
    def _setup_lrn(self, layer):
        param = layer.lrn_param
        if param.norm_region != param.ACROSS_CHANNELS:
            raise RuntimeError('Within-channel LRN is not supported')

        alpha = layer.lrn_param.alpha
        k = layer.lrn_param.k
        beta = layer.lrn_param.beta
        n = layer.lrn_param.local_size

        # or self.model.add(LRN2D(alpha=alpha/n, k=k, beta=beta, n=n)) ?
        self.model.add(LRN2D(alpha=alpha, k=k, beta=beta, n=n))
        # TODO: add output dim or not

    @_layer('ReLU', 'RELU')
    def _setup_relu(self, layer):
        slope = layer.relu_param.negative_slope

        if slope != 0:
            self.model.add(LeakyReLU(alpha=slope))
            if debug:
                print 'self.model.add(LeakyReLU(alpha={0}))'.format(slope)
        else:
            self.model.add(Activation('relu'))
            if debug:
                print "self.model.add(Activation('relu'))"

    @_layer('Softmax', 'SOFTMAX')
    @_layer('SoftmaxWithLoss', 'SOFTMAX_LOSS')
    def _setup_softmax_with_loss(self, layer):
        if layer.softmax_param.axis != 1:
            raise RuntimeError(
                'Softmax along non-channel axis is not supported')

        self.model.add(Activation('softmax'))
        if debug:
            print "self.model.add(Activation('softmax'))"

    @_layer('Split', 'SPLIT')
    def _setup_split(self, layer):
        self.model.add(Activation('linear'))
        # TODO: add output dim or not

    @_layer('Tanh', 'TANH')
    def _setup_tanh(self, layer):
        self.model.add(Activation('tanh'))

    @_layer('Sigmoid', 'SIGMOID')
    def _setup_sigmoid(self, layer):
        self.model.add(Activation('sigmoid'))

    @_layer('InnerProduct', 'INNER_PRODUCT')
    def _setup_inner_product(self, layer):
        if layer.blobs:
            blobs = layer.blobs
            nb_filter, stack_size, nb_col, nb_row = blobs[0].num, blobs[0].channels, blobs[0].height, blobs[0].width

            weights_p = np.array(blobs[0].data).reshape(nb_filter, stack_size, nb_col, nb_row)[0, 0, :, :].T
            weights_b = np.array(blobs[1].data)
            weights = [weights_p.astype(dtype=np.float32), weights_b.astype(dtype=np.float32)]
        else:
            weights = None

        param = layer.inner_product_param
        num_output = param.num_output

        if len(self.layer_output_dim) > 1:
            self.model.add(Flatten())
            self.model.add(Dense(np.prod(self.layer_output_dim), num_output, weights=weights))
            if debug:
                print 'self.model.add(Flatten())'
                print 'self.model.add(Dense({0}, {1}), weights={2}))'\
                    .format(np.prod(self.layer_output_dim), num_output, weights)
        else:
            self.model.add(Dense(np.prod(self.layer_output_dim), num_output))
            if debug:
                print 'self.model.add(Dense({0}, {1})))'\
                    .format(np.prod(self.layer_output_dim), num_output)

        self.layer_output_dim = [num_output]


def _get_ksize(param):
    if param.kernel_h > 0 and param.kernel_w > 0:
        return param.kernel_h, param.kernel_w
    else:
        return param.kernel_size, param.kernel_size


def _get_stride(param):
    if param.stride_h > 0 and param.stride_w > 0:
        return param.stride_h, param.stride_w
    else:
        return param.stride, param.stride


def _get_pad(param):
    if param.pad_h > 0 and param.pad_w > 0:
        return param.pad_h, param.pad_w
    else:
        return param.pad, param.pad
