import unittest
from keras.optimizers import SGD
import numpy as np
import os
import cv2

# from keras.datasets.data_utils import get_file
from caffe2keras.caffe2keras import Model2Keras
from caffe2keras.caffe2keras import Proto2Keras
from caffe2keras.caffeloader import CaffeLoader
import vgg_16_keras

prototxt = '/mnt/share/projects/keras_test/chainer-imagenet-vgg-master/VGG_ILSVRC_16_layers_deploy.prototxt'
model_file = '/mnt/share/projects/keras_test/chainer-imagenet-vgg-master/VGG_ILSVRC_16_layers.caffemodel'

prototxt_19 = '/mnt/share/projects/keras_test/chainer-imagenet-vgg-master/VGG_ILSVRC_19_layers_deploy.prototxt'
model_file_19 = '/mnt/share/projects/keras_test/chainer-imagenet-vgg-master/VGG_ILSVRC_19_layers.caffemodel'

model = vgg_16_keras.VGG_16()


class TestCaffeLoader(unittest.TestCase):
    def old_fun_use(self):
        p2k = Proto2Keras(prototxt)
        keras_model = p2k.model
        m2k = Model2Keras(keras_model, prototxt, model_file)
        m2k.load_caffe_params()
        im = cv2.resize(cv2.imread('Cats.jpg'), (224, 224))
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)

        # Test pretrained model
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        keras_model.compile(optimizer=sgd, loss='categorical_crossentropy')

        out = keras_model.predict(im)
        top5 = np.argsort(out)[0][::-1][:5]
        probs = np.sort(out)[0][::-1][:5]
        print 'yes'
        words = open('synset_words.txt').readlines()
        words = [(w[0], ' '.join(w[1:])) for w in [w.split() for w in words]]
        words = np.asarray(words)

        for w, p in zip(words[top5], probs):
            print('{}\tprobability:{}'.format(w, p))

    def new_fun_use(self):
        print 'new_fun_use'
        cl = CaffeLoader(prototxt_path=prototxt_19, caffemodel_path=model_file_19)
        model = cl.load()
        im = cv2.resize(cv2.imread('Cats.jpg'), (224, 224))
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)

        # Test pretrained model
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        out = model.predict(im)
        top5 = np.argsort(out)[0][::-1][:5]
        probs = np.sort(out)[0][::-1][:5]
        print 'yes'
        words = open('synset_words.txt').readlines()
        words = [(w[0], ' '.join(w[1:])) for w in [w.split() for w in words]]
        words = np.asarray(words)

        for w, p in zip(words[top5], probs):
            print('{}\tprobability:{}'.format(w, p))


class TestModel2Keras(unittest.TestCase):
    m2k = Model2Keras(model, prototxt, model_file)

    def test_load_caffe_params(self):
        self.m2k.load_caffe_params()
        # test part
        im = cv2.resize(cv2.imread('Cats.jpg'), (224, 224))
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)

        # Test pretrained model
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        out = model.predict(im)
        top5 = np.argsort(out)[0][::-1][:5]
        probs = np.sort(out)[0][::-1][:5]
        print 'yes'
        words = open('synset_words.txt').readlines()
        words = [(w[0], ' '.join(w[1:])) for w in [w.split() for w in words]]
        words = np.asarray(words)

        for w, p in zip(words[top5], probs):
            print('{}\tprobability:{}'.format(w, p))

    def test_save_weights(self):
        self.m2k.save_weights('cutom_weights.h5')
        print 'done'


class TestProto2Keras(unittest.TestCase):
    p2k = Proto2Keras('../protos/VGG_ILSVRC_16_layers_deploy.prototxt')

    def test_proto_2_keras(self):
        model = self.p2k.model
        sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(loss='categorical_crossentropy', optimizer=sgd)
        # custom fit
        # model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, show_accuracy=True, callbacks=[remote])

        # test part
        im = cv2.resize(cv2.imread('Cats.jpg'), (224, 224))
        im = im.transpose((2, 0, 1))
        im = np.expand_dims(im, axis=0)

        # Test pretrained model
        sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
        model.compile(optimizer=sgd, loss='categorical_crossentropy')

        out = model.predict(im)
        top5 = np.argsort(out)[0][::-1][:5]
        probs = np.sort(out)[0][::-1][:5]
        print 'yes'
        words = open('synset_words.txt').readlines()
        words = [(w[0], ' '.join(w[1:])) for w in [w.split() for w in words]]
        words = np.asarray(words)

        for w, p in zip(words[top5], probs):
            print('{}\tprobability:{}'.format(w, p))


if __name__ == '__main__':
    print('Test caffe model conversion')
    unittest.main()

