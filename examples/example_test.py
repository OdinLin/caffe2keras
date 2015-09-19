import unittest
from keras.optimizers import SGD
import numpy as np
import os
import cv2

from keras.datasets.data_utils import get_file
from caffe2keras.caffe2keras import Model2Keras, Proto2Keras
import vgg_16_keras

prototxt = '../VGG_ILSVRC_16_layers_deploy.prototxt'
model_file = '../VGG_ILSVRC_16_layers.caffemodel'

model = vgg_16_keras.VGG_16()


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

