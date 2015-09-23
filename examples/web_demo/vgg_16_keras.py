from keras.models import Sequential
from keras.layers.core import Flatten, Dense, Dropout
from keras.layers.convolutional import Convolution2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD
import cv2, numpy as np

def VGG_16(weights_path=None):
    model = Sequential()
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 3, 3, 3, activation='relu'))  # (224 + 2 - 3)/1 +1 = 244
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, 64, 3, 3, activation='relu'))  # 224
    model.add(MaxPooling2D((2,2), stride=(2,2)))  # 112

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 64, 3, 3, activation='relu'))  # (112+2-3)/1+1=122
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, 128, 3, 3, activation='relu'))  #112
    model.add(MaxPooling2D((2,2), stride=(2,2)))  #56

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 128, 3, 3, activation='relu'))  # 56
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 256, 3, 3, activation='relu'))  # 56
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, 256, 3, 3, activation='relu'))  # 56
    model.add(MaxPooling2D((2,2), stride=(2,2)))  #28

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 256, 3, 3, activation='relu'))  #28
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))  #28
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))  #28
    model.add(MaxPooling2D((2,2), stride=(2,2)))  #14

    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))  #14
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))  #14
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, 512, 3, 3, activation='relu'))  #14
    model.add(MaxPooling2D((2,2), stride=(2,2)))  #7

    model.add(Flatten())
    model.add(Dense(512*7*7, 4096, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(4096, 4096, activation='relu'))

    model.add(Dropout(0.5))
    model.add(Dense(4096, 1000, activation='softmax'))

    if weights_path:
        model.load_weights(weights_path)

    return model

if __name__ == "__main__":
    im = cv2.resize(cv2.imread('Cats.jpg'), (224, 224))
    im = im.transpose((2,0,1))
    im = np.expand_dims(im, axis=0)

    # Test pretrained model
    model = VGG_16('vgg16_weights.h5')
    sgd = SGD(lr=0.1, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(optimizer=sgd, loss='categorical_crossentropy')
    out = model.predict(im)
    print out
    print 'finish'