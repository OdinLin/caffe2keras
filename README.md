# simple tool to translate caffe model to keras

It is inspired by:

- keras's caffe branch
- https://github.com/pfnet/chainer/blob/master/chainer/functions/caffe (functions and examples)
- https://gist.github.com/baraldilorenzo (VGG16 model for Keras)
- https://github.com/mitmul/chainer-imagenet-vgg

Example:

please browse the examples directory

## Dependencies:

- Python:
    - keras
    - google.protobuf

- caffe:
    - caffe's python lib (`caffe/python`)
     
## Some Problems:

- 1.The Proto2Keras can accept a caffe's prototxt and output a standed keras model(layer by layer translate).
 while the caffe model is VGG like, so we can't output this keras model and then use `load_caffe_params` funcation
 to load caffe model's params.It can only output model and train the data ourself.
 
- 2.Some function in `load_caffe_params` and `save_weights`  which I think is Inefficient.

- 3.The Proto2Keras is combine keras's caffe branch and chainer's caffe function, which some of them unfamiliar for me.
 So I just keep it.
 
- 4.The last and most import is I test this in a very very few example, and certainly there are a lot of bugs.
 So feel free to contact me(heyflypigATgmail.com), I am a newbie to DeepLearning.
