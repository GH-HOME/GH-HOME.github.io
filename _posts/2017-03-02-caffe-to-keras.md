---
layout: post
title: From Caffe to Keras - VGG16 example
categories: Neural_Networks Keras Tutorial VGG16 Caffe
tags: Neural Networks Keras Tutorial VGG16 Caffe
comments: true
---

Caffe is really famous due to its incredible collection of pretrained model called [ModelZoo](https://github.com/BVLC/caffe/wiki/Model-Zoo). Keras has also some [pretrained models](https://keras.io/applications/) in Imagenet: Xception, VGG16, VGG19, ResNet50 and InceptionV3. However, it would be awesome to add the ModelZoo pretrained networks to Keras. In this tutorial I will explain my personal solution to this problem without using any other tool, just using Caffe, Keras and Python. BTW, I use the Theano background in this tutorial.

## First step: getting the weights

You can extract the weights in various formats from Caffe, I selected HDF5 format because it's easy to use and has the HDF5View tool to visualize what are you actually saving. We need Caffe and PyCaffe installed. Use the following code in Python to create first the Caffe model and then to save the weights as an HDF5 file:

{% highlight python %}
netname = 'vgg16.prototxt'
paramname = 'vgg16.caffemodel'
net = caffe.Net(netname, paramname, caffe.TEST)
net.save_hdf5('/home/adrian/project/caffedata.h5')
{% endhighlight %}

## Second step: create the model

This is the easy step: I just downloaded the model from Keras and adapted it: I added the ZeroPadding2D that was missing, I changed the 'border_mode' parameter to 'valid' instead of 'same' and I added the 2 Dropout between the fully connected layers.

## Third step: copying the weights to the model

This is where things get complicated. Both Caffe and Theano have some differences, so we have to make some tweaks in the weights. I create the VGG16 model first:

{% highlight python %}
channels, width, height = ..., ..., ...
data = Input(shape=(channels, width, height), dtype='float32', name='input')
vgg16 = VGG16(weights=None, include_top=True, input_tensor=data, classes=101)
x = vgg16.output
{% endhighlight %}

I have no interest in the weights so I use None. Make sure you include the top (fully connected layers), that you specify if needed the new input shape by passing a value to the input_tensor variable and the amount of classes if you are in a different problem. To copy the weights this is code is used:

{% highlight python %}
layerskeras = ['block1_conv1', 'block1_conv2', 'block2_conv1', 'block2_conv2', 'block3_conv1', 'block3_conv2', 'block3_conv3', 'block4_conv1', 'block4_conv2', 'block4_conv3', 'block5_conv1', 'block5_conv2', 'block5_conv3', 'fc1', 'fc2', 'predictions']
layerscaffe = ['conv1_1', 'conv1_2', 'conv2_1', 'conv2_2', 'conv3_1', 'conv3_2', 'conv3_3', 'conv4_1', 'conv4_2', 'conv4_3', 'conv5_1', 'conv5_2', 'conv5_3', 'fc6', 'fc7', 'fc8']

i = 0
h5 = h5py.File('/home/anunez/project/caffedata.h5')

for layer in layerscaffe[:-3]:
	w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']
	w2 = np.transpose(w2, (0,1,2,3))
	w2 = w2[:, :, ::-1, ::-1]
	b2 = np.asarray(b2)
	model.get_layer(layerskeras[i]).W.set_value(w2)
	model.get_layer(layerskeras[i]).b.set_value(b2)
	i += 1

for layer in layerscaffe[-3:]:
	w2, b2 = h5['data'][layer]['0'], h5['data'][layer]['1']      
	w2 = np.transpose(w2,(1,0))
	b2 = np.asarray(b2)
	model.get_layer(layerskeras[i]).W.set_value(w2)
	model.get_layer(layerskeras[i]).b.set_value(b2)
        i += 1
{% endhighlight %}

I will explain this code little by little. In general:

* First, I have 2 arrays of layer names. If you change the layer names in the VGG16 of Keras to adapt it to Caffe's names you can skip this.
* Then, we will do 2 loops: one for the convolutional layers and the other for the fully connected layers.

The first loop goes loading the weights we saved in the 'caffedata.h5' file into the variables w2 and b2 (weights and biases, respectively). First important change: remember the image dimensions ordering: we have to use a transpose operation of the numpy module (previously called permute) to get the correct permutation of dimensions. We also need to flip the values of the dimensions width and height. I found this explanation in this [post](https://gab41.lab41.org/taking-keras-to-the-zoo-9a76243152cb#.es7mhsx34). Check it out for more details. 

Next, we cast b2 to a numpy array so that both w2 and b2 have the same format. To set this weights we use the 'get_layer' function of the model (with the name in Keras), get the parameter (W or b) and use 'set_value' to set our weights.

Now we only need to do more or less the same for the fully connected layer. In this case, we only need to swap the dimensions of w2 (remember that it only has 2 dimensions) and set them like in the previous loop.

To make it work remember that you have to compile your model!
