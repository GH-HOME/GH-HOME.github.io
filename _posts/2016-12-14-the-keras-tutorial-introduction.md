---
layout: post
title: The Keras Tutorial - Introduction
categories: Neural_Networks Keras Tutorial
tags: Neural Networks Keras Tutorial
comments: true
---

[Keras](https://keras.io/) is a high-level neural networks library written in Python and built on top of [Theano](http://deeplearning.net/software/theano/) or [Tensorflow](https://www.tensorflow.org/). That means you need one of them as a backend for Keras to work. I have been working with Neural Networks for a while, I have tried  [Caffe](http://caffe.berkeleyvision.org/), Tensorflow and [Torch](http://torch.ch/) and now I'm working with Keras. Its main advantage is the minimalism of the code as it allows the creation of big networks with a few lines of code. It allows multi-input and multi-ouput networks, convolutional and recurrent neural networks, embeddings, etc. I'm really comfortable working with it so I thought it would be nice to write some paragraphs to explain how it works and the tricks I found. I hope this tutorial is helpful for new users.

### Index

* Download and Install Keras
* Backend
* Our first Neural Network
* The basic layers

## Download and Install Keras

Before we actually install Keras let's make our life easier and install the pip program.

{% highlight ruby %}
sudo apt-get install python-pip
{% endhighlight %}

Now we can easily install the dependencies:

{% highlight ruby %}
pip install numpy scipy scikit-learn pillow h5py
{% endhighlight %}

Next we install Keras:

{% highlight ruby %}
pip install keras
{% endhighlight %}

You can check the version of Keras you are using by typing the following command in the terminal:

{% highlight ruby %}
python -c "import keras; print keras.__version__"
{% endhighlight %} 

Moreover, you can upgrade Keras with the following command:

{% highlight ruby %}
sudo pip install --upgrade keras
{% endhighlight %}

## Backend

As I mentioned earlier, Keras works on top of Tensorflow (by default) or Theano. I tend to use Theano as my backend. To specify which [backend](https://keras.io/backend/) we want to use we have to edit the keras.json file:

{% highlight ruby %}
nano ~/.keras/keras.json
{% endhighlight %}

Example of the content of my keras.json file using Theano:

{% highlight ruby %}
{
    "image_dim_ordering": "th", 
    "epsilon": 1e-07, 
    "floatx": "float32", 
    "backend": "theano"
}
{% endhighlight %}

In the 'backend' variable we can specify 'theano' or 'tensorflow'. The 'image_dim_ordering' ('th' for theano and 'tf' for tensorflow) is used to specify the arrangement of the dimensions in images, i.e. in Theano we use the order (channels, width, height) whereas in Tensorflow the order is (width, height, channels). In the case of theano, you have to create a file '~/.theanorc':

{% highlight ruby %}
touch ~/.theanorc
{% endhighlight %}

And include the following:

{% highlight ruby %}
[global]
floatX = float32
device = gpu
optimizer = fast_run
 
[lib]
cnmem = 0.9
 
[nvcc]
fastmath = True
 
[blas]
ldflags = -llapack -lblas
{% endhighlight %}

If you want to use the CPU change 'gpu0' to 'cpu'. With this final steps we should have keras ready to work with Theano.

## Our first Neural Network

First I will say that Keras has two different structures to create Neural Networks: the Sequential Model and the Functional API. We will work with the second one through the tutorial as it allows more freedom. If you have heard about the Graph Model I have to say that it was removed (therefore it's now deprecated) in the version 1.0 ([link](https://github.com/fchollet/keras/issues/2802#issuecomment-221314411)).

So let's start coding! Our first neural network is going to have a single input and a final dense layer (which will be the output). The example code can also be shown in the [Keras Models section](https://keras.io/models/model/) but we will go through each of the lines to understand better what we are doing.

{% highlight python %}
from keras.models import Model
from keras.layers import Input, Dense

a = Input(shape=(32,))
b = Dense(32)(a)
model = Model(input=a, output=b)
{% endhighlight %}

The first two lines are the imports. No need for explanation, we need to import all the layers, optimizers, functions for the initilisation of layers, etc. we want to use.

The Functional API forces you to include an input layer. Here you have to specify the shape of your input (without the batch size). In this example we only have a 1D input of 32 values. The comma after the value 32 is also mandatory in some cases to avoid errors. Your input layer is stored in the variable 'a', so you can use this variable as input to other variables to create links (this allows the network to branch out easily).

Now we include a Dense layer by calling the function and providing the number of neurons for that layer (in this case 32). The next step is to stack the output layer or dense layer on top of the input layer, i.e., we have to connect them. In this type of model we do this by providing the variable of the last layer at the end of the new layer (check the 'a' between parenthesis after the Dense layer).

Finally, we have to instantiate the Model or create a container for it. We can do this with the Model function, providing the input and output. In this case providing the variables of the input and output layers. We can also provide a Python list for multi-input and output.

{% highlight python %}
a = Input(shape=(32,))
b = Input(shape=(32,))
...
z = merge([a,b], mode='concat', concat_axis = -1)
...
c = Dense(32)(z)
model = Model(input=[a, b], output=c)
{% endhighlight %}

This is an example of a multi-input neural network, in this case a and b. At some point in the code we have to fuse both of them into one stream, which we cal z. Finally, we will output c.

Anyway, now that the model is created we have to configure it. To do so we have to call the function 'compile' of the model:

{% highlight python %}
compile(self, optimizer, loss, metrics=None, loss_weights=None, sample_weight_mode=None)
{% endhighlight %}

We have to pass an optimizer and a loss function at least. You can check the [optimizers section](https://keras.io/optimizers/) and the [loss, objective or cost function section](https://keras.io/objectives/) of the webpage of Keras for information about the available functions.

Once we have the model ready is time to actually train it. This can be done in various ways, we explain the simple one: using the 'fit' function of the model:

{% highlight python %}
fit(self, x, y, batch_size=32, nb_epoch=10, verbose=1, callbacks=None, validation_split=0.0, validation_data=None, shuffle=True, class_weight=None, sample_weight=None, initial_epoch=0)
{% endhighlight %}

This function expects at least the input data X (which can be a list if it's a muti-input network), the true labels Y (again, it can be an array). You can also specify the batch size, number of epochs, the validation split (between 0 and 1, percentage of the training data that will be used for validation) or validation data (a pair of validation data and true labels), etc.

Finally, you can evaluate your trained model with the 'evaluate' function:

{% highlight python %}
evaluate(self, x, y, batch_size=32, verbose=1, sample_weight=None)
{% endhighlight %}

You have to provide it with test data or data that has not been used during training. The accuracy provided by this process measures the real quality of your model.

And this concludes our first neural network tutorial! I will write now about some basic stuff about Neural Networks: layers, initialisation, etc. I hope it's clear enough. For any question or suggestions please use the comments section below.

## The basic layers

* Input layer: The input layer specifies the shape of the input. This replaces the old and mandatory parameter 'input_shape' that had to be added to the first layer of the network. The parameter shape expects the shape without the batch size. The final comma is also necessary.

{% highlight python %}
keras.layers.Input(shape, batch_shape, dtype, name)
{% endhighlight %}

* Dense layer: The Dense layer or Fully-connected layer is the most basic Neural Network layer composed by a number of neurons specified by the parameter output_dim (greater than 0).

{% highlight python %}
keras.layers.core.Dense(output_dim, init='glorot_uniform', activation=None, weights=None, W_regularizer=None, b_regularizer=None, activity_regularizer=None, W_constraint=None, b_constraint=None, bias=True, input_dim=None)
{% endhighlight %}

* Activation (layer): Not really a layer. It applies an activation function (which must be included as a parameter) to the previous layer's neurons. Its use can be avoided by specifying the parameter 'activation' in the previous layer. Anyway, it comes in handy when you want to apply a function between the linear and non-linear operation. You can check in [the code](https://github.com/fchollet/keras/blob/master/keras/activations.py) the activation functions included in Keras. These are available: softmax, elu, softplus, softsign, relu, tanh, sigmoid, hard_sigmoid and linear. Note that the linear function is the identity function as, by default, the previous layer applies the linear activation function by default.

{% highlight python %}
keras.layers.core.Activation(activation)
{% endhighlight %}

* Flatten layer: Not really a layer. It's a reshape operation to transform a nD array into a 2D array with shape (batch size, features). This operation is usually followed by a Dense layer.

{% highlight python %}
keras.layers.core.Flatten()
{% endhighlight %}

* Merge layer: The merge layer fuses various tensors into a single one. This is useful to create multiple streams (each one with its own input) and then merge them. The first argument is a list of the tensors, the mode specifies how to merge them (concatenation, sum...) and the concat_axis tells the layer which axis to pick to make the concatenation (-1 by default).

{% highlight python %}
keras.layers.merge(layers, mode, concat_axis)
{% endhighlight %}

* Dropout layer: Not really a layer. From the paper ['Dropout: A Simple Way to Prevent Neural Networks from Overfitting, 2014'](http://www.jmlr.org/papers/volume15/srivastava14a/srivastava14a.pdf) The dropout operation is a way of preventing overfitting and hence improving the generalisation. During training time, it drops with a probability p (given as a parameter) the previous layer's neurons, i.e., their activations become 0.

{% highlight python %}
keras.layers.core.Dropout(p)
{% endhighlight %}

## Initialisation

The initialisation of a neural network (non-convex function) is important as we try to make it converge. We should initialise all neurons with a specific method. The [initialisation section](https://keras.io/initializations/) in the webpage of Keras provides a list of all the initialisation options implemented in Keras. First, you have to import them:

{% highlight python %}
from keras.initializations import uniform
{% endhighlight %}

An example of importing the uniform initialisation function. These functions are given as the value of the parameter 'init' that some Keras layers have, e.g., the Dense layer.
