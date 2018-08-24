---
layout: post
title: Using Python generators in Keras
categories: Neural_Networks Keras Tutorial Caffe Python
tags: Neural Networks Keras Tutorial Caffe Python
comments: true
---

## What is a generator?

In Python, we define a generator as a function that returns a value each time we call its next() function (check [the Python wiki](https://wiki.python.org/moin/Generators)). To illustrate this, let me give you an example:

{% highlight python %}
def myGenerator():
	for i in range(5):
		yield i

generator = myGenerator()
for _ in range(5):
	print(generator.next())
{% endhighlight %}

This function is a generator that returns values from 0 to 4 each time we call its next() function. It uses the 'yield' reserved word to return the values. Whenever you have a 'yield' command in a function it automatically becomes a generator.

What makes it special? When next() is called the function executes a part of the code until the first yield. When the following next() is called the function doesn't start over again, it starts from the last yield called. Therefore, it wouldn't print a sequence of 0s like in a normal function, but the sequence from 0 to 4.

## Generators in Keras

What is the use of these generators in Keras? Well, Keras has various functions to train and evaluate a model, and also to make predictions (check the full list [here](https://keras.io/models/model/)). Among them we have the fit_, evaluate_ and predict_generator functions. They accept a generator as input instead of a list of data (like fit, evaluate and predict) or batches of data (like train_on_batch and test_on_batch). 

* We can read data from our computer without creating big loops with fit or train_on_batch functions (same for test purpouses).
* When training with lots of data we can get each batch inside the generator and yield it, becoming really transparent for the main function where we load the data, the model and call the training and test.
* Useful for data augmentation on training time. In fact, this method parallelizes the data augmentation in CPU with the training in GPU.

The fit_generator function also accepts a validation generator like the fit function, which is pretty nice.

A generator for these Keras functions has to loop indefinitely (with a *while True* statement, for example). Therefore, we have to specify when do we want to stop training/evaluating. This requires to specify the amount of data we will be using per epoch.

I will show you an easy example where we load images in a generator and we train our model with them. Suppose we use 100 images.

{% highlight python %}
def generator():
	files = getfiles()
	for file in files:
		yield imread(file)

nb_files = 100
myGenerator = generator()
history = model.fit_generator(generator=myGenerator, steps_per_epoch=nb_files, epochs=1)
{% endhighlight %}

Here we create the generator first. Then we instantiate it in the *generator* variable and we give it to the *fit_generator function. We also specify the amount of data per epoch (100) and the number of epochs. Now, lets add a validation generator to the example:

{% highlight python %}
def trainingGenerator():
	files = getfiles('training')
	for file in files:
		yield imread(file)

def validationGenerator():
	files = getfiles('validation')
	for file in files:
		yield imread(file)

nb_training_files = 100
nb_val_files = 10
myTrainingGenerator = trainingGenerator()
myValidationGenerator = validationGenerator()
history = model.fit_generator(generator=myTrainingGenerator, steps_per_epoch=nb_training_files, epochs=1, validation_data=myValidationGenerator, validation_steps=nb_val_files)
{% endhighlight %}

Then testing with the evaluate_generator is pretty much the same. I hope this tips may be useful to train neural networks with lots of data.
