---
layout: post
title: Tensorflow Object Detection API - Finetuning with your own dataset
categories: Tensorflow Tutorial Python Object Detection
tags: Tensorflow Tutorial Python Object Detection
comments: true
---

Note: I'm using Ubuntu 16.04 and Tensorflow-GPU 1.6.0 installed via pip for this tutorial.

Tensorflow has its own Object Detection API with tutorials and a ModelZoo, you can find it [here](https://github.com/tensorflow/models/tree/master/research/object_detection). With so much documentation it can be difficult to actually get your model working on your own dataset, so I will try to summarize my experience using it. The tutorial will by composed of the following parts:

1. Installing the Object Detection API
2. Preparing the dataset
3. Convert data to TFRecord format
4. Prepare the directory structure
5. Finetune the network with your dataset
6. Export the finetuned graph

## 1. Installing the Object Detection API

You can download the API with the command:

{% highlight ruby %}
git clone ttps://github.com/tensorflow/models.git
{% endhighlight %}

You will also need to install the API, follow the steps presented [HERE](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/installation.md). If you pass the test provided in the *object_detection/builders/model_builder_test.py* script you can continue to the next step.

## 2. Preparing the dataset

#### (Optional) Labelling your dataset

In my case, I had a dataset without any bounding box annotation, so I used the [labelImg](https://github.com/tzutalin/labelImg) tool to create easily bounding box annotations in VOC Pascal format. The tool is very easy to use and comfortable. Remember to save in Pascal VOC the labels (.xml format), not in YOLO format. I try to keep two separate folder structures: one for images and other one for annotations. It is important that you can easily find correspondences between images and annotations.

### 3. Convert data to TFRecord format

The Tensorflow Object Detection API requires the use of the TFRecord formatting of the data. The repository actually provides a script to transform your data format into TFRecord, but you have to extract by yourself the data (bounding box annotation, class of the bounding boxes...) inside the script.

You will also need a label_map file, which maps the name of a class with an id. An example of the content for a classification of dogs and cats could be:

{% highlight ruby %}
item {
  id: 1
  name: 'cat'
  display_name: 'Cat'
}
item {
  id: 2
  name: 'dog'
  display_name: 'Dog'
}
{% endhighlight %}

You can add all the labels you require, but remember that the first id must be 1. If you put a 0 that class will be discarded.

### 4. Prepare the directory structure

The recommended folder structure by the Tensorflow Object Detection API developers for training and validation is the following:

{% highlight ruby %}
+data
  -label_map file 
  -train TFRecord file
  -eval TFRecord file 
+models
  + model
    -pipeline config file 
    +train 
    +eval 
{% endhighlight %}

where '+' is a folder and '-' is a file. 

* label_map file: mapping from class name to id, explained in Section 3.1.
* train/test TFRecord file: training/test set in TFRecord format, you should obtain these with the script to convert your dataset to TFRecord format.
* pipeline config file: a pipeline.config file should already be inside the folder of the model you download from the ModelZoo. You just need to change some paths.
* train/val folders: empty folders to store checkpoints from training and evaluation.

## 5. Finetune the network with your dataset

First pick the model you want to finetune from the [Object Detection Modelzoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and download it. I used the Faster-RCNN with a Resnet101 pretrained in COCO. In this tutorial we will not be using segmentation, just object detection, so Mask-RCNN is out of the scope of it. Store all the files in the 'models/model' folder.

#### 5.1. Tweaking of parameters

You can edit some parameters and configuration options in the 'models/model/pipeline.config' file. You will need to modify at least:
* num_classes: specify the number of classes/objects in your dataset.
* label_map_path: the path to your label map file, it should be in the 'data' folder. This parameter appears twice.
* input_path: you have two of these variable, one under the 'train_input_reader' part and another under 'eval_input_reader'. You need to specify the path to the TFRecord files of the training set and the evaluation set. The second one is only necessary to evaluate your model with the evaluation script provided in the API. Tese files should be in the 'data' folder too.

#### 5.2. Training

To train the model we have just configured go to 'models/research' and from there use the following command in the terminal:

{% highlight ruby %}
python object_detection/train.py \
    --pipeline_config_path=../model/pipeline.config \
    --train_dir=../model/train
{% endhighlight %}

You have to specify the path to the pipeline.config file we have configured in Section 5.1 and the empty 'train' folder created in Section 4. It takes some time to start. The loss per iteration will be shown in the terminal. Some checkpoints are stored in the 'train' folder after some time.

#### 5.2. Runing Tensorboard during training time

You can run Tensorboard to see the training Mean Average Precision (mAP), the Average Recall (AR) and even visualizations of the bounding boxes predicted by the network in the training set using:

{% highlight ruby %}
tensorboard --logdir=train/
{% endhighlight %}

Note that the folder train has been created in Section 4 and it should be inside the path 'models/model'. If you have the error

{% highlight ruby %}
ImportError: cannot import name run_main
{% endhighlight %}

when you run Tensorboard install the following:

{% highlight ruby %}
pip install tb-nightly
{% endhighlight %}

#### 5.3. Evaluate your model while you train it

The loss that you get in the terminal while fine-tuning the model is not really helpful. Instead, you can evaluate the model in a separate evaluation set. Remember that you should have a separate 'test/eval.record' file under the 'data' folder and an 'eval' folder inside 'models/model'. To be able to evaluate while you train first take into account that, by default, Tensorflow uses all the GPU memory. We need to add one line of code in the 'models/research/object_detection/trainer.py' script. Go to [this line](https://github.com/tensorflow/models/blob/master/research/object_detection/trainer.py#L357) and add:

{% highlight ruby %}
session_config.gpu_options.allow_growth=True
{% endhighlight %}

With this you should be able to launch the training script 'train.py' and the evaluation script 'eval.py' at the same time. Wait until the training starts printing the iterations' loss before using the evaluation script. To execute the last one use the following command from 'models/research':

{% highlight ruby %}
python object_detection/eval.py \
        --checkpoint_dir=../model/train \
        --eval_dir=../model/eval \
        --pipeline_config_path=../model/pipeline.config
{% endhighlight %}

The parameter 'checkpoint_dir' points to the directory where we store the checkpoints generated while training. 'eval_dir' contains the empty folder created in Section 4 and the 'pipeline.config' file is the same as the one used in training. In the terminal you will see a summary of the mAP and AR. You can also launch tensorboard to visualize the predictions done over the images in the test/evaluation set (of the 'data/test.record' file).

{% highlight ruby %}
tensorboard --logdir=eval/
{% endhighlight %}

## 6. Export the finetuned graph

After the model is fine-tuned you can freeze and export it to use it in your code. First to freeze it use the following command from the 'models/research' path.

{% highlight ruby %}
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path ../model/pipeline.config --trained_checkpoint_prefix ../model/train/model.ckpt-xxxx --output_directory ../model/fine_tuned_model/
{% endhighlight %}

where 'model.ckpt-xxxx' is a file stored in the 'models/model/train' folder of our directory structure and 'fine_tuned_model' is a folder that you need to create under 'data' to store our checkpoint. You can pick the checkpoint you like and replace the 'xxxx' with the iteration number (it must exist under 'models/model/train'). Now to load this checkpoint in Python and also load the important tensorflow ops of the network use the following script:

{% highlight python %}
import cv2
import tensorflow as tf
import numpy as np

model_path = model_path = '../model/fine_tuned_model/saved_model/'
with tf.Session(graph=tf.Graph()) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING], model_path) 

    detection_graph = tf.get_default_graph()
    input_tensor = detection_graph.get_tensor_by_name('image_tensor:0')
    predict_class_op = detection_graph.get_tensor_by_name('detection_classes:0')
    num_detections_op = detection_graph.get_tensor_by_name('num_detections:0')
    detection_boxes_op = detection_graph.get_tensor_by_name('detection_boxes:0')
    detection_scores_op = detection_graph.get_tensor_by_name('detection_scores:0')

    image = load_image(path_to_image)
    feed_dict = {input_tensor: np.expand_dims(image,0)}
    classes = sess.run(predict_class_op, feed_dict)
{% endhighlight %}

In this example we obtain the classes predicted by the network for a given image. Notice that you will obtain an array of integers. Each integer is the ID number of the class specified in the label map file. This means that the first class is 1, you need to subtract a 1 from the class in order to start from 0.

I have noticed that using SciPy or OpenCV functions to load the images makes the network to have a really bad performance. Therefore our 'load_image' function should be implemented in the following way:

{% highlight python %}
def load_image(sess, image_path):
	image_file = open(image_path,'rb')
	image_raw = image_file.read()

	image_tf = tf.image.decode_jpeg(image_raw).eval(session=sess)
	image_tf_accurate = tf.image.decode_jpeg(image_raw,dct_method="INTEGER_ACCURATE").eval(session=sess)
{% endhighlight %}

Note that if you do not pass the session as argument and you create another session your 'load_image' function will be extremely slow.

{% highlight python %}
def load_image(sess, image_path):
	image_file = open(image_path,'rb')
	image_raw = image_file.read()

	image_tf = tf.image.decode_jpeg(image_raw).eval(session=sess)
	image_tf_accurate = tf.image.decode_jpeg(image_raw,dct_method="INTEGER_ACCURATE").eval(session=sess)
{% endhighlight %}

#### 6.1. Visualize the predicted Bounding Boxes

There is an utility to visualize the bounding boxes in PASCAL VOC format.

{% highlight python %}
import object_detection.utils.visualization_utils as vis_util 
label_map_path = 'data/gaze+_label_map.pbtxt'
nb_objects = ...

label_map = label_map_util.load_labelmap(label_map_path)
categories = label_map_util.convert_label_map_to_categories(label_map=label_map, max_num_classes=nb_objects)
category_index = label_map_util.create_category_index(categories)

...

classes = sess.run(predict_class_op,feed_dict)[0]
num_detections = sess.run(num_detections_op,feed_dict)[0]
bounding_boxes = sess.run(detection_boxes_op,feed_dict)[0]
scores = sess.run(detection_scores_op,feed_dict)[0]

...

vis_util.visualize_boxes_and_labels_on_image_array(
	img,
	bounding_boxes,
	np.squeeze(classes).astype(np.int32),
	scores,
	category_index,
	use_normalized_coordinates=False,
	line_thickness=4,
	min_score_thresh=0.5)
{% endhighlight %}

This function requires an image 'img', the output from the different ops (classes, bounding_boxes and scores) and a category_index object (this object is created using the label map file in the 'data' folder). You can adjust the thickness of the bounding boxes with the 'line_thickness' and also limit the bounding boxes that appear only to those with a score higher than 'min_score_thresh'.
