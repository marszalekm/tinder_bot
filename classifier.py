#!/usr/bin/env python3
import os


os.system("make_image_classifier " +
  "--image_dir training_images " +
  #"--tfhub_module https://tfhub.dev/google/tf2-preview/mobilenet_v2/feature_vector/4 " +
  #"--tfhub_module https://tfhub.dev/google/tf2-preview/inception_v3/feature_vector/4 " +
  "--tfhub_module https://tfhub.dev/tensorflow/resnet_50/feature_vector/1 " +
  "--image_size 224 " +
  "--saved_model_dir  resnet_50-feature_vector/new_model " +
  "--labels_output_file resnet_50-feature_vector/class_labels.txt " +
  "--tflite_output_file resnet_50-feature_vector/new_mobile_model.tflite")
