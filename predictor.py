#!/usr/bin/env python3
import tensorflow as tf


#model_path = "/media/pszenowies/Prywatne/Python&Linux/python-scripts/tinder_bot/tf2-preview-inception_v3-feature_vector/new_model"
model_path = "/media/pszenowies/Prywatne/Python&Linux/python-scripts/tinder_bot/tf2-preview-mobilenet_v2-feature_vector/new_model"

face_model = tf.saved_model.load(model_path)
face_model.summary()

