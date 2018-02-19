import tensorflow as tf
from tensorflow.contrib import slim
from tensorflow.python.ops import sparse_ops

from tensorflow.contrib.slim.python.slim.nets import inception

from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model.signature_def_utils import build_signature_def
from tensorflow.python.saved_model import utils

import os
from os import listdir
from os.path import isfile, join

slim = tf.contrib.slim

num_classes = 6012
checkpoint = 'data/2016_08/model.ckpt'


def process_images(serialized_images):
    def decode(jpeg_str, central_fraction=0.875, image_size=299):
        decoded = tf.cast(
                tf.image.decode_jpeg(jpeg_str, channels=3), tf.float32)
        cropped = tf.image.central_crop(
                decoded, central_fraction=central_fraction)
        resized = tf.squeeze(
                tf.image.resize_bilinear(
                        tf.expand_dims(cropped, [0]), [image_size, image_size],
                        align_corners=False), [0])
        resized.set_shape((image_size, image_size, 3))
        normalized = tf.subtract(tf.multiply(resized, 1.0 / 127.5), 1.0)

        return normalized

    def process(images, image_size=299):
        images = tf.map_fn(decode, images, dtype=tf.float32)

        return images

    images = process(serialized_images)

    with slim.arg_scope(inception.inception_v3_arg_scope()):
        logits, end_points = inception.inception_v3(
                images, num_classes=num_classes, is_training=False)

    features = tf.reshape(end_points['PreLogits'], [-1, 2048])
    class_predictions = tf.nn.sigmoid(logits)

    return features, class_predictions


serialized_images = tf.placeholder(tf.string, shape=[None], name="images")

features, predictions = process_images(serialized_images)

pred, indices = tf.nn.top_k(predictions, k=5)

features = tf.identity(features, name="features")

labels = tf.constant("labels.txt")
tf.add_to_collection(tf.GraphKeys.ASSET_FILEPATHS, labels)
table = tf.contrib.lookup.index_to_string_table_from_file(
        vocabulary_file=labels, default_value="UNKNOWN")

values = table.lookup(tf.to_int64(indices))

preddy = tf.identity(values, name="predictions")

saver = tf.train.Saver()
with tf.Session() as sess:
    sess.run([
            tf.local_variables_initializer(),
            tf.global_variables_initializer(),
            tf.tables_initializer()
    ])
    #tf.initialize_all_tables().run()
    saver.restore(sess, checkpoint)

    # this part specified the saved model
    feature_sig = build_signature_def(
            {
                    'image': utils.build_tensor_info(serialized_images)
            }, {'features': utils.build_tensor_info(features)}, 'features')
    # this part specified the saved model
    name_sig = build_signature_def(
            {
                    'image': utils.build_tensor_info(serialized_images)
            }, {'names': utils.build_tensor_info(preddy)}, 'names')

    builder = saved_model_builder.SavedModelBuilder('savedmodel')
    builder.add_meta_graph_and_variables(
            sess, [tf.saved_model.tag_constants.SERVING],
            signature_def_map={'features': feature_sig,
                               'names': name_sig},
            assets_collection=tf.get_collection(tf.GraphKeys.ASSET_FILEPATHS),
            legacy_init_op=tf.tables_initializer())
    builder.save()

# Here we load the saved graph as a sanity check
graph = tf.Graph()
with tf.Session(graph=graph) as sess:
    tf.saved_model.loader.load(sess, [tf.saved_model.tag_constants.SERVING],
                               'savedmodel')
    with open('img.jpg', 'rb') as f:
        res = sess.run(
                [
                        graph.get_tensor_by_name('features:0'),
                        graph.get_tensor_by_name('predictions:0')
                ],
                feed_dict={
                        graph.get_tensor_by_name('images:0'): [f.read()]
                })
        print(res)
