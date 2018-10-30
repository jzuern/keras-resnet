# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Cifar example using Keras for model definition."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import app
from absl import flags
import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf
import os


from tensorflow.contrib.keras import layers
from tensorflow.contrib.keras import models


# Cloud TPU Cluster Resolvers
flags.DEFINE_string(
    'tpu', default=None,
    help='The Cloud TPU to use for training. This should be either the name '
    'used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.')
flags.DEFINE_string(
    "gcp_project", default=None,
    help="Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")
flags.DEFINE_string(
    "tpu_zone", default=None,
    help="GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

# Model specific paramenters
flags.DEFINE_integer("batch_size", 128,
                     "Mini-batch size for the computation. Note that this "
                     "is the global batch size and not the per-shard batch.")
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_string("train_file", "/content/cifar-10-data/train.tfrecords", "Path to cifar10 training data.")
flags.DEFINE_integer("train_steps", 100000,
                     "Total number of steps. Note that the actual number of "
                     "steps is the next multiple of --iterations greater "
                     "than this value.")
flags.DEFINE_bool("use_tpu", True, "Use TPUs rather than plain CPUs")
flags.DEFINE_string("model_dir", '/content/model-dir', "Estimator model_dir")
flags.DEFINE_integer("iterations_per_loop", 100,
                     "Number of iterations per TPU training loop.")
flags.DEFINE_integer("num_shards", 8, "Number of shards (TPU chips).")


FLAGS = flags.FLAGS

cardinality = 32


def residual_network(x):
    """
    ResNeXt by default. For ResNet set `cardinality` = 1 above.
    
    """
    def add_common_layers(y):
        y = layers.BatchNormalization()(y)
        y = layers.LeakyReLU()(y)

        return y

    def grouped_convolution(y, nb_channels, _strides):
        # when `cardinality` == 1 this is just a standard convolution
        if cardinality == 1:
            return layers.Conv2D(nb_channels, kernel_size=(3, 3), strides=_strides, padding='same')(y)
        
        assert not nb_channels % cardinality
        _d = nb_channels // cardinality

        # in a grouped convolution layer, input and output channels are divided into `cardinality` groups,
        # and convolutions are separately performed within each group
        groups = []
        for j in range(cardinality):
            group = layers.Lambda(lambda z: z[:, :, :, j * _d:j * _d + _d])(y)
            groups.append(layers.Conv2D(_d, kernel_size=(3, 3), strides=_strides, padding='same')(group))
            
        # the grouped convolutional layer concatenates them as the outputs of the layer
        y = layers.concatenate(groups)

        return y

    def residual_block(y, nb_channels_in, nb_channels_out, _strides=(1, 1), _project_shortcut=False):
        """
        Our network consists of a stack of residual blocks. These blocks have the same topology,
        and are subject to two simple rules:
        - If producing spatial maps of the same size, the blocks share the same hyper-parameters (width and filter sizes).
        - Each time the spatial map is down-sampled by a factor of 2, the width of the blocks is multiplied by a factor of 2.
        """
        shortcut = y

        # we modify the residual building block as a bottleneck design to make the network more economical
        y = layers.Conv2D(nb_channels_in, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        y = add_common_layers(y)

        # ResNeXt (identical to ResNet when `cardinality` == 1)
        y = grouped_convolution(y, nb_channels_in, _strides=_strides)
        y = add_common_layers(y)

        y = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=(1, 1), padding='same')(y)
        # batch normalization is employed after aggregating the transformations and before adding to the shortcut
        y = layers.BatchNormalization()(y)

        # identity shortcuts used directly when the input and output are of the same dimensions
        if _project_shortcut or _strides != (1, 1):
            # when the dimensions increase projection shortcut is used to match dimensions (done by 1×1 convolutions)
            # when the shortcuts go across feature maps of two sizes, they are performed with a stride of 2
            shortcut = layers.Conv2D(nb_channels_out, kernel_size=(1, 1), strides=_strides, padding='same')(shortcut)
            shortcut = layers.BatchNormalization()(shortcut)

        y = layers.add([shortcut, y])

        # relu is performed right after each batch normalization,
        # expect for the output of the block where relu is performed after the adding to the shortcut
        y = layers.LeakyReLU()(y)

        return y

    # conv1
    x = layers.Conv2D(64, kernel_size=(7, 7), strides=(2, 2), padding='same')(x)
    x = add_common_layers(x)

    # conv2
    x = layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)
    for i in range(3):
        project_shortcut = True if i == 0 else False
        x = residual_block(x, 128, 256, _project_shortcut=project_shortcut)

    # conv3
    for i in range(4):
        # down-sampling is performed by conv3_1, conv4_1, and conv5_1 with a stride of 2
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 256, 512, _strides=strides)

    # conv4
    for i in range(6):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 512, 1024, _strides=strides)

    # conv5
    for i in range(3):
        strides = (2, 2) if i == 0 else (1, 1)
        x = residual_block(x, 1024, 2048, _strides=strides)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(1)(x)

    return x


def model_fn(features, labels, mode, params):

    """Define a CIFAR model in Keras."""
    del params  # unused

    logits = residual_network(features)


    # Instead of constructing a Keras model for training, build our loss function
    # and optimizer in Tensorflow.
    #
    # N.B.  This construction omits some features that are important for more
    # complex models (e.g. regularization, batch-norm).  Once
    # `model_to_estimator` support is added for TPUs, it should be used instead.

    loss = tf.reduce_mean(
        tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits=logits, labels=labels
        )
    )


    optimizer = tf.train.AdamOptimizer()
    if FLAGS.use_tpu:
        optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

    train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())

    if FLAGS.use_tpu:
        return tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=loss,
            train_op=train_op,
            predictions={
                "classes": tf.argmax(input=logits, axis=1),
                "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
            }
        )
  
    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=train_op,
        predictions={
            "classes": tf.argmax(input=logits, axis=1),
            "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
        }
    )

def input_fn(params):
    """Read CIFAR input data from a TFRecord dataset."""
    del params
    batch_size = FLAGS.batch_size

    def parser(serialized_example):

        """Parses a single tf.Example into image and label tensors."""
        features = tf.parse_single_example(
            serialized_example,
            features={
                "image": tf.FixedLenFeature([], tf.string),
                "label": tf.FixedLenFeature([], tf.int64),
            })
        image = tf.decode_raw(features["image"], tf.uint8)
        image.set_shape([3*32*32])
        image = tf.cast(image, tf.float32) * (1. / 255) - 0.5
        image = tf.transpose(tf.reshape(image, [3, 32, 32]))
        label = tf.cast(features["label"], tf.int32)
        return image, label

    dataset = tf.data.TFRecordDataset([FLAGS.train_file])
    dataset = dataset.map(parser, num_parallel_calls=batch_size)
    dataset = dataset.prefetch(4 * batch_size).cache().repeat()
    dataset = dataset.batch(FLAGS.batch_size, drop_remainder=True)
    dataset = dataset.prefetch(1)
    return dataset


def main(argv):
    del argv  # Unused.

    if FLAGS.use_tpu:
        print("resolving tpu cluster")
        tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(tpu='grpc://' + os.environ['COLAB_TPU_ADDR'])

        print("defining run confing")
        run_config = tf.contrib.tpu.RunConfig(
            cluster=tpu_cluster_resolver,
            model_dir=FLAGS.model_dir,
            save_checkpoints_secs=3600,
            session_config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True),
            tpu_config=tf.contrib.tpu.TPUConfig(
                iterations_per_loop=FLAGS.iterations_per_loop,
                num_shards=FLAGS.num_shards),
        )

        print("creating TPU estimator")
        estimator = tf.contrib.tpu.TPUEstimator(
            model_fn=model_fn,
            use_tpu=FLAGS.use_tpu,
            config=run_config,
            train_batch_size=FLAGS.batch_size)
  
    else:
        print("creating CPU/GPU estimator")
        run_config = tf.estimator.RunConfig(
            save_checkpoints_secs=120)

        estimator = tf.estimator.Estimator(
            model_fn=model_fn,
            model_dir=FLAGS.model_dir,
            config=run_config)


    print("training estimator")
    estimator.train(input_fn=input_fn, max_steps=FLAGS.train_steps)


if __name__ == "__main__":

    tf.logging.set_verbosity(tf.logging.DEBUG)
    app.run(main)