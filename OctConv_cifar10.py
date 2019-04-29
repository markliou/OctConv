import tensorflow as tf
import numpy as np
import os

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 32

# Network Parameters
num_input = 784 # MNIST data input (img shape: 28*28)
num_classes = 10 # MNIST total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit


def oct_conv(alpha):
    
    




# Create the neural network
def conv_net(x, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 32, 32, 3])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1 = tf.layers.conv2d(x, 32, 5, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1 = tf.layers.max_pooling2d(conv1, 2, 2)

        # Convolution Layer with 64 filters and a kernel size of 3
        conv2 = tf.layers.conv2d(conv1, 64, 3, activation=tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv2 = tf.layers.max_pooling2d(conv2, 2, 2)

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv2)

        # Fully connected layer (in tf contrib folder for now)
        fc1 = tf.layers.dense(fc1, 1024)
        # Apply Dropout (if is_training is False, dropout is not applied)
        fc1 = tf.layers.dropout(fc1, rate=dropout, training=is_training)

        # Output layer, class prediction
        out = tf.layers.dense(fc1, n_classes)

    return out

# entry point
CIFAR10_imgs = tf.placeholder(tf.float32, [None, 32, 32 ,3])
CIFAR10_labels = tf.placeholder(tf.float32, [None])

# use TF.dataset to handle minst
CIFAR10_dataset = tf.data.Dataset.from_tensor_slices({'imgs':CIFAR10_imgs, 'labs':CIFAR10_labels})
CIFAR10_dataset.batch(batch_size)
CIFAR10_dataset = CIFAR10_dataset.apply(tf.contrib.data.shuffle_and_repeat(buffer_size=10000))
CIFAR10_dataset = CIFAR10_dataset.prefetch(buffer_size=100) # prefech
CIFAR10_dataset_iter = CIFAR10_dataset.make_initializable_iterator()
CIFAR10_dataset_fetch = CIFAR10_dataset_iter.get_next()

logits_train = conv_net(CIFAR10_dataset_fetch['imgs'], num_classes, dropout, reuse=False, is_training=True)
logits_test = conv_net(CIFAR10_imgs, num_classes, dropout=0, reuse=True, is_training=False)

# Predictions
pred_classes = tf.cast(tf.argmax(logits_test, axis=1), tf.float32)
pred_probas = tf.nn.softmax(logits_test)

# loss
loss_op = tf.reduce_mean(
            tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=logits_train, 
                labels=tf.reshape(tf.cast(CIFAR10_dataset_fetch['labs'], dtype=tf.int32), [-1])
            )
          )
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, global_step=tf.train.get_global_step())

# Evaluate the accuracy of the model
acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(CIFAR10_labels,tf.shape(pred_classes)), pred_classes), tf.float32))

# setting the device parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())
sess.graph.finalize() 

from tensorflow.examples.tutorials.mnist import input_data
(CIFAR10_tr_imgs, CIFAR10_tr_labs), (CIFAR10_ts_imgs, CIFAR10_ts_labs) = tf.keras.datasets.cifar10.load_data()
sess.run(CIFAR10_dataset_iter.initializer, feed_dict={CIFAR10_imgs: CIFAR10_tr_imgs, 
                                                     CIFAR10_labels: np.reshape(CIFAR10_tr_labs, -1)}) # initialize tf.data module
# training
for training_step in range(num_steps):
    closs, _ = sess.run([loss_op, train_op])
    if training_step%1000 == 0:
        print('step:{} loss:{}'.format(training_step, closs))

# test
acc = sess.run(acc_op, feed_dict={CIFAR10_imgs: CIFAR10_ts_imgs,
                                  CIFAR10_labels: np.reshape(CIFAR10_ts_labs, -1)})

sess.close()
print("Testing Accuracy:",acc)


