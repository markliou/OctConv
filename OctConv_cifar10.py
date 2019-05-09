import tensorflow as tf
import numpy as np
import os

# Training Parameters
learning_rate = 0.001
num_steps = 2000
batch_size = 32

# Network Parameters
num_classes = 10 # Cifar10 total classes (0-9 digits)
dropout = 0.25 # Dropout, probability to drop a unit
alpha = 0.8


# OctConv 
#######
# The OctConv should be compose with 3 parts:
# 1) The input part: the channels have only single frequency, but output multi-frequency channels
# 2) The building block part: this part swallow multi-frequency channels, and output multi-frequency channels
# 3) The final part : this part swallow multi-frequency channels, and single frequency channels
# Tips:
# using pooling would be better than 2-stride convolution due to the original paper
def OctConv_ini(x, alpha, channel_no, kernel_size = 3, activation = tf.nn.relu):
    H_channel_no = int(channel_no * alpha // 1)
    L_channel_no = int(channel_no - H_channel_no)
    H2H = tf.keras.layers.Conv2D(H_channel_no, kernel_size, padding="SAME", activation=activation)(x)
    H2L = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(x)
    H2L = tf.keras.layers.Conv2D(L_channel_no, kernel_size, padding="SAME", activation=activation)(H2L)
    return H2H, H2L
pass 

def OctConv_block(Hx, Lx, alpha, channel_no,  kernel_size = 3, activation = tf.nn.relu):
    H_channel_no = int(channel_no * alpha // 1)
    L_channel_no = int(channel_no - H_channel_no)
    H2H = tf.keras.layers.Conv2D(H_channel_no, kernel_size, padding="SAME", activation=None)(Hx)
    H2L = tf.keras.layers.AveragePooling2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(Hx)
    H2L = tf.keras.layers.Conv2D(L_channel_no, kernel_size, padding="SAME", activation=None)(H2L)
    L2L = tf.keras.layers.Conv2D(L_channel_no, kernel_size, padding="SAME", activation=None)(Lx)
    L2H = tf.keras.layers.Conv2D(H_channel_no, kernel_size, padding="SAME", activation=None)(Lx)
    L2H = tf.keras.layers.UpSampling2D(interpolation='bilinear')(L2H)
    return activation((H2H + L2H)/2), activation((L2L + H2L)/2)
pass

def OctConv_final(Hx, Lx, channel_no, kernel_size = 3, activation = tf.nn.relu):
    H2H = tf.keras.layers.Conv2D(channel_no, kernel_size, padding="SAME", activation=None)(Hx)
    L2H = tf.keras.layers.Conv2D(channel_no, kernel_size, padding="SAME", activation=None)(Lx)
    L2H = tf.keras.layers.UpSampling2D(interpolation='bilinear')(L2H)
    return activation((H2H + L2H)/2)
pass

# Create the neural network
def conv_net(x, alpha, n_classes, dropout, reuse, is_training):
    # Define a scope for reusing the variables
    with tf.variable_scope('ConvNet', reuse=reuse):

        # MNIST data input is a 1-D vector of 784 features (28*28 pixels)
        # Reshape to match picture format [Height x Width x Channel]
        # Tensor input become 4-D: [Batch Size, Height, Width, Channel]
        x = tf.reshape(x, shape=[-1, 32, 32, 3])

        # Convolution Layer with 32 filters and a kernel size of 5
        conv1H, conv1L =OctConv_ini(x, alpha, 32, kernel_size = 5, activation = tf.nn.relu)
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv1H = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(conv1H)
        conv1L = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(conv1L)

        conv2H, conv2L = OctConv_block(conv1H, conv1L, alpha, 32, kernel_size = 5, activation = tf.nn.relu)
        conv3H, conv3L = OctConv_block(conv2H, conv2L, alpha, 32, kernel_size = 5, activation = tf.nn.relu)
        

        # Convolution Layer with 64 filters and a kernel size of 3
        conv4H, conv4L = OctConv_block(conv3H, conv3L, alpha, 64, kernel_size = 3, activation = tf.nn.relu)
        
        # Max Pooling (down-sampling) with strides of 2 and kernel size of 2
        conv4H = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(conv4H)
        conv4L = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2,2), padding='SAME')(conv4L)

        conv5H, conv5L = OctConv_block(conv4H, conv4L, alpha, 64, kernel_size = 3, activation = tf.nn.relu)
        conv6 = OctConv_final(conv5H, conv5L, 64,  kernel_size = 3, activation = tf.nn.relu)
        

        # Flatten the data to a 1-D vector for the fully connected layer
        fc1 = tf.contrib.layers.flatten(conv6)

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

# conv_net(x, alpha, n_classes, dropout, reuse, is_training)
logits_train = conv_net(CIFAR10_dataset_fetch['imgs'], alpha, num_classes, dropout, reuse=False, is_training=True)
logits_test = conv_net(CIFAR10_imgs, alpha, num_classes, dropout=0, reuse=True, is_training=False)

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
cnn_vars =  tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='ConvNet')
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op, var_list=cnn_vars, global_step=tf.train.get_global_step())

# Evaluate the accuracy of the model
acc_op = tf.reduce_mean(tf.cast(tf.equal(tf.reshape(CIFAR10_labels,tf.shape(pred_classes)), pred_classes), tf.float32))



# setting the device parameters
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
sess.run(tf.global_variables_initializer())

# tensorboard
tf.summary.scalar('loss', loss_op)
merged = tf.summary.merge_all()
TFB_summary = tf.summary.FileWriter('VCNN', graph=sess.graph)

sess.graph.finalize() 

from tensorflow.examples.tutorials.mnist import input_data
(CIFAR10_tr_imgs, CIFAR10_tr_labs), (CIFAR10_ts_imgs, CIFAR10_ts_labs) = tf.keras.datasets.cifar10.load_data()
sess.run(CIFAR10_dataset_iter.initializer, feed_dict={CIFAR10_imgs: CIFAR10_tr_imgs, 
                                                     CIFAR10_labels: np.reshape(CIFAR10_tr_labs, -1)}) # initialize tf.data module
# training
for training_step in range(num_steps):
    closs, _ = sess.run([loss_op, train_op])

    # call the tensorboad operator
    TFB_process = sess.run(merged)
    TFB_summary.add_summary(TFB_process, training_step)

    if training_step%1000 == 0:
        print('step:{} loss:{}'.format(training_step, closs))

# test
acc = sess.run(acc_op, feed_dict={CIFAR10_imgs: CIFAR10_ts_imgs,
                                  CIFAR10_labels: np.reshape(CIFAR10_ts_labs, -1)})


sess.close()
print("Testing Accuracy:",acc)


