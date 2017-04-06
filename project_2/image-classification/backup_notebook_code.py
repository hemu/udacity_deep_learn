"""
Exploration notes

Q: What are all possible labels?
A: With our successfully trained network, we should be able to accept images as inputs and then
classify the object the image contains. Hmm I'm guessing the labels will just be from 0-9 since we are looking at
[airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck]. And we'll have to one hot encode these so
they will end up looking something like this [0,0,0,1,0,0,0,0,0,0]. These should be it since these are the only
labels present in our training set.

Q: What is the range of values for the image data?
A: The range should be that of 8-bit pixels (0 - 255)

Q: Are the labels in order or random?
A: I think the order of the labels should match the order of the training data right? Otherwise they aren't very
helpful when we train.

Q: Is this going to be cool when we're done?
A: Very much so, this is gonna be awesome.

"""

def normalize(x):
    """
    Normalize a list of sample image data in the range of 0 to 1
    : x: List of image data.  The image shape is (32, 32, 3)
    : return: Numpy array of normalize data
    """
    # TODO: Implement Function
    # Normalize by max possible value, which should be 255
    return x / 255


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_normalize(normalize)


def one_hot_encode(x):
    """
    One hot encode a list of sample labels. Return a one-hot encoded vector for each label.
    : x: List of sample Labels
    : return: Numpy array of one-hot encoded labels
    """
    # TODO: Implement Function
    return np.eye(10)[x]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_one_hot_encode(one_hot_encode)


import tensorflow as tf

def neural_net_image_input(image_shape):
    """
    Return a Tensor for a batch of image input
    : image_shape: Shape of the images
    : return: Tensor for image input.
    """
    # TODO: Implement Function
    inp_shape = list(image_shape)
    inp_shape.insert(0, None)
    return tf.placeholder(tf.float32, shape=inp_shape, name="x")


def neural_net_label_input(n_classes):
    """
    Return a Tensor for a batch of label input
    : n_classes: Number of classes
    : return: Tensor for label input.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, shape=[None, n_classes], name="y")


def neural_net_keep_prob_input():
    """
    Return a Tensor for keep probability
    : return: Tensor for keep probability.
    """
    # TODO: Implement Function
    return tf.placeholder(tf.float32, name="keep_prob")


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tf.reset_default_graph()
tests.test_nn_image_inputs(neural_net_image_input)
tests.test_nn_label_inputs(neural_net_label_input)
tests.test_nn_keep_prob_inputs(neural_net_keep_prob_input)

def conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides):
    """
    Apply convolution then max pooling to x_tensor
    :param x_tensor: TensorFlow Tensor
    :param conv_num_outputs: Number of outputs for the convolutional layer
    :param conv_ksize: kernal size 2-D Tuple for the convolutional layer
    :param conv_strides: Stride 2-D Tuple for convolution
    :param pool_ksize: kernal size 2-D Tuple for pool
    :param pool_strides: Stride 2-D Tuple for pool
    : return: A tensor that represents convolution and max pooling of x_tensor
    """
    # TODO: Implement Function

    x_shape = x_tensor.get_shape().as_list()
    k_h, k_w = conv_ksize
    conv_stride_h, conv_stride_w = conv_strides
    pk_h, pk_w = pool_ksize
    ps_h, ps_w = pool_strides

    W = tf.Variable(tf.truncated_normal([k_h, k_w, x_shape[-1], conv_num_outputs]))
    b = tf.Variable(tf.truncated_normal([conv_num_outputs]))
    # tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, data_format=None, name=None)
    conv = tf.nn.conv2d(x_tensor, W, [1, conv_stride_h, conv_stride_w, 1], 'SAME') + b
    # add nonlinear activation
    conv = tf.nn.relu(conv)
    # add pooling
    # tf.nn.max_pool(value, ksize, strides, padding, data_format='NHWC', name=None)
    conv = tf.nn.max_pool(conv, [1, pk_h, pk_w, 1], [1, ps_h, ps_w, 1], 'SAME')

    return conv

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_con_pool(conv2d_maxpool)


def flatten(x_tensor):
    """
    Flatten x_tensor to (Batch Size, Flattened Image Size)
    : x_tensor: A tensor of size (Batch Size, ...), where ... are the image dimensions.
    : return: A tensor of size (Batch Size, Flattened Image Size).
    """
    # TODO: Implement Function
    b, h, w, d = x_tensor.get_shape().as_list()
    return tf.reshape(x_tensor, [-1, w*h*d])


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""



tests.test_flatten(flatten)def fully_conn(x_tensor, num_outputs):
    """
    Apply a fully connected layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    # TODO: Implement Function
    x_h, x_w = x_tensor.get_shape().as_list()
    W = tf.Variable(tf.truncated_normal([x_w, num_outputs]))
    b = tf.Variable(tf.truncated_normal([num_outputs]))

    return tf.nn.relu(tf.matmul(x_tensor, W) + b)


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_fully_conn(fully_conn)


def output(x_tensor, num_outputs):
    """
    Apply a output layer to x_tensor using weight and bias
    : x_tensor: A 2-D tensor where the first dimension is batch size.
    : num_outputs: The number of output that the new tensor should be.
    : return: A 2-D tensor where the second dimension is num_outputs.
    """
    x_h, x_w = x_tensor.get_shape().as_list()
    W = tf.Variable(tf.truncated_normal([x_w, num_outputs]))
    b = tf.Variable(tf.truncated_normal([num_outputs]))

    return tf.matmul(x_tensor, W) + b


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_output(output)


def conv_net(x, keep_prob):
    """
    Create a convolutional neural network model
    : x: Placeholder tensor that holds image data.
    : keep_prob: Placeholder tensor that hold dropout keep probability.
    : return: Tensor that represents logits
    """
    # TODO: Apply 1, 2, or 3 Convolution and Max Pool layers
    #    Play around with different number of outputs, kernel size and stride
    # Function Definition from Above:
    #    conv2d_maxpool(x_tensor, conv_num_outputs, conv_ksize, conv_strides, pool_ksize, pool_strides)

    convL_1 = conv2d_maxpool(x,       30, (4, 4), (1, 1), (2, 2), (1, 1))
    convL_2 = conv2d_maxpool(convL_1, 30, (4, 4), (1, 1), (2, 2), (1, 1))

    # TODO: Apply a Flatten Layer
    # Function Definition from Above:
    #   flatten(x_tensor)
    flattenL_1 = flatten(convL_2)

    # TODO: Apply 1, 2, or 3 Fully Connected Layers
    #    Play around with different number of outputs
    # Function Definition from Above:
    #   fully_conn(x_tensor, num_outputs)

    fullL_1 = fully_conn(flattenL_1, 30)
    fullL_2 = fully_conn(fullL_1, 30)

    dropoutL = tf.nn.dropout(fullL_2, keep_prob)

    # TODO: Apply an Output Layer
    #    Set this to the number of classes
    # Function Definition from Above:
    #   output(x_tensor, num_outputs)

    outputL = output(dropoutL, 10)

    # TODO: return output
    return outputL


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""

##############################
## Build the Neural Network ##
##############################

# Remove previous weights, bias, inputs, etc..
tf.reset_default_graph()

# Inputs
x = neural_net_image_input((32, 32, 3))
y = neural_net_label_input(10)
keep_prob = neural_net_keep_prob_input()

# Model
logits = conv_net(x, keep_prob)

# Name logits Tensor, so that is can be loaded from disk after training
logits = tf.identity(logits, name='logits')

# Loss and Optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y))
optimizer = tf.train.AdamOptimizer().minimize(cost)

# Accuracy
correct_pred = tf.equal(tf.argmax(logits, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32), name='accuracy')

tests.test_conv_net(conv_net)


def train_neural_network(session, optimizer, keep_probability, feature_batch, label_batch):
    """
    Optimize the session on a batch of images and labels
    : session: Current TensorFlow session
    : optimizer: TensorFlow optimizer function
    : keep_probability: keep probability
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    """
    session.run(optimizer, feed_dict={x: feature_batch, y: label_batch, keep_prob: keep_probability})


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_train_nn(train_neural_network)


def print_stats(session, feature_batch, label_batch, cost, accuracy):
    """
    Print information about loss and validation accuracy
    : session: Current TensorFlow session
    : feature_batch: Batch of Numpy image data
    : label_batch: Batch of Numpy label data
    : cost: TensorFlow cost function
    : accuracy: TensorFlow accuracy function
    """

    loss = session.run(cost, feed_dict={x: feature_batch, y: label_batch, keep_prob: 1.0})
    acc = session.run(accuracy,  feed_dict={x: valid_features, y: valid_labels, keep_prob: 1.0})
    print("validation accuracy: %g" % acc)
    print("validation loss: %g" % loss)
