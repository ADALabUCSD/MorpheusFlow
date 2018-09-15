import tensorflow as tf

def neural_network(x, y, num_input, num_classes, learning_rate, num_neurons=256):
    # Network Parameters
    n_hidden_1 = num_neurons  # 1st layer number of neurons
    n_hidden_2 = num_neurons  # 2nd layer number of neurons

    # Store layers weight & bias
    weights = {
        'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
        'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
        'out': tf.Variable(tf.random_normal([n_hidden_2, num_classes]))
    }
    biases = {
        'b1': tf.Variable(tf.random_normal([n_hidden_1])),
        'b2': tf.Variable(tf.random_normal([n_hidden_2])),
        'out': tf.Variable(tf.random_normal([num_classes]))
    }

    # Create model
    def neural_net(x):
        # Hidden fully connected layer with 256 neurons
        layer_1 = tf.add(tf.sparse_tensor_dense_matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        # Hidden fully connected layer with 256 neurons
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        # Output fully connected layer with a neuron for each class
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer

    # Construct model
    logits = neural_net(x)

    # Define loss and optimizer
    loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        logits=logits, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train_op = optimizer.minimize(loss_op)

    return train_op, loss_op

def logistic_regression(x, y, num_input, num_classes, learning_rate, sparse=True):
    # Set model weights
    W = tf.Variable(tf.zeros([num_input, num_classes]))
    b = tf.Variable(tf.zeros([num_classes]))

    # Construct model
    if sparse:
        pred = tf.nn.relu(tf.sparse_tensor_dense_matmul(x, W) + b)  # Softmax
    else:
        pred = tf.nn.relu(tf.matmul(x, W) + b)  # Softmax

    # Minimize error using cross entropy
    cost = tf.reduce_mean(-tf.reduce_sum(y * tf.log(pred), reduction_indices=1))
    # Gradient Descent
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)

    return optimizer, cost
