import tensorflow as tf
from examples.algorithm import neural_network
from morpheusflow.dataset import Dataset
from examples.data_wrapper import expedia


def iteration(data_fn, model_fn):
    # Parameters
    learning_rate = 0.1
    batch_size = 100
    training_epochs = 20
    s, r, k, Y, ns, num_input = data_fn()
    num_classes = 1
    total_batch = int(ns / batch_size)

    # init model
    x = tf.sparse_placeholder("float", [None, num_input])
    y = tf.placeholder("float", [None, num_classes])
    train_op, loss_op = model_fn(x, y, num_input, num_classes, learning_rate)

    # init dataset
    dataset = Dataset(s, k, r, Y).batch(batch_size).repeat(training_epochs)
    init = tf.global_variables_initializer()

    # Start training
    with tf.Session() as sess:
        # Run the initializer
        sess.run(init)
        # Training cycle
        for epoch in range(training_epochs):
            print("epoch:", epoch)
            # Loop over all batches
            for i in range(total_batch):
                batch_xs, batch_ys = dataset.get_next()
                _, c = sess.run([train_op, loss_op], feed_dict={x: batch_xs, y: batch_ys})

        print("Optimization Finished!")


iteration(expedia, neural_network)
