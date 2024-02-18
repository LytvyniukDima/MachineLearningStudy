import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
import matplotlib.pyplot as plt

# Import data
mnist = input_data.read_data_sets("datasets/MNIST_data/", one_hot=True)

# Define placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_ = tf.placeholder(tf.float32, [None, 10])

# Define hidden layers
# First hidden layer with 500 units and ReLU activation
W1 = tf.Variable(tf.random_normal([784, 500]))
b1 = tf.Variable(tf.random_normal([500]))
z1 = tf.matmul(x, W1) + b1
h1 = tf.nn.relu(z1)

# Second hidden layer with 100 units and ReLU activation
W2 = tf.Variable(tf.random_normal([500, 100]))
b2 = tf.Variable(tf.random_normal([100]))
z2 = tf.matmul(h1, W2) + b2
h2 = tf.nn.relu(z2)

# Output layer with 10 units and softmax activation
W3 = tf.Variable(tf.random_normal([100, 10]))
b3 = tf.Variable(tf.random_normal([10]))
y = tf.matmul(h2, W3) + b3

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=y_))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# Define training parameters
batch_size = 100
total_epochs = 10
learning_rate = 0.001

# Initialize lists to store loss values
loss_history = []

# Train the model
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for epoch in range(total_epochs):
        epoch_loss = 0
        total_batches = int(mnist.train.num_examples / batch_size)

        # Train the model
        for _ in range(total_batches):
            batch_xs, batch_ys = mnist.train.next_batch(batch_size)
            _, c = sess.run([train_step, loss], feed_dict={x: batch_xs, y_: batch_ys})
            epoch_loss += c
            loss_history.append(c)

        print('Epoch', epoch, 'completed out of', total_epochs, 'loss:', epoch_loss)

    # Evaluate the model
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    print('Test accuracy: {}'.format(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels})))


plt.plot(loss_history)
plt.title('Cross-Entropy Loss Over Training')
plt.xlabel('Training Steps')
plt.ylabel('Cross-Entropy Loss')
plt.show()