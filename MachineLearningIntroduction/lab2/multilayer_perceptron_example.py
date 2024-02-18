import tensorflow.compat.v1 as tf
from tensorflow.examples.tutorials.mnist import input_data

# Suppress TensorFlow logging messages
tf.logging.set_verbosity(tf.logging.ERROR)

# Load MNIST data
mnist = input_data.read_data_sets("./datasets/MNIST_data/", one_hot=True)

# Define input and output placeholders
x = tf.placeholder(tf.float32, [None, 784])
y_true = tf.placeholder(tf.float32, [None, 10])

# Define model architecture
def neural_network_model(data):
    hidden_layer1 = {'weights': tf.Variable(tf.random_normal([784, 500])),
                     'biases': tf.Variable(tf.random_normal([500]))}

    hidden_layer2 = {'weights': tf.Variable(tf.random_normal([500, 100])),
                     'biases': tf.Variable(tf.random_normal([100]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([100, 10])),
                    'biases': tf.Variable(tf.random_normal([10]))}

    layer1 = tf.add(tf.matmul(data, hidden_layer1['weights']), hidden_layer1['biases'])
    layer1 = tf.nn.relu(layer1)

    layer2 = tf.add(tf.matmul(layer1, hidden_layer2['weights']), hidden_layer2['biases'])
    layer2 = tf.nn.relu(layer2)

    output = tf.add(tf.matmul(layer2, output_layer['weights']), output_layer['biases'])

    return output

# Define training parameters
batch_size = 100
total_epochs = 10
learning_rate = 0.001

# Define loss function and optimizer
logits = neural_network_model(x)
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=y_true))
optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

# Initialize variables
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:
    sess.run(init)

    for epoch in range(total_epochs):
        epoch_loss = 0
        total_batches = int(mnist.train.num_examples / batch_size)

        for _ in range(total_batches):
            epoch_x, epoch_y = mnist.train.next_batch(batch_size)
            _, c = sess.run([optimizer, loss], feed_dict={x: epoch_x, y_true: epoch_y})
            epoch_loss += c

        print('Epoch', epoch, 'completed out of', total_epochs, 'loss:', epoch_loss)

    # Test model
    correct = tf.equal(tf.argmax(logits, 1), tf.argmax(y_true, 1))
    accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
    print('Accuracy:', accuracy.eval({x: mnist.test.images, y_true: mnist.test.labels}))
