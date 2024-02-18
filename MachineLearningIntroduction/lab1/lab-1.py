import tensorflow as tf # When we import TensorFlow, a default graph is made

tf.compat.v1.disable_eager_execution()

# Configure a session to not use too much GPU memory
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True

with tf.compat.v1.Session(config=config) as sess:
    a = tf.constant(3.0, dtype=tf.float32) # add a constant-op to the graph
    b = tf.constant(4.0, dtype=tf.float32) # add another constant-op to the graph
    sum_a_b = tf.add(a,b) # create a TensorFlow op that adds tensors a,b and produces a new tensor

    first_const, sum_result = sess.run([a, sum_a_b])
    print("The first constant tensor has value: {}".format(first_const))
    print("The result of the add operation has value: {}".format(sum_result))


x = tf.compat.v1.placeholder(tf.float32, shape=[2,1])
W = tf.constant([[3.0, 4.0], [5.0, 6.0]], dtype=tf.float32)
y = tf.matmul(W, x) # perform matrix-vector multiplication W * x

with tf.compat.v1.Session(config=config) as sess:
    print("x is [[1.0], [2.0]]:")
    print(sess.run(y, feed_dict={x: [[1.0], [2.0]]}))  # input a feed_dict for placeholder x -- must be at least rank-2!
    print("x is [[2.0], [4.0]]:")
    print(sess.run(y, feed_dict={x: [[2.0], [4.0]]}))  # we can change input to graph from here


x = tf.compat.v1.placeholder(tf.float32, shape=[2,1])
init_value = tf.compat.v1.random_normal(shape = [2, 2]) # will draw a 2 x 2 matrix with entries from a standard normal distn
W = tf.Variable(init_value) # Within the graph, initialize W with the values drawn from a standard normal above
y = tf.matmul(W, x)

with tf.compat.v1.Session(config=config) as sess:
    sess.run(tf.compat.v1.global_variables_initializer()) # necessary step now that we have variables
    print("Our random matrix W:\n")
    print(sess.run(W)) # Notice that we don't have to use a feed_dict here, because x is not part of computing W
    print("\nResult of our matrix multiplication, y:\n")
    print(sess.run(y, feed_dict={x: [[1.0], [2.0]]}))


x = tf.compat.v1.placeholder(tf.float32)
W = tf.compat.v1.get_variable(name="W", shape = [2, 2], initializer=tf.random_normal_initializer) # note we give the variable a name
y = tf.matmul(W, x)

with tf.compat.v1.Session() as sess:
    sess.run(tf.compat.v1.global_variables_initializer())
    print(sess.run(W))
    print(sess.run(y, feed_dict={x: [[1.0], [2.0]]}))