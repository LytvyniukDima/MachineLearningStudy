import tensorflow as tf # When we import TensorFlow, a default graph is made

tf.compat.v1.disable_eager_execution()

hello = tf.constant("Hello, TensorFlow!") # Add a constant-operation to the (default) graph
sess = tf.compat.v1.Session() # Create a session from which to run the graph
print(sess.run(hello))
sess.close()