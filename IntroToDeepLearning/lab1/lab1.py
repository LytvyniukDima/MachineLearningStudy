# Exercise: A Single Neuron

# Setup plotting
import matplotlib.pyplot as plt
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

plt.style.use('seaborn-whitegrid')
# Set Matplotlib defaults
plt.rc('figure', autolayout=True)
plt.rc('axes', labelweight='bold', labelsize='large',
       titleweight='bold', titlesize=18, titlepad=10)

# Setup feedback system
from learntools.core import binder
binder.bind(globals())
from learntools.deep_learning_intro.ex1 import *

red_wine = pd.read_csv('../input/dl-course-data/red-wine.csv')
red_wine.head()

print(red_wine.shape) # (rows, columns)

# YOUR CODE HERE
input_shape = [11]

# Check your answer
q_1.check()

# YOUR CODE HERE
model = keras.Sequential([
    layers.Dense(1, input_shape=input_shape),
])

# Check your answer
q_2.check()

# YOUR CODE HERE
w, b = model.weights
print("Weights\n{}\n\nBias\n{}".format(w, b))

# Check your answer
q_3.check()

# Optional
model = keras.Sequential([
    layers.Dense(1, input_shape=[1]),
])

x = tf.linspace(-1.0, 1.0, 100)
y = model.predict(x)

plt.figure(dpi=100)
plt.plot(x, y, 'k')
plt.xlim(-1, 1)
plt.ylim(-1, 1)
plt.xlabel("Input: x")
plt.ylabel("Target y")
w, b = model.weights # you could also use model.get_weights() here
plt.title("Weight: {:0.2f}\nBias: {:0.2f}".format(w[0][0], b[0]))
plt.show()