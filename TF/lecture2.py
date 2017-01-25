import tensorflow as tf
import numpy as np

# Graph Visualization Using TensorBoard
# ==============================================================================

# a simple program and its visualization using TensorBoard
# the filewriter must be built after the graph has been created and before doing
# anything. Using the 2 modules notion of last lecture, it must be creating in
# GraphRunner and at the begining of it.

# the 'name' argument makes everything more beautiful in TensorBoard

a = tf.constant([2, 2], name='a')
b = tf.constant([3, 6], name='b')
x = tf.add(a, b, name='add')
with tf.Session() as session:
    writer = tf.summary.FileWriter('/tmp/tf/lec2')
    writer.add_graph(session.graph)
    print session.run(x)

# close the writer
writer.close()


# Constant types
# ==============================================================================
# creating constant tensors
tf.zeros, tf.zeros_like
tf.ones, tf.ones_like
tf.constant
# creating a constant sequence
# note that these are not iterable!
tf.range
tf.linspace
# creating random constants
tf.random_normal
tf.random_uniform
tf.multinomial
tf.set_random_seed

# Math Operations
# ==============================================================================
# elementwise operations
tf.add, tf.subtract, tf.multiply  # tf.mul is also the same as tf.multiply
# and many more
# note that this works, but only if dtype of b is set.
# this is broadcasting
with tf.Session() as session:
    a = tf.ones((1, 2))
    b = tf.constant(1, dtype=tf.float32)

    print session.run(tf.add(a, b))

# Data types
# ==============================================================================
# Numpy and Tensorflow are dtype frineds
tf.float32 == np.float32
# One catch is `string` data type, where TF and NP are different

# Variables
# ==============================================================================
# Parameters of a neural net are variables.

# This gives you the protobuf of the graph.
# you can see that the value of the constants are stored in this protobuf.
# the initial value of a variable is stored as a constant in the graph.
a = tf.constant([1.0, 2.0], name='a')
b = tf.Variable(initial_value=[3.0, 4.0], name='b')
c = tf.get_variable('c', shape=(1, 2))
print tf.get_default_graph().as_graph_def()

# before using the variabel you have to initialize it
with tf.Session() as session:
    print session.run(a + b)
    # This gives an error
    # FailedPreconditionError: Attempting to use uninitialized value b

# the correct way is this
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print session.run(a + b)

# you can do selective initialization also
with tf.Session() as session:
    session.run(b.initializer)  # for a single variable

    init_bc = tf.variables_initializer([a, b])  # for a set of variable
    session.run(init_bc)

# getting and setting a variable's value
with tf.Session() as session:
    session.run(tf.global_variables_initializer())
    print b.eval()
    session.run(b.assign([-1.0, -2.0]))  # you must run the assign operation
    print b.eval()

# Also there is `assign_add` and `assign_sub`

# One Variable depending on the other
W = tf.Variable(tf.random_normal((10, 10)))
U = tf.Variable(W * 2)

# Interactive Session
# ==============================================================================
