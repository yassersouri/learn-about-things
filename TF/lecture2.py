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
# NOTE: Did you notice that tf.constant (small c) is a operation, but
# tf.Variable (capital V) is a class?
# tf.Variable has many operations: init (x.initializer), read(x.value()),
# write(x.assign()) and more (x.assign_add())
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
# The difference between InteraciveSession and Session is that the former makes
# itself the default session automatically.

session = tf.InteractiveSession()
a = tf.constant(5.0)
b = tf.constant(6.0)
c = a * b
print b.eval()
session.close()

# Control Dependencies
# ==============================================================================
# This is just to ensure that some operation is run before the other.
session = tf.Session()
session.as_default()
g = session.graph

a = tf.constant(5.0)
b = tf.constant(6.0)
with g.control_dependencies([a, b]):
    c = tf.zeros((2, 2))  # or something else.

# Placeholders and feed_dict
# ==============================================================================
# Placeholder is used for input to the neural network, say the training data and
# it's label. Comparing to Variable which is used to define the weights of the
# network.
# You don't have to provide the initial value for Placeholders, but you need to
# feed them to the session with "feed_dict"
# The API is like this: tf.placeholder(dtype, shape=None, name=None)
# Note that shape can be None. This seems easy at first, but makes debugging
# hard. It is recommended to set them.

# TODO: I wonder if it is possible to partly set the shape, like when you don't
# specify the batch size but the other 3 dimensions.
# I found that it is possible to set the shape to something like (None, 3), have
# not found any documentation on it.

a = tf.placeholder(tf.float32, shape=(3,))
b = tf.constant([5, 5, 5], tf.float32)
c = tf.add(a, b)

# This will result in an error
with tf.Session() as session:
    print session.run(c)

# The correct way to run it
with tf.Session() as session:
    print session.run(c, feed_dict={a: [1, 2, 3]})

# The thing is, usually we want to give multiple values to the network
# So you can for loop it.
a_values = [[1, 2, 3], [4, 5, 6]]

with tf.Session() as session:
    for a_value in a_values:
        print session.run(c, feed_dict={a: a_value})

# You can also (quite easily) debug or test parts of your graph with feeding
# the network with necessary inputs. This way TF will not run certain parts
a = tf.constant(1)
b = tf.constant(2)
c = tf.constant(3)
d = tf.mul(a, b)
e = tf.mul(d, c)

# Can we just feed d and not compute it?
with tf.Session() as session:
    print session.graph.is_feedable(d)
    # The output in this case is true

# Then we can do this
with tf.Session() as session:
    print session.run(e, feed_dict={d: 10})
# In this case d is not computed, but feeded to the network

# The trap of the lazy loading
# ==============================================================================
# Make sure you separate graph definition from its execution.
# Some blog post which makes things way complicated: http://danijar.com/structuring-your-tensorflow-models/
