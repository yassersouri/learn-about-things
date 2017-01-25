import tensorflow as tf


# TF code is usually two stages: 1. Create the graph. 2. Run the graph.
# I think, these two stages can be well encapsulated. Specially part 2.
# If one can create a good GraphRunner, with "Pause", "Resume" capability
# only assuming a simple structure for the graph (how losses, optimizers,
# etc. are defined) it would be nice.
# For example the Graph object will create a merge of all summaries, then
# through an API, provides it to GraphRunner. The GraphRunner will create
# two different tf.summary.FileWriter's and write to them with different
# intervals.

# 0-d tensor is a scalar.

# Running something
a = tf.add(3, 5)
session = tf.InteractiveSession()  # this is because I am using Hydrogen
print session.run(a)
session.close()

# Better code for running something in a session
a = tf.add(3, 5)
with tf.Session() as session:
    print session.run(a)

# a much larger graph
x = 2
y = 3

op1 = tf.add(x, y)
op2 = tf.mul(x, y)
op3 = tf.pow(op2, op1)
with tf.Session() as session:
    print session.run(op3)

# Showing graphs in TensorBoard
with tf.Session() as session:
    fw = tf.summary.FileWriter(logdir='/tmp/tf/', graph=session.graph)

# Use the following to manually create a graph for a specific device
with tf.device('/gpu:0'):
    a = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='a')
    b = tf.constant([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], name='b')
    c = tf.mul(a, b)

with tf.Session(config=tf.ConfigProto(log_device_placement=True)) as session:
    fw = tf.summary.FileWriter(logdir='/tmp/tf/', graph=session.graph)
    print session.run(c)


# Sessions try to use all available resources by default.
# Data sharing between sessions is hard.
# Better not to do it.
