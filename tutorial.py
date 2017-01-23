import numpy as np
import scipy.stats
import tensorflow as tf
import seaborn as sns
import click
import matplotlib.pyplot as plt
from matplotlib import animation

seed = 42
np.random.seed(seed)
tf.set_random_seed(seed)


class DataDistribution(object):
    """
    Generates 'x' from a single gaussian distribution.
    """
    def __init__(self, mu=4, sigma=0.5):
        """
        params:
            - mu: the mean of the gaussian
            - sigma: the stddev of the gaussian
        """
        self.mu = mu
        self.sigma = sigma

    def sample(self, N):
        samples = np.random.normal(self.mu, self.sigma, N)
        samples.sort()  # ?
        return samples


class GeneratorDistribution(object):
    """
    Generates 'z'

    The approach which it generates 'z' is not clear to me, why it works.
    """
    def __init__(self, range):
        self.range = range

    def sample(self, N):
        """
        Stratified sampling approach
        """
        return np.linspace(-self.range, self.range, N) + np.random.random(N) * 0.01


def linear(input, output_dim, scope='linear', init_stddev=1.0, init_const=0.0):
    """
    A Fully Connected Linear layer

    params:
        - input: the incoming tensor to this layer
        - the dimension of the output tensor
        - scope: The variable scope for weights of this layer, default: 'linear'
        - init_stddev: the std of the gaussian noise used for initializing the weights 'w'
        - init_const: the init value of the weight: 'b'
    """
    norm = tf.random_normal_initializer(stddev=init_stddev)
    const = tf.constant_initializer(init_const)

    with tf.variable_scope(scope):
        w = tf.get_variable('w', [input.get_shape()[1], output_dim], initializer=norm)
        b = tf.get_variable('b', [output_dim], initializer=const)
        return tf.matmul(input, w) + b


def generator(input, h_dim):
    """
    Creates the generator subnetwork
    """
    with tf.variable_scope('generator'):
        h0 = tf.nn.softplus(linear(input, h_dim, 'l1'))  # Why softplus?
        h1 = linear(h0, 1, 'l2')
    return h1


def discriminator(input, h_dim, minibatch_layer=False):
    """
    Creates the discriminator subnetwork.
    If 'minibatch_layer' is set to True, minibatch_features is going to be added to the network.
    """
    with tf.variable_scope('discriminator'):
        h0 = tf.tanh(linear(input, h_dim * 2, 'l1'))
        h1 = tf.tanh(linear(h0, h_dim * 2, 'l2'))

        if minibatch_layer:
            h2 = minibatch(h1)
        else:
            h2 = tf.tanh(linear(h1, h_dim * 2, 'l3'))

        h3 = tf.sigmoid(linear(h2, 1, 'l4'))
    return h3


def minibatch(input, num_kernels=5, kernel_dim=5):
    """
    So what is going on exactly? I knew about the concept of minibatch_features,
    but this kind of implementation seems a bit complicated to me.
    """
    with tf.variable_scope('minibatch'):
        x = linear(input, num_kernels * kernel_dim, 'l1', init_stddev=0.02)
        activation = tf.reshape(x, (-1, num_kernels, kernel_dim))
        diffs = tf.expand_dims(activation, 3) - tf.expand_dims(tf.transpose(activation, [1, 2, 0]), 0)
        abs_diffs = tf.reduce_sum(tf.abs(diffs), 2)
        minibatch_features = tf.reduce_sum(tf.exp(-abs_diffs), 2)

        return tf.concat(1, [input, minibatch_features])


def optimizer(loss, var_list, init_lr):
    decay = 0.95
    num_decay_steps = 150
    batch = tf.Variable(0, trainable=False)
    learning_rate = tf.train.exponential_decay(init_lr, batch, num_decay_steps, decay, staircase=True)
    optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=batch, var_list=var_list)
    return optimizer


class GAN(object):
    def __init__(self, data, gen, num_steps, batch_size, minibatch, log_every, anim_path):
        self.data = data
        self.gen = gen
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.minibatch = minibatch
        self.log_every = log_every
        self.mlp_hidden_size = 4
        self.anim_path = anim_path
        self.anim_frames = []

        # can use a higher learning rate when not using the minibatch layer
        if self.minibatch:
            self.learning_rate = 0.005
        else:
            self.learning_rate = 0.03

        self._create_model()

    def _create_model(self):
        with tf.variable_scope('D_pre'):
            self.pre_input = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.pre_labels = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            D_pre = discriminator(self.pre_input, self.mlp_hidden_size, self.minibatch)
            self.pre_loss = tf.reduce_mean(tf.square(D_pre - self.pre_labels))
            self.pre_opt = optimizer(self.pre_loss, None, self.learning_rate)

        with tf.variable_scope('G'):
            self.z = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.G = generator(self.z, self.mlp_hidden_size)

        with tf.variable_scope('D') as scope:
            self.x = tf.placeholder(tf.float32, shape=(self.batch_size, 1))
            self.D1 = discriminator(self.x, self.mlp_hidden_size, self.minibatch)
            scope.reuse_variables()
            self.D2 = discriminator(self.G, self.mlp_hidden_size, self.minibatch)

        self.loss_d = tf.reduce_mean(-tf.log(self.D1) - tf.log(1 - self.D2))
        self.loss_g = tf.reduce_mean(-tf.log(self.D2))

        vars = tf.trainable_variables()
        self.d_pre_params = [v for v in vars if v.name.startswith('D_pre/')]
        self.d_params = [v for v in vars if v.name.startswith('D/')]
        self.g_params = [v for v in vars if v.name.startswith('G/')]

        self.opt_d = optimizer(self.loss_d, self.d_params, self.learning_rate)
        self.opt_g = optimizer(self.loss_g, self.g_params, self.learning_rate)

    def train(self):
        with tf.Session() as session:
            tf.global_variables_initializer().run()

            num_pretrain_steps = 1000
            for step in xrange(num_pretrain_steps):
                d = (np.random.random(self.batch_size) - 0.5) * 10.0  # why not sample from data?
                labels = scipy.stats.norm.pdf(d, loc=self.data.mu, scale=self.data.sigma)
                pretrain_loss, _ = session.run([self.pre_loss, self.pre_opt], {
                    self.pre_input: np.reshape(d, (self.batch_size, 1)),
                    self.pre_labels: np.reshape(labels, (self.batch_size, 1))
                })
            self.weightsD = session.run(self.d_pre_params)

            for i, v in enumerate(self.d_params):
                session.run(v.assign(self.weightsD[i]))

            for step in xrange(self.num_steps):
                x = self.data.sample(self.batch_size)
                z = self.gen.sample(self.batch_size)
                loss_d, _ = session.run([self.loss_d, self.opt_d], {
                    self.x: np.reshape(x, (self.batch_size, 1)),
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                z = self.gen.sample(self.batch_size)
                loss_g, _ = session.run([self.loss_g, self.opt_g], {
                    self.z: np.reshape(z, (self.batch_size, 1))
                })

                if step % self.log_every == 0:
                    print('{}: {}\t{}'.format(step, loss_d, loss_g))
            self._plot_distributions(session)

    def _samples(self, session, num_points=10000, num_bins=100):
        '''
        Return a tuple (db, pd, pg), where db is the current decision
        boundary, pd is a histogram of samples from the data distribution,
        and pg is a histogram of generated samples.
        '''
        xs = np.linspace(-self.gen.range, self.gen.range, num_points)
        bins = np.linspace(-self.gen.range, self.gen.range, num_bins)

        # decision boundary
        db = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            db[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.D1, {
                self.x: np.reshape(
                    xs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })

        # data distribution
        d = self.data.sample(num_points)
        pd, _ = np.histogram(d, bins=bins, density=True)

        # generated samples
        zs = np.linspace(-self.gen.range, self.gen.range, num_points)
        g = np.zeros((num_points, 1))
        for i in range(num_points // self.batch_size):
            g[self.batch_size * i:self.batch_size * (i + 1)] = session.run(self.G, {
                self.z: np.reshape(
                    zs[self.batch_size * i:self.batch_size * (i + 1)],
                    (self.batch_size, 1)
                )
            })
        pg, _ = np.histogram(g, bins=bins, density=True)

        return db, pd, pg

    def _plot_distributions(self, session):
        db, pd, pg = self._samples(session)
        db_x = np.linspace(-self.gen.range, self.gen.range, len(db))
        p_x = np.linspace(-self.gen.range, self.gen.range, len(pd))
        f, ax = plt.subplots(1)
        ax.plot(db_x, db, label='decision boundary')
        ax.set_ylim(0, 1)
        plt.plot(p_x, pd, label='real data')
        plt.plot(p_x, pg, label='generated data')
        plt.title('1D Generative Adversarial Network')
        plt.xlabel('Data values')
        plt.ylabel('Probability density')
        plt.legend()
        plt.show()


def main():
    num_steps = 1200
    batch_size = 12
    minibatch = False
    log_every = 10
    anim_path = None
    gan = GAN(DataDistribution(), GeneratorDistribution(range=8), num_steps, batch_size, minibatch, log_every, anim_path)
    gan.train()

if __name__ == '__main__':
    main()
