import tensorflow as tf
import numpy as np

class Model:
    def __init__(self, learning_rate):
        #placeholder for input batch of board states
        self.states = tf.placeholder(shape=[None, 9], dtype=tf.float32, name='states')
        #placeholder for expected output batch of action values
        self.qsa = tf.placeholder(shape=[None,9], dtype=tf.float32)
        #fully connected layers between states and qsa prediction

        self.fc1 = tf.layers.dense(self.states, 15, activation=tf.nn.relu)
        self.fc2 = tf.layers.dense(self.fc1, 15, activation=tf.nn.relu)
        self.logits = tf.layers.dense(self.fc2, 9)#logits means no activation function like relu
        
        self.action_values = tf.identity(self.logits, name='action_values')
        self.loss = tf.losses.mean_squared_error(self.qsa, self.action_values)
        self.optimizer = tf.train.AdamOptimizer(learning_rate).minimize(self.loss)
        self.var_init = tf.global_variables_initializer()

        self.losses = []#for plotting

    def train_batch(self, state_batch, qsa_batch, sess):
        sess.run(self.optimizer, feed_dict={self.states: state_batch, self.qsa: qsa_batch})
        self.losses.append(np.mean(sess.run(self.loss, feed_dict={self.states: state_batch, self.qsa: qsa_batch})))
        
    #given state, predict value of each action
    def predict_one(self, state, sess):
        return sess.run(self.logits, feed_dict={self.states: state})

    #we need this to estimate cumulative reward
    def predict_batch(self, states, sess):
        return sess.run(self.logits, feed_dict={self.states: states})

    def save(self, sess, name):
        saver = tf.train.Saver()
        saver.save(sess,'C:/Anoop/Python Projects/TicTacToe_DeepQ/' + name)

    def plot_losses(self, fname):
        import matplotlib.pyplot as plt
        plt.plot(self.losses)
        plt.savefig(fname)
        plt.show()
