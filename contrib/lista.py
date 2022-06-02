import numpy as np

import tensorflow as tf
from tensorflow.keras.layers import Layer, Input
import tensorflow.keras.backend as K

def softshrink(x, theta):
    return tf.keras.activations.relu(x - theta) - tf.keras.activations.relu(-x - theta)

class LISTA_Block(Layer):
    def __init__(self, n_atoms, L, theta, n_iters, positive_code=False, **kwargs):
        super(LISTA_Block, self).__init__(**kwargs)

        self.dense = Dense(n_atoms, activation=None, use_bias=False)

        self.theta = tf.Variable(
            initial_value=theta,
            trainable=True,
            name='theta',
        )

        self.L = L
        self.n_iters = n_iters
        self.positive_code = positive_code

    def call(self, x):
        y = softshrink(x, self.theta)

        if self.n_iters > 1:
            for idx in range(self.n_iters):
                y = softshrink(x +\
                    self.dense(y), self.theta)

        if self.positive_code:
            y = K.clip(y, min_value=0, max_value=np.inf)

        return y

    def get_config(self):
        config = super().get_config()
        config.update(
            {
                'L': self.L,
                'theta': self.theta.numpy(),
                'n_iters': self.n_iters,
                'positive_code': self.positive_code,
            }
        )

        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)