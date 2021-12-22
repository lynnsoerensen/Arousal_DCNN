from keras.layers import Layer
import keras.backend as K
from asn.conversion.utils import normalize_transfer
import tensorflow as tf

class ASNTransfer_arousal(Layer):
    """Parametric Adaptive Spiking Neuron Transfer function with added gain on input or output.
    31.08.2020
    """
    def __init__(self, m_f=0.1, threshold=0.0, loc='out', **kwargs):
        self.supports_masking = True
        self.loc = loc

        self.m_f = m_f
        self.tau_gamma = 15.0
        self.tau_eta = 50.0
        self.theta0 = m_f
        self.h = normalize_transfer(m_f) # 0.1244 for m_f= 0.1,
        self.threshold = threshold  # for ReLU cut-off

        self.c1 = 2*self.m_f*self.tau_gamma*self.tau_gamma
        self.c2 = 2*self.theta0*self.tau_eta*self.tau_gamma
        self.c3 = self.tau_gamma*(self.m_f*self.tau_gamma + 2*(self.m_f + 1)*self.tau_eta)
        self.c4 = self.theta0*self.tau_gamma*self.tau_eta + self.theta0*self.tau_eta*self.tau_eta

        self.c0 = self.h/(K.exp((self.c1*0.5*self.theta0 + self.c2)/(self.c3*0.5*self.theta0 + self.c4)) - 1)

        super(ASNTransfer_arousal, self).__init__(**kwargs)

    def call(self, inputs):
        x = inputs[0]
        gain = inputs[1]
        if self.loc == 'in':
            x = x * gain

        reluval = x * K.cast(x >= self.threshold, K.floatx())
        act_val = self.h/(K.exp((self.c1*reluval + self.c2)/(self.c3*reluval + self.c4)) - 1.) - self.c0 + self.h/2.0

        if self.loc == 'out':
            r_act_val = K.relu(act_val) * gain
        else:
            r_act_val = K.relu(act_val)

        return r_act_val

class ReLU_arousal(Layer):
    """ReLU function with added gain on output.
    23.11.2021
    """
    def __init__(self, loc='out', **kwargs):
        self.supports_masking = True
        #self.gain = gain
        self.loc = loc

        super(ReLU_arousal, self).__init__(**kwargs)


    def call(self, inputs):
        x = inputs[0]
        gain = inputs[1]
        if self.loc == 'in':
            #x = x * self.gain
            x = x * gain

        if self.loc == 'out':
            #r_act_val = K.relu(act_val) * self.gain
            r_act_val = K.relu(x) * gain
        else:
            r_act_val = K.relu(x)

        return r_act_val

class ArousalValue(Layer):

    def __init__(self, output_dim, **kwargs):
       self.output_dim = output_dim
       super(ArousalValue, self).__init__(**kwargs)

    def build(self, input_shapes):
       self.kernel = self.add_weight(name='kernel', shape=self.output_dim, initializer='ones', trainable=True)
       super(ArousalValue, self).build(input_shapes)

    def call(self, inputs):
       return self.kernel

    def compute_output_shape(self, input_shape):
       return self.output_dim


class spatialJumble(Layer):
    """
    This layers spatially jumbles all activations, while preserving the channel distributions
    """

    def __init__(self, rate=1.0, **kwargs):
        self.rate = rate
        super(spatialJumble, self).__init__(**kwargs)

    def call(self, inputs, **kwargs):
        shape = inputs.get_shape().as_list()  # a list: [None, 9, 2]
        dim = shape[1] * shape[2]

        idx = tf.range(start=0, limit=shape[3], dtype=tf.int32)
        idx = tf.random_shuffle(idx)
        idx_sel = tf.reshape(idx[:tf.cast(tf.floor(self.rate * shape[3]), dtype=tf.int32)], (-1,1))

        # make a boolean idx
        idx_vector = tf.scatter_nd(idx_sel, tf.ones(idx_sel.shape[0]), tf.constant([shape[3]], dtype=tf.int32))

        out = tf.random_shuffle(tf.reshape(inputs[:, :, :, 0], [dim, -1, 1])) * idx_vector[0] + tf.reshape(inputs[:, :, :, 0], [dim, -1, 1]) * (1 - idx_vector[0])

        # Loop over all channels
        for f in range(1, shape[3]):
            out = K.concatenate((out, tf.random_shuffle(tf.reshape(inputs[:, :, :, f], [dim, -1, 1])) * idx_vector[f] + tf.reshape(inputs[:, :, :, f], [dim, -1, 1]) * (1 - idx_vector[f])),
                                 axis=2)

        # Reformat back to the original shape
        out = tf.reshape(out, [-1, shape[1], shape[2], shape[3]])

        return out

