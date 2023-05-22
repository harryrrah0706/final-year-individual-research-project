###############################################################
# Imports
###############################################################
from sionna.utils import BinarySource, PlotBER, ebnodb2no, insert_dims, log10, expand_to_rank
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.channel import OFDMChannel, AWGN
from sionna.channel import gen_single_sector_topology as gen_topology
from sionna.channel.tr38901 import AntennaArray, CDL, Antenna, UMi, UMa, RMa
from sionna.ofdm import RemoveNulledSubcarriers, ZFPrecoder
from sionna.ofdm import ResourceGrid, ResourceGridMapper, ResourceGridDemapper, LSChannelEstimator, LMMSEEqualizer
from sionna.mimo import StreamManagement
from keras import Model
from keras.layers import Layer, Conv2D, LayerNormalization, Dense
from tensorflow.nn import relu
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay, CosineDecayRestarts, PiecewiseConstantDecay
import time
import pickle
import sionna
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import sionna as sn


###############################################################
# GPU configuration
###############################################################
gpus = tf.config.list_physical_devices('GPU')
print('Number of GPUs available :', len(gpus))
if gpus:
    gpu_num = 0  # Number of the GPU to be used
    try:
        tf.config.set_visible_devices(gpus[gpu_num], 'GPU')
        print('Only GPU number', gpu_num, 'used.')
        tf.config.experimental.set_memory_growth(gpus[gpu_num], True)
    except RuntimeError as e:
        print(e)
###############################################################
# Residual block
###############################################################
class ResidualBlock(Layer):

    def __init__(self):
        super().__init__()

        self.layer_norm_1 = LayerNormalization(axis=(-1, -2, -3))
        self.conv_1 = Conv2D(filters=128,
                              kernel_size=[3, 3],
                              padding='same',
                              activation=None)
        self.layer_norm_2 = LayerNormalization(axis=(-1, -2, -3))
        self.conv_2 = Conv2D(filters=128,
                              kernel_size=[3, 3],
                              padding='same',
                              activation=None)

    def call(self, inputs):
        z = self.layer_norm_1(inputs)
        z = relu(z)
        z = self.conv_1(z)
        z = self.layer_norm_2(z)
        z = relu(z)
        z = self.conv_2(z)
        z = z + inputs

        return z

###############################################################
# Neural receiver
###############################################################
class ResNet(Layer):

    def __init__(self):
        super().__init__()

        self.input_conv = Conv2D(filters=128,
                                  kernel_size=[3, 3],
                                  padding='same',
                                  activation=None)
        self.res_block_1 = ResidualBlock()
        self.res_block_2 = ResidualBlock()
        self.res_block_3 = ResidualBlock()
        self.res_block_4 = ResidualBlock()
        self.res_block_5 = ResidualBlock()
        self.res_block_6 = ResidualBlock()
        self.output_conv = Conv2D(filters=SIMS['num_bits_per_symbol'],
                                   kernel_size=[3, 3],
                                   padding='same',
                                   activation=None)

    def call(self, y):

        nn_input = tf.stack([tf.math.real(y), tf.math.imag(y)], axis=-1)
        nn_input = insert_dims(nn_input, 1, 1)
        z = self.input_conv(nn_input)
        z = self.res_block_1(z)
        z = self.res_block_2(z)
        z = self.res_block_3(z)
        z = self.res_block_4(z)
        z = self.res_block_5(z)
        z = self.res_block_6(z)
        z = self.output_conv(z)
        z = tf.reshape(z, [tf.shape(y)[0], -1]) # [batch size, number of bits per block]

        return z

sionna.config.xla_compat=True
class E2ESystem(Model): # Inherits from Keras Model

    def __init__(self,
                 training,
                 num_bits_per_symbol,
                 evaluation):

        super().__init__() # Must call the Keras model initializer
        
        # System Configuration
        self.num_bits_per_symbol = num_bits_per_symbol
        self.training = training
        self.evaluation = evaluation

        # Transmitter
        self.binary_source = BinarySource()
        self.constellation = Constellation("qam", self.num_bits_per_symbol, trainable=True)
        self.mapper = Mapper(constellation=self.constellation)

        # Channel
        self.awgn_channel = AWGN()
        
        # Receiver
        self.demapper = Demapper("app", constellation=self.constellation)
        self.neural_demapper = ResNet()
        self.bce = BinaryCrossentropy(from_logits=True)



    @tf.function(jit_compile=True)
    def __call__(self, batch_size, ebno_db):

        # no channel coding used; we set coderate=1.0
        no = sn.utils.ebnodb2no(ebno_db,
                                num_bits_per_symbol=self.num_bits_per_symbol,
                                coderate=1.0)
        
        # Transmitter
        bits = self.binary_source([batch_size, 1200])
        x = self.mapper(bits)

        # Channel
        y = self.awgn_channel([x, no])

        # Receiver
        if self.training:
            llr = self.neural_demapper(y)
            loss = self.bce(bits, llr)
            return loss
        elif self.evaluation:
            llr = self.neural_demapper(y)
            return bits, llr
        else:
            llr = self.demapper([y, no])
            return bits, llr
        

# System configuration
iterations = 50000
ebno_db_min = -5
ebno_db_max = 30
batch_size = 256
monte_carlo = 5000
SIMS = {"training" : True,
        "evaluation" : False,
        "num_bits_per_symbol" : 8}



if SIMS['training']:
    model = E2ESystem(training=SIMS["training"],
                        num_bits_per_symbol=SIMS["num_bits_per_symbol"],
                        evaluation=SIMS["evaluation"])
    loss_store = []

    optimizer = tf.keras.optimizers.Adam()
    for i in range(iterations):
        ebno_db = tf.random.uniform(shape=[batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        with tf.GradientTape() as tape:
            loss = model(batch_size, ebno_db)
        grads = tape.gradient(loss, model.trainable_weights)
        optimizer.apply_gradients(zip(grads, model.trainable_weights))
        if i % 100 == 0:
            loss_store.append(loss)
            print(f"{i}/{iterations}  Loss: {loss:.2E}")
    weights = model.get_weights()
    with open(f'awgn_model_weights_default', 'wb') as f:
        pickle.dump(weights, f)
    with open(f'awgn_model_loss_default', 'wb') as f:
        pickle.dump(loss_store, f)

else:
    ber_plots = PlotBER()
    ebno_db = np.linspace(ebno_db_min, ebno_db_max, 100)

    model = E2ESystem(training=SIMS["training"],
                        num_bits_per_symbol=SIMS['num_bits_per_symbol'],
                        evaluation=SIMS["evaluation"])
    if SIMS["evaluation"] == True:
        model(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))
        with open(f'awgn_model_weights_default', 'rb') as f:
            weights = pickle.load(f)
            model.set_weights(weights)

    ber, bler = ber_plots.simulate(model,
                                    ebno_dbs=ebno_db,
                                    batch_size=batch_size,
                                    num_target_block_errors=1000,
                                    soft_estimates=True,
                                    max_mc_iter=monte_carlo,
                                    add_bler=True,
                                    add_ber=True)
    if SIMS["evaluation"] == True:
        with open(f'awgn_neural_receiver_default_BLER', "wb") as f:
            pickle.dump(bler, f)
        with open(f'awgn_neural_receiver_default_BER', "wb") as f:
            pickle.dump(ber, f)
    else:
        with open(f"awgn_baseline_BLER", "wb") as f:
            pickle.dump(bler, f)
        with open(f"awgn_baseline_BER", "wb") as f:
            pickle.dump(ber, f)