###############################################################
# Imports
###############################################################
from sionna.utils import BinarySource, PlotBER, ebnodb2no, insert_dims, log10, expand_to_rank
from sionna.mapping import Mapper, Demapper, Constellation
from sionna.fec.ldpc.decoding import LDPC5GDecoder
from sionna.fec.ldpc.encoding import LDPC5GEncoder
from sionna.channel import OFDMChannel, AWGN, subcarrier_frequencies, ApplyOFDMChannel, ApplyTimeChannel
from sionna.channel import gen_single_sector_topology as gen_topology, time_lag_discrete_time_channel
from sionna.channel.tr38901 import AntennaArray, CDL, Antenna, UMi, UMa, RMa
from sionna.ofdm import RemoveNulledSubcarriers, ZFPrecoder, OFDMDemodulator, OFDMModulator
from sionna.ofdm import ResourceGrid, ResourceGridMapper, ResourceGridDemapper, LSChannelEstimator, LMMSEEqualizer
from sionna.mimo import StreamManagement
from keras import Model
from keras.layers import Layer, Conv2D, LayerNormalization, Dense
from tensorflow.nn import relu
from keras.losses import BinaryCrossentropy
from keras.optimizers import Adam
from keras.optimizers.schedules.learning_rate_schedule import ExponentialDecay, PolynomialDecay, PiecewiseConstantDecay, CosineDecayRestarts
import time
import pickle
import sionna
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

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
s
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
        self.output_conv = Conv2D(filters=SIMS["num_bits_per_symbol"],
                                   kernel_size=[3, 3],
                                   padding='same',
                                   activation=None)

    def call(self, inputs):

        y, no = inputs
        y = tf.squeeze(y, axis=1)
        no = log10(no)
        y = tf.transpose(y, [0, 2, 3, 1])
        no = insert_dims(no, 3, 1)
        no = tf.tile(no, [1, y.shape[1], y.shape[2], 1])
        z = tf.concat([tf.math.real(y),
                       tf.math.imag(y),
                       no], axis=-1)

        z = self.input_conv(z)
        z = self.res_block_1(z)
        z = self.res_block_2(z)
        z = self.res_block_3(z)
        z = self.res_block_4(z)
        z = self.res_block_5(z)
        z = self.res_block_6(z)
        z = self.output_conv(z)
        z = insert_dims(z, 2, 1)

        return z

###############################################################
# End-to-end System
###############################################################
sionna.config.xla_compat=True
class E2ESystem(Model):

    def __init__(self):
        super().__init__()

        # Provided parameters
        self.scenario = SIMS["scenario"]
        self.perfect_csi = SIMS["perfect_csi"]
        self.training = SIMS["training"]
        self.evaluation = SIMS["evaluation"]
        self.num_bits_per_symbol = SIMS["num_bits_per_symbol"]

        # System parameters
        self.num_ut = 1
        self.num_bs = 1
        self.num_ut_ant = 1
        self.num_bs_ant = 2
        self.carrier_frequency = 3.5e9
        self.subcarrier_spacing = 30e3
        self.speed = 0
        self.fft_size = 144
        self.num_ofdm_symbols = 14
        self.delay_spread = 100e-9
        self.num_guard_carriers = [5, 6]
        self.pilot_ofdm_symbol_indices = [2, 11]
        self.num_streams_per_tx = self.num_ut_ant
        self.dc_null = True
        self.cyclic_prefix_length = 0
        self.pilot_pattern = "kronecker"
        self.coderate = 0.5
        self.rx_tx_association = np.ones([1, self.num_ut])
        self.direction = "uplink"

        self.sm = StreamManagement(self.rx_tx_association,
                                   self.num_streams_per_tx)

        self.rg = ResourceGrid(num_ofdm_symbols=self.num_ofdm_symbols,
                               fft_size=self.fft_size,
                               subcarrier_spacing = self.subcarrier_spacing,
                               num_tx=self.num_ut,
                               num_streams_per_tx=self.num_streams_per_tx,
                               cyclic_prefix_length=self.cyclic_prefix_length,
                               num_guard_carriers=self.num_guard_carriers,
                               dc_null=self.dc_null,
                               pilot_pattern=self.pilot_pattern,
                               pilot_ofdm_symbol_indices=self.pilot_ofdm_symbol_indices)
        
        self.n = int(self.rg.num_data_symbols * self.num_bits_per_symbol)
        self.k = int(self.n * self.coderate)
        
        self.ut_array = Antenna(polarization="single",
                                polarization_type="V",
                                antenna_pattern="38.901",
                                carrier_frequency=self.carrier_frequency)

        self.bs_array = AntennaArray(num_rows=1,
                                     num_cols=int(self.num_bs_ant/2),
                                     polarization="dual",
                                     polarization_type="cross",
                                     antenna_pattern="38.901",
                                     carrier_frequency=self.carrier_frequency)
        
        if self.scenario in ["A", "B", "C", "D", "E"]:
            self.channel_model = CDL(model=self.scenario,
                                     delay_spread=self.delay_spread,
                                     carrier_frequency=self.carrier_frequency,
                                     ut_array=self.ut_array,
                                     bs_array=self.bs_array,
                                     direction=self.direction,
                                     min_speed=self.speed)
        elif self.scenario == "umi":
            self.channel_model = UMi(carrier_frequency=self.carrier_frequency,
                                     o2i_model="low",
                                     ut_array=self.ut_array,
                                     bs_array=self.bs_array,
                                     direction=self.direction,
                                     enable_pathloss=False,
                                     enable_shadow_fading=False)
        elif self.scenario == "uma":
            self.channel_model = UMa(carrier_frequency=self.carrier_frequency,
                                     o2i_model="low",
                                     ut_array=self.ut_array,
                                     bs_array=self.bs_array,
                                     direction=self.direction,
                                     enable_pathloss=False,
                                     enable_shadow_fading=False)
        elif self.scenario == "rma":
            self.channel_model = RMa(carrier_frequency=self.carrier_frequency,
                                     ut_array=self.ut_array,
                                     bs_array=self.bs_array,
                                     direction=self.direction,
                                     enable_pathloss=False,
                                     enable_shadow_fading=False)

        ############################
        ## Transmitter
        ############################
        self.binary_source = BinarySource()
        self.encoder = LDPC5GEncoder(self.k, self.n)
        self.mapper = Mapper("qam", self.num_bits_per_symbol)
        self.rg_mapper = ResourceGridMapper(self.rg)

        ############################
        ## Channel
        ############################    
        self.channel = OFDMChannel(channel_model=self.channel_model,
                                   resource_grid=self.rg,
                                   add_awgn=True,
                                   normalize_channel=True,
                                   return_channel=True)

        ############################
        ## Receiver
        ############################
        self.neural_receiver = ResNet()
        self.ls_est = LSChannelEstimator(self.rg, interpolation_type="nn")
        self.lmmse_equ = LMMSEEqualizer(self.rg, self.sm)
        self.demapper = Demapper("app", "qam", self.num_bits_per_symbol)
        self.rg_demapper = ResourceGridDemapper(self.rg, self.sm)
        self.decoder = LDPC5GDecoder(self.encoder, hard_out=True)
        self.remove_nulled_scs = RemoveNulledSubcarriers(self.rg)
        self.bce = BinaryCrossentropy(from_logits=True)

    def new_topology(self, batch_size):
        topology = gen_topology(batch_size,
                                self.num_ut,
                                self.scenario,
                                min_ut_velocity=0.0,
                                max_ut_velocity=0.0)

        self.channel_model.set_topology(*topology)

    @tf.function(jit_compile=True)
    def call(self, batch_size, ebno_db):
        
        if self.scenario in ["umi", "uma", "rma"]:
            self.new_topology(batch_size)
        
        if len(ebno_db.shape) == 0:
            ebno_db = tf.fill([batch_size], ebno_db)
        no = ebnodb2no(ebno_db, self.num_bits_per_symbol, self.coderate, self.rg)
        
        ############################
        ## Transmitter
        ############################
        if self.training:
            c = self.binary_source([batch_size, self.num_ut, self.num_streams_per_tx, self.n])
        else:
            b = self.binary_source([batch_size, self.num_ut, self.num_streams_per_tx, self.k])
            c = self.encoder(b)
        x = self.mapper(c)
        x_rg = self.rg_mapper(x)

        ############################
        ## Channel
        ############################
        y, h = self.channel([x_rg, no])

        ############################
        ## Receiver
        ############################
        if self.training or self.evaluation:
            llr = self.neural_receiver([y, no])
            llr = self.rg_demapper(llr)
            llr = tf.reshape(llr, [batch_size, self.num_ut, self.num_ut_ant, self.n])
        else:
            if self.perfect_csi:
                h_hat, err_var= self.remove_nulled_scs(h), 0
            else:
                h_hat, err_var = self.ls_est([y, no])
            x_hat, no_eff = self.lmmse_equ([y, h_hat, err_var, no])
            llr = self.demapper([x_hat, no_eff])

        if self.training:
            loss = self.bce(c, llr)
            return loss
        else:
            b_hat = self.decoder(llr)
            return b,b_hat

###############################################################
# Preconfiguring the model
###############################################################
SIMS = {"scenario" : "umi",
        "perfect_csi" : False,
        "training" : True,
        "evaluation" : False,
        "num_bits_per_symbol" : 10}

###############################################################
# Training configuration
###############################################################
num_training_iterations = 50000
batch_size = 128
ebno_db_min = 0
ebno_db_max = 15
monte_carlo = 200
target_block_error = 1000
model = E2ESystem()

###############################################################
# Training the neural receiver
###############################################################
if SIMS["training"] == True:

    optimizer = tf.keras.optimizers.Adam()

    loss_store = []
    for i in range(num_training_iterations):
        ebno_db = tf.random.uniform(shape=[batch_size], minval=ebno_db_min, maxval=ebno_db_max)
        with tf.GradientTape() as tape:
            loss = model(batch_size, ebno_db)
        weights = model.trainable_weights
        grads = tape.gradient(loss, weights)
        optimizer.apply_gradients(zip(grads, weights))
        if i % 100 == 0:
            loss_store.append(loss)
        if i % 1000 == 0:
            print(optimizer.learning_rate)
        print(f"{i}/{num_training_iterations}  Loss: {loss:.2E}")


    weights = model.get_weights()
    with open(f"ofdm_model_weights_{SIMS['scenario']}_{SIMS['num_bits_per_symbol']}", 'wb') as f:
        pickle.dump(weights, f)
    with open(f"ofdm_model_loss_{SIMS['scenario']}_{SIMS['num_bits_per_symbol']}", 'wb') as f:
        pickle.dump(loss_store, f)


###############################################################
# Simulation
###############################################################
else:
    ber_plots = PlotBER()
    ebno_db = np.linspace(ebno_db_min, ebno_db_max, 30)

    if SIMS["evaluation"] == True:
        model(tf.constant(1, tf.int32), tf.constant(10.0, tf.float32))
        with open(f"ofdm_model_weights_{SIMS['scenario']}_{SIMS['num_bits_per_symbol']}", 'rb') as f:
            weights = pickle.load(f)
            model.set_weights(weights)

        ber, bler = ber_plots.simulate(model,
                                       ebno_dbs=ebno_db,
                                       batch_size=batch_size,
                                       num_target_block_errors=target_block_error,
                                       soft_estimates=True,
                                       max_mc_iter=monte_carlo,
                                       add_bler=True,
                                       add_ber=True)
        with open(f"ofdm_neural_receiver_{SIMS['scenario']}_{SIMS['num_bits_per_symbol']}_BLER", "wb") as f:
            pickle.dump(bler, f)
        with open(f"ofdm_neural_receiver_{SIMS['scenario']}_{SIMS['num_bits_per_symbol']}_BER", "wb") as f:
            pickle.dump(ber, f)

    else:
        if SIMS['perfect_csi']:
            state = 'perfect_csi'
        else:
            state = 'ls_estimation'
        ber, bler = ber_plots.simulate(model,
                                       ebno_dbs=ebno_db,
                                       batch_size=batch_size,
                                       num_target_block_errors=target_block_error,
                                       soft_estimates=True,
                                       max_mc_iter=monte_carlo,
                                       add_bler=True,
                                       add_ber=True)
        with open(f"ofdm_{state}_{SIMS['scenario']}_{SIMS['num_bits_per_symbol']}_BLER", "wb") as f:
            pickle.dump(bler, f)
        with open(f"ofdm_{state}_{SIMS['scenario']}_{SIMS['num_bits_per_symbol']}_BER", "wb") as f:
            pickle.dump(ber, f)


plt.show()