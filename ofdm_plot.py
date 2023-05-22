import pickle
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.python.ops.numpy_ops import np_config
np_config.enable_numpy_behavior()

font = {'family' : 'serif',
        'color' : 'black',
        'size' : 20}

font1 = {'family' : 'serif',
        'color' : 'black',
        'size' : 9}

fig, ax = plt.subplots()


step = 30
ebno_db = np.linspace(0, 15, step)

system = "ofdm"
scenario = "A"
ber_or_bler = "BER"
bits = "10"

with open(f"{system}_ls_estimation_{scenario}_{bits}_{ber_or_bler}", "rb") as f:
    ls_estimation = pickle.load(f)

with open(f"{system}_perfect_csi_{scenario}_{bits}_{ber_or_bler}", "rb") as f:
    perfect_csi = pickle.load(f)



with open(f"{system}_neural_receiver_{scenario}_{bits}_{ber_or_bler}", "rb") as f:
    trained = pickle.load(f)

# with open(f"{system}_neural_receiver_{scenario}_{bits}_(ebno)", "rb") as f:
#     trained7 = pickle.load(f)




# with open(f"ofdm_neural_receiver_umi_10_(10with100000)", "rb") as f:
#     trained1 = pickle.load(f)

# with open(f"ofdm_neural_receiver_umi_10_(10ebno)", "rb") as f:
#     trained2 = pickle.load(f)

# with open(f"ofdm_neural_receiver_umi_10_(6with100000)", "rb") as f:
#     trained3 = pickle.load(f)

# with open(f"ofdm_neural_receiver_8_umi_10_BER", "rb") as f:
#     trained4 = pickle.load(f)

# with open(f"ofdm_neural_receiver_16_umi_10_BER", "rb") as f:
#     trained5 = pickle.load(f)

# # with open(f"ofdm_neural_receiver_10_umi_10_BER", "rb") as f:
# #     trained6 = pickle.load(f)




# with open(f"tmp", "rb") as f:
#     tmp = pickle.load(f)

ax.semilogy(ebno_db, ls_estimation, '-',c=f'C0', label=f"LS estimation")
ax.semilogy(ebno_db, perfect_csi, '-', c=f'C1', label=f"Perfect CSI")

ax.semilogy(ebno_db, trained, '-', c=f'C2', label=f"12 RB, 100,000 iter, 10-15 dB")
# ax.semilogy(ebno_db, trained7, '-', c=f'C7', label=f"10")

# ax.semilogy(ebno_db, trained5, '-', c=f'C6', label=f"8 RB, 50,000 iter, 0-15 Eb/No")
# ax.semilogy(ebno_db, trained4, '-', c=f'C5', label=f"12 RB, 50,000 iter, 0-15 Eb/No")
# ax.semilogy(ebno_db, trained3, '-', c=f'C4', label=f"16 RB, 50,000 iter, 0-15 Eb/No")
# ax.semilogy(ebno_db, trained1, '-', c=f'C2', label=f"12 RB, 100,000 iter, 0-15 Eb/No")
# ax.semilogy(ebno_db, trained2, '-', c=f'C3', label=f"12 RB, 100,000 iter, 10-15 Eb/No")
# # ax.semilogy(ebno_db, trained6, '-', c=f'C7', label=f"10")

# ax.semilogy(ebno_db, tmp, '-', c=f'C7', label=f"tmp")


# plt.ylim((1e-4,1))

ax.set_xlabel(r"$E_b/N_0$ (dB)", fontdict=font)
ax.set_ylabel(ber_or_bler, fontdict=font)
ax.grid(which="both")
fig.set_size_inches(6,6)
ax.set_title(f"CDL-A Neural Receivers ({2**int(bits)}-QAM)", fontdict=font)
ax.legend(fontsize=12, loc="lower left")
fig.savefig(f"ofdm_{scenario}_{bits}.png", format="png", dpi=1200)





# with open('ofdm_constellation_untrained', 'rb')as f:
#     x = pickle.load(f).points.tolist()
#     real, imag = [i.imag for i in x], [i.real for i in x]
# ax.scatter(real, imag, s=50)

# with open('awgn_constellation_cyclical', 'rb')as f:
#     x = pickle.load(f).points.tolist()
#     real, imag = [i.real for i in x], [i.imag for i in x]
# ax.scatter(real, imag, s=50, c=f'C1')

# with open('awgn_constellation_6_blocks_exponential', 'rb')as f:
#     x = pickle.load(f).points.tolist()
#     real, imag = [i.real for i in x], [i.imag for i in x]
# ax.scatter(real, imag, s=50, c=f'C1')

# with open('awgn_constellation_6_blocks', 'rb')as f:
#     x = pickle.load(f).points.tolist()
#     real, imag = [i.real for i in x], [i.imag for i in x]
# ax.scatter(real, imag, s=50, c=f'C1')

# with open('awgn_constellation_8_blocks', 'rb')as a:
#     x = pickle.load(a).points.tolist()
#     real, imag = [i.real for i in x], [i.imag for i in x]
# ax.scatter(real, imag, s=50, c=f'C1')

# with open('awgn_constellation_default', 'rb')as a:
#     x = pickle.load(a).points.tolist()
#     real, imag = [i.real for i in x], [i.imag for i in x]
# ax.scatter(real, imag, s=50, c=f'C1')

# with open('awgn_constellation_6_blocks_warm_restart', 'rb')as f:
#     x = pickle.load(f).points.tolist()
#     real, imag = [i.real for i in x], [i.imag for i in x]
# ax.scatter(real, imag, s=50, c=f'C1')


# ax.grid(which="both")
# fig.set_size_inches(6,6)
# # ax.set_xticks([])
# # ax.set_yticks([])
# ax.set_title(f"6 RB, warm restart LR, fixed BS", fontdict=font)
# fig.savefig(f"ofdm_constellation_.png", format="png", dpi=1200)
# # plt.show()