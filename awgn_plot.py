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


# step = 100
# ebno_db = np.linspace(-5, 30, step)

# system = "awgn"
# ber_or_bler = "BER"

# with open(f"{system}_untrained_{ber_or_bler}", "rb") as f:
#     untrained = pickle.load(f)

# # with open(f"{system}_neural_receiver_default_{ber_or_bler}", "rb") as f:
# #     trained1 = pickle.load(f)

# # with open(f"{system}_neural_receiver_cyclical_{ber_or_bler}", "rb") as f:
# #     trained3 = pickle.load(f)

# # with open(f"{system}_neural_receiver_8_blocks_{ber_or_bler}", "rb") as f:
# #     trained4 = pickle.load(f)

# with open(f"{system}_neural_receiver_6_blocks_{ber_or_bler}", "rb") as f:
#     trained5 = pickle.load(f)

# with open(f"{system}_neural_receiver_6_blocks_warm_restart_{ber_or_bler}", "rb") as f:
#     trained7 = pickle.load(f)

# with open(f"{system}_neural_receiver_6_blocks_exponential_{ber_or_bler}", "rb") as f:
#     trained2 = pickle.load(f)

# with open(f"{system}_baseline_{ber_or_bler}", "rb") as f:
#     baseline = pickle.load(f)

# # with open(f"tmp", "rb") as f:
# #     baseline1 = pickle.load(f)


# ax.semilogy(ebno_db, untrained, '-', c=f'C4', label=f"Untrained")
# # ax.semilogy(ebno_db, trained1, '-', c=f'C0', label=f"Fixed batch size")
# ax.semilogy(ebno_db, trained5, '-', c=f'C0', label=f"6 RB constant LR")
# ax.semilogy(ebno_db, trained2, '-', c=f'C1', label=f"6 RB exponential LR")
# ax.semilogy(ebno_db, trained7, '-', c=f'C2', label=f"6 RB warm restart LR")
# # ax.semilogy(ebno_db, trained3, '-', c=f'C1', label=f"Cyclical decaying batch size")
# # ax.semilogy(ebno_db, trained4, '-', c=f'C2', label=f"8 Residual Blocks")
# ax.semilogy(ebno_db, baseline, '-', c=f'C3', label=f"Conventional baseline")
# # ax.semilogy(ebno_db, baseline1, '-', c=f'C7', label=f"tmp")

# ax.set_xlabel(r"$E_b/N_0$ (dB)", fontdict=font)
# ax.set_ylabel(ber_or_bler, fontdict=font)
# ax.grid(which="both")
# fig.set_size_inches(6,6)
# ax.set_title(f"Neural Reiceivers (256-QAM)", fontdict=font)
# ax.legend(fontsize=15, loc="lower left")
# fig.savefig(f"awgn_neural_receiver_different_lr.png", format="png", dpi=1200)





with open('awgn_constellation_untrained', 'rb')as f:
    x = pickle.load(f).points.tolist()
    real, imag = [i.imag for i in x], [i.real for i in x]
ax.scatter(real, imag, s=50)

# with open('awgn_constellation_cyclical', 'rb')as f:
#     x = pickle.load(f).points.tolist()
#     real, imag = [i.real for i in x], [i.imag for i in x]
# ax.scatter(real, imag, s=50, c=f'C1')

with open('awgn_constellation_6_blocks_exponential', 'rb')as f:
    x = pickle.load(f).points.tolist()
    real, imag = [i.real for i in x], [i.imag for i in x]
ax.scatter(real, imag, s=50, c=f'C1')

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


ax.grid(which="both")
fig.set_size_inches(6,6)
# ax.set_xlabel("real", fontdict=font)
# ax.set_ylabel("imag", fontdict=font)
# ax.set_xticks([])
# ax.set_yticks([])
ax.set_title(f"6 RB, warm restart LR, fixed BS", fontdict=font)
fig.savefig(f"awgn_constellation_6_blocks_warm_restart.png", format="png", dpi=1200)
# plt.show()