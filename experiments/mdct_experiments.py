# %%
%load_ext autoreload
%autoreload 2

import numpy as np
from scipy.signal import triang
from scipy.signal.windows import cosine
import matplotlib.pyplot as plt
from filterbank.transforms import synthesisFB, analysisFB, \
    mdct, zeropad, crop
from torch import from_numpy, ones, sum

# %% ####################################################################
# MDCT filter bank with different window functions
fl = 5120
x = from_numpy(triang(fl))
C = 8                                      # Channels
H = from_numpy(mdct(C))                    # Transform
x = zeropad(x,(C,C))
R = 4
windows = {'rect' : ones((H.shape[-1])), # Window functions
           'cos'  : from_numpy(cosine(H.shape[-1]))}
Zxxs = [analysisFB(x,R,window,H) for window \
     in windows.values()]
x_recs = [synthesisFB(Zxx,R,R,window,H) for \
    (Zxx, window) in zip(Zxxs,windows.values())]

for (x_rec, window) in zip(x_recs,windows.keys()):
    assert np.allclose(x_rec[C:-C],x[C:-C], atol=0.000000000001)
    plt.plot(x_rec.real, label=window)
plt.plot(x, label='original', linestyle='--')
plt.legend()

# %% ####################################################################
# Non perfect reconstructing filter bank (Aliasing reduc-
# tion not possible because successive frames do not
# cover aliased area)
fl = 5120
x = from_numpy(triang(fl))
C = 8                                       # Channels
H = from_numpy(mdct(C))                     # Transform
x = zeropad(x,(C,C))
R = 9                     
window = from_numpy(cosine(H.shape[-1]))    # Window function

Zxx = analysisFB(x,R,window,H)
x_rec = synthesisFB(Zxx,R,R,window,H)

#plt.plot(x, label='original')
plt.plot(x_rec, label='x_rec')
plt.plot(x, label='original')
plt.legend()

#%% #####################################################################
#  Band decompostion into 8 bands
N = 1024
n = np.linspace(0,N,N,endpoint=False, dtype=int)
k = 1
C = 8
R = 1
x = from_numpy(np.cos( np.pi / C * \
(n - C//2 + 1 / 2) * (k + 1 / 2)))
H = from_numpy(mdct(C))                                 # Transform

#Apply filter bank
x = zeropad(x,(C,C))
Zxx = analysisFB(x,R,window,H)
x_rec = synthesisFB(Zxx, R, R, window, H, reduction=False)
x_rec = crop(x_rec, (C,C))
x = crop(x,(C,C))

#assert np.allclose(x,sum(x_rec,0))

fig, axes = plt.subplots(4, 1, figsize=(20,20))
axes[0].plot(x_rec[0,...])
axes[1].plot(x_rec[1,...])
axes[2].plot(x_rec[2,...])
axes[3].plot(x-sum(x_rec,0))


# # %% #####################################################################
# # Downsampling and perfect reconstructing filter bank
# # Todo: Why is no downsampling perfectly working at the ends? 
# fl = 5120
# x = from_numpy(triang(fl))
# C = 8                                   # Channels
# H = from_numpy(mdct(C))                                # Transform
# R = 4                                   # Decimation factor
# I = 1                                   # Interpolation factor
# window = ones((H.shape[-1]))            # Window function

# # Expand signal by additional frame for reconstruction (TAC)
# x = zeropad(x,(C,C))

# Zxx = analysisFB(x,R,window,H)
# x_rec = synthesisFB(Zxx,I,R,window,H)
# x = x[C:-C]
# x_rec = x_rec[C:-C]

# assert allclose(x_rec, x[::R//I])
# plt.plot(x_rec, label='x_rec')
# plt.plot(x[::R//I], label='original')
# plt.legend()



# %%

# %%
