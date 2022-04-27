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
N = 5120                                                # Signal length                 
x = from_numpy(triang(N))                               # Signal
C = 8                                                   # Number of channels/frequencies
M = 2*C                                                 # Window length
H = from_numpy(mdct(C))                                 # Transform
R = 4                                                   # Decimation factor
windows = {'rect' : ones((H.shape[-1])),                # Window functions
           'cos'  : from_numpy(cosine(H.shape[-1]))}

# Zero padding (For reconstruction we need to add additional frames at the
# boundary so that alias cancellation does work)
x = zeropad(x,(M-R,M-R))

#Apply filter bank
Zxxs = [analysisFB(x,R,window,H) for window \
     in windows.values()]
x_recs = [synthesisFB(Zxx,R,R,window,H) for \
    (Zxx, window) in zip(Zxxs,windows.values())]

x_recs = [crop(x_rec,(M-R,M-R)) for x_rec in x_recs]
x = crop(x,(M-R,M-R))

for (x_rec, window) in zip(x_recs,windows.keys()):
    assert np.allclose(x_rec,x)
    plt.plot(x_rec.real, label=window)
plt.plot(x, label='original', linestyle='--')
plt.legend()

# %% ####################################################################
# Non perfect reconstructing filter bank (Aliasing reduc-
# tion not possible because successive frames do not
# cover aliased area)
N = 5120                                    # Signal length
x = from_numpy(triang(N))                   # Signal
C = 8                                       # Number of channels/frequencies
M = C*2                                     # Window length
H = from_numpy(mdct(C))                     # Transform
R = 9                                       # Decimation factor
window = from_numpy(cosine(M))              # Window function

x = zeropad(x,(M-R,M-R))

Zxx = analysisFB(x,R,window,H)
x_rec = synthesisFB(Zxx,R,R,window,H)

plt.plot(x_rec, label='x_rec')
plt.plot(x, label='original')
plt.legend()

#%% #####################################################################
#  Band decompostion into 8 bands
N = 1024                                            # Signal length
C = 8                                               # Number of channels/frequencies
M = C*2                                             # Window length
R = 1                                               # Decimation factor
H = from_numpy(mdct(C))                             # Transform
window = ones(H.shape[-1])                          # Window function

# Generate cosine signal which corresponds to one frequency of the
# mdct
n = np.linspace(0,N,N,endpoint=False, dtype=int)
k = 1
x = from_numpy(np.cos( np.pi / C * \
(n - C//2 + 1 / 2) * (k + 1 / 2)))

#Apply filter bank
x = zeropad(x,(C,C))
Zxx = analysisFB(x,R,window,H)
x_rec = synthesisFB(Zxx, R, R, window, H, reduction=False)
x_rec = crop(x_rec, (M-R,M-R))
x = crop(x,(M-R,M-R))

assert np.allclose(x,sum(x_rec,0))

fig, axes = plt.subplots(4, 1, figsize=(20,20))
axes[0].set_title('Band 0')
axes[0].plot(x_rec[0,...])
axes[1].set_title('Band 1')
axes[1].plot(x_rec[1,...])
axes[2].set_title('Band 2')
axes[2].plot(x_rec[2,...])
axes[3].set_title('Diff')
axes[3].plot(x-sum(x_rec,0), label='Diff')

# %% #####################################################################
# Downsampling and perfect reconstructing filter bank
N = 5120                                # Signal length
C = 8                                   # Channels
M = C*2                                 # Window length
H = from_numpy(mdct(C))                 # Transform
R = 4                                   # Decimation factor
I = 2                                   # Interpolation factor
window = ones((M))                      # Window function
x = from_numpy(triang(N))               # Signal

# Expand signal by additional samples for reconstruction (TAC)
x = zeropad(x,(M-R,M-R))

Zxx = analysisFB(x,R,window,H)
x_rec = synthesisFB(Zxx,I,R,window,H)
x = crop(x,(M-R,M-R))
x_rec = crop(x_rec,((M-R)//R*I,(M-R)//R*I))

assert np.allclose(x_rec, x[::R//I])
plt.plot(x_rec, label='x_rec')
plt.plot(x[::R//I], label='original')
plt.legend()



# %%

# %%
