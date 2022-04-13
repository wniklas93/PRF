# %%
%load_ext autoreload
%autoreload 2

from numpy import allclose
from scipy.signal import triang
from scipy.signal.windows import cosine
import matplotlib.pyplot as plt
from filterbank.transforms import synthesisFB, analysisFB, \
    dft

from torch import from_numpy, ones

# %%
# Define signal and transform for experiment
fl = 1024
x = from_numpy(triang(fl).astype(complex))
C = 16                                       # Number of channel
H = from_numpy(dft(C))                      # Dft transform matrix

# %%
#Dft filter bank with different window functions
R = 8                                           # Decimation factor                                  
windows = {'rect' : ones(C),                 # Window function
           'cos' : from_numpy(cosine(C))}  
                 
Zxxs = [analysisFB(x,R,window,H) for window \
     in windows.values()]
x_recs = [synthesisFB(Zxx,R,R,window,H.conj()) for \
    (Zxx, window) in zip(Zxxs,windows.values())]

for (x_rec, window) in zip(x_recs,windows):
    assert allclose(x,x_rec)
    plt.plot(x_rec.real, label=window)
plt.plot(x.real, label='original')
plt.legend()

# %%
Rs = [1,2,4,8,16]                                  # Decimation factor
H = from_numpy(dft(C))                                         # Transform matrix
window = ones(C)                                   # Window function
                             
Zxxs = [analysisFB(x,R,window,H) for R \
     in Rs]
x_recs = [synthesisFB(Zxx,R,R,window,H.conj()) for \
    (Zxx, R) in zip(Zxxs,Rs)]

for (x_rec, R) in zip(x_recs,Rs[:-1]):
    assert allclose(x_rec,x)
    plt.plot(x_rec.real, label=str(R))
plt.plot(x.real, label='original')
plt.legend()

# %%
# Non-perfect DFT filter bank because of too great decimation factor
R = 32                                             # Decimation factor
window = ones(C)                                # Window function
                             
Zxx = analysisFB(x,R,window,H)
x_rec = synthesisFB(Zxx,R,R,window,H.conj())

plt.plot(x.real, label='original')
plt.plot(x_rec.real, label='x_rec')
# %%
# DFT filter bank downsampling
R = 8                                             # Decimation factor
I = 1                                             # Interpolation
window = ones(C)                               # Window function
                             
Zxx = analysisFB(x,R,window,H)
x_rec = synthesisFB(Zxx,I,R,window,H.conj())

plt.plot(x[::R].real, label='original', linestyle='--')
plt.plot(x_rec.real, label='x_rec')
plt.legend()
# %%
