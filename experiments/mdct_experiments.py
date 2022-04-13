# %%
%load_ext autoreload
%autoreload 2

import numpy as np
from scipy.signal import triang
from scipy.signal.windows import cosine
import matplotlib.pyplot as plt
from filterbank.transforms import synthesisFB, analysisFB, \
    mdct

# %%
# MDCT filter bank with different window functions
fl = 5120
x = triang(fl)
C = 8                                      # Channels
H = mdct(C)                                 # Transform

# Expand signal by additional frame for reconstruction (TAC)
x = np.pad(x,(C,C),constant_values=(0,0))

# %%
# MDCT filter bank with different window functions
R = 4
windows = {'rect' : np.ones((H.shape[-1])), # Window functions
           'cos'  : cosine(H.shape[-1])}
Zxxs = [analysisFB(x,R,window,H) for window \
     in windows.values()]
x_recs = [synthesisFB(Zxx,R,R,window,H) for \
    (Zxx, window) in zip(Zxxs,windows.values())]

for (x_rec, window) in zip(x_recs,windows.keys()):
    assert np.allclose(x_rec[C:-C],x[C:-C], atol=0.000000000001)
    plt.plot(x_rec.real, label=window)
plt.plot(x, label='original', linestyle='--')
plt.legend()

# %%
# Non perfect reconstructing filter bank (Aliasing reduc-
# tion not possible because successive frames do not
# cover aliased area)
R = 9                     
window = cosine(H.shape[-1])                       # Window function

Zxx = analysisFB(x,R,window,H)
x_rec = synthesisFB(Zxx,R,R,window,H)

#plt.plot(x, label='original')
plt.plot(x_rec, label='x_rec')
plt.plot(x, label='original')
plt.legend()

# %%
# Downsampling and perfect reconstructing filter bank
# Todo: Why is no downsampling perfectly working at the ends? 
fl = 5120
x = triang(fl)
C = 8                                      # Channels
H = mdct(C)                                # Transform
R = 1                                      # Decimation factor
I = 1                                      # Interpolation factor
window = np.ones((H.shape[-1]))            # Window function

# Expand signal by additional frame for reconstruction (TAC)
x = np.pad(x,(C,C),constant_values=(0,0))

Zxx = analysisFB(x,R,window,H)
x_rec = synthesisFB(Zxx,I,R,window,H)
x = x[C:-C]
x_rec = x_rec[C:-C]

assert np.allclose(x_rec, x[::R//I])
plt.plot(x_rec, label='x_rec')
plt.plot(x[::R//I], label='original')
plt.legend()



# %%
