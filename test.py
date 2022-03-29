# %%
import numpy as np
import matplotlib.pyplot as plt
# %%


def uniformPRFB(N, nBands, o, analysisFB):
    # Create analysis filter bank
    H = analysisFB(N, nBands, o)
    # Create synthesis filter bank
    F = synthesisFB(H)
    return H, F


def rectAnalysisFB(N, nBands, o):
    B = N/(nBands-1)
    H = np.zeros((nBands, N))

    # edges
    edges = np.cumsum([B/2] + (nBands-2)*[B])
    starts = np.ceil(np.append([0], edges - o/2)).astype(int)
    stops = np.append(edges + o/2, [-1]).astype(int)

    # Filter range of one channel
    for i, (start, stop) in enumerate(zip(starts, stops)):
        H[i, start:stop] = 1
    H[-1, -1] = 1

    # Increasing and decreasing ramps
    for i, edge in enumerate(edges):
        start = edge-o/2
        stop = edge+o/2
        n = np.arange(int(np.ceil(start)),
                      int(stop))

        # Decreasing ramp
        H[i, n] -= (n-start)/o

        # Increasing ramp
        H[i+1, n] = (n-start)/o

    return H


def synthesisFB(H):
    '''

    '''
    nBands = H.shape[0]

    # Twiddle factor (Aliasing caused by wrap around)
    h = np.fft.irfft(H)
    n = np.linspace(0, h.shape[-1], h.shape[-1], endpoint=False)
    e = np.exp(2j*np.pi*np.arange(nBands)/nBands*n[:, None])
    hm = e.T[:, None, :] * h[None, :, :]
    Hm = np.fft.fft(hm, axis=-1)

    # State underdetermined equation system: Hm * F = c
    Hm = np.reshape(Hm, (nBands, -1))
    c = np.zeros((nBands))
    c[0] = nBands

    # Determine synthesis filters
    F = np.linalg.lstsq(Hm, c, rcond=None)[0]

    assert np.allclose(Hm.astype(complex)@F, c), '''Hm @ F = c is not met!'''

    F = F.reshape(nBands, -1)
    f = np.fft.ifft(F)
    F = np.fft.rfft(f)

    # Todo: Verify this statement
    assert np.max(F.imag) < 0.0000001, '''
    F should not be complex!'''

    return F.real


def polyphaseUPRFB(H, F, R):
    h = np.fft.irfft(H)
    e = np.stack([h[:, i::R] for i in range(R)], axis=1)
    E = np.fft.rfft(e)
    E = np.reshape(E, (E.shape[0], -1))

    f = np.fft.irfft(F)
    f = np.stack([f[:, i::R] for i in range(R)], axis=1)
    R = np.fft.rfft(f)
    R = np.reshape(R, (R.shape[0], -1))

    return E, R


# %%
N = 513
o = 10
nBands = 8
H = rectAnalysisFB(N, nBands, o)
F = synthesisFB(H)
E, R = polyphaseUPRFB(H, F, 8)


# %%
